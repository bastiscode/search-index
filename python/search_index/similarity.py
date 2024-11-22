import os
import numpy as np
import json
import random
from itertools import islice
from typing import Iterator

import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from search_index import SearchIndex, normalize
from search_index._internal import IndexData


def select_faiss_index(d: int, n: int) -> tuple[str, faiss.Index]:
    """

    Selects the appropriate Faiss index for the given number of
    dimensions and datapoints.

    """
    if False and n < 100_000:
        return "Flat", faiss.IndexIDMap2(
            faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
        )

    elif n < 1_000_000:
        n_clusters = round(4 * n**0.5)
        index = f"IVF{n_clusters},Flat"

    elif n <= 10_000_000:
        index = "IVF65536_HNSW32,Flat"

    elif n <= 100_000_000:
        index = "IVF262144_HNSW32,Flat"

    else:
        index = "IVF1048576_HNSW32,Flat"

    return index, faiss.index_factory(d, index, faiss.METRIC_INNER_PRODUCT)


def select_faiss_binary_index(d: int, n: int) -> tuple[str, faiss.IndexBinary]:
    """

    Selects the appropriate Faiss binary index for the given number of datapoints.

    """
    if False and n < 100_000:
        return "BFlat", faiss.IndexBinaryIDMap2(faiss.index_binary_factory(d, "BFlat"))

    elif n < 1_000_000:
        n_clusters = round(4 * n**0.5)
        index = f"BIVF{n_clusters}"

    elif n < 10_000_000:
        index = "BIVF65536_HNSW32"

    elif n < 100_000_000:
        index = "BIVF262144_HNSW32"

    else:
        index = "BIVF1048576_HNSW32"

    return index, faiss.index_binary_factory(d, index)


class EmbeddingModel:
    def __init__(
        self,
        model: str,
        device: str,
        precision: str,
        embedding_dim: int | None = None,
    ):
        assert precision in ["float32", "ubinary"], "invalid precision"
        self.encoder = SentenceTransformer(
            model,
            device=device,
        )
        self.precision = precision
        self.dim = self.encoder.get_sentence_embedding_dimension()
        assert self.dim is not None, "unable to get embedding dimension"
        if embedding_dim is not None and embedding_dim < self.dim:
            self.dim = embedding_dim

        if self.precision == "ubinary":
            assert self.dim % 8 == 0, "embedding dimension must be a multiple of 8"
            self._dim = self.dim // 8
        else:
            self._dim = self.dim

    def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> torch.Tensor:
        return self.encoder.encode(
            texts,
            batch_size=batch_size or len(texts),
            normalize_embeddings=True,
            precision=self.precision,
        )[:, : self._dim]


class SimilarityIndex(SearchIndex):
    def __init__(
        self, model: EmbeddingModel, data: IndexData, index: faiss.Index
    ) -> None:
        self.model = model
        self.data = data
        self.index = index

    @staticmethod
    def build(
        data_file: str,
        index_dir: str,
        use_synonyms: bool = True,
        use_columns: tuple[int, ...] = (),
        model: str | None = None,
        embedding_dim: int | None = None,
        batch_size: int = 32,
        device: str = "cuda",
        precision: str = "float32",
        show_progress: bool = False,
    ) -> None:
        """

        Builds the index from the given file and saves
        it in the index dir.

        The file should contain one record per line, in the following format:
            name\tscore\tsynonyms\tinfo1\tinfo2\t...

        Synonyms are expected to be separated by three semicolons.

        An example line:
            Albert Einstein\t275\tEinstein;;;A. Einstein\tGerman physicist\t
        """
        if model is None:
            if embedding_dim is None:
                model = "mixedbread-ai/mxbai-embed-large-v1"
            else:
                model = "mixedbread-ai/mxbai-embed-2d-large-v1"

        emb_model = EmbeddingModel(model, device, precision, embedding_dim)
        data = IndexData(data_file)

        # calculate index size
        index_size = len(data) * (1 + len(use_columns))
        # add synonyms to index size
        # estimate avg. number of synonyms per record
        # by randomly sampling up to 1000 records
        if use_synonyms:
            synonyms = []
            for idx in random.sample(range(len(data)), min(len(data), 1000)):
                syns: str = data.get_val(idx, 2)
                if syns:
                    synonyms.append(syns.count(";;;") + 1)
                else:
                    synonyms.append(0)

            avg_synonyms = sum(synonyms) / max(1, len(synonyms))
            index_size += round(avg_synonyms * len(data))

        def data_iter(
            desc: str,
            indices: tuple[Iterator[int], int] | None = None,
        ) -> tuple[int, str]:
            if indices is not None:
                index_iter, total = indices
            else:
                index_iter = range(len(data))
                total = len(data)

            for i in tqdm(
                index_iter,
                total=total,
                desc=desc,
                disable=not show_progress,
                leave=False,
            ):
                split = data.get_row(i).split("\t")
                yield i, normalize(split[0])

                if use_synonyms:
                    for synonym in split[2].split(";;;"):
                        yield i, normalize(synonym)

                for col in use_columns:
                    assert col < len(split), f"column {col} out of range"
                    yield i, normalize(split[col])

        if precision == "float32":
            index_name, index = select_faiss_index(emb_model.dim, index_size)
        else:
            index_name, index = select_faiss_binary_index(emb_model.dim, index_size)

        if "IVF" in index_name:
            if faiss.get_num_gpus() > 0:
                # try moving to GPU, but can fail on older GPUs
                try:
                    clustering_index = faiss.index_cpu_to_all_gpus(
                        faiss.IndexFlat(index.d, faiss.METRIC_INNER_PRODUCT)
                    )
                    index.clustering_index = clustering_index
                except Exception:
                    pass

            train_texts = []
            train_embeddings = []
            cp = index.cp
            # interpolate number of training samples between mean and max
            # of clustering parameters
            num_train_samples = min(
                index_size, cp.min_points_per_centroid * index.nlist
            )
            data_samples = min(len(data), num_train_samples)
            for id, text in islice(
                data_iter(
                    "Getting train embeddings",
                    (
                        random.sample(range(len(data)), data_samples),
                        data_samples,
                    ),
                ),
                num_train_samples,
            ):
                train_texts.append(text)
                if len(train_texts) < batch_size:
                    continue

                train_embeddings.append(emb_model.embed(train_texts))
                train_texts.clear()

            if len(train_texts) > 0:
                train_embeddings.append(emb_model.embed(train_texts))

            index.train(np.concatenate(train_embeddings))

        ids = []
        texts = []
        for id, text in data_iter("Indexing data"):
            ids.append(id)
            texts.append(text)
            if len(ids) < batch_size:
                continue

            embeddings = emb_model.embed(texts)
            index.add_with_ids(embeddings, ids)

            ids.clear()
            texts.clear()

        if len(ids) > 0:
            embeddings = emb_model.embed(texts)
            index.add_with_ids(embeddings, ids)

        os.makedirs(index_dir, exist_ok=True)
        index_file = os.path.join(index_dir, "faiss.index")
        if precision == "float32":
            faiss.write_index(index, index_file)
        else:
            faiss.write_index_binary(index, index_file)

        with open(os.path.join(index_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "model": model,
                    "precision": precision,
                    "embedding_dim": embedding_dim,
                },
                f,
            )

    @staticmethod
    def load(data_file: str, index_dir: str, device: str = "cuda") -> "SimilarityIndex":
        """

        Loads the index from the given data file and index directory.

        """
        data = IndexData(data_file)

        with open(os.path.join(index_dir, "config.json")) as f:
            config = json.load(f)

        index_file = os.path.join(index_dir, "faiss.index")
        if config["precision"] == "float32":
            index = faiss.read_index(index_file)
        else:
            index = faiss.read_index_binary(index_file)

        model = EmbeddingModel(**config, device=device)
        return SimilarityIndex(model, data, index)

    def find_matches(
        self, query: str, k: int = 10, nprobe: int = 16
    ) -> list[tuple[int, float]]:
        """

        Returns a sorted list of tuples containing IDs
        and ranking key for all matches for the given query.

        """
        query_embeddings = self.model.embed([query])
        if isinstance(self.index, (faiss.IndexIVF, faiss.IndexBinaryIVF)):
            self.index.nprobe = min(nprobe, self.index.nlist)
        scores, indices = self.index.search(query_embeddings, k)

        # deduplicate based on id
        seen = set()
        deduped = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                break
            elif index in seen:
                continue
            seen.add(index)
            deduped.append((index, score))

        return deduped

    def get_name(self, id: int) -> str:
        """

        Returns the name for the given ID.

        """
        return self.data.get_val(id, 0)

    def get_row(self, id: int) -> str:
        """

        Returns the line from the data file for the given ID.
        ID must be between 0 and the index length.

        """
        return self.data.get_row(id)

    def get_val(self, id: int, col: int) -> str:
        """

        Returns the column value for the given ID.

        """
        return self.data.get_val(id, col)

    def sub_index_by_ids(self, ids: list[int]) -> "SimilarityIndex":
        """

        Creates a sub-index containing only the given IDs.

        """
        return self

    def __len__(self) -> int:
        """

        Returns the number of items in the index.

        """
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        """

        Iterates over the index data.

        """
        yield from self.data

    def get_type(self) -> str:
        """

        Returns the type of the index.

        """
        return "similarity"
