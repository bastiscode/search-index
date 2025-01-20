import json
import os
import random
from typing import Iterable, Iterator

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange

from search_index import SearchIndex, normalize
from search_index._internal import IndexData


def select_faiss_index(d: int, n: int) -> tuple[str, faiss.Index]:
    """

    Selects the appropriate Faiss index for the given number of
    dimensions and datapoints.

    """
    if n < 1_000_000:
        return "Flat", faiss.IndexIDMap2(
            faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
        )

    n_clusters = round(4 * n**0.5)
    index = f"IVF{n_clusters}"
    if n_clusters >= 2**16:
        # use HNSW32 instead of flat quantizer for large number of clusters
        index += "_HNSW32"
    index += ",Flat"
    return index, faiss.index_factory(d, index, faiss.METRIC_INNER_PRODUCT)


def select_faiss_binary_index(d: int, n: int) -> tuple[str, faiss.IndexBinary]:
    """

    Selects the appropriate Faiss binary index for the given number of datapoints.

    """
    if n < 1_000_000:
        return "BFlat", faiss.IndexBinaryIDMap2(faiss.index_binary_factory(d, "BFlat"))

    n_clusters = round(4 * n**0.5)
    index = f"BIVF{n_clusters}"
    if n_clusters >= 2**16:
        # use HNSW32 instead of flat quantizer for large number of clusters
        index += "_HNSW32"
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
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim))
        if batch_size is None:
            batch_size = len(texts)
        # sort texts by length to minimize padding
        indices = np.argsort([-len(text) for text in texts])
        sorted_texts = [texts[i] for i in indices]
        full_embeddings = []
        # doing our own loop here because sentence transformers
        # only converts to target precision at the end, which
        # might OOM for large datasets
        for i in trange(
            0,
            len(sorted_texts),
            batch_size,
            desc="Calculating embeddings",
            disable=not show_progress,
        ):
            batch = sorted_texts[i : i + batch_size]
            embeddings = self.encoder.encode(
                batch,
                normalize_embeddings=True,
                batch_size=len(batch),
                precision=self.precision,
                show_progress_bar=False,
            )[:, : self._dim]
            full_embeddings.extend(embeddings)

        embeddings = np.vstack(full_embeddings)
        inv_indices = np.argsort(indices)
        # make sure inv indices correctly restores the original order
        assert all(t == sorted_texts[i] for t, i in zip(texts, inv_indices))
        return embeddings[inv_indices]


class SimilarityIndex(SearchIndex):
    def __init__(
        self,
        model: EmbeddingModel,
        data: IndexData,
        index: faiss.Index,
        index_name: str,
        subset: set[int] | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.index = index
        self.index_name = index_name
        self.subset = subset

    @staticmethod
    def build(
        data_file: str,
        index_dir: str,
        use_synonyms: bool = True,
        use_columns: tuple[int, ...] | None = None,
        model: str | None = None,
        embedding_dim: int | None = None,
        batch_size: int = 32,
        device: str = "cuda",
        train_on_gpu: bool = False,
        precision: str | None = None,
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
        data = IndexData(data_file)

        def data_iter(indices: Iterable[int] | None = None) -> tuple[int, list[str]]:
            if indices is None:
                indices = range(len(data))

            for i in indices:
                split = data.get_row(i).split("\t")
                text = [normalize(split[0])]

                if use_synonyms:
                    for synonym in split[2].split(";;;"):
                        if synonym:
                            text.append(normalize(synonym))

                if not use_columns:
                    yield i, text
                    continue

                for col in use_columns:
                    assert col > 2, (
                        "column index must be greater than 2, because "
                        "0, 1, and 2 are reserved for name, score, and synonyms"
                    )
                    assert col < len(split), f"column {col} out of range"
                    if split[col]:
                        text.append(normalize(split[col]))

                yield i, text

        # calculate index size
        index_size = sum(len(text) for _, text in data_iter())

        # set some sensible defaults
        if precision is None:
            # set precision based on index size
            precision = "float32" if index_size < 1_000_000 else "ubinary"

        if model is None:
            if embedding_dim is None:
                model = "mixedbread-ai/mxbai-embed-large-v1"
            else:
                model = "mixedbread-ai/mxbai-embed-2d-large-v1"

        emb_model = EmbeddingModel(model, device, precision, embedding_dim)

        if precision == "float32":
            index_name, index = select_faiss_index(emb_model.dim, index_size)
        else:
            index_name, index = select_faiss_binary_index(emb_model.dim, index_size)

        if show_progress:
            print(
                f"Building a {index_name} index for {len(data):,} records "
                f"with a total of {index_size:,} entries"
            )

        added_ids = set()
        if "IVF" in index_name:
            if faiss.get_num_gpus() > 0 and train_on_gpu:
                if show_progress:
                    print(
                        f"Setting up clustering index on {faiss.get_num_gpus()} GPUs "
                        "for training"
                    )
                try:
                    ci = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(index.d))
                    index.clustering_index = ci
                except Exception as e:
                    print(f"Failed to move clustering index to GPUs: {e}")

            train_ids = []
            train_texts = []
            train_size = min(
                index_size, round(1.1 * index.cp.min_points_per_centroid * index.nlist)
            )
            train_factor = train_size / index_size
            data_samples = int(train_factor * len(data))

            train_samples = random.sample(range(len(data)), data_samples)
            for id, text in tqdm(
                data_iter(train_samples),
                desc="Getting train data",
                total=data_samples,
                disable=not show_progress,
            ):
                train_ids.extend((id for _ in range(len(text))))
                train_texts.extend(text)

            train_embeddings = emb_model.embed(
                train_texts,
                batch_size=batch_size,
                show_progress=show_progress,
            )

            if show_progress:
                print(
                    f"Training {index_name} index with {index.nlist:,} clusters on "
                    f"{len(train_embeddings):,} embeddings from {data_samples:,} records"
                )
            index.train(train_embeddings)

            # add train embeddings to index
            index.add_with_ids(train_embeddings, train_ids)
            added_ids.update(train_ids)

        if len(added_ids) < len(data):
            index_ids = []
            index_texts = []
            for id, text in tqdm(
                data_iter(),
                desc="Getting index data",
                total=len(data),
                disable=not show_progress,
            ):
                if id in added_ids:
                    continue
                index_ids.extend((id for _ in range(len(text))))
                index_texts.extend(text)

            embeddings = emb_model.embed(
                index_texts,
                batch_size=batch_size,
                show_progress=show_progress,
            )
            index.add_with_ids(embeddings, index_ids)

        os.makedirs(index_dir, exist_ok=True)
        index_file = os.path.join(index_dir, "faiss.index")
        if precision == "float32":
            faiss.write_index(index, index_file)
        else:
            faiss.write_index_binary(index, index_file)

        with open(os.path.join(index_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "index_name": index_name,
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

        index_name = config.pop("index_name")
        model = EmbeddingModel(**config, device=device)
        return SimilarityIndex(model, data, index, index_name)

    def find_matches(
        self,
        query: str,
        k: int = 10,
        nprobe: int = 10,
        min_score: float = float("-inf"),
    ) -> list[tuple[int, float]]:
        """

        Returns a sorted list of tuples containing IDs
        and ranking key for all matches for the given query.

        """
        # we want to scale k because we might have ids in the top k
        # results that point to the same data point, in which case
        # we get less than k unique results; this is an approximation
        # to scale k based on the number of indexed vectors per data point
        k_factor = self.index.ntotal / max(1, len(self.data))
        # scale also by 2 to be sure
        k_scaled = round(k * k_factor * 2)

        if self.subset is not None:
            selector = faiss.IDSelectorBatch(list(self.subset))
        else:
            selector = None

        is_ivf = "IVF" in self.index_name
        is_binary = self.index_name.startswith("B")
        assert is_binary == (
            self.model.precision == "ubinary"
        ), "Model and index mismatch"

        search_kwargs = {}
        if is_binary:
            if is_ivf:
                # ivf binary index
                # does not support search params yet, so set nprobe directly
                self.index.nprobe = nprobe

            # selector not yet supported for binary indices, handled below
            # in deduplication currently
        else:
            if is_ivf:
                # ivf float index
                search_kwargs["params"] = faiss.SearchParametersIVF(
                    sel=selector, nprobe=nprobe
                )
            else:
                # flat float index
                search_kwargs["params"] = faiss.SearchParameters(sel=selector)

        query_embeddings = self.model.embed([query])
        scores, indices = self.index.search(query_embeddings, k_scaled, **search_kwargs)

        # deduplicate based on id
        seen = set()
        deduped = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                break
            elif index in seen:
                continue
            # this is required because binary indices do not support
            # ID selectors yet, so we might get indices outside the subset
            elif self.subset is not None and index not in self.subset:
                continue
            elif len(deduped) >= k:
                break
            elif score < min_score:
                # break because scores are sorted
                break

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
        assert all(0 <= id < len(self) for id in ids), "invalid ID in ID list"
        if self.subset is not None:
            subset = self.subset.intersection(ids)
        else:
            subset = set(ids)

        return SimilarityIndex(
            self.model,
            self.data,
            self.index,
            self.index_name,
            subset,
        )

    def __len__(self) -> int:
        """

        Returns the number of items in the index.

        """
        if self.subset is not None:
            return len(self.subset)
        else:
            return len(self.data)

    def __iter__(self) -> Iterator[str]:
        """

        Iterates over the index data.

        """
        if self.subset is not None:
            for id in sorted(self.subset):
                yield self.data.get_row(id)
        else:
            yield from self.data

    def get_type(self) -> str:
        """

        Returns the type of the index.

        """
        return "similarity"
