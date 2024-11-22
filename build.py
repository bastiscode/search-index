import argparse

from search_index import PrefixIndex, QGramIndex, SimilarityIndex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument("index_dir", type=str, help="Index directory")
    parser.add_argument(
        "--type",
        type=str,
        default="prefix",
        choices=["prefix", "qgram", "similarity"],
        help="Index type",
    )
    parser.add_argument(
        "--prefix-score",
        type=str,
        choices=["occurrence", "bm25", "tfidf", "count"],
        default="occurrence",
        help="Scoring function for prefix index",
    )
    parser.add_argument(
        "--sim-model",
        type=str,
        default=None,
        help="Model for similarity index",
    )
    parser.add_argument(
        "--sim-precision",
        type=str,
        default="float32",
        choices=["float32", "ubinary"],
        help="Precision for similarity index",
    )
    parser.add_argument(
        "--sim-dimensions",
        type=int,
        default=None,
        help="Number of dimensions for similarity index",
    )
    parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=64,
        help="Batch size for similarity index",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.type == "qgram":
        QGramIndex.build(args.input_file, args.index_dir)
    elif args.type == "prefix":
        PrefixIndex.build(args.input_file, args.index_dir, score=args.prefix_score)
    else:
        SimilarityIndex.build(
            args.input_file,
            args.index_dir,
            precision=args.sim_precision,
            batch_size=args.sim_batch_size,
            embedding_dim=args.sim_dimensions,
            show_progress=True,
        )
