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
    parser.add_argument("--no-synonyms", action="store_true", help="Disable synonyms")
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
        "--sim-use-columns",
        type=int,
        nargs="+",
        default=None,
        help="Additional columns to index for similarity index",
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
        QGramIndex.build(
            args.input_file, args.index_dir, use_synonyms=not args.no_synonyms
        )
    elif args.type == "prefix":
        PrefixIndex.build(
            args.input_file,
            args.index_dir,
            score=args.prefix_score,
            use_synonyms=not args.no_synonyms,
        )
    else:
        SimilarityIndex.build(
            args.input_file,
            args.index_dir,
            precision=args.sim_precision,
            batch_size=args.sim_batch_size,
            embedding_dim=args.sim_dimensions,
            use_synonyms=not args.no_synonyms,
            use_columns=args.sim_use_columns,
            show_progress=True,
        )
