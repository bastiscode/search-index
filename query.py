import argparse
import time

from search_index import PrefixIndex, QGramIndex, SimilarityIndex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument("index_dir", type=str, help="Index directory")
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument(
        "--type",
        type=str,
        default="prefix",
        choices=["prefix", "qgram", "similarity"],
        help="Index type",
    )
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument(
        "-n", type=int, default=10, help="Number of benchmark iterations"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    kwargs = {}
    if args.type == "qgram":
        idx = QGramIndex.load(args.input_file, args.index_dir)
    elif args.type == "prefix":
        idx = PrefixIndex.load(args.input_file, args.index_dir)
    else:
        idx = SimilarityIndex.load(args.input_file, args.index_dir)

    print(f"Loaded index with {len(idx):,} records.")

    matches = idx.find_matches(args.query, **kwargs)
    print(f"Found {len(matches):,} matches.")
    for i, (id, key) in enumerate(matches[:5]):
        if i > 0:
            print()
        print(
            f"{i + 1}. match: id={id}, key={key}\n{idx.get_name(id)} "
            f"{idx.get_val(id, 3)}"
        )

    if args.bench:
        print()
        print(f"Benchmarking index for {args.n} iterations...")
        start = time.perf_counter()
        for _ in range(args.n):
            _ = idx.find_matches(args.query, **kwargs)
        end = time.perf_counter()
        diff_ms = (end - start) * 1000
        print(f"Took {diff_ms / args.n:.2f}ms on average")

    sub_idx = idx.sub_index_by_ids([id for id, _ in matches])
    print(
        f"\nBuilt sub index with {len(sub_idx):,} entries from "
        f"{len(matches):,} matches."
    )
    sub_matches = sub_idx.find_matches(args.query, **kwargs)
    assert set(sub_matches) == set(matches), (
        f"Sub index matches ({len(sub_matches):,}) differ from "
        f"original matches ({len(matches):,})"
    )
    for i, (id, key) in enumerate(sub_matches[:5]):
        if i > 0:
            print()
        print(
            f"{i + 1}. sub match: id={id}, key={key}\n{sub_idx.get_name(id)} "
            f"{sub_idx.get_val(id, 3)}"
        )

    if args.bench:
        print()
        print(f"Benchmarking sub index for {args.n} iterations...")
        start = time.perf_counter()
        for _ in range(args.n):
            _ = sub_idx.find_matches(args.query, **kwargs)
        end = time.perf_counter()
        diff_ms = (end - start) * 1000
        print(f"Took {diff_ms / args.n:.2f}ms on average")
