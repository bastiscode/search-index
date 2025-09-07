import argparse
import logging
import os
import time

from search_index import IndexData, PrefixIndex, SimilarityIndex
from search_index import __version__ as version


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
    parser.add_argument(
        "--prefix-min-keyword-length",
        type=int,
        default=None,
        help="Minimum keyword length for prefix index",
    )
    parser.add_argument(
        "--prefix-no-refinement",
        action="store_true",
        help="Disable refinement step for prefix index",
    )
    parser.add_argument(
        "--sim-nprobe",
        type=int,
        default=10,
        help="Number of clusters to check for similarity index",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Run benchmark",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--offsets-file",
        type=str,
        default=None,
        help="Offsets file (default: <index_dir>/data.offsets)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Set log level",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.log_level:
        logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
        logging.getLogger().setLevel(args.log_level.upper())

    if args.offsets_file is None:
        args.offsets_file = os.path.join(args.index_dir, "data.offsets")

    data = IndexData.load(args.input_file, args.offsets_file)

    kwargs = {}
    if args.type == "prefix":
        idx = PrefixIndex.load(data, args.index_dir)
        if int(version.split(".")[1]) <= 3:
            kwargs["min_keyword_length"] = args.prefix_min_keyword_length
            kwargs["no_refinement"] = args.prefix_no_refinement
    else:
        idx = SimilarityIndex.load(data, args.index_dir)
        kwargs["nprobe"] = args.sim_nprobe

    print(f"Loaded index with {len(idx):,} records.")

    matches = idx.find_matches(args.query, **kwargs)
    print(f"Found {len(matches):,} matches.")
    for i, (id, score, col) in enumerate(matches[:5]):
        if i > 0:
            print()
        print(
            f"{i + 1}. match: id={id}, score={score}, via={idx.get_val(id, col)}\n"
            f"{idx.get_name(id)} {idx.get_identifier(id)}"
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

    sub_idx = idx.sub_index_by_ids([id for id, *_ in matches])
    print(
        f"\nBuilt sub index with {len(sub_idx):,} entries from "
        f"{len(matches):,} matches."
    )
    sub_matches = sub_idx.find_matches(args.query, **kwargs)
    assert set(sub_matches) == set(matches), (
        f"Sub index matches ({len(sub_matches):,}) differ from "
        f"original matches ({len(matches):,})"
    )
    for i, (id, score, col) in enumerate(sub_matches[:5]):
        if i > 0:
            print()
        print(
            f"{i + 1}. sub match: id={id}, key={score}, via={sub_idx.get_val(id, col)}\n"
            f"{sub_idx.get_name(id)} {sub_idx.get_identifier(id)}"
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
