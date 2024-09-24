import argparse

from search_index import PrefixIndex, QGramIndex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument("index_dir", type=str, help="Index directory")
    parser.add_argument("--type", type=str, default="prefix",
                        choices=["prefix", "qgram"],
                        help="Index type (prefix or qgram)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.type == "qgram":
        QGramIndex.build(args.input_file, args.index_dir)
    else:
        PrefixIndex.build(args.input_file, args.index_dir)
