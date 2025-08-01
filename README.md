# Python package implementing search indices backed by Rust

## Installation

Make sure you have
[Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
installed.

Install from PyPI (only built for Python 3.12+ on Linux)

```
pip install search-index
```

## Manual installation

Make sure you have Faiss and Rust installed, then do the following:

```bash
git clone https://github.com/bastiscode/search-index.git
cd search-index
pip install maturin[patchelf]
maturin develop --release
```

## Usage

See `build.py` and `query.py` for examples on how to build and query an index.

## Search indices

Prefix keyword index with one of the following score functions:

- keyword occurrence

Similarity index with one of the following index types:

- flat index (brute-force search)
- inverted index (bucket search)
- inverted index + HNSW

Both search indices also support searching only in a subset of the indexed records.
