## Python package implementing search indices backed by Rust

### Installation

Install from PyPI (only built for Python 3.10+ on Linux)

```
pip install search-index
```

### Manual installation

Make sure you have Rust installed, then do the following:

```bash
git clone https://github.com/bastiscode/search-index.git
cd search-index
pip install maturin[patchelf]
maturin develop --release
```

### Usage

See `build.py` and `query.py` for examples on how to build and query an index.

#### Search indices

Q-gram index using either
    - infix edit distance
    - prefix edit distance

Prefix keyword index using either
    - keyword occurrence
    - keyword count
    - tf-idf
    - bm25

Similarity index using either
    - brute force search
    - inverted index search
    - inverted index + HNSW search

All search indices also support searching only in a subset of the indexed records.
