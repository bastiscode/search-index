from importlib import metadata

from search_index._internal import (
    IndexData,
    Mapping,
    PrefixIndex,
    normalize,
)
from search_index.base import SearchIndex  # noqa
from search_index.similarity import SimilarityIndex  # noqa

__all__ = [
    "IndexData",
    "Mapping",
    "PrefixIndex",
    "SearchIndex",
    "SimilarityIndex",
    "normalize",
]


try:
    __version__ = metadata.version("search_index")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
