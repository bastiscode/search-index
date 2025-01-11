from typing import Any, Iterator

from search_index.base import SearchIndex

def normalize(s: str) -> str:
    """

    Normalizes the given string.

    """
    pass

def ped(prefix: str, string: str, delta: int | None = None) -> int:
    """

    Computes the prefix edit distance between the given prefix and string
    using the given delta.

    """
    pass

def ied(infix: str, string: str) -> tuple[int, int]:
    """

    Computes the infix edit distance between the given infix and string.

    """
    pass

class IndexData:
    """

    Data for a search index.

    """
    def __init__(self, file: str) -> None:
        """

        Initializes the data from the given file.

        """
        pass

    def __len__(self) -> int:
        """

        Returns the number of rows in the data.

        """
        pass

    def __getitem__(self, key: int) -> str:
        """

        Returns the row at the given index.

        """
        pass

    def __iter__(self) -> Iterator[str]:
        """

        Returns an iterator over the rows.

        """
        pass

    def get_row(self, idx: int) -> str | None:
        """

        Returns the row at the given index.

        """
        pass

    def get_val(self, idx: int, column: int) -> str | None:
        """

        Returns the value at the given index and column.

        """
        pass

class Mapping:
    """

    A mapping from a identifier column of index data to its index.

    """
    def __init__(self, data: IndexData, identifier_column: int) -> None:
        """

        Initializes the mapping from the given data and identifier column.

        """
        pass

    def get(self, identifier: str) -> int | None:
        """

        Returns the index for the given identifier.

        """
        pass

class QGramIndex(SearchIndex):
    """

    A q-gram index for fuzzy prefix or infix search.

    """

    @property
    def q(self) -> int:
        """

        The q in q-grams.

        """
        pass

    @property
    def distance(self) -> str:
        """

        The distance function used.

        """
        pass

    @staticmethod
    def build(
        data_file: str,
        index_dir: str,
        q: int = 3,
        distance: str = "ied",
        use_synonyms: bool = True,
        **kwargs: Any,
    ) -> None:
        """

        Builds the index from the given file and saves
        it in the index dir.

        """
        pass

class PrefixIndex(SearchIndex):
    """

    A prefix index for keyword prefix search.

    """
    @property
    def score(self) -> str:
        """

        The scoring function used.

        """
        pass

    @property
    def min_keyword_length(self) -> int:
        """

        The minimum keyword length, all keywords shorter than this
        are ignored.

        """
        pass

    @staticmethod
    def build(
        data_file: str,
        index_dir: str,
        use_synonyms: bool = True,
        **kwargs: Any,
    ) -> None:
        """

        Builds the index from the given file and saves
        it in the index dir.

        """
        pass

    def find_matches(
        self,
        query: str,
        score: str = "occurrence",
        k: float = 1.75,
        b: float = 0.75,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """

        Returns a sorted list of tuples containing IDs
        and scores for all matches for the given query.

        """
        pass
