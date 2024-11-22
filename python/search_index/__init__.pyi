from typing import Any
from search_index import SearchIndex

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

    @staticmethod
    def build(
        data_file: str,
        index_dir: str,
        score: str = "occurrence",
        k: float = 1.75,
        b: float = 0.75,
        use_synonyms: bool = True,
        **kwargs: Any,
    ) -> None:
        """

        Builds the index from the given file and saves
        it in the index dir.

        """
        pass
