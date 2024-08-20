from typing import Any
from search_index.index import SearchIndex


def normalize(s: str) -> str:
    """

    Normalizes the given string.

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
    def use_syns(self) -> bool:
        """

        Whether synonyms are used.

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
        **kwargs: Any
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

    @staticmethod
    def build(
        data_file: str,
        index_dir: str,
        use_synonyms: bool = True,
        **kwargs: Any
    ) -> None:
        """

        Builds the index from the given file and saves
        it in the index dir.

        """
        pass
