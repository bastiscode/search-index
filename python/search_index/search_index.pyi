from typing import Any, Iterator


class SearchIndex:
    """

    A search index.

    """

    @staticmethod
    def build(data_file: str, index_dir: str, **kwargs: Any) -> None:
        """

        Builds the index from the given file and saves
        it in the index dir.

        The file should contain one record per line, in the following format:
            name\tscore\tsynonyms\tinfo1\tinfo2\t...

        Synonyms are separated by a semicolon.

        An example line:
            Albert Einstein\t275\tEinstein;A. Einstein\tGerman physicist\t
        """
        pass

    @staticmethod
    def load(data_file: str, index_dir: str, **kwargs: Any) -> "SearchIndex":
        """

        Loads the index from the given data file and index directory.

        """
        pass

    def find_matches(
        self,
        query: str,
        **kwargs: Any
    ) -> list[tuple[int, Any]]:
        """

        Returns a sorted list of tuples containing IDs
        and ranking key for all matches for the given query.

        """
        pass

    def get_name(self, id: int) -> str:
        """

        Returns the name or synonym for the given ID.

        """
        pass

    def get_row(
        self,
        id: int
    ) -> str:
        """

        Returns the line from the data file for the given ID.
        ID must be between 0 and the index length.

        """
        pass

    def get_val(
        self,
        id: int,
        col: int
    ) -> str:
        """

        Returns the column value for the given ID.

        """
        pass

    def sub_index_by_ids(
        self,
        ids: list[int]
    ) -> "SearchIndex":
        """

        Creates a sub-index contating only the given IDs.

        """
        pass

    def __len__(self) -> int:
        """

        Returns the number of items in the index.

        """
        pass

    def __iter__(self) -> Iterator[str]:
        """

        Iterates over the index data.

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
        use_synonyms: bool = True,
        distance: str = "ped",
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
