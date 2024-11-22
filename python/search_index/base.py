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

        Synonyms are expected to be separated by three semicolons.

        An example line:
            Albert Einstein\t275\tEinstein;;;A. Einstein\tGerman physicist\t
        """
        ...

    @staticmethod
    def load(data_file: str, index_dir: str, **kwargs: Any) -> "SearchIndex":
        """

        Loads the index from the given data file and index directory.

        """
        ...

    def find_matches(self, query: str, **kwargs: Any) -> list[tuple[int, Any]]:
        """

        Returns a sorted list of tuples containing IDs
        and ranking key for all matches for the given query.

        """
        ...

    def get_name(self, id: int) -> str:
        """

        Returns the name for the given ID.

        """
        ...

    def get_row(self, id: int) -> str:
        """

        Returns the line from the data file for the given ID.
        ID must be between 0 and the index length.

        """
        ...

    def get_val(self, id: int, col: int) -> str:
        """

        Returns the column value for the given ID.

        """
        ...

    def sub_index_by_ids(self, ids: list[int]) -> "SearchIndex":
        """

        Creates a sub-index contating only the given IDs.

        """
        ...

    def __len__(self) -> int:
        """

        Returns the number of items in the index.

        """
        ...

    def __iter__(self) -> Iterator[str]:
        """

        Iterates over the index data.

        """
        ...

    def get_type(self) -> str:
        """

        Returns the type of the index.

        """
        ...
