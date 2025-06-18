from typing import Any, Iterator

from search_index import IndexData


class SearchIndex:
    """

    A search index.

    """

    @staticmethod
    def build(data: IndexData, index_dir: str, **kwargs: Any) -> None:
        """

        Builds the index from the given data and saves
        it in the index dir.

        """
        ...

    @staticmethod
    def load(data: IndexData, index_dir: str, **kwargs: Any) -> "SearchIndex":
        """

        Loads the index with the given data and index directory.

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
