from typing import Any, Iterator

from search_index.base import SearchIndex

def normalize(s: str) -> str:
    """

    Normalizes the given string.

    """
    pass

class IndexData:
    """

    Data for a search index.

    """
    @staticmethod
    def build(data_file: str, offset_file: str) -> None:
        """

        Builds the index data from the given data and saves line
        offsets in the offset file.

        """
        pass

    @staticmethod
    def load(data_file: str, offset_file: str) -> "IndexData":
        """

        Loads the data from the given data and offset file.

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

    def __iter__(self) -> Iterator[list[str]]:
        """

        Returns an iterator over the rows.

        """
        pass

    def get_row(self, idx: int) -> list[str] | None:
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

class PrefixIndex(SearchIndex):
    """

    A prefix index for keyword prefix search.

    """
    @staticmethod
    def build(
        data: IndexData,
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
        min_keyword_length: int | None = None,
        no_refinement: bool = False,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """

        Returns a sorted list of tuples containing IDs
        and scores for all matches for the given query.

        """
        pass
