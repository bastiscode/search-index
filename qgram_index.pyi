class QGramIndex:
    """

    A QGram-Index.

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

    def __init__(
        self,
        q: int,
        use_syns: bool = True,
        distance: str = "ped"
    ) -> None:
        """

        Creates an empty qgram index,
        either for prefix or regular search.

        """
        pass

    def build(self, data_file: str) -> None:
        """

        Builds the index from the given file.

        The file should contain one entity per line, in the following format:
            name\tscore\tsynonyms\tinfo1\tinfo2\t...

        Synonyms are separated by a semicolon.

        An example line:
            Albert Einstein\t275\tEinstein;A. Einstein\tGerman physicist\t..."

        The entity IDs are zero-based.

        """
        pass

    def save(self, index_file: str) -> None:
        """

        Saves the index to the given file.

        """
        pass

    @staticmethod
    def load(index_file: str, data_file: str) -> "QGramIndex":
        """

        Loads the index from the given file and data file.

        """
        pass

    def find_matches(
        self,
        query: str,
        delta: int | None = None
    ) -> list[tuple[int, int]]:
        """

        Finds all entities y with PED(x, y) or ED(x, y) <= delta for a given
        integer delta and a given query x. The query should be non-empty.
        The delta must be chosen such that the threshold for filtering
        PED / ED computations is greater than zero. That way, it suffices to
        only consider names which have at least one q-gram in common with
        the query.

        It returns a list of (ID, PED / ED) ordered first by PED / ED
        ascending and then by score descending.

        """
        pass

    def get_name_by_id(self, id: int) -> str:
        """

        Returns the name or synonym for the given ID.
        If the index was built without synonyms, the synonym is always
        equal to the name.

        """
        pass

    def get_idx_by_id(self, id: int) -> int:
        """

        Returns the index for the given ID.
        If the index was built without synonyms, the index is always
        equal to the ID.

        """
        pass

    def get_data_by_idx(
        self,
        idx: int
    ) -> str:
        """

        Returns the line from the data file for the given index.
        Index must be between 0 and the index length.

        """
        pass

    def get_data_by_id(
        self,
        id: int
    ) -> str:
        """

        Returns the line from the data file for the given ID.
        Safe to call this with an ID from a find_matches result.
        If the index is not using synonyms, this is the same as
        get_data_by_idx.

        """
        pass

    def sub_index_by_indices(
        self,
        indices: list[int]
    ) -> "QGramIndex":
        """

        Creates a sub-index from the data stored at the given indices.

        """
        pass

    def __len__(self) -> int:
        """

        Returns the number of entities in the index.

        """
        pass
