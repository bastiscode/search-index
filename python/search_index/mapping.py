import mmap
from search_index import SearchIndex


class Mapping:
    def __init__(
        self,
        search_index: SearchIndex,
        identifier_column: int,
        mapping_file: str
    ) -> None:
        self.search_index = search_index
        self.identifier_column = identifier_column
        with open(mapping_file, "r+b") as f:
            self.mapping = mmap.mmap(f.fileno(), 0)

    @staticmethod
    def build(
        search_index: SearchIndex,
        mapping_file: str,
        identifier_column: int
    ) -> None:
        """

        Builds the mapping from the given search index using the
        specified identifier column and saves it in the given file.

        """
        map: dict[str, int] = {}
        for i, data in enumerate(search_index):
            data = search_index.get_data_by_idx(i)
            split = data.rstrip("\r\n").split("\t")
            assert len(split) > identifier_column, \
                f"identifier column not found for index {i}"
            identifier = split[identifier_column]
            assert identifier not in map, f"duplicate identifier {identifier}"
            map[identifier] = i

        # sort by identifier
        identifiers = sorted(map.items())

        # save mapping to file in binary format
        with open(mapping_file, "w+b") as f:
            for _, index in identifiers:
                f.write(index.to_bytes(8, "little"))

    def get(self, identifier: str) -> int | None:
        """

        Returns the index of the given identifier.

        """
        # perform a binary search
        lower = 0
        upper = len(self.mapping) // 8
        while lower < upper:
            middle = (lower + upper) // 2
            bytes = self.mapping[middle * 8: (middle + 1) * 8]
            index = int.from_bytes(bytes, "little")
            ident = self.search_index.get_column_by_idx(
                index,
                self.identifier_column
            )
            if ident is None:
                return None
            elif ident < identifier:
                lower = middle + 1
            elif ident > identifier:
                upper = middle
            else:
                return index

        return None
