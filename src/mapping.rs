use std::{
    cmp::Ordering,
    fs::File,
    io::{BufWriter, Write},
};

use anyhow::anyhow;
use memmap2::Mmap;
use pyo3::prelude::*;

use crate::data::IndexData;

#[pyclass]
pub struct Mapping {
    data: IndexData,
    permutation: Vec<usize>,
    identifier_column: usize,
}

const U64_SIZE: usize = size_of::<u64>();

#[pymethods]
impl Mapping {
    #[staticmethod]
    pub fn build(
        data: IndexData,
        mapping_file: &str,
        identifier_column: usize,
    ) -> anyhow::Result<()> {
        let mut mapping_file = BufWriter::new(File::create(mapping_file)?);
        let identifier_bytes = u64::try_from(identifier_column)?.to_le_bytes();
        mapping_file.write_all(&identifier_bytes)?;

        let mut permutation = Vec::with_capacity(data.len());
        let mut identifiers = Vec::with_capacity(data.len());
        for index in 0..data.len() {
            if let Some(identifier) = data.get_val(index, identifier_column) {
                identifiers.push(identifier);
                permutation.push(index);
            } else {
                return Err(anyhow!("missing identifier at index {}", index));
            }
        }

        permutation.sort_by_key(|&i| identifiers[i]);

        for index in permutation {
            let index_bytes = u64::try_from(index)?.to_le_bytes();
            mapping_file.write_all(&index_bytes)?;
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(data: IndexData, mapping_file: &str) -> anyhow::Result<Self> {
        let mut permutation = Vec::with_capacity(data.len());

        let mapping_bytes = unsafe { Mmap::map(&File::open(mapping_file)?)? };
        let mut chunks = mapping_bytes.chunks_exact(U64_SIZE);

        let mut identifier_bytes = [0; U64_SIZE];
        identifier_bytes.copy_from_slice(
            chunks
                .next()
                .ok_or_else(|| anyhow!("mapping does not contain identifier column"))?,
        );
        let identifier_column = u64::from_le_bytes(identifier_bytes) as usize;
        for index_bytes in chunks {
            let index = u64::from_le_bytes(index_bytes.try_into()?) as usize;
            permutation.push(index);
        }

        Ok(Self {
            data,
            permutation,
            identifier_column,
        })
    }

    pub fn get(&self, identifier: &str) -> Option<usize> {
        let mut lower = 0;
        let mut upper = self.permutation.len();
        while lower < upper {
            let mid = (lower + upper) / 2;
            let mid_identifier = self
                .data
                .get_val(self.permutation[mid], self.identifier_column)?;
            match mid_identifier.cmp(identifier) {
                Ordering::Less => lower = mid + 1,
                Ordering::Equal => return Some(self.permutation[mid]),
                Ordering::Greater => upper = mid,
            }
        }
        None
    }
}

#[cfg(test)]
mod test {

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_mapping() {
        let dir = env!("CARGO_MANIFEST_DIR");
        let data_file = format!("{dir}/test.tsv");

        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let offset_file = temp_dir
            .path()
            .join("test.offsets")
            .to_str()
            .unwrap()
            .to_string();

        IndexData::build(&data_file, &offset_file).expect("Failed to build index data");
        let data = IndexData::load(&data_file, &offset_file).expect("Failed to load index data");

        let temp_file = temp_dir.path().join("mapping.bin");
        Mapping::build(data.clone(), temp_file.to_str().unwrap(), 3)
            .expect("Failed to build mapping");

        let mapping = Mapping::load(data.clone(), temp_file.to_str().unwrap())
            .expect("Failed to load mapping");

        let id = mapping
            .get("<http://www.wikidata.org/entity/Q76>")
            .expect("Failed to find mapping");
        assert_eq!(data.get_val(id, 0), Some("Barack Obama"));
    }
}
