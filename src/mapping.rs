use std::{
    cmp::Ordering,
    fs::File,
    io::{BufWriter, Write},
};

use anyhow::anyhow;
use itertools::Itertools;
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
        let mut permutation = Vec::with_capacity(data.len());
        let mut identifiers = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            let identifier = data
                .get_val(i, identifier_column)
                .ok_or_else(|| anyhow!("identifier column out of bounds"))?;
            identifiers.push(identifier);
            permutation.push(i);
        }

        let mut mapping_file = BufWriter::new(File::create(mapping_file)?);
        let identifier_bytes = u64::try_from(identifier_column)?.to_le_bytes();
        mapping_file.write_all(&identifier_bytes)?;

        for index in permutation.into_iter().sorted_by_key(|&i| identifiers[i]) {
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
        while let Some(index_bytes) = chunks.next() {
            let index = u64::from_le_bytes(
                index_bytes
                    .try_into()
                    .map_err(|_| anyhow!("invalid index in mapping"))?,
            ) as usize;
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
