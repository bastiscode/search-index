use std::cmp::Ordering;

use anyhow::anyhow;
use pyo3::prelude::*;

use crate::data::IndexData;

#[pyclass]
pub struct Mapping {
    data: IndexData,
    permutation: Vec<usize>,
    identifier_column: usize,
}

#[pymethods]
impl Mapping {
    #[new]
    pub fn new(data: IndexData, identifier_column: usize) -> PyResult<Self> {
        let mut permutation = Vec::with_capacity(data.len());
        let mut identifiers = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            let identifier = data
                .get_val(i, identifier_column)
                .ok_or_else(|| anyhow!("identifier column out of bounds"))?;
            identifiers.push(identifier);
            permutation.push(i);
        }

        permutation.sort_by_key(|&i| identifiers[i]);

        Ok(Mapping {
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
