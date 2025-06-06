use csv::ReaderBuilder;
use pyo3::prelude::*;
use std::{fs::File, sync::Arc};

use anyhow::anyhow;
use memmap2::Mmap;

struct Inner {
    mmap: Mmap,
    offsets: Vec<usize>,
}

#[pyclass]
#[derive(Clone)]
pub struct IndexData {
    inner: Arc<Inner>,
}

#[pymethods]
impl IndexData {
    #[new]
    pub fn new(file: &str) -> anyhow::Result<Self> {
        // iterate over the file and store the offest of each line
        let mmap = unsafe { Mmap::map(&File::open(file)?)? };
        let mut offsets = vec![];

        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(mmap.as_ref());

        for record in reader.records() {
            let record = record.map_err(|e| anyhow!("failed to read record: {e}"))?;
            let Some(position) = record.position() else {
                return Err(anyhow!("failed to get position of record"));
            };
            offsets.push(position.byte() as usize);
        }

        Ok(IndexData {
            inner: Arc::new(Inner { mmap, offsets }),
        })
    }

    pub fn get_row(&self, idx: usize) -> Option<Vec<String>> {
        let offset = self.inner.offsets.get(idx)?;
        let data = self.inner.mmap.get(*offset..)?;
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .from_reader(data);
        let mut record = reader.records().next()?.ok()?;
        record.trim();
        Some(record.iter().map(|s| s.to_string()).collect())
    }

    pub fn get_val(&self, idx: usize, column: usize) -> Option<String> {
        let offset = self.inner.offsets.get(idx)?;
        let data = self.inner.mmap.get(*offset..)?;
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .from_reader(data);
        let record = reader.records().next()?.ok()?;
        record.get(column).map(|s| s.to_string())
    }

    pub fn __iter__(&self) -> PyIndexDataIter {
        PyIndexDataIter {
            inner: self.clone(),
            idx: 0,
        }
    }

    pub fn __getitem__(&self, idx: usize) -> anyhow::Result<Vec<String>> {
        self.get_row(idx)
            .ok_or_else(|| anyhow!("index out of bounds"))
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }
}

impl IndexData {
    pub fn len(&self) -> usize {
        self.inner.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> IndexDataIter {
        IndexDataIter { data: self, idx: 0 }
    }
}

pub struct IndexDataIter<'a> {
    data: &'a IndexData,
    idx: usize,
}

impl Iterator for IndexDataIter<'_> {
    type Item = Vec<String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.data.get_row(self.idx).inspect(|_| {
            self.idx += 1;
        })
    }
}

#[pyclass]
#[pyo3(name = "IndexDataIter")]
pub struct PyIndexDataIter {
    inner: IndexData,
    idx: usize,
}

#[pymethods]
impl PyIndexDataIter {
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __next__(&mut self) -> Option<Vec<String>> {
        self.inner.get_row(self.idx).inspect(|_| {
            self.idx += 1;
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_index_data() {
        let dir = env!("CARGO_MANIFEST_DIR");
        let data = IndexData::new(&format!("{dir}/test.tsv")).expect("Failed to load data");
        assert_eq!(data.len(), 99);
        assert_eq!(data.get_val(0, 0), Some("United States".to_string()));
        assert_eq!(data.get_val(2, 1), Some("412".to_string()));
    }
}
