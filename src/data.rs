use pyo3::prelude::*;
use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::Arc,
};

use anyhow::anyhow;
use memmap2::Mmap;
const U64_SIZE: usize = size_of::<u64>();

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
    #[staticmethod]
    pub fn build(data_file: &str, offset_file: &str) -> anyhow::Result<()> {
        // iterate over the file and store the offest of each line
        let mmap = unsafe { Mmap::map(&File::open(data_file)?)? };

        let mut offsets = vec![];
        // trim final newlines
        let trim = mmap.iter().rev().take_while(|&&b| b == b'\n').count();
        let mut lines = mmap[..mmap.len() - trim].split(|&b| b == b'\n');
        // skip the first line (header)
        let mut offset = lines.next().map(|line| line.len() + 1).unwrap_or(0);
        for line in lines {
            // check that line is valid utf8
            let line =
                std::str::from_utf8(line).map_err(|e| anyhow!("found invalid utf8: {}", e))?;
            if line.split('\t').count() != 5 {
                return Err(anyhow!(
                    "invalid line format, expected 5 tab-separated columns per line: {}",
                    line
                ));
            }
            offsets.push(offset);
            offset += line.len() + 1;
        }

        // write offsets to file
        let mut offset_file = BufWriter::new(File::create(offset_file)?);
        for &offset in &offsets {
            let offset_bytes = u64::try_from(offset)?.to_le_bytes();
            offset_file.write_all(&offset_bytes)?;
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(data_file: &str, offset_file: &str) -> anyhow::Result<Self> {
        let mmap = unsafe { Mmap::map(&File::open(data_file)?)? };
        let offset_bytes = unsafe { Mmap::map(&File::open(offset_file)?)? };

        let mut offsets = vec![];
        for chunk in offset_bytes.chunks_exact(U64_SIZE) {
            let offset = u64::from_le_bytes(chunk.try_into()?) as usize;
            offsets.push(offset);
        }

        Ok(IndexData {
            inner: Arc::new(Inner { mmap, offsets }),
        })
    }

    #[pyo3(name = "get_row")]
    pub(crate) fn _get_row(&self, idx: usize) -> Option<Vec<String>> {
        self.get_row(idx)
            .map(|row| row.into_iter().map(|s| s.to_string()).collect())
    }

    #[pyo3(name = "get_val")]
    pub(crate) fn _get_val(&self, idx: usize, column: usize) -> Option<String> {
        self.get_val(idx, column).map(|val| val.to_string())
    }

    pub fn __iter__(&self) -> PyIndexDataIter {
        PyIndexDataIter {
            inner: self.clone(),
            idx: 0,
        }
    }

    pub fn __getitem__(&self, idx: usize) -> anyhow::Result<Vec<String>> {
        self._get_row(idx)
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

    pub fn get_row(&self, idx: usize) -> Option<Vec<&str>> {
        let start = self.inner.offsets.get(idx).copied()?;
        let end = self
            .inner
            .offsets
            .get(idx + 1)
            .copied()
            .map_or_else(|| self.inner.mmap.len(), |o| o - 1);

        let row = unsafe { std::str::from_utf8_unchecked(&self.inner.mmap[start..end]) };
        Some(row.split('\t').collect())
    }

    pub fn get_val(&self, idx: usize, column: usize) -> Option<&str> {
        let start = self.inner.offsets.get(idx).copied()?;
        let end = self
            .inner
            .offsets
            .get(idx + 1)
            .copied()
            .unwrap_or(self.inner.mmap.len());

        let row = unsafe { std::str::from_utf8_unchecked(&self.inner.mmap[start..end]) };
        row.split('\t').nth(column)
    }

    pub fn iter(&self) -> IndexDataIter {
        IndexDataIter { data: self, idx: 0 }
    }
}

pub struct IndexDataIter<'a> {
    data: &'a IndexData,
    idx: usize,
}

impl<'a> Iterator for IndexDataIter<'a> {
    type Item = Vec<&'a str>;

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
        self.inner._get_row(self.idx).inspect(|_| {
            self.idx += 1;
        })
    }
}

#[cfg(test)]
mod test {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_index_data() {
        let dir = env!("CARGO_MANIFEST_DIR");
        let data_file = format!("{dir}/test.tsv");

        let temp_dir = tempdir().expect("Failed to create temp dir");
        let offset_file = temp_dir
            .path()
            .join("test.offsets")
            .to_str()
            .unwrap()
            .to_string();

        IndexData::build(&data_file, &offset_file).expect("Failed to build index data");
        let data = IndexData::load(&data_file, &offset_file).expect("Failed to load index data");
        assert_eq!(data.len(), 99);
        assert_eq!(data.get_val(0, 0), Some("United States"));
        assert_eq!(data.get_val(2, 1), Some("412"));
        assert_eq!(data.get_val(98, 0), Some("Barack Obama"));
    }
}
