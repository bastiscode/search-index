use std::{cmp::Ordering, sync::Arc};

use any_ascii::any_ascii;
use pyo3::prelude::*;

use crate::data::IndexData;

#[pyclass]
pub struct IndexIter {
    data: Arc<IndexData>,
    sub_index: Option<Arc<[usize]>>,
    index: usize,
}

impl IndexIter {
    pub(crate) fn new(data: Arc<IndexData>, sub_index: Option<Arc<[usize]>>) -> Self {
        Self {
            data,
            sub_index,
            index: 0,
        }
    }
}

#[pymethods]
impl IndexIter {
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __next__(&mut self) -> Option<String> {
        let id = if let Some(sub_index) = self.sub_index.as_ref() {
            if self.index >= sub_index.len() {
                return None;
            }
            sub_index[self.index]
        } else {
            if self.index >= self.data.len() {
                return None;
            }
            self.index
        };
        self.index += 1;
        self.data.get_row(id).map(|s| s.to_string())
    }
}

pub(crate) fn normalize(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let norm = any_ascii(name);
    if norm.is_empty() {
        // if ascii conversion produces an empty string, return lowercase
        // as fallback
        name.to_lowercase()
    } else {
        norm.to_lowercase()
    }
}

pub(crate) fn list_intersection(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = vec![];
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    // add remaining elements
    while i < a.len() {
        result.push(a[i]);
        i += 1;
    }
    while j < b.len() {
        result.push(b[j]);
        j += 1;
    }
    result
}
