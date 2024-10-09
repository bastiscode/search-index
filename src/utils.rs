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

#[pyfunction]
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
    // intersect two sorted lists of ids
    // assumes that the input lists are sorted
    // and that the ids are unique per list
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
    result
}

pub(crate) fn list_merge(
    a: &[(usize, f32)],
    b: &[(usize, f32)],
    agg_fn: impl Fn(f32, f32) -> f32,
) -> Vec<(usize, f32)> {
    // merge two sorted lists of (id, score) pairs
    // into a single sorted list
    // assumes that the input lists are sorted by id and that
    // the ids are unique per list
    let mut result = vec![];
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        let (a_id, a_score) = a[i];
        let (b_id, b_score) = b[j];
        match a_id.cmp(&b_id) {
            Ordering::Less => {
                result.push((a_id, a_score));
                i += 1;
            }
            Ordering::Greater => {
                result.push((b_id, b_score));
                j += 1;
            }
            Ordering::Equal => {
                result.push((a_id, agg_fn(a_score, b_score)));
                i += 1;
                j += 1;
            }
        }
    }
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

fn idf(doc_freq: u32, doc_count: u32) -> Option<f32> {
    if doc_count == 0 || doc_freq == 0 {
        return None;
    }
    Some((doc_count as f32 / doc_freq as f32).log2())
}

pub(crate) fn tfidf(term_freq: u32, doc_freq: u32, doc_count: u32) -> Option<f32> {
    if term_freq == 0 {
        return None;
    }
    Some(term_freq as f32 * idf(doc_freq, doc_count)?)
}

pub(crate) fn bm25(
    term_freq: u32,
    doc_freq: u32,
    doc_count: u32,
    avg_doc_len: f32,
    doc_len: u32,
    k: f32,
    b: f32,
) -> Option<f32> {
    assert!((0.0..=1.0).contains(&b), "b must be in [0, 1]");
    assert!(k >= 0.0, "k must be non-negative");
    if term_freq == 0 || avg_doc_len == 0.0 {
        return None;
    }
    let idf = idf(doc_freq, doc_count)?;
    let tf = term_freq as f32;
    let alpha = 1.0 - b + b * doc_len as f32 / avg_doc_len;
    let tf_star = if k > 0.0 {
        tf * (1.0 + 1.0 / k) / (alpha + tf / k)
    } else {
        1.0
    };
    Some(tf_star * idf)
}

#[cfg(test)]
mod test {
    use super::{bm25, tfidf};

    #[test]
    fn test_tfidf() {
        let score = tfidf(1, 16, 64);
        assert_eq!(score, Some(2.0));

        let score = tfidf(1, 16, 0);
        assert_eq!(score, None);

        let score = tfidf(1, 0, 64);
        assert_eq!(score, None);

        let score = tfidf(0, 16, 64);
        assert_eq!(score, None);
    }

    #[test]
    fn test_bm25() {
        // regular tfidf
        let k = f32::INFINITY;
        let b = 0.0;
        let score = bm25(1, 16, 64, 8.0, 8, k, b);
        assert_eq!(score, Some(2.0));
        let score = bm25(1, 16, 64, 5.0, 17, k, b);
        assert_eq!(score, Some(2.0));

        // binary
        let k = 0.0;
        let b = 0.0;
        let score = bm25(1, 16, 64, 8.0, 8, k, b);
        assert_eq!(score, Some(2.0));

        let score = bm25(0, 16, 64, 5.0, 17, k, b);
        assert_eq!(score, None);

        let score = bm25(1, 0, 64, 5.0, 17, k, b);
        assert_eq!(score, None);

        let score = bm25(1, 16, 0, 5.0, 17, k, b);
        assert_eq!(score, None)
    }
}
