use std::{cmp::Ordering, sync::Arc};

use any_ascii::any_ascii;
use pyo3::prelude::*;

use crate::data::IndexData;

#[pyclass]
pub struct IndexIter {
    data: IndexData,
    sub_index: Option<Arc<[usize]>>,
    index: usize,
}

impl IndexIter {
    pub(crate) fn new(data: IndexData, sub_index: Option<Arc<[usize]>>) -> Self {
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

pub(crate) fn lower_bound(
    mut start: usize,
    mut end: usize,
    cmp: impl Fn(usize) -> Ordering,
) -> Option<(usize, bool)> {
    let mut answer = None;

    while start < end {
        let mid = (start + end) / 2;
        match cmp(mid) {
            Ordering::Less => {
                start = mid + 1;
            }
            ord @ (Ordering::Equal | Ordering::Greater) => {
                answer = Some((mid, ord == Ordering::Equal));
                end = mid;
            }
        }
    }
    answer
}

pub(crate) fn upper_bound(
    mut start: usize,
    mut end: usize,
    cmp: impl Fn(usize) -> Ordering,
) -> Option<usize> {
    let mut answer = None;

    while start < end {
        let mid = (start + end) / 2;
        match cmp(mid) {
            Ordering::Less | Ordering::Equal => {
                start = mid + 1;
            }
            Ordering::Greater => {
                answer = Some(mid);
                end = mid;
            }
        }
    }
    answer
}

#[cfg(test)]
mod test {
    use super::{bm25, lower_bound, tfidf, upper_bound};

    #[test]
    fn test_bounds() {
        let values = vec![1, 1, 2, 3, 3, 5, 8, 8, 9];
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&3)),
            Some((3, true))
        );
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&4)),
            Some((5, false))
        );
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&0)),
            Some((0, false))
        );
        assert_eq!(lower_bound(0, values.len(), |i| values[i].cmp(&10)), None);
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&8)),
            Some((6, true))
        );

        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&3)), Some(5));
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&4)), Some(5));
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&10)), None);
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&8)), Some(8));

        let values: Vec<i32> = vec![];
        assert_eq!(lower_bound(0, values.len(), |i| values[i].cmp(&3)), None);
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&3)), None);

        let values = vec![2, 2];
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&2)),
            Some((0, true))
        );
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&2)), None);

        let values = vec![1, 2, 2, 4];
        assert_eq!(
            lower_bound(0, values.len(), |i| values[i].cmp(&2)),
            Some((1, true))
        );
        assert_eq!(upper_bound(0, values.len(), |i| values[i].cmp(&2)), Some(3));
    }

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
