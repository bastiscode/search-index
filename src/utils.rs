use std::{cmp::Ordering, sync::Arc};

use any_ascii::any_ascii;
use itertools::Itertools;
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

    pub fn __next__(&mut self) -> Option<Vec<String>> {
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
        self.data._get_row(id)
    }
}

#[pyfunction]
pub(crate) fn normalize(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let norm = any_ascii(name);
    let norm = if norm.is_empty() {
        // if ascii conversion produces an empty string, return lowercase
        // as fallback
        name.to_lowercase()
    } else {
        norm.to_lowercase()
    };
    // remove all punctuation characters around words
    // but keep punctuation inside words and words containing
    // only punctuation
    norm.split_whitespace()
        .map(|word| {
            let trimmed = word
                .trim_end_matches(|c: char| c.is_ascii_punctuation())
                .trim_start_matches(|c: char| c.is_ascii_punctuation());
            // only punctuation
            if trimmed.is_empty() {
                word.to_string()
            } else {
                trimmed.to_string()
            }
        })
        .join(" ")
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
    use super::{lower_bound, normalize, upper_bound};

    #[test]
    fn test_normalize() {
        // empty string
        assert_eq!(normalize(""), "");

        // basic lowercase conversion
        assert_eq!(normalize("Hello World"), "hello world");

        // remove punctuation at word boundaries
        assert_eq!(normalize("Hello, World!"), "hello world");
        assert_eq!(normalize("(Hello) [World]"), "hello world");

        // keep punctuation inside words
        assert_eq!(normalize("it's a test"), "it's a test");
        assert_eq!(normalize("semi-automated"), "semi-automated");

        // handle words with only punctuation
        assert_eq!(normalize("Hello --- World"), "hello --- world");

        // handle non-ASCII characters
        assert_eq!(normalize("Café"), "cafe");
        assert_eq!(normalize("Größe"), "grosse");
        assert_eq!(normalize("Niño"), "nino");

        // handle emojis
        assert_eq!(normalize("Hello 😊 World"), "hello blush world");

        // combination of cases
        assert_eq!(normalize("!Hello, Größe-Test!"), "hello grosse-test");
    }

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
}
