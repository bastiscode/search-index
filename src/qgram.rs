use anyhow::anyhow;
use core::cmp::Reverse;
use itertools::Itertools;
use memmap2::Mmap;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::iter::repeat;
use std::path::Path;
use std::sync::Arc;

use crate::data::IndexData;
use crate::utils::{list_intersection, normalize, IndexIter};

#[pyclass]
#[derive(Clone)]
pub struct QGramIndex {
    #[pyo3(get)]
    q: usize,
    #[pyo3(get)]
    distance: Distance,
    data: Arc<IndexData>,
    qgrams: Arc<Mmap>,
    qgram_offsets: Arc<[usize]>,
    qgram_list_lengths: Arc<[usize]>,
    names: Arc<Mmap>,
    name_offsets: Arc<[usize]>,
    name_to_index: Arc<[usize]>,
    sub_index: Option<Arc<[usize]>>,
}

const U64_SIZE: usize = size_of::<u64>();
const U32_SIZE: usize = size_of::<u32>();
const U16_SIZE: usize = size_of::<u16>();

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distance {
    #[default]
    Ped,
    Ied,
}

impl IntoPy<PyObject> for Distance {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Distance::Ped => "ped",
            Distance::Ied => "ied",
        }
        .into_py(py)
    }
}

impl FromPyObject<'_> for Distance {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let s = ob.extract::<String>()?;
        match s.as_str() {
            "ped" | "PED" => Ok(Distance::Ped),
            "ied" | "IED" => Ok(Distance::Ied),
            _ => Err(anyhow!("invalid distance type").into()),
        }
    }
}

impl QGramIndex {
    #[inline]
    fn compute_q_grams(name: &str, q: usize, distance: Distance, full: bool) -> Vec<String> {
        assert!(!name.is_empty(), "name must not be empty");
        assert!(q > 0, "q must be greater than 0");
        let chars: Vec<_> = match distance {
            Distance::Ped => repeat('$').take(q - 1).chain(name.chars()).collect(),
            Distance::Ied => name.chars().collect(),
        };
        let start = if chars.len() < q && full {
            assert_eq!(
                distance,
                Distance::Ied,
                "smaller q-grams only allowed for IED"
            );
            1
        } else {
            q.min(chars.len())
        };
        (start..=q.min(chars.len()))
            .flat_map(|i| chars.windows(i).map(|window| window.iter().collect()))
            .collect()
    }

    fn size(&self) -> usize {
        self.qgram_offsets.len()
    }

    #[inline]
    fn get_qgram(&self, index: usize) -> (&[u8], (usize, usize)) {
        let start = self.qgram_offsets[index];
        let next_start = self
            .qgram_offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| self.qgrams.len());
        let length = self.qgram_list_lengths[index];
        let end = next_start - (U32_SIZE + U16_SIZE) * length;
        (&self.qgrams[start..end], (end, next_start))
    }

    fn get_inverted_list(&self, q_gram: &str, count: usize) -> Option<Vec<(usize, usize)>> {
        // do binary search in offsets
        let mut lower = 0;
        let mut upper = self.size();
        while lower < upper {
            let mid = (lower + upper) / 2;
            let (mid_qgram, (start, end)) = self.get_qgram(mid);
            match mid_qgram.cmp(q_gram.as_bytes()) {
                Ordering::Less => lower = mid + 1,
                Ordering::Greater => upper = mid,
                Ordering::Equal => {
                    let inv_list = self.parse_inverted_list(start, end, count);
                    return Some(inv_list);
                }
            }
        }
        None
    }

    fn parse_inverted_list(&self, start: usize, end: usize, count: usize) -> Vec<(usize, usize)> {
        self.qgrams[start..end]
            .chunks_exact(U32_SIZE + U16_SIZE)
            .filter_map(|bytes| {
                let mut id_bytes = [0; U32_SIZE];
                id_bytes.copy_from_slice(&bytes[..U32_SIZE]);
                let id = u32::from_le_bytes(id_bytes) as usize;
                if let Some(sub_index) = self.sub_index.as_ref() {
                    if sub_index.binary_search(&id).is_err() {
                        return None;
                    }
                }
                let mut freq_bytes = [0; U16_SIZE];
                freq_bytes.copy_from_slice(&bytes[U32_SIZE..]);
                let freq = u16::from_le_bytes(freq_bytes) as usize;
                Some((id, count.min(freq)))
            })
            .collect()
    }

    fn merge_lists(&self, q_grams: &[String]) -> Vec<(usize, usize)> {
        q_grams
            .iter()
            .fold(HashMap::new(), |mut counts, q_gram| {
                counts
                    .get_mut(&q_gram)
                    .map(|count: &mut usize| *count += 1)
                    .unwrap_or_else(|| {
                        counts.insert(q_gram, 1);
                    });
                counts
            })
            .into_iter()
            .filter_map(|(q_gram, count)| self.get_inverted_list(q_gram, count))
            .flatten()
            .sorted()
            .fold(vec![], |mut cur, (id, freq)| {
                match cur.last_mut() {
                    Some((last_id, last_freq)) if last_id == &id => {
                        *last_freq += freq;
                    }
                    _ => cur.push((id, freq)),
                }
                cur
            })
    }

    fn get_normalized_name(&self, name_id: usize) -> Option<&str> {
        let start = self.name_offsets.get(name_id).copied()?;
        let end = self
            .name_offsets
            .get(name_id + 1)
            .copied()
            .unwrap_or_else(|| self.names.len());
        std::str::from_utf8(&self.names[start..end]).ok()
    }

    fn get_index(&self, name_id: usize) -> Option<usize> {
        self.name_to_index.get(name_id).copied()
    }
}

pub type Ranking = (usize, usize, usize);

#[pymethods]
impl QGramIndex {
    #[staticmethod]
    #[pyo3(signature = (data_file, index_dir, q = 3, distance = Distance::Ied, use_synonyms = true))]
    pub fn build(
        data_file: &str,
        index_dir: &str,
        q: usize,
        distance: Distance,
        use_synonyms: bool,
    ) -> anyhow::Result<()> {
        let data = IndexData::new(data_file)?;
        let mut inverted_lists: HashMap<String, Vec<(u32, u16)>> = HashMap::new();

        let index_dir = Path::new(index_dir);
        let mut name_file = BufWriter::new(File::create(index_dir.join("index.names"))?);
        let mut name_offset_file =
            BufWriter::new(File::create(index_dir.join("index.name-offsets"))?);

        let mut name_offset = 0;
        let mut name_id = 0;
        for (i, row) in data.iter().enumerate() {
            let mut split = row.split("\t");
            let name = normalize(
                split
                    .next()
                    .ok_or_else(|| anyhow!("name not found in row {i}: {row}"))?,
            );
            let mut names = vec![name];
            if use_synonyms {
                names.extend(
                    split
                        .nth(1)
                        .ok_or_else(|| anyhow!("synonyms not found in row {i}: {row}"))?
                        .split(';')
                        .map(normalize),
                );
            }
            for name in names.into_iter().filter(|s| !s.is_empty()) {
                let q_grams = Self::compute_q_grams(&name, q, distance, true);
                for qgram in q_grams {
                    let list = inverted_lists.entry(qgram).or_default();
                    match list.last_mut() {
                        Some((last_id, last_freq)) if last_id == &name_id => {
                            *last_freq = last_freq.saturating_add(1);
                        }
                        _ => {
                            list.push((name_id, 1));
                        }
                    }
                }
                if name_id == u32::MAX {
                    return Err(anyhow!("too many names, max {} supported", u32::MAX));
                }
                name_id += 1;
                name_file.write_all(name.as_bytes())?;
                let name_offset_bytes = u64::try_from(name_offset)?.to_le_bytes();
                name_offset_file.write_all(&name_offset_bytes)?;
                name_offset += name.len();
                let name_to_index_bytes = u64::try_from(i)?.to_le_bytes();
                name_offset_file.write_all(&name_to_index_bytes)?;
            }
        }

        // sort inverted lists by q-gram
        let mut qgram_index_file = BufWriter::new(File::create(index_dir.join("index.qgrams"))?);
        let mut qgram_offset_file =
            BufWriter::new(File::create(index_dir.join("index.qgram-offsets"))?);
        let mut qgram_offset = 0;
        for (qgram, inv_list) in inverted_lists
            .into_iter()
            .sorted_by(|(a, _), (b, _)| a.cmp(b))
        {
            qgram_index_file.write_all(qgram.as_bytes())?;
            let offset_bytes = u64::try_from(qgram_offset)?.to_le_bytes();
            qgram_offset_file.write_all(&offset_bytes)?;
            qgram_offset += qgram.len();
            let length_bytes = u64::try_from(inv_list.len())?.to_le_bytes();
            qgram_offset_file.write_all(&length_bytes)?;
            for (id, freq) in inv_list {
                let id_bytes = id.to_le_bytes();
                qgram_index_file.write_all(&id_bytes)?;
                qgram_offset += id_bytes.len();
                let freq_bytes = freq.to_le_bytes();
                qgram_index_file.write_all(&freq_bytes)?;
                qgram_offset += freq_bytes.len();
            }
        }
        let mut config_file = BufWriter::new(File::create(index_dir.join("index.config"))?);
        let mut buffer = vec![];
        let q_bytes = u64::try_from(q)?.to_le_bytes();
        buffer.extend_from_slice(&q_bytes);
        let distance_bytes = match distance {
            Distance::Ped => 0u64,
            Distance::Ied => 1u64,
        }
        .to_le_bytes();
        buffer.extend_from_slice(&distance_bytes);
        config_file.write_all(&buffer)?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(data_file: &str, index_dir: &str) -> anyhow::Result<Self> {
        let data = Arc::new(IndexData::new(data_file)?);
        let index_dir = Path::new(index_dir);
        let qgrams = Arc::new(unsafe { Mmap::map(&File::open(index_dir.join("index.qgrams"))?)? });
        let mut qgram_offsets = vec![];
        let mut qgram_list_lengths = vec![];
        let offset_bytes =
            unsafe { Mmap::map(&File::open(index_dir.join("index.qgram-offsets"))?)? };
        for chunk in offset_bytes.chunks_exact(U64_SIZE * 2) {
            let mut qgram_offset_bytes = [0; U64_SIZE];
            qgram_offset_bytes.copy_from_slice(&chunk[..U64_SIZE]);
            let qgram_offset = u64::from_le_bytes(qgram_offset_bytes) as usize;
            qgram_offsets.push(qgram_offset);
            let mut length_bytes = [0; U64_SIZE];
            length_bytes.copy_from_slice(&chunk[U64_SIZE..]);
            let length = u64::from_le_bytes(length_bytes) as usize;
            qgram_list_lengths.push(length);
        }
        let names = Arc::new(unsafe { Mmap::map(&File::open(index_dir.join("index.names"))?)? });
        let mut name_offsets = vec![];
        let mut name_to_index = vec![];
        let offset_bytes =
            unsafe { Mmap::map(&File::open(index_dir.join("index.name-offsets"))?)? };
        for chunk in offset_bytes.chunks_exact(U64_SIZE * 2) {
            let mut name_offset_bytes = [0; U64_SIZE];
            name_offset_bytes.copy_from_slice(&chunk[..U64_SIZE]);
            let name_offset = u64::from_le_bytes(name_offset_bytes) as usize;
            name_offsets.push(name_offset);
            let mut index_bytes = [0; U64_SIZE];
            index_bytes.copy_from_slice(&chunk[U64_SIZE..]);
            let index = u64::from_le_bytes(index_bytes) as usize;
            name_to_index.push(index);
        }
        let config = unsafe { Mmap::map(&File::open(index_dir.join("index.config"))?)? };
        let mut q_bytes = [0; U64_SIZE];
        q_bytes.copy_from_slice(&config[..U64_SIZE]);
        let q = u64::from_le_bytes(q_bytes) as usize;
        let mut distance_bytes = [0; U64_SIZE];
        distance_bytes.copy_from_slice(&config[U64_SIZE..]);
        let distance = match u64::from_le_bytes(distance_bytes) {
            0 => Distance::Ped,
            1 => Distance::Ied,
            _ => return Err(anyhow!("invalid distance")),
        };
        Ok(Self {
            q,
            distance,
            data,
            qgrams,
            qgram_offsets: qgram_offsets.into(),
            qgram_list_lengths: qgram_list_lengths.into(),
            names,
            name_offsets: name_offsets.into(),
            name_to_index: name_to_index.into(),
            sub_index: None,
        })
    }

    pub fn find_matches(
        &self,
        query: &str,
        delta: Option<usize>,
    ) -> anyhow::Result<Vec<(usize, Ranking)>> {
        let query = normalize(query);
        if query.is_empty() {
            return Err(anyhow!("query must not be empty"));
        }
        let q_grams = Self::compute_q_grams(&query, self.q, self.distance, false);
        let delta = delta.unwrap_or_else(|| q_grams.len() / (self.q + 1));
        let thres = q_grams.len().saturating_sub(self.q * delta);
        if thres == 0 {
            return Err(anyhow!(
                "threshold for filtering distance computations must be positive, lower delta"
            ));
        }

        let merged = self.merge_lists(&q_grams);

        let matches: Vec<_> = merged
            .into_par_iter()
            .filter_map(|(name_id, num_qgrams)| {
                if num_qgrams < thres {
                    return None;
                }
                let name = self.get_normalized_name(name_id)?;
                let (dist, aux) = match self.distance {
                    Distance::Ped => (ped(&query, name, delta), 0),
                    Distance::Ied => ied(&query, name),
                };
                if dist <= delta {
                    Some((name_id, (dist, aux)))
                } else {
                    None
                }
            })
            .collect();

        Ok(matches
            .into_iter()
            .filter_map(|(name_id, dist)| {
                let index = self.get_index(name_id)?;
                Some((index, dist))
            })
            .sorted()
            .unique_by(|&(index, ..)| index)
            .filter_map(|(index, (first, second))| {
                let score = self.data.get_val(index, 1).and_then(|s| s.parse().ok())?;
                Some((index, (first, second, score)))
            })
            .sorted_by_key(|&(_, (first, second, score))| (first, second, Reverse(score)))
            .collect())
    }

    pub fn get_name(&self, id: usize) -> anyhow::Result<&str> {
        self.data
            .get_val(id, 0)
            .ok_or_else(|| anyhow!("invalid id"))
    }

    pub fn get_row(&self, id: usize) -> anyhow::Result<&str> {
        self.data.get_row(id).ok_or_else(|| anyhow!("invalid id"))
    }

    pub fn get_val(&self, id: usize, column: usize) -> anyhow::Result<&str> {
        self.data
            .get_val(id, column)
            .ok_or_else(|| anyhow!("invalid id or column"))
    }

    pub fn sub_index_by_ids(&self, mut ids: Vec<usize>) -> anyhow::Result<Self> {
        if !ids.iter().all(|&id| id < self.data.len()) {
            return Err(anyhow!("invalid ids"));
        }
        ids.sort();
        if let Some(sub_index) = self.sub_index.as_ref() {
            ids = list_intersection(sub_index, &ids);
        }
        let mut index = self.clone();
        index.sub_index = Some(ids.into());
        Ok(index)
    }

    pub fn __len__(&self) -> usize {
        self.sub_index
            .as_ref()
            .map_or(self.data.len(), |sub_index| sub_index.len())
    }

    pub fn __iter__(&self) -> IndexIter {
        IndexIter::new(self.data.clone(), self.sub_index.clone())
    }
}

#[inline]
fn ped(prefix: &str, string: &str, delta: usize) -> usize {
    let x: Vec<_> = prefix.chars().collect();
    let y: Vec<_> = string.chars().collect();
    let n = x.len() + 1;
    let m = (n + delta).min(y.len() + 1);

    let mut matrix = vec![0; m * n];

    for row in 0..n {
        matrix[row * m] = row;
    }
    matrix
        .iter_mut()
        .take(m)
        .enumerate()
        .for_each(|(i, v)| *v = i);

    for row in 1..n {
        for col in 1..m {
            let s = if x[row - 1] == y[col - 1] { 0 } else { 1 };

            let rep_cost = matrix[m * (row - 1) + (col - 1)] + s;
            let add_cost = matrix[m * row + (col - 1)] + 1;
            let del_cost = matrix[m * (row - 1) + col] + 1;

            matrix[m * row + col] = rep_cost.min(add_cost).min(del_cost);
        }
    }

    let mut delta_min = delta + 1;
    for col in 0..m {
        let v = matrix[m * (n - 1) + col];
        if v < delta_min {
            delta_min = v;
        }
    }
    delta_min
}

#[inline]
fn ied(infix: &str, string: &str) -> (usize, usize) {
    let x: Vec<_> = infix.chars().collect();
    let y: Vec<_> = string.chars().collect();
    let n = x.len() + 1;
    let m = y.len() + 1;

    let mut matrix = vec![0; m * n];

    for row in 0..n {
        matrix[row * m] = row;
    }

    for row in 1..n {
        for col in 1..m {
            let s = if x[row - 1] == y[col - 1] { 0 } else { 1 };

            let rep_cost = matrix[m * (row - 1) + (col - 1)] + s;
            let add_cost = matrix[m * row + (col - 1)] + 1;
            let del_cost = matrix[m * (row - 1) + col] + 1;

            matrix[m * row + col] = rep_cost.min(add_cost).min(del_cost);
        }
    }

    // find min over last row
    let ied = matrix[(n - 1) * m..]
        .iter()
        .copied()
        .min()
        .unwrap_or_default();
    (ied, string.len().saturating_sub(infix.len()) + ied)
}

#[cfg(test)]
mod tests {
    use super::{ied, ped};

    #[test]
    fn test_ped() {
        assert_eq!(ped("frei", "frei", 0), 0);
        assert_eq!(ped("frei", "freiburg", 0), 0);
        assert_eq!(ped("frei", "breifurg", 4), 1);
        assert_eq!(ped("freiburg", "stuttgart", 2), 3);
        assert_eq!(ped("", "freiburg", 10), 0);
        assert_eq!(ped("", "", 10), 0);
    }

    #[test]
    fn test_ied() {
        assert_eq!(ied("frei", "frei"), (0, 0));
        assert_eq!(ied("frei", "freiburg"), (0, 4));
        assert_eq!(ied("frei", "breifurg"), (1, 5));
        assert_eq!(ied("freiburg", "stuttgart"), (7, 8));
        assert_eq!(ied("", "freiburg"), (0, 8));
        assert_eq!(ied("", ""), (0, 0));
        assert_eq!(ied("cat", "dog"), (3, 3));
        assert_eq!(ied("cat", "the cat sat on the mat"), (0, 19));
        assert_eq!(ied("university", "the University of Barcelona"), (1, 18));
        assert_eq!(ied("einstein", "albert einstein jr."), (0, 11));
        assert_eq!(ied("university", "uni"), (7, 7));
        assert_eq!(ied("uriversity", "uni"), (8, 8));
    }
}
