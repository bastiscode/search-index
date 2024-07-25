use anyhow::anyhow;
use core::cmp::Reverse;
use itertools::Itertools;
use memmap2::Mmap;
use pyo3::prelude::*;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::sync::Arc;

#[pyclass]
#[derive(Default, Serialize, Deserialize)]
pub struct QGramIndex {
    #[pyo3(get)]
    q: usize,
    padding: Vec<char>,
    inverted_lists: HashMap<String, Vec<(u32, u32)>>,
    #[serde(skip)]
    mmap: Option<Arc<Mmap>>,
    data: Vec<usize>,
    syn_to_ent: Vec<(u32, u16)>,
    #[pyo3(get)]
    use_syns: bool,
    #[pyo3(get)]
    distance: Distance,
}

#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Distance {
    #[default]
    PED,
    IED,
}

impl IntoPy<PyObject> for Distance {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Distance::PED => "ped",
            Distance::IED => "ied",
        }
        .into_py(py)
    }
}

impl FromPyObject<'_> for Distance {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let s = ob.extract::<String>()?;
        match s.as_str() {
            "ped" | "PED" => Ok(Distance::PED),
            "ied" | "IED" => Ok(Distance::IED),
            _ => Err(anyhow!("invalid distance type").into()),
        }
    }
}

impl QGramIndex {
    fn new(q: usize, use_syns: bool, distance: Distance) -> anyhow::Result<Self> {
        if q == 0 {
            return Err(anyhow!("q must be positive"));
        }
        Ok(QGramIndex {
            q,
            padding: vec!['$'; q - 1],
            use_syns,
            distance,
            ..Self::default()
        })
    }

    fn compute_q_grams(&self, name: &str, full: bool) -> Vec<String> {
        assert!(!name.is_empty());
        let padded: Vec<_> = match self.distance {
            Distance::PED => self
                .padding
                .clone()
                .into_iter()
                .chain(name.chars())
                .collect(),
            Distance::IED => name.chars().collect(),
        };
        let q = self.q.min(padded.len());
        let start = if q < self.q && full { 1 } else { q };
        (start..=q)
            .flat_map(|i| padded.windows(i).map(|window| window.iter().collect()))
            .collect()
    }

    #[inline]
    fn get_next_line<'l>(&self, bytes: &'l [u8]) -> anyhow::Result<&'l str> {
        let len = bytes
            .iter()
            .position(|&b| b == b'\n')
            .unwrap_or(bytes.len());
        Ok(std::str::from_utf8(&bytes[..len])?)
    }

    #[inline]
    fn get_bytes_by_idx(&self, idx: u32) -> anyhow::Result<(usize, &[u8])> {
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| anyhow!("index not built or loaded"))?;
        let offset = self
            .data
            .get(usize::try_from(idx)?)
            .ok_or_else(|| anyhow!("invalid idx"))?;
        let slice = &mmap[*offset..];
        Ok((*offset, slice))
    }

    #[inline]
    fn get_bytes_by_id(&self, id: u32) -> anyhow::Result<(usize, &[u8])> {
        self.syn_to_ent
            .get(usize::try_from(id)?)
            .ok_or_else(|| anyhow!("invalid id"))
            .and_then(|&(idx, _)| self.get_bytes_by_idx(idx))
    }

    fn get_score_by_id(&self, id: u32) -> anyhow::Result<usize> {
        // score is the second field in a line
        self.get_bytes_by_id(id).and_then(|(_, bytes)| {
            bytes
                .split(|&b| b == b'\t')
                .nth(1)
                .ok_or_else(|| anyhow!("missing score field"))
                .and_then(|score| std::str::from_utf8(score).map_err(Into::into))
                .and_then(|score| score.parse().map_err(Into::into))
        })
    }

    fn normalize(&self, name: &str) -> String {
        name.to_lowercase()
    }

    fn merge_lists(&self, q_grams: Vec<String>) -> Vec<(u32, u32)> {
        q_grams
            .into_iter()
            .fold(HashMap::new(), |mut counts, q_gram| {
                *counts.entry(q_gram).or_insert(0) += 1;
                counts
            })
            .into_iter()
            .filter_map(|(q_gram, count)| {
                let intersect_list = self.inverted_lists.get(&q_gram)?;
                Some(
                    intersect_list
                        .iter()
                        .map(move |&(id, freq)| (id, count.min(freq))),
                )
            })
            .flatten()
            .sorted()
            .fold(vec![], |mut cur, (id, freq)| {
                match cur.last_mut() {
                    Some((last_id, last_freq)) if last_id == &id => *last_freq += freq,
                    _ => cur.push((id, freq)),
                }
                cur
            })
    }

    fn add_line(
        &mut self,
        line: &str,
        obj_id: u32,
        offset: usize,
        mut syn_id: u32,
    ) -> anyhow::Result<u32> {
        self.data.push(offset);
        let mut fields = line.trim_end_matches(['\r', '\n']).split('\t');
        // name is first field
        let name = fields.next().ok_or(anyhow!("missing name field"))?;
        if name.is_empty() {
            return Err(anyhow!("name must not be empty"));
        }
        let mut names = vec![name];
        // syns are third field
        let syns = fields.nth(1).ok_or(anyhow!("missing synonyms field"))?;
        if self.use_syns {
            names.extend(syns.split(';'));
        }

        let start_syn_id = syn_id;
        for (syn_idx, name) in names.into_iter().enumerate() {
            // assumes names are already properly normalized (see normalize fn)
            let norm = self.normalize(name);
            if norm.is_empty() && syn_idx == 0 {
                return Err(anyhow!(
                    "normalized name '{norm}' in line {} is empty",
                    obj_id + 1
                ));
            } else if norm.is_empty() {
                continue;
            } else if syn_idx > u16::MAX as usize {
                return Err(anyhow!(
                    "too many synonyms for '{name}' in line {}, maximum is {}",
                    obj_id + 1,
                    u16::MAX
                ));
            }
            for qgram in self.compute_q_grams(&norm, true) {
                let list = self.inverted_lists.entry(qgram).or_default();
                match list.last_mut() {
                    Some((last_syn_id, last_freq)) if last_syn_id == &syn_id => {
                        *last_freq += 1;
                    }
                    _ => {
                        list.push((syn_id, 1));
                    }
                }
            }

            self.syn_to_ent.push((obj_id, u16::try_from(syn_idx)?));
            syn_id += 1;
        }
        Ok(syn_id - start_syn_id)
    }

    pub fn sub_index_by_indices(&self, indices: &[u32]) -> anyhow::Result<Self> {
        let mut sub_index = Self::new(self.q, self.use_syns, self.distance)?;
        let mut syn_id = 0;
        for (obj_id, idx) in indices.iter().enumerate() {
            let (offset, bytes) = self.get_bytes_by_idx(*idx)?;
            let line = self.get_next_line(bytes)?;
            let n = sub_index.add_line(line, u32::try_from(obj_id)?, offset, syn_id)?;
            syn_id += n;
        }
        sub_index.mmap.clone_from(&self.mmap);
        Ok(sub_index)
    }
}

#[pymethods]
impl QGramIndex {
    #[new]
    #[pyo3(signature = (q, use_syns = true, distance = Distance::PED))]
    pub fn py_new(q: usize, use_syns: bool, distance: Distance) -> anyhow::Result<Self> {
        Self::new(q, use_syns, distance)
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    pub fn build(&mut self, data_file: &str) -> anyhow::Result<()> {
        if self.mmap.is_some() {
            return Err(anyhow!("index already built or loaded"));
        }
        let file = File::open(data_file)?;
        self.mmap = Some(Arc::new(unsafe { Mmap::map(&file)? }));
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut offset = reader.read_line(&mut line)?;
        let mut obj_id = 0;
        let mut syn_id = 0;
        loop {
            line.clear();
            let line_len = reader.read_line(&mut line)?;
            if line_len == 0 {
                break;
            }
            let n = self.add_line(&line, obj_id, offset, syn_id)?;
            offset += line_len;
            obj_id += 1;
            syn_id += n;
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(index_file: &str, data_file: &str) -> anyhow::Result<()> {
        let index_mmap = unsafe { Mmap::map(&File::open(index_file)?)? };
        let mut index: Self = rmp_serde::decode::from_slice(&index_mmap)?;
        index.mmap = Some(Arc::new(unsafe { Mmap::map(&File::open(data_file)?)? }));
        Ok(())
    }

    pub fn save(&self, index_file: &str) -> anyhow::Result<()> {
        let mut writer = BufWriter::new(File::create(index_file)?);
        rmp_serde::encode::write(&mut writer, self)?;
        Ok(())
    }

    pub fn find_matches(
        &self,
        query: &str,
        delta: Option<usize>,
    ) -> anyhow::Result<Vec<(u32, usize)>> {
        let query = self.normalize(query);
        if query.is_empty() {
            return Err(anyhow!("normalized query is empty"));
        }
        let q_grams = self.compute_q_grams(&query, false);
        let delta = delta.unwrap_or_else(|| q_grams.len() / (self.q + 1));
        let thres = u32::try_from(q_grams.len().saturating_sub(self.q * delta))?;
        if thres == 0 {
            return Err(anyhow!(
                "threshold for filtering distance computations must be positive, lower delta"
            ));
        }

        let merged = self.merge_lists(q_grams);

        let mut matches: Vec<_> = merged
            .into_par_iter()
            .filter_map(|(syn_id, num_qgrams)| {
                if num_qgrams < thres {
                    return None;
                }
                let name = self.get_name_by_id(syn_id).ok()?;
                let norm = self.normalize(name);
                let dist = match self.distance {
                    Distance::PED => ped(&query, &norm, delta),
                    Distance::IED => ied(&query, &norm),
                };
                if dist <= delta {
                    Some((syn_id, dist))
                } else {
                    None
                }
            })
            .collect();

        if self.use_syns {
            matches = matches
                .into_iter()
                .sorted_by_key(|&(syn_id, _)| self.syn_to_ent[syn_id as usize])
                .fold(vec![], |mut acc, (syn_id, dist)| {
                    if acc.is_empty() {
                        acc.push((syn_id, dist));
                    } else if let Some(&(last_syn_id, ..)) = acc.last() {
                        if self.syn_to_ent[last_syn_id as usize] != self.syn_to_ent[syn_id as usize]
                        {
                            acc.push((syn_id, dist));
                        }
                    }
                    acc
                });
        }

        matches.sort_by_key(|&(syn_id, dist)| (dist, Reverse(self.get_score_by_id(syn_id).ok())));
        Ok(matches)
    }

    pub fn get_data_by_id(&self, id: u32) -> anyhow::Result<String> {
        self.get_bytes_by_id(id)
            .and_then(|(_, bytes)| self.get_next_line(bytes))
            .map(|s| s.to_string())
    }

    pub fn get_data_by_idx(&self, idx: u32) -> anyhow::Result<String> {
        self.get_bytes_by_idx(idx)
            .and_then(|(_, bytes)| self.get_next_line(bytes))
            .map(|s| s.to_string())
    }

    #[pyo3(name = "sub_index_by_indices")]
    pub fn py_sub_index_by_indices(&self, indices: Vec<u32>) -> anyhow::Result<Self> {
        self.sub_index_by_indices(&indices)
    }

    pub fn get_name_by_id(&self, id: u32) -> anyhow::Result<&str> {
        // name is the first field in a line
        let &(idx, syn_idx) = self
            .syn_to_ent
            .get(usize::try_from(id)?)
            .ok_or_else(|| anyhow!("invalid id"))?;
        self.get_bytes_by_idx(idx).and_then(|(_, bytes)| {
            let mut split = bytes.split(|&b| b == b'\t');
            if syn_idx == 0 {
                split
                    .next()
                    .ok_or_else(|| anyhow!("missing name field"))
                    .and_then(|name| std::str::from_utf8(name).map_err(Into::into))
            } else {
                let syns = split
                    .nth(2)
                    .ok_or_else(|| anyhow!("missing syn field"))
                    .and_then(|syns| std::str::from_utf8(syns).map_err(Into::into))?;
                syns.split(';')
                    .nth(usize::from(syn_idx - 1))
                    .ok_or_else(|| anyhow!("missing syn field"))
            }
        })
    }

    pub fn get_idx_by_id(&self, id: u32) -> anyhow::Result<u32> {
        self.syn_to_ent
            .get(usize::try_from(id)?)
            .map(|&(idx, _)| idx)
            .ok_or_else(|| anyhow!("invalid id"))
    }
}

#[inline]
fn ped(x: &str, y: &str, delta: usize) -> usize {
    let x: Vec<_> = x.chars().collect();
    let y: Vec<_> = y.chars().collect();
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
fn ied(x: &str, y: &str) -> usize {
    let x: Vec<_> = x.chars().collect();
    let y: Vec<_> = y.chars().collect();
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
    matrix[(n - 1) * m..]
        .iter()
        .copied()
        .min()
        .unwrap_or_default()
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
        assert_eq!(ied("frei", "frei"), 0);
        assert_eq!(ied("frei", "freiburg"), 0);
        assert_eq!(ied("frei", "breifurg"), 1);
        assert_eq!(ied("freiburg", "stuttgart"), 7);
        assert_eq!(ied("", "freiburg"), 0);
        assert_eq!(ied("", ""), 0);
        assert_eq!(ied("cat", "dog"), 3);
        assert_eq!(ied("cat", "the cat sat on the mat"), 0);
        assert_eq!(ied("university", "the University of Barcelona"), 1);
        assert_eq!(ied("einstein", "albert einstein jr."), 0);
    }
}

#[pymodule]
fn qgram_index(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    let _ = m.add_class::<QGramIndex>();
    Ok(())
}
