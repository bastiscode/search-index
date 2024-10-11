use anyhow::anyhow;
use itertools::Itertools;
use memmap2::Mmap;
use ordered_float::OrderedFloat;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::{
    cmp::{Ordering, Reverse},
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    iter::once,
    path::Path,
    sync::Arc,
};

use crate::utils::{bm25, tfidf};
use crate::{
    data::IndexData,
    utils::{list_intersection, list_merge, normalize, IndexIter},
};

#[pyclass]
#[derive(Clone)]
pub struct PrefixIndex {
    #[pyo3(get)]
    score: Score,
    data: Arc<IndexData>,
    index: Arc<Mmap>,
    offsets: Arc<[usize]>,
    num_ids: Arc<[usize]>,
    name_to_index: Arc<[usize]>,
    sub_index: Option<Arc<[usize]>>,
}

const U64_SIZE: usize = size_of::<u64>();
const U32_SIZE: usize = size_of::<u32>();
const F32_SIZE: usize = size_of::<f32>();

impl PrefixIndex {
    #[inline]
    fn prefix_cmp(word: &[u8], prefix: &[u8]) -> Ordering {
        // prefix comparison
        // 1. return equal if prefix is prefix of word or equal
        // 2. return less if word is less than prefix
        // 3. return greater if word is greater than prefix
        let mut wi = 0;
        let mut pi = 0;
        while wi < word.len() && pi < prefix.len() {
            match word[wi].cmp(&prefix[pi]) {
                Ordering::Equal => {
                    wi += 1;
                    pi += 1;
                }
                ordering => return ordering,
            }
        }
        if pi == prefix.len() {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }

    #[inline]
    fn parse_scores(&self, start: usize, end: usize) -> Vec<(usize, f32)> {
        self.index[start..end]
            .chunks_exact(U32_SIZE + F32_SIZE)
            .filter_map(|bytes| {
                let mut id_bytes = [0; U32_SIZE];
                let mut score_bytes = [0; F32_SIZE];
                id_bytes.copy_from_slice(&bytes[..U32_SIZE]);
                let id = u32::from_le_bytes(id_bytes) as usize;
                score_bytes.copy_from_slice(&bytes[U32_SIZE..]);
                let score = f32::from_le_bytes(score_bytes);
                self.sub_index
                    .as_ref()
                    .map_or(Some((id, score)), |sub_index| {
                        let idx = self.get_index(id)?;
                        if sub_index.binary_search(&idx).is_err() {
                            None
                        } else {
                            Some((id, score))
                        }
                    })
            })
            .collect()
    }

    #[inline]
    fn get_keyword(&self, index: usize) -> (&[u8], (usize, usize)) {
        let start = self.offsets[index];
        let next_start = self
            .offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| self.index.len());
        let num_ids = self.num_ids[index];
        let end = next_start - num_ids * (U32_SIZE + F32_SIZE);
        (&self.index[start..end], (end, next_start))
    }

    #[inline]
    fn get_scores_on_prefix_match(&self, index: usize, prefix: &[u8]) -> Option<Vec<(usize, f32)>> {
        let (index_keyword, (start, end)) = self.get_keyword(index);
        if let Ordering::Equal = Self::prefix_cmp(index_keyword, prefix) {
            Some(self.parse_scores(start, end))
        } else {
            None
        }
    }

    fn size(&self) -> usize {
        self.offsets.len()
    }

    fn get_exact_matches(&self, keyword: &str) -> Vec<(usize, f32)> {
        let mut lower = 0;
        let mut upper = self.size();
        while lower < upper {
            let mid = (lower + upper) / 2;
            let (mid_keyword, (start, end)) = self.get_keyword(mid);
            match Self::prefix_cmp(mid_keyword, keyword.as_bytes()) {
                Ordering::Less => lower = mid + 1,
                Ordering::Greater => upper = mid,
                Ordering::Equal => return self.parse_scores(start, end),
            }
        }
        vec![]
    }

    fn get_prefix_matches(&self, prefix: &str) -> Vec<(usize, f32)> {
        let mut lower = 0;
        let mut upper = self.size();
        while lower < upper {
            let mid = (lower + upper) / 2;
            let (mid_keyword, (start, end)) = self.get_keyword(mid);
            match Self::prefix_cmp(mid_keyword, prefix.as_bytes()) {
                Ordering::Less => lower = mid + 1,
                Ordering::Greater => upper = mid,
                Ordering::Equal => {
                    // find all prefix matches
                    return (0..mid)
                        .rev()
                        .map(|idx| self.get_scores_on_prefix_match(idx, prefix.as_bytes()))
                        .take_while(|ids| ids.is_some())
                        .flatten()
                        .chain(once(self.parse_scores(start, end)))
                        .chain(
                            // mid to right
                            (mid + 1..self.size())
                                .map(|idx| self.get_scores_on_prefix_match(idx, prefix.as_bytes()))
                                .take_while(|ids| ids.is_some())
                                .flatten(),
                        )
                        .reduce(|a, b| {
                            list_merge(&a, &b, |a, b| {
                                if self.score == Score::Occurrence {
                                    a.max(b)
                                } else {
                                    a + b
                                }
                            })
                        })
                        .unwrap_or_default();
                }
            }
        }
        vec![]
    }

    #[inline]
    fn get_index(&self, name_id: usize) -> Option<usize> {
        self.name_to_index.get(name_id).copied()
    }
}

pub type Ranking = (f32, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Score {
    Occurrence,
    Count,
    TfIdf,
    BM25,
}

impl FromPyObject<'_> for Score {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let score = ob.extract::<String>()?;
        match score.as_str() {
            "count" | "Count" => Ok(Self::Count),
            "occurrence" | "Occurrence" => Ok(Self::Occurrence),
            "tfidf" | "TfIdf" => Ok(Self::TfIdf),
            "bm25" | "BM25" => Ok(Self::BM25),
            _ => Err(PyErr::new::<PyValueError, _>("invalid score type")),
        }
    }
}

impl IntoPy<PyObject> for Score {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Self::Occurrence => "Occurrence",
            Self::Count => "Count",
            Self::TfIdf => "TfIdf",
            Self::BM25 => "BM25",
        }
        .into_py(py)
    }
}

#[pymethods]
impl PrefixIndex {
    // implement functions from pyi interface
    #[staticmethod]
    #[pyo3(signature = (data_file, index_dir, score = Score::Occurrence, k = 1.5, b = 0.75, use_synonyms = true))]
    pub fn build(
        data_file: &str,
        index_dir: &str,
        score: Score,
        k: f32,
        b: f32,
        use_synonyms: bool,
    ) -> anyhow::Result<()> {
        let index_dir = Path::new(index_dir);
        let data = IndexData::new(data_file)?;
        let mut map: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
        let mut lengths = vec![];
        let mut name_to_index_file =
            BufWriter::new(File::create(index_dir.join("index.name-to-index"))?);
        let mut name_id = 0;
        for (i, row) in data.iter().enumerate() {
            let mut split = row.split('\t');
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
                        .split(";;;")
                        .map(normalize),
                );
            }
            for name in names {
                let mut length = 0;
                for keyword in name.split_whitespace().filter(|s| !s.is_empty()) {
                    let list = map.entry(keyword.to_string()).or_default();
                    match list.last_mut() {
                        Some((last_id, last_freq)) if last_id == &name_id => {
                            *last_freq += if score == Score::Occurrence { 0.0 } else { 1.0 };
                        }
                        _ => {
                            list.push((name_id, 1.0));
                        }
                    }
                    length += 1;
                }
                if name_id == u32::MAX {
                    return Err(anyhow!("too many names, max {} supported", u32::MAX));
                }
                name_id += 1;
                lengths.push(length);
                let name_to_index_bytes = u64::try_from(i)?.to_le_bytes();
                name_to_index_file.write_all(&name_to_index_bytes)?;
            }
        }

        let sum_length: u32 = lengths.iter().sum();
        let avg_length = sum_length as f32 / lengths.len().max(1) as f32; // write the map to different files on disk so we can load it memory mapped
        let doc_count = u32::try_from(lengths.len())?;

        for items in map.values_mut() {
            let doc_freq = items.len() as u32;
            for (id, tf) in items.iter_mut() {
                match score {
                    Score::TfIdf => {
                        // convert tf to u64
                        let term_freq = tf.round() as u32;
                        *tf = tfidf(term_freq, doc_freq, doc_count).unwrap_or(0.0);
                    }
                    Score::BM25 => {
                        let term_freq = tf.round() as u32;
                        let doc_length = lengths[*id as usize];
                        *tf = bm25(term_freq, doc_freq, doc_count, avg_length, doc_length, k, b)
                            .unwrap_or(0.0);
                    }
                    _ => {}
                }
            }
        }
        // first sort by key to have them in lexicographical order
        let mut index_file = BufWriter::new(File::create(index_dir.join("index.data"))?);
        let mut offset_file = BufWriter::new(File::create(index_dir.join("index.offsets"))?);
        let mut offset = 0;
        for (keyword, ids) in map.into_iter().sorted_by(|(a, _), (b, _)| a.cmp(b)) {
            index_file.write_all(keyword.as_bytes())?;
            let offset_bytes = u64::try_from(offset)?.to_le_bytes();
            offset_file.write_all(&offset_bytes)?;
            offset += keyword.len();
            let num_id_bytes = u64::try_from(ids.len())?.to_le_bytes();
            offset_file.write_all(&num_id_bytes)?;
            for (id, score) in ids {
                let id_bytes = id.to_le_bytes();
                index_file.write_all(&id_bytes)?;
                offset += id_bytes.len();
                let score_bytes = score.to_le_bytes();
                index_file.write_all(&score_bytes)?;
                offset += score_bytes.len();
            }
        }
        let mut config_file = BufWriter::new(File::create(index_dir.join("index.config"))?);
        let score_bytes = match score {
            Score::Occurrence => 0u32,
            Score::Count => 1u32,
            Score::TfIdf => 2u32,
            Score::BM25 => 3u32,
        }
        .to_le_bytes();
        config_file.write_all(&score_bytes)?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(data_file: &str, index_dir: &str) -> anyhow::Result<Self> {
        let data = Arc::new(IndexData::new(data_file)?);
        let index_dir = Path::new(index_dir);
        let index = Arc::new(unsafe { Mmap::map(&File::open(index_dir.join("index.data"))?)? });
        let mut offsets = vec![];
        let mut num_ids = vec![];
        let offset_bytes = unsafe { Mmap::map(&File::open(index_dir.join("index.offsets"))?)? };
        for chunk in offset_bytes.chunks_exact(2 * U64_SIZE) {
            let mut offset = [0; U64_SIZE];
            let mut num = [0; U64_SIZE];
            offset.copy_from_slice(&chunk[..U64_SIZE]);
            num.copy_from_slice(&chunk[U64_SIZE..]);
            offsets.push(u64::from_le_bytes(offset) as usize);
            num_ids.push(u64::from_le_bytes(num) as usize);
        }
        let mut name_to_index = vec![];
        let name_to_index_bytes =
            unsafe { Mmap::map(&File::open(index_dir.join("index.name-to-index"))?)? };
        for chunk in name_to_index_bytes.chunks_exact(U64_SIZE) {
            let mut index = [0; U64_SIZE];
            index.copy_from_slice(chunk);
            name_to_index.push(u64::from_le_bytes(index) as usize);
        }
        let config = unsafe { Mmap::map(&File::open(index_dir.join("index.config"))?)? };
        let mut score_bytes = [0; U32_SIZE];
        score_bytes.copy_from_slice(&config[..U32_SIZE]);
        let score = match u32::from_le_bytes(score_bytes) {
            0 => Score::Occurrence,
            1 => Score::Count,
            2 => Score::TfIdf,
            3 => Score::BM25,
            _ => return Err(anyhow!("invalid score type")),
        };
        Ok(Self {
            data,
            index,
            score,
            offsets: offsets.into(),
            num_ids: num_ids.into(),
            name_to_index: name_to_index.into(),
            sub_index: None,
        })
    }

    pub fn find_matches(&self, query: &str) -> anyhow::Result<Vec<(usize, Ranking)>> {
        let matches = normalize(query)
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .unique()
            .fold(HashMap::new(), |mut map, keyword| {
                let matches = if keyword.len() < 4 {
                    self.get_exact_matches(keyword)
                } else {
                    self.get_prefix_matches(keyword)
                };
                for (id, score) in matches {
                    *map.entry(id).or_insert(0.0) += score;
                }
                map
            });
        Ok(matches
            .into_iter()
            .filter_map(|(name_id, score)| {
                let index = self.get_index(name_id)?;
                Some((index, score))
            })
            .sorted_by_key(|&(index, score)| (index, Reverse(OrderedFloat(score))))
            .unique_by(|&(index, ..)| index)
            .filter_map(|(index, score)| {
                let id_score = self.data.get_val(index, 1).and_then(|s| s.parse().ok())?;
                Some((index, (score, id_score)))
            })
            .sorted_by_key(|&(id, (score, id_score))| {
                (Reverse(OrderedFloat(score)), Reverse(id_score), id)
            })
            .collect())
    }

    pub fn get_type(&self) -> &str {
        "prefix"
    }

    pub fn get_name(&self, id: usize) -> anyhow::Result<&str> {
        self.data.get_val(id, 0).ok_or_else(|| anyhow!("inalid id"))
    }

    pub fn get_row(&self, id: usize) -> anyhow::Result<&str> {
        self.data.get_row(id).ok_or_else(|| anyhow!("invalid id"))
    }

    pub fn get_val(&self, id: usize, column: usize) -> anyhow::Result<&str> {
        self.data
            .get_val(id, column)
            .ok_or_else(|| anyhow!("invalid id or column"))
    }

    pub fn sub_index_by_ids(&self, ids: Vec<usize>) -> anyhow::Result<Self> {
        if !ids.iter().all(|&id| id < self.data.len()) {
            return Err(anyhow!("invalid ids"));
        }
        let mut ids: Vec<_> = ids.into_iter().unique().sorted().collect();
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
