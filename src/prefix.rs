use anyhow::anyhow;
use itertools::Itertools;
use memmap2::Mmap;
use ordered_float::OrderedFloat;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyString, PyTuple},
};
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
    utils::{list_intersection, normalize, IndexIter},
};

const MIN_KEYWORD_LEN: usize = 2;
// should be larger or equal to MIN_KEYWORD_LEN
const MIN_PREFIX_MATCHES_LEN: usize = 3;

#[pyclass]
#[derive(Clone)]
pub struct PrefixIndex {
    #[pyo3(get)]
    score: Score,
    chunk_size: usize,
    data: Arc<IndexData>,
    index: Arc<Mmap>,
    offsets: Arc<[usize]>,
    lengths: Arc<[u32]>,
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
    fn parse_scores(&self, start: usize, end: usize) -> Vec<(usize, ScoreData)> {
        self.index[start..end]
            .chunks_exact(self.chunk_size)
            .filter_map(|bytes| {
                let mut id_bytes = [0; U32_SIZE];
                id_bytes.copy_from_slice(&bytes[..U32_SIZE]);
                let id = u32::from_le_bytes(id_bytes) as usize;
                let score = match self.score {
                    Score::Occurrence => ScoreData::Int(1),
                    Score::Count => {
                        let mut count_bytes = [0; U32_SIZE];
                        count_bytes.copy_from_slice(&bytes[U32_SIZE..]);
                        ScoreData::Int(u32::from_le_bytes(count_bytes))
                    }
                    Score::TfIdf | Score::BM25 => {
                        let mut score_bytes = [0; F32_SIZE];
                        score_bytes.copy_from_slice(&bytes[U32_SIZE..]);
                        ScoreData::Float(OrderedFloat(f32::from_le_bytes(score_bytes)))
                    }
                };
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
        let end = next_start - num_ids * self.chunk_size;
        (&self.index[start..end], (end, next_start))
    }

    #[inline]
    fn get_scores_on_prefix_match(
        &self,
        index: usize,
        prefix: &[u8],
    ) -> Option<Vec<(usize, ScoreData)>> {
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

    fn get_exact_matches(&self, keyword: &str) -> Option<(usize, Vec<(usize, ScoreData)>)> {
        let mut lower = 0;
        let mut upper = self.size();
        while lower < upper {
            let mid = (lower + upper) / 2;
            let (mid_keyword, (start, end)) = self.get_keyword(mid);
            match mid_keyword.cmp(keyword.as_bytes()) {
                Ordering::Less => lower = mid + 1,
                Ordering::Greater => upper = mid,
                Ordering::Equal => return Some((mid, self.parse_scores(start, end))),
            }
        }
        None
    }

    fn get_prefix_matches(&self, prefix: &str) -> Vec<(usize, Vec<(usize, ScoreData)>)> {
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
                        .map(|idx| {
                            self.get_scores_on_prefix_match(idx, prefix.as_bytes())
                                .map(|scores| (idx, scores))
                        })
                        .take_while(|ids| ids.is_some())
                        .flatten()
                        .chain(once((mid, self.parse_scores(start, end))))
                        .chain(
                            // mid to right
                            (mid + 1..self.size())
                                .map(|idx| {
                                    self.get_scores_on_prefix_match(idx, prefix.as_bytes())
                                        .map(|scores| (idx, scores))
                                })
                                .take_while(|ids| ids.is_some())
                                .flatten(),
                        )
                        .collect();
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

#[derive(Clone, Copy, Debug, Eq)]
pub enum Ranking {
    // frequency, diff, id_score
    ByFrequency(u32, u32, usize),
    // score, id_score
    ByScore(OrderedFloat<f32>, usize),
}

impl PartialEq for Ranking {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ByFrequency(a, b, c), Self::ByFrequency(x, y, z)) => a == x && b == y && c == z,
            (Self::ByScore(a, b), Self::ByScore(x, y)) => a == x && b == y,
            _ => unreachable!("ranking type mismatch"),
        }
    }
}

impl PartialOrd for Ranking {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranking {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::ByFrequency(a, b, c), Self::ByFrequency(x, y, z)) => {
                // freq -> higher is better, diff -> lower is better, id_score -> higher is better
                (a, Reverse(b), c).cmp(&(x, Reverse(y), z))
            }
            (Self::ByScore(a, b), Self::ByScore(x, y)) => (a, b).cmp(&(x, y)),
            _ => unreachable!("ranking type mismatch"),
        }
    }
}

impl<'py> IntoPyObject<'py> for Ranking {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Self::ByFrequency(freq, diff, id_score) => {
                let tuple = (freq, diff, id_score);
                tuple.into_pyobject(py)
            }
            Self::ByScore(score, id_score) => {
                let tuple = (score.into_inner(), id_score);
                tuple.into_pyobject(py)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Score {
    Occurrence,
    Count,
    TfIdf,
    BM25,
}

impl FromPyObject<'_> for Score {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let score: String = ob.extract()?;
        match score.as_str() {
            "count" | "Count" => Ok(Self::Count),
            "occurrence" | "Occurrence" => Ok(Self::Occurrence),
            "tfidf" | "TfIdf" => Ok(Self::TfIdf),
            "bm25" | "BM25" => Ok(Self::BM25),
            _ => Err(PyErr::new::<PyValueError, _>("invalid score type")),
        }
    }
}

impl<'py> IntoPyObject<'py> for Score {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Self::Occurrence => "Occurrence",
            Self::Count => "Count",
            Self::TfIdf => "TfIdf",
            Self::BM25 => "BM25",
        }
        .into_pyobject(py)
    }
}

// define enum for score data
//  - u32 for occurrence and count
//  - f32 for tfidf and bm25
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord)]
enum ScoreData {
    Int(u32),
    Float(OrderedFloat<f32>),
}

impl ScoreData {
    fn new(score: Score) -> Self {
        match score {
            Score::Occurrence | Score::Count => Self::Int(0),
            Score::TfIdf | Score::BM25 => Self::Float(OrderedFloat(0.0)),
        }
    }

    fn to_le_bytes(self) -> [u8; 4] {
        match self {
            Self::Int(val) => val.to_le_bytes(),
            Self::Float(val) => val.to_le_bytes(),
        }
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
        let mut map: HashMap<String, Vec<(u32, ScoreData)>> = HashMap::new();
        let mut lengths = vec![];
        let mut name_to_index_file =
            BufWriter::new(File::create(index_dir.join("index.name-to-index"))?);
        let mut lengths_file = BufWriter::new(File::create(index_dir.join("index.lengths"))?);
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
                let mut length: u32 = 0;
                for keyword in name
                    .split_whitespace()
                    .filter(|s| s.len() >= MIN_KEYWORD_LEN)
                {
                    let list = map.entry(keyword.to_string()).or_default();
                    match list.last_mut() {
                        Some((last_id, ScoreData::Int(last_freq))) if last_id == &name_id => {
                            // for occurence score:
                            // - frequency if it occurrs at least once in the document
                            // - length is the number of unique keywords in the document
                            if score != Score::Occurrence {
                                *last_freq += 1;
                                length += 1;
                            }
                        }
                        Some((_, ScoreData::Float(_))) => {
                            unreachable!("score data is always int at this point")
                        }
                        _ => {
                            list.push((name_id, ScoreData::Int(1)));
                            length += 1;
                        }
                    }
                }
                if name_id == u32::MAX {
                    return Err(anyhow!("too many names, max {} supported", u32::MAX));
                }
                name_id += 1;
                lengths.push(length);
                let name_to_index_bytes = u32::try_from(i)?.to_le_bytes();
                name_to_index_file.write_all(&name_to_index_bytes)?;
                lengths_file.write_all(&length.to_le_bytes())?;
            }
        }

        if score == Score::TfIdf || score == Score::BM25 {
            let doc_count = name_id;
            let total_length: f32 = lengths.iter().map(|l| *l as f32).sum();
            let avg_length = total_length / doc_count.max(1) as f32;

            for items in map.values_mut() {
                let doc_freq = items.len() as u32;
                for (name_id, score_data) in items.iter_mut() {
                    let ScoreData::Int(term_freq) = score_data else {
                        unreachable!("score data is always int at this point")
                    };
                    match score {
                        Score::TfIdf => {
                            // convert tf to u64
                            let tf = tfidf(*term_freq, doc_freq, doc_count).unwrap_or(0.0);
                            *score_data = ScoreData::Float(OrderedFloat(tf));
                        }
                        Score::BM25 => {
                            let doc_length = lengths[*name_id as usize];
                            let bm25 = bm25(
                                *term_freq, doc_freq, doc_count, avg_length, doc_length, k, b,
                            )
                            .unwrap_or(0.0);
                            *score_data = ScoreData::Float(OrderedFloat(bm25));
                        }
                        _ => unreachable!("only execute for tfidf and bm25"),
                    }
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
            for (id, score_data) in ids {
                let id_bytes = id.to_le_bytes();
                index_file.write_all(&id_bytes)?;
                offset += id_bytes.len();
                if score != Score::Occurrence {
                    let score_bytes = score_data.to_le_bytes();
                    index_file.write_all(&score_bytes)?;
                    offset += score_bytes.len();
                }
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
        for chunk in name_to_index_bytes.chunks_exact(U32_SIZE) {
            let mut index = [0; U32_SIZE];
            index.copy_from_slice(chunk);
            let index = u32::from_le_bytes(index) as usize;
            name_to_index.push(index);
        }

        let mut lengths = vec![];
        let length_bytes = unsafe { Mmap::map(&File::open(index_dir.join("index.lengths"))?)? };
        for chunk in length_bytes.chunks_exact(U32_SIZE) {
            let mut length = [0; U32_SIZE];
            length.copy_from_slice(chunk);
            lengths.push(u32::from_le_bytes(length));
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
        let chunk_size = match score {
            Score::Occurrence => U32_SIZE,
            Score::Count => U32_SIZE + U32_SIZE,
            _ => U32_SIZE + F32_SIZE,
        };
        Ok(Self {
            data,
            index,
            score,
            chunk_size,
            offsets: offsets.into(),
            lengths: lengths.into(),
            num_ids: num_ids.into(),
            name_to_index: name_to_index.into(),
            sub_index: None,
        })
    }

    pub fn find_matches(&self, query: &str) -> anyhow::Result<Vec<(usize, Ranking)>> {
        Ok(normalize(query)
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .unique()
            .fold(HashMap::new(), |mut map, keyword| {
                let matching = if keyword.len() < MIN_KEYWORD_LEN {
                    // skip too small keywords
                    return map;
                } else if keyword.len() < MIN_PREFIX_MATCHES_LEN {
                    // exact matches for keywords with this length
                    // to avoid too many prefix matches
                    let Some(m) = self.get_exact_matches(keyword) else {
                        // no exact matches
                        return map;
                    };
                    vec![m]
                } else {
                    self.get_prefix_matches(keyword)
                };
                for (keyword, keyword_matches) in matching {
                    for (id, score) in keyword_matches {
                        let current = map
                            .entry((keyword, id))
                            .or_insert(ScoreData::new(self.score));
                        match (current, score) {
                            (ScoreData::Int(a), ScoreData::Int(b)) => {
                                *a = b.max(*a);
                            }
                            (ScoreData::Float(a), ScoreData::Float(b)) => {
                                *a = b.max(*a);
                            }
                            _ => unreachable!("score data type mismatch"),
                        }
                    }
                }
                map
            })
            .into_iter()
            .fold(HashMap::new(), |mut map, ((_, id), score)| {
                // sum score across all keyword for each id
                let current = map.entry(id).or_insert(ScoreData::new(self.score));
                match (current, score) {
                    (ScoreData::Int(a), ScoreData::Int(b)) => {
                        *a += b;
                    }
                    (ScoreData::Float(a), ScoreData::Float(b)) => {
                        *a += b;
                    }
                    _ => unreachable!("score data type mismatch"),
                }
                map
            })
            .into_iter()
            .filter_map(|(id, score)| {
                let index = self.get_index(id)?;
                let id_score = self.data.get_val(index, 1).and_then(|s| s.parse().ok())?;
                let ranking = match score {
                    ScoreData::Int(score) => Ranking::ByFrequency(
                        score,
                        self.lengths[id as usize].saturating_sub(score),
                        id_score,
                    ),
                    ScoreData::Float(score) => Ranking::ByScore(score, id_score),
                };
                Some((index, ranking))
            })
            .sorted_by_key(|&(index, ranking)| (index, Reverse(ranking)))
            .unique_by(|&(index, ..)| index)
            .sorted_by_key(|&(.., ranking)| Reverse(ranking))
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
