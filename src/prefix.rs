use anyhow::anyhow;
use itertools::Itertools;
use memmap2::Mmap;
use ordered_float::OrderedFloat;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyString};
use std::{
    cmp::{Ordering, Reverse},
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

use crate::utils::{bm25, lower_bound, tfidf, upper_bound};
use crate::{
    data::IndexData,
    utils::{list_intersection, normalize, IndexIter},
};

const MIN_KEYWORD_LEN: usize = 3;
// should be larger or equal to MIN_KEYWORD_LEN
const MIN_PREFIX_MATCHES_LEN: usize = 3;

#[pyclass]
#[derive(Clone)]
pub struct PrefixIndex {
    #[pyo3(get)]
    score: Score,
    #[pyo3(get)]
    data: IndexData,
    index: Arc<Mmap>,
    offsets: Arc<[usize]>,
    lengths: Arc<[u32]>,
    num_ids: Arc<[usize]>,
    id_to_index: Arc<[usize]>,
    sub_index: Option<Arc<[usize]>>,
}

struct ItemMatch {
    id: usize,
    freq: u32,
    score: f32,
}

struct WordItemMatches {
    word_id: usize,
    matches: Vec<ItemMatch>,
}

struct Match {
    keyword_id: usize,
    word_id: usize,
    exact: bool,
    freq: u32,
    score: f32,
}

const U64_SIZE: usize = size_of::<u64>();
const U32_SIZE: usize = size_of::<u32>();
const F32_SIZE: usize = size_of::<f32>();
const CHUNK_SIZE: usize = U32_SIZE + U32_SIZE + F32_SIZE;

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
    fn parse_matches(&self, word_id: usize) -> Vec<ItemMatch> {
        let end = self
            .offsets
            .get(word_id + 1)
            .copied()
            .unwrap_or_else(|| self.index.len());
        let num_ids = self.num_ids[word_id];
        let start = end - num_ids * CHUNK_SIZE;
        self.index[start..end]
            .chunks_exact(CHUNK_SIZE)
            .filter_map(|bytes| {
                let mut id_bytes = [0; U32_SIZE];
                id_bytes.copy_from_slice(&bytes[..U32_SIZE]);
                let id = u32::from_le_bytes(id_bytes) as usize;
                if let Some(sub_index) = self.sub_index.as_ref() {
                    let idx = self.get_index(id)?;
                    if sub_index.binary_search(&idx).is_err() {
                        return None;
                    }
                }
                let mut freq_bytes = [0; U32_SIZE];
                freq_bytes.copy_from_slice(&bytes[U32_SIZE..2 * U32_SIZE]);
                let freq = u32::from_le_bytes(freq_bytes);
                let mut score_bytes = [0; F32_SIZE];
                score_bytes.copy_from_slice(&bytes[2 * U32_SIZE..]);
                let score = f32::from_le_bytes(score_bytes);
                Some(ItemMatch { id, freq, score })
            })
            .collect()
    }

    #[inline]
    fn get_keyword(&self, index: usize) -> &[u8] {
        let start = self.offsets[index];
        let next_start = self
            .offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| self.index.len());
        let num_ids = self.num_ids[index];
        let end = next_start - num_ids * CHUNK_SIZE;
        &self.index[start..end]
    }

    fn size(&self) -> usize {
        self.offsets.len()
    }

    fn get_exact_matches(&self, keyword: &str) -> Option<WordItemMatches> {
        match lower_bound(0, self.size(), |idx| {
            self.get_keyword(idx).cmp(keyword.as_bytes())
        }) {
            None | Some((_, false)) => None,
            Some((word_id, true)) => Some(WordItemMatches {
                word_id,
                matches: self.parse_matches(word_id),
            }),
        }
    }

    fn get_prefix_matches(&self, prefix: &str) -> (Option<WordItemMatches>, Vec<WordItemMatches>) {
        let mut exact_matches = None;
        let mut prefix_matches = vec![];
        let lower = match lower_bound(0, self.size(), |idx| {
            self.get_keyword(idx).cmp(prefix.as_bytes())
        }) {
            None => return (exact_matches, prefix_matches),
            Some((word_id, true)) => {
                exact_matches = Some(WordItemMatches {
                    word_id,
                    matches: self.parse_matches(word_id),
                });
                word_id + 1
            }
            Some((index, false)) => index,
        };

        let upper = upper_bound(lower, self.size(), |idx| {
            Self::prefix_cmp(self.get_keyword(idx), prefix.as_bytes())
        })
        .unwrap_or_else(|| self.size());

        for word_id in lower..upper {
            prefix_matches.push(WordItemMatches {
                word_id,
                matches: self.parse_matches(word_id),
            });
        }
        (exact_matches, prefix_matches)
    }

    #[inline]
    fn get_index(&self, id: usize) -> Option<usize> {
        self.id_to_index.get(id).copied()
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
            Self::Occurrence => "occurrence",
            Self::Count => "count",
            Self::TfIdf => "tfidf",
            Self::BM25 => "bm25",
        }
        .into_pyobject(py)
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
        let mut map: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut lengths = vec![];
        let mut id_to_index_file =
            BufWriter::new(File::create(index_dir.join("index.id-to-index"))?);
        let mut lengths_file = BufWriter::new(File::create(index_dir.join("index.lengths"))?);
        let mut id = 0;
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
            let index_bytes = u32::try_from(i)?.to_le_bytes();
            for name in names {
                let mut length: u32 = 0;
                for keyword in name
                    .split_whitespace()
                    .filter(|s| s.len() >= MIN_KEYWORD_LEN)
                {
                    let list = map.entry(keyword.to_string()).or_default();
                    match list.last_mut() {
                        Some((last_id, last_freq)) if last_id == &id => {
                            *last_freq += 1;
                            length += 1;
                        }
                        _ => {
                            list.push((id, 1));
                            length += 1;
                        }
                    }
                }
                if id == u32::MAX {
                    return Err(anyhow!("too many names, max {} supported", u32::MAX));
                }
                id += 1;
                lengths.push(length);
                id_to_index_file.write_all(&index_bytes)?;
                lengths_file.write_all(&length.to_le_bytes())?;
            }
        }

        // some stuff frequired for tf idf and bm25
        let doc_count = id;
        let total_length: f32 = lengths.iter().map(|l| *l as f32).sum();
        let avg_length = total_length / doc_count.max(1) as f32;

        // first sort by key to have them in lexicographical order
        let mut index_file = BufWriter::new(File::create(index_dir.join("index.data"))?);
        let mut offset_file = BufWriter::new(File::create(index_dir.join("index.offsets"))?);
        let mut offset = 0;
        map.into_iter()
            .map(|(keyword, inv_list)| {
                let doc_freq = inv_list.len() as u32;
                let inv_list: Vec<_> = inv_list
                    .into_iter()
                    .map(|(id, freq)| match score {
                        Score::Occurrence | Score::Count => (id, freq, freq as f32),
                        Score::TfIdf => {
                            let tf = tfidf(freq, doc_freq, doc_count).unwrap_or(0.0);
                            (id, freq, tf)
                        }
                        Score::BM25 => {
                            let doc_length = lengths[id as usize];
                            let bm25 =
                                bm25(freq, doc_freq, doc_count, avg_length, doc_length, k, b)
                                    .unwrap_or(0.0);
                            (id, freq, bm25)
                        }
                    })
                    .collect();
                (keyword, inv_list)
            })
            .sorted_by(|(a, _), (b, _)| a.cmp(b))
            .try_for_each(|(keyword, ids)| -> anyhow::Result<_> {
                index_file.write_all(keyword.as_bytes())?;
                let offset_bytes = u64::try_from(offset)?.to_le_bytes();
                offset_file.write_all(&offset_bytes)?;
                offset += keyword.len();
                let num_id_bytes = u64::try_from(ids.len())?.to_le_bytes();
                offset_file.write_all(&num_id_bytes)?;
                for (id, freq, score) in ids {
                    index_file.write_all(&id.to_le_bytes())?;
                    offset += U32_SIZE;

                    index_file.write_all(&freq.to_le_bytes())?;
                    offset += U32_SIZE;

                    index_file.write_all(&score.to_le_bytes())?;
                    offset += F32_SIZE;
                }
                Ok(())
            })?;

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
        let data = IndexData::new(data_file)?;
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
        let mut id_to_index = vec![];
        let id_to_index_bytes =
            unsafe { Mmap::map(&File::open(index_dir.join("index.id-to-index"))?)? };
        for chunk in id_to_index_bytes.chunks_exact(U32_SIZE) {
            let mut index = [0; U32_SIZE];
            index.copy_from_slice(chunk);
            let index = u32::from_le_bytes(index) as usize;
            id_to_index.push(index);
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
        Ok(Self {
            data,
            index,
            score,
            offsets: offsets.into(),
            lengths: lengths.into(),
            num_ids: num_ids.into(),
            id_to_index: id_to_index.into(),
            sub_index: None,
        })
    }

    pub fn find_matches(&self, query: &str) -> anyhow::Result<Vec<(usize, f32)>> {
        let mut num_keywords = 0usize;
        Ok(normalize(query)
            .split_whitespace()
            .filter(|s| s.len() >= MIN_KEYWORD_LEN)
            .enumerate()
            .fold(
                HashMap::<usize, Vec<_>>::new(),
                |mut map, (keyword_id, keyword)| {
                    num_keywords += 1;
                    let (exact_matches, prefix_matches) = if keyword.len() < MIN_PREFIX_MATCHES_LEN
                    {
                        // exact matches for keywords with this length
                        // to avoid too many prefix matches
                        (self.get_exact_matches(keyword), vec![])
                    } else {
                        self.get_prefix_matches(keyword)
                    };

                    // group matches by id
                    if let Some(WordItemMatches { word_id, matches }) = exact_matches {
                        for kw_match in matches {
                            map.entry(kw_match.id).or_default().push(Match {
                                keyword_id,
                                word_id,
                                exact: true,
                                freq: kw_match.freq,
                                score: kw_match.score,
                            });
                        }
                    }

                    for WordItemMatches { word_id, matches } in prefix_matches {
                        for kw_match in matches {
                            map.entry(kw_match.id).or_default().push(Match {
                                keyword_id,
                                word_id,
                                exact: false,
                                freq: kw_match.freq,
                                score: kw_match.score,
                            });
                        }
                    }

                    map
                },
            )
            .into_iter()
            .filter_map(|(id, matches)| {
                // scores is a list of tuples (query_id, keyword_id, is_exact, freq, score)
                let index = self.get_index(id)?;
                // let id_score = self.data.get_val(index, 1).and_then(|s| s.parse().ok())?;
                let score = match self.score {
                    Score::Occurrence => {
                        // count number of
                        let num_query_matched = matches
                            .iter()
                            .unique_by(|Match { keyword_id, .. }| keyword_id)
                            .count();
                        let num_query_unmatched = num_keywords - num_query_matched;

                        let mut num_exact_matches = 0;
                        let mut num_prefix_matches = 0;
                        // bring exact matches to front and then make them unique
                        // by word_id
                        for m in matches
                            .iter()
                            .sorted_by_key(|Match { exact, .. }| !exact)
                            .unique_by(|Match { word_id, .. }| word_id)
                        {
                            if m.exact {
                                num_exact_matches += m.freq;
                            } else {
                                num_prefix_matches += m.freq;
                            }
                        }
                        let num_item_unmatched =
                            self.lengths[id] - num_exact_matches - num_prefix_matches;
                        1.0 * num_exact_matches as f32 + 0.75 * num_prefix_matches as f32
                            - 0.5 * num_query_unmatched as f32
                            - 0.25 * num_item_unmatched as f32
                    }
                    _ => !unimplemented!(),
                };
                Some((index, score))
                // sort this list by query_id, keyword_id, is_exact
            })
            .sorted_by_key(|&(index, ranking)| (index, Reverse(OrderedFloat(ranking))))
            .unique_by(|&(index, ..)| index)
            .sorted_by_key(|&(.., ranking)| Reverse(OrderedFloat(ranking)))
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

    #[getter]
    pub fn min_keyword_length(&self) -> usize {
        MIN_KEYWORD_LEN
    }
}
