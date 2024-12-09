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
    data: IndexData,
    index: Arc<Mmap>,
    offsets: Arc<[usize]>,
    lengths: Arc<[u32]>,
    avg_length: f32,
    list_lengths: Arc<[usize]>,
    id_to_index: Arc<[usize]>,
    sub_index: Option<Arc<[usize]>>,
}

#[derive(Debug)]
struct ItemMatch {
    id: usize,
    freq: u32,
    score: f32,
}

#[derive(Debug)]
struct WordItemMatches {
    word_id: usize,
    matches: Vec<ItemMatch>,
}

#[derive(Debug)]
struct Match {
    keyword_id: usize,
    word_id: usize,
    exact: bool,
    freq: u32,
    score: f32,
}

const U64_SIZE: usize = size_of::<u64>();
const U32_SIZE: usize = size_of::<u32>();

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
    fn parse_matches(&self, word_id: usize, score: Score) -> Vec<ItemMatch> {
        let end = self
            .offsets
            .get(word_id + 1)
            .copied()
            .unwrap_or_else(|| self.index.len());
        let list_length = self.list_lengths[word_id];
        let start = end - list_length * U32_SIZE;

        let mut doc_freq = 0;
        let mut last_id = None;
        let inv_list: Vec<_> = self.index[start..end]
            .chunks_exact(U32_SIZE)
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

                if last_id != Some(id) {
                    doc_freq += 1;
                    last_id = Some(id);
                }

                Some(id)
            })
            .collect();

        let doc_count = self.data.len() as u32;
        let mut freq = 0;
        let mut matches = vec![];
        for i in 0..inv_list.len() {
            freq += 1;
            let id = inv_list[i];
            if inv_list.get(i + 1) == Some(&id) {
                continue;
            }

            let score = match score {
                Score::Count | Score::Occurrence => freq as f32,
                Score::TfIdf => tfidf(freq, doc_freq, doc_count).unwrap_or(0.0),
                Score::BM25 => bm25(
                    freq,
                    doc_freq,
                    doc_count,
                    self.avg_length,
                    self.lengths[id],
                    1.5,
                    0.75,
                )
                .unwrap_or(0.0),
            };

            matches.push(ItemMatch { id, freq, score });
            freq = 0;
        }
        println!(
            "doc_freq: {doc_freq}, doc_count: {doc_count} matches: {:#?}",
            matches
        );
        matches
    }

    #[inline]
    fn get_keyword(&self, index: usize) -> &[u8] {
        let start = self.offsets[index];
        let next_start = self
            .offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| self.index.len());
        let num_ids = self.list_lengths[index];
        let end = next_start - num_ids * U32_SIZE;
        &self.index[start..end]
    }

    fn size(&self) -> usize {
        self.offsets.len()
    }

    fn get_matches(
        &self,
        prefix: &str,
        score: Score,
    ) -> (Option<WordItemMatches>, Vec<WordItemMatches>) {
        let mut exact_matches = None;
        let mut prefix_matches = vec![];
        let lower = match lower_bound(0, self.size(), |idx| {
            self.get_keyword(idx).cmp(prefix.as_bytes())
        }) {
            None => return (exact_matches, prefix_matches),
            Some((word_id, true)) => {
                exact_matches = Some(WordItemMatches {
                    word_id,
                    matches: self.parse_matches(word_id, score),
                });
                word_id + 1
            }
            Some((index, false)) => index,
        };

        // only exact matches for short keywords, because they would
        // have too many prefix matches
        if prefix.len() < MIN_PREFIX_MATCHES_LEN {
            return (exact_matches, prefix_matches);
        }

        let upper = upper_bound(lower, self.size(), |idx| {
            Self::prefix_cmp(self.get_keyword(idx), prefix.as_bytes())
        })
        .unwrap_or_else(|| self.size());

        for word_id in lower..upper {
            prefix_matches.push(WordItemMatches {
                word_id,
                matches: self.parse_matches(word_id, score),
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
    #[pyo3(signature = (data_file, index_dir, use_synonyms = true))]
    pub fn build(data_file: &str, index_dir: &str, use_synonyms: bool) -> anyhow::Result<()> {
        let index_dir = Path::new(index_dir);
        let data = IndexData::new(data_file)?;
        let mut inv_lists: HashMap<String, Vec<u32>> = HashMap::new();
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
                for word in name.split_whitespace() {
                    let inv_list = inv_lists.entry(word.to_string()).or_default();
                    inv_list.push(id);
                    length += 1;
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

        // first sort by key to have them in lexicographical order
        let mut index_file = BufWriter::new(File::create(index_dir.join("index.data"))?);
        let mut offset_file = BufWriter::new(File::create(index_dir.join("index.offsets"))?);
        let mut offset = 0;
        for (keyword, inv_list) in inv_lists.into_iter().sorted_by(|(a, _), (b, _)| a.cmp(b)) {
            index_file.write_all(keyword.as_bytes())?;
            let offset_bytes = u64::try_from(offset)?.to_le_bytes();
            offset_file.write_all(&offset_bytes)?;
            offset += keyword.len();
            let inv_list_length = u64::try_from(inv_list.len())?.to_le_bytes();
            offset_file.write_all(&inv_list_length)?;

            for id in inv_list {
                index_file.write_all(&id.to_le_bytes())?;
                offset += U32_SIZE;
            }
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(data_file: &str, index_dir: &str) -> anyhow::Result<Self> {
        let data = IndexData::new(data_file)?;
        let index_dir = Path::new(index_dir);
        let index = Arc::new(unsafe { Mmap::map(&File::open(index_dir.join("index.data"))?)? });
        let mut offsets = vec![];
        let mut list_lengths = vec![];
        let offset_bytes = unsafe { Mmap::map(&File::open(index_dir.join("index.offsets"))?)? };
        for chunk in offset_bytes.chunks_exact(2 * U64_SIZE) {
            let mut offset = [0; U64_SIZE];
            let mut length = [0; U64_SIZE];
            offset.copy_from_slice(&chunk[..U64_SIZE]);
            length.copy_from_slice(&chunk[U64_SIZE..]);
            offsets.push(u64::from_le_bytes(offset) as usize);
            list_lengths.push(u64::from_le_bytes(length) as usize);
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
        let total_length: u32 = lengths.iter().sum();
        let avg_length = total_length as f32 / lengths.len().max(1) as f32;

        Ok(Self {
            data,
            index,
            avg_length,
            offsets: offsets.into(),
            lengths: lengths.into(),
            list_lengths: list_lengths.into(),
            id_to_index: id_to_index.into(),
            sub_index: None,
        })
    }

    #[pyo3(signature = (query, score = Score::Occurrence, k = 1.5, b = 0.75))]
    pub fn find_matches(
        &self,
        query: &str,
        score: Score,
        k: f32,
        b: f32,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let mut num_keywords = 0usize;
        Ok(normalize(query)
            .split_whitespace()
            .filter(|s| s.len() >= MIN_KEYWORD_LEN)
            .enumerate()
            .fold(
                HashMap::<_, Vec<_>>::new(),
                |mut map, (keyword_id, keyword)| {
                    num_keywords += 1;
                    let (exact_matches, prefix_matches) = self.get_matches(keyword, score);

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
                let score = match score {
                    Score::Occurrence => {
                        let num_keywords_matched = matches
                            .iter()
                            .unique_by(|Match { keyword_id, .. }| keyword_id)
                            .count();
                        let num_keywords_unmatched = num_keywords - num_keywords_matched;

                        let num_words_matched: u32 = matches
                            .iter()
                            .unique_by(|Match { word_id, .. }| word_id)
                            .map(|Match { freq, .. }| freq)
                            .sum();
                        let num_words_unmatched = self.lengths[id] - num_words_matched;

                        let mut num_keyword_exact_matches = 0;
                        let mut num_keyword_prefix_matches = 0;
                        for m in matches
                            .iter()
                            .sorted_by_key(|Match { exact, .. }| !exact)
                            .unique_by(|Match { keyword_id, .. }| keyword_id)
                        {
                            if m.exact {
                                num_keyword_exact_matches += 1;
                            } else {
                                num_keyword_prefix_matches += 1;
                            }
                        }

                        1.0 * num_keyword_exact_matches as f32
                            + 0.75 * num_keyword_prefix_matches as f32
                            - 0.5 * num_keywords_unmatched as f32
                            - 0.25 * num_words_unmatched as f32
                    }
                    _ => !unimplemented!(),
                };
                Some((index, score))
                // sort this list by query_id, keyword_id, is_exact
            })
            .sorted_by_key(|&(index, score)| (index, Reverse(OrderedFloat(score))))
            .unique_by(|&(index, ..)| index)
            .sorted_by_key(|&(index, score)| (Reverse(OrderedFloat(score)), index))
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
