use anyhow::anyhow;
use itertools::Itertools;
use log::debug;
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
    time::Instant,
};

use crate::utils::{bm25, lower_bound, tfidf, upper_bound};
use crate::{
    data::IndexData,
    utils::{list_intersection, normalize, IndexIter},
};

const MIN_KEYWORD_LEN: usize = 3;

#[pyclass]
#[derive(Clone)]
pub struct PrefixIndex {
    #[pyo3(get)]
    data: IndexData,
    keywords: Arc<Mmap>,
    inv_lists: Arc<Mmap>,
    keyword_offsets: Arc<[usize]>,
    inv_list_offsets: Arc<[usize]>,
    lengths: Arc<[u32]>,
    avg_length: f32,
    id_to_index: Arc<[usize]>,
    sub_index: Option<Arc<[usize]>>,
}

#[derive(Debug)]
struct ItemMatch {
    id: usize,
    freq: u32,
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
}

const U32_SIZE: usize = size_of::<u32>();

impl PrefixIndex {
    #[inline]
    fn prefix_cmp(word: &str, prefix: &str) -> Ordering {
        // prefix comparison
        // 1. return equal if prefix is prefix of word or equal
        // 2. return less if word is less than prefix
        // 3. return greater if word is greater than prefix
        let mut wi = 0;
        let mut pi = 0;

        let word = word.as_bytes();
        let prefix = prefix.as_bytes();

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
    fn get_name_or_synonym(&self, id: usize) -> Option<&str> {
        let index = self.get_index(id)?;
        let Some((name_id, true)) = lower_bound(0, self.id_to_index.len(), |i| {
            self.id_to_index[i].cmp(&index)
        }) else {
            return None;
        };
        assert!(name_id <= id);
        if name_id == id {
            return self.data.get_val(index, 0);
        }
        // check synonyms
        let synonyms = self.data.get_val(index, 2)?;
        let offset = id - name_id - 1;
        synonyms.split(";;;").nth(offset)
    }

    #[inline]
    fn parse_inverted_list(&self, word_id: usize) -> Vec<ItemMatch> {
        let start = self.inv_list_offsets[word_id];
        let end = self
            .inv_list_offsets
            .get(word_id + 1)
            .copied()
            .unwrap_or_else(|| self.inv_lists.len());
        let list_length = (end - start) / U32_SIZE;

        let mut matches = Vec::with_capacity(list_length);
        let mut freq = 0;
        let mut last_id: Option<&u32> = None;

        // read the inverted list
        let (head, inv_list, tail) = unsafe { self.inv_lists[start..end].align_to::<u32>() };
        assert!(
            head.is_empty() && tail.is_empty(),
            "inverted list not aligned"
        );

        for id in inv_list {
            if let Some(sub_index) = self.sub_index.as_ref() {
                let idx = self.get_index(*id as usize).expect("Invalid id");
                if sub_index.binary_search(&idx).is_err() {
                    continue;
                }
            }

            if let Some(last) = last_id {
                if last != id {
                    matches.push(ItemMatch {
                        id: *last as usize,
                        freq,
                    });
                    freq = 1;
                } else {
                    freq += 1;
                }
            } else {
                freq = 1;
            }
            last_id = Some(id);
        }

        // dont forget the last run
        if let Some(last) = last_id {
            matches.push(ItemMatch {
                id: *last as usize,
                freq,
            });
        }

        matches
    }

    #[inline]
    fn get_word(&self, word_id: usize) -> &str {
        let start = self.keyword_offsets[word_id];
        let end = self
            .keyword_offsets
            .get(word_id + 1)
            .copied()
            .unwrap_or_else(|| self.keywords.len());
        unsafe { std::str::from_utf8_unchecked(&self.keywords[start..end]) }
    }

    fn size(&self) -> usize {
        self.keyword_offsets.len()
    }

    fn get_word_id(&self, word: &str) -> Option<usize> {
        match lower_bound(0, self.size(), |word_id| self.get_word(word_id).cmp(word)) {
            Some((word_id, true)) => Some(word_id),
            _ => None,
        }
    }

    fn get_matches(&self, prefix: &str) -> (Option<WordItemMatches>, Vec<WordItemMatches>) {
        let mut exact_matches = None;
        let mut prefix_matches = vec![];
        let lower_id =
            match lower_bound(0, self.size(), |word_id| self.get_word(word_id).cmp(prefix)) {
                None => return (exact_matches, prefix_matches),
                Some((word_id, true)) => {
                    exact_matches = Some(WordItemMatches {
                        word_id,
                        matches: self.parse_inverted_list(word_id),
                    });
                    word_id.saturating_add(1)
                }
                Some((word_id, false)) => word_id,
            };

        let upper_id = upper_bound(lower_id, self.size(), |word_id| {
            Self::prefix_cmp(self.get_word(word_id), prefix)
        })
        .unwrap_or_else(|| self.size());

        prefix_matches.extend((lower_id..upper_id).map(|word_id| WordItemMatches {
            word_id,
            matches: self.parse_inverted_list(word_id),
        }));
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
        let mut keyword_file = BufWriter::new(File::create(index_dir.join("index.keywords"))?);
        let mut inv_list_file = BufWriter::new(File::create(index_dir.join("index.inv-lists"))?);
        let mut offset_file = BufWriter::new(File::create(index_dir.join("index.offsets"))?);
        let mut keyword_offset = 0;
        let mut inv_list_offset = 0;
        for (keyword, inv_list) in inv_lists.into_iter().sorted_by(|(a, _), (b, _)| a.cmp(b)) {
            // write keyword and offset
            keyword_file.write_all(keyword.as_bytes())?;
            let keyword_offset_bytes = u64::try_from(keyword_offset)?.to_le_bytes();
            offset_file.write_all(&keyword_offset_bytes)?;
            keyword_offset += keyword.len();

            // write inverted list and offset
            for id in &inv_list {
                inv_list_file.write_all(&id.to_le_bytes())?;
            }
            let inv_list_offset_bytes = u64::try_from(inv_list_offset)?.to_le_bytes();
            offset_file.write_all(&inv_list_offset_bytes)?;
            inv_list_offset += inv_list.len() * U32_SIZE;
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(data_file: &str, index_dir: &str) -> anyhow::Result<Self> {
        let data = IndexData::new(data_file)?;
        let index_dir = Path::new(index_dir);

        let keywords =
            Arc::new(unsafe { Mmap::map(&File::open(index_dir.join("index.keywords"))?)? });
        let inv_lists =
            Arc::new(unsafe { Mmap::map(&File::open(index_dir.join("index.inv-lists"))?)? });

        let mut keyword_offsets = vec![];
        let mut inv_list_offsets = vec![];

        let offset_bytes = unsafe { Mmap::map(&File::open(index_dir.join("index.offsets"))?)? };
        let (head, offsets, tail) = unsafe { offset_bytes.align_to::<u64>() };
        assert!(head.is_empty() && tail.is_empty(), "offsets not aligned");

        for (keyword_offset, inv_list_offset) in offsets.iter().tuples() {
            keyword_offsets.push(*keyword_offset as usize);
            inv_list_offsets.push(*inv_list_offset as usize);
        }
        let keyword_offsets = Arc::from(keyword_offsets);
        let inv_list_offsets = Arc::from(inv_list_offsets);

        let id_to_index_bytes =
            unsafe { Mmap::map(&File::open(index_dir.join("index.id-to-index"))?)? };
        let (head, id_to_index, tail) = unsafe { id_to_index_bytes.align_to::<u32>() };
        assert!(
            head.is_empty() && tail.is_empty(),
            "id_to_index not aligned"
        );

        let id_to_index: Vec<_> = id_to_index.iter().map(|&id| id as usize).collect();
        let id_to_index = Arc::from(id_to_index);

        let length_bytes = unsafe { Mmap::map(&File::open(index_dir.join("index.lengths"))?)? };
        let (head, lengths, tail) = unsafe { length_bytes.align_to::<u32>() };
        assert!(head.is_empty() && tail.is_empty(), "lengths not aligned");

        let lengths = lengths.to_vec();
        let total_length: u32 = lengths.iter().sum();
        let avg_length = total_length as f32 / lengths.len().max(1) as f32;
        let lengths = Arc::from(lengths);

        Ok(Self {
            data,
            keywords,
            inv_lists,
            avg_length,
            keyword_offsets,
            inv_list_offsets,
            id_to_index,
            lengths,
            sub_index: None,
        })
    }

    #[pyo3(signature = (
        query,
        score = Score::Occurrence,
        k = 1.5,
        b = 0.75,
        min_keyword_length = None,
        no_refinement = false
    ))]
    pub fn find_matches(
        &self,
        query: &str,
        score: Score,
        k: f32,
        b: f32,
        min_keyword_length: Option<usize>,
        no_refinement: bool,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let min_keyword_length = min_keyword_length
            .map(|len| len.max(1))
            .unwrap_or(MIN_KEYWORD_LEN);
        let start = Instant::now();
        let query_norm = normalize(query);
        let (long_keywords, mut short_keywords): (Vec<_>, Vec<_>) = query_norm
            .split_whitespace()
            .enumerate()
            .partition(|(_, keyword)| keyword.len() >= min_keyword_length);

        if no_refinement {
            // not refiment, just use the long keywords
            short_keywords.clear();
        }

        let num_keywords = long_keywords.len() + short_keywords.len();

        let mut item_matches: HashMap<_, Vec<_>> = HashMap::new();
        for (keyword_id, keyword) in long_keywords {
            let (exact_matches, prefix_matches) = self.get_matches(keyword);

            // group matches by id
            if let Some(WordItemMatches { word_id, matches }) = exact_matches {
                for kw_match in matches {
                    item_matches.entry(kw_match.id).or_default().push(Match {
                        keyword_id,
                        word_id,
                        exact: true,
                        freq: kw_match.freq,
                    });
                }
            }

            for WordItemMatches { word_id, matches } in prefix_matches {
                for kw_match in matches {
                    item_matches.entry(kw_match.id).or_default().push(Match {
                        keyword_id,
                        word_id,
                        exact: false,
                        freq: kw_match.freq,
                    });
                }
            }
        }

        if !short_keywords.is_empty() {
            for (id, matches) in item_matches.iter_mut() {
                let Some(name) = self.get_name_or_synonym(*id) else {
                    continue;
                };
                let name_norm = normalize(name);
                let words: HashMap<_, _> =
                    name_norm
                        .split_whitespace()
                        .fold(HashMap::new(), |mut map, word| {
                            map.entry(word).and_modify(|freq| *freq += 1).or_insert(1);
                            map
                        });

                // add new keyword matches
                for &(keyword_id, keyword) in short_keywords.iter() {
                    for (word, &freq) in words.iter() {
                        if !word.starts_with(keyword) {
                            continue;
                        }
                        let Some(word_id) = self.get_word_id(word) else {
                            continue;
                        };
                        matches.push(Match {
                            keyword_id,
                            word_id,
                            exact: word.len() == keyword.len(),
                            freq,
                        });
                    }
                }
            }
        }

        let item_matches: Vec<_> = item_matches
            .into_iter()
            .filter_map(|(id, matches)| {
                // scores is a list of tuples (query_id, keyword_id, is_exact, freq, score)
                let index = self.get_index(id)?;
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
            })
            .sorted_by_key(|&(index, score)| (index, Reverse(OrderedFloat(score))))
            .unique_by(|&(index, ..)| index)
            .sorted_by_key(|&(index, score)| (Reverse(OrderedFloat(score)), index))
            .collect();

        debug!(
            "Got {} matches for query '{query}' in {:.2}ms",
            item_matches.len(),
            start.elapsed().as_secs_f32() * 1000.0
        );

        Ok(item_matches)
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
