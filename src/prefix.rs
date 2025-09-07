use anyhow::anyhow;
use itertools::Itertools;
use log::debug;
use memmap2::Mmap;
use ordered_float::OrderedFloat;
use pathfinding::{kuhn_munkres::kuhn_munkres, matrix::Matrix};
use pyo3::prelude::*;
use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeSet, HashMap, HashSet, hash_map::Entry},
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

use crate::utils::{lower_bound, upper_bound};
use crate::{
    data::IndexData,
    utils::{IndexIter, normalize},
};

#[pyclass]
#[derive(Clone)]
pub struct PrefixIndex {
    #[pyo3(get)]
    data: IndexData,
    keywords: Arc<Mmap>,
    keyword_offsets: Arc<[usize]>,
    inv_lists: Arc<Mmap>,
    inv_list_offsets: Arc<[usize]>,
    lengths: Arc<[u32]>,
    id_to_index: Arc<[usize]>,
    sub_index: Option<Arc<BTreeSet<usize>>>,
}

struct InvList<'a> {
    word: usize,
    exact: bool,
    length: usize,
    inv_list: &'a [u8],
}

impl InvList<'_> {
    fn parse(&self) -> anyhow::Result<&[u32]> {
        let (head, body, tail) = unsafe { self.inv_list.align_to::<u32>() };
        if !head.is_empty() || !tail.is_empty() {
            return Err(anyhow!("Inverted list not aligned"));
        }
        Ok(body)
    }
}

#[derive(Debug)]
struct Item {
    candidate: Option<Candidate>,
    matches: HashMap<(usize, usize), Vec<(usize, f32)>>,
}

impl Item {
    const ES: f32 = 1.0; // exact match score
    const PS: f32 = 0.75; // prefix match score
    const KP: f32 = 0.5; // keyword penalty
    const WP: f32 = 0.25; // word penalty

    fn new() -> Self {
        Self {
            candidate: None,
            matches: HashMap::new(),
        }
    }

    fn update(&mut self, word: usize, occurrence: usize, keyword: usize, exact: bool) {
        self.matches
            .entry((word, occurrence))
            .or_default()
            .push((keyword, if exact { Self::ES } else { Self::PS }));
    }

    fn candidate(&self, id: u32, index: usize, num_keywords: usize, length: u32) -> Candidate {
        let score = self.assignment(num_keywords);

        let num_matches = num_keywords.min(self.matches.len());
        let unmatched_keywords = num_keywords.saturating_sub(num_matches);
        // make sure to not penalize twice for unmatched keyword and word
        let unmatched_words = (length as usize)
            .saturating_sub(num_matches)
            .saturating_sub(unmatched_keywords);

        let penalty = unmatched_keywords as f32 * Self::KP + unmatched_words as f32 * Self::WP;

        let score = score - penalty;
        Candidate::new(id, score, index)
    }

    fn assignment(&self, num_keywords: usize) -> f32 {
        // make sure to have at least as many columns as rows
        let num_columns = num_keywords.max(self.matches.len());
        let weights = Matrix::from_rows(self.matches.values().map(|matches| {
            let mut row = vec![OrderedFloat(0.0); num_columns];
            for (keyword, score) in matches {
                row[*keyword] = OrderedFloat(*score);
            }
            row
        }))
        .expect("Cannot fail");

        let (score, _) = kuhn_munkres(&weights);
        score.into_inner()
    }
}

#[derive(Debug, Copy, Clone)]
struct Candidate {
    score: OrderedFloat<f32>,
    index: usize,
    id: u32,
}

impl Candidate {
    fn new(id: u32, score: f32, index: usize) -> Self {
        Self {
            score: OrderedFloat(score),
            index,
            id,
        }
    }

    fn add(&self, score: f32) -> Self {
        Self {
            score: self.score + score,
            index: self.index,
            id: self.id,
        }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index && self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.index.cmp(&self.index))
            .then_with(|| self.id.cmp(&other.id))
    }
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
    fn get_inverted_list(&self, word: usize) -> (&[u8], usize) {
        let start = self.inv_list_offsets[word];
        let end = self
            .inv_list_offsets
            .get(word + 1)
            .copied()
            .unwrap_or_else(|| self.inv_lists.len());

        let data = &self.inv_lists[start..end];
        let length = data.len() / U32_SIZE;
        (data, length)
    }

    #[inline]
    fn get_word(&self, word: usize) -> &str {
        let start = self.keyword_offsets[word];
        let end = self
            .keyword_offsets
            .get(word + 1)
            .copied()
            .unwrap_or_else(|| self.keywords.len());
        unsafe { std::str::from_utf8_unchecked(&self.keywords[start..end]) }
    }

    fn num_keywords(&self) -> usize {
        self.keyword_offsets.len()
    }

    pub fn len(&self) -> usize {
        if let Some(sub_index) = self.sub_index.as_ref() {
            sub_index.len()
        } else {
            self.data.len()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get_matches(&self, prefix: &str) -> Vec<InvList> {
        let mut matches = vec![];

        let lower = match lower_bound(0, self.num_keywords(), |word| {
            self.get_word(word).cmp(prefix)
        }) {
            None => return matches,
            Some((word, true)) => {
                let (inv_list, length) = self.get_inverted_list(word);
                matches.push(InvList {
                    word,
                    exact: true,
                    length,
                    inv_list,
                });
                word.saturating_add(1)
            }
            Some((word, false)) => word,
        };

        let upper = upper_bound(lower, self.num_keywords(), |word| {
            Self::prefix_cmp(self.get_word(word), prefix)
        })
        .unwrap_or_else(|| self.num_keywords());

        matches.extend((lower..upper).map(|word| {
            let (inv_list, length) = self.get_inverted_list(word);
            InvList {
                word,
                exact: false,
                length,
                inv_list,
            }
        }));

        matches
    }

    fn get_column_for_id(&self, id: u32) -> usize {
        // get the offset (label index in data row) for the given id
        // by checking how many previous ids map to the same index
        let mut id = id as usize;
        let index = self.id_to_index[id];
        let mut offset = 0;
        while id > 0 && self.id_to_index[id - 1] == index {
            offset += 1;
            id -= 1;
        }
        offset + 1 // +1 to account for identifier column
    }
}

#[pymethods]
impl PrefixIndex {
    // implement functions from pyi interface
    #[staticmethod]
    pub fn build(data: IndexData, index_dir: &str) -> anyhow::Result<()> {
        let index_dir = Path::new(index_dir);
        let mut inv_lists: HashMap<String, Vec<u32>> = HashMap::new();

        let mut id_to_index_file =
            BufWriter::new(File::create(index_dir.join("index.id-to-index"))?);
        let mut lengths_file = BufWriter::new(File::create(index_dir.join("index.lengths"))?);

        let mut id = 0;
        for (i, row) in data.iter().enumerate() {
            let index_bytes = u32::try_from(i)?.to_le_bytes();
            for name in row.into_iter().skip(1).map(normalize) {
                let mut length: u32 = 0;
                for word in name.split_whitespace() {
                    let inv_list = inv_lists.entry(word.to_string()).or_default();
                    inv_list.push(id);
                    length += 1;
                }
                if id == u32::MAX {
                    return Err(anyhow!("too many labels, max {} supported", u32::MAX));
                }
                id += 1;
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

            // write inverted list offset
            let inv_list_offset_bytes = u64::try_from(inv_list_offset)?.to_le_bytes();
            offset_file.write_all(&inv_list_offset_bytes)?;

            // write inverted list
            for id in &inv_list {
                inv_list_file.write_all(&id.to_le_bytes())?;
            }
            inv_list_offset += inv_list.len() * U32_SIZE;
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(data: IndexData, index_dir: &str) -> anyhow::Result<Self> {
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

        let lengths = Arc::from(lengths);

        Ok(Self {
            data,
            keywords,
            keyword_offsets,
            inv_lists,
            inv_list_offsets,
            id_to_index,
            lengths,
            sub_index: None,
        })
    }

    #[pyo3(signature = (
        query,
        k = 100,
    ))]
    pub fn find_matches(&self, query: &str, k: usize) -> anyhow::Result<Vec<(usize, f32, usize)>> {
        let start = Instant::now();

        // scale k by facotr of avg. ids per indexed item
        // add factor of 2 to be safe
        let k_factor = self.id_to_index.len() as f32 / self.data.len() as f32;
        let k_scaled = (k as f32 * k_factor * 2.0).ceil() as usize;

        let query = normalize(query);
        let keywords: Vec<_> = query.split_whitespace().collect();
        let num_keywords = keywords.len();

        let mut items: HashMap<u32, Item> = HashMap::new();

        let keyword_matches: Vec<_> = keywords
            .iter()
            .map(|kw| self.get_matches(kw))
            // most selective first
            .sorted_by_cached_key(|inv_lists| {
                inv_lists
                    .iter()
                    .map(|inv_list| inv_list.length)
                    .sum::<usize>()
            })
            .collect();

        debug!(
            "Getting matches took {:.2}ms",
            start.elapsed().as_secs_f32() * 1000.0
        );

        let mut top_k: BTreeSet<Candidate> = BTreeSet::new();

        let worst_candidate = |top_k: &BTreeSet<_>| -> Option<Candidate> {
            if top_k.len() == k_scaled {
                top_k.first().copied()
            } else if top_k.len() > k_scaled {
                panic!("top_k has more than k elements, should not happen");
            } else {
                None
            }
        };

        let mut skip = HashSet::new();

        for (keyword, inv_lists) in keyword_matches.into_iter().enumerate() {
            let keywords_left = num_keywords - keyword - 1;
            let max_future_score = Item::ES * keywords_left as f32;

            for inv_list in inv_lists {
                let mut last_id = None;
                let mut occurrence = 0;
                for &id in inv_list.parse()? {
                    if skip.contains(&id) {
                        continue;
                    }

                    if Some(id) == last_id {
                        occurrence += 1;
                    } else {
                        occurrence = 0;
                    }
                    last_id = Some(id);

                    let index = self.id_to_index[id as usize];
                    if let Some(sub_index) = self.sub_index.as_ref() {
                        if !sub_index.contains(&index) {
                            continue;
                        }
                    };

                    let length = self.lengths[id as usize];

                    // update item and get current score
                    let mut entry = items.entry(id);
                    let item = match entry {
                        Entry::Occupied(ref mut entry) => entry.get_mut(),
                        Entry::Vacant(entry) => {
                            if let Some(worst) = worst_candidate(&top_k) {
                                // compute an upper bound score for this newly matched id:
                                // 1. assume all future keywords match exactly
                                // 2. add current score for exact or prefix match
                                // 3. subtract penalty for all previous keywords missed
                                // TODO: currently ignores length of doc (potential word penalty or
                                // fewer future matches), maybe add that too?
                                let mut upper_bound_score = max_future_score;
                                upper_bound_score +=
                                    if inv_list.exact { Item::ES } else { Item::PS };
                                upper_bound_score -= keyword as f32 * Item::KP;

                                let upper_bound = Candidate::new(id, upper_bound_score, index);
                                if upper_bound <= worst {
                                    // even the upper bound is not enough to enter the top k
                                    skip.insert(id);
                                    continue;
                                }
                            }
                            entry.insert(Item::new())
                        }
                    };
                    item.update(inv_list.word, occurrence, keyword, inv_list.exact);
                    let current = item.candidate(id, index, num_keywords, length);
                    let old = item.candidate.replace(current);
                    if let Some(old) = old {
                        // remove old candidate from top_k, might not be present
                        top_k.remove(&old);
                    }

                    let Some(worst) = worst_candidate(&top_k) else {
                        // top_k not full yet, just insert
                        top_k.insert(current);
                        continue;
                    };

                    // upper bound for current candidate if all future keywords match exactly
                    let upper_bound = current.add(max_future_score);
                    if upper_bound <= worst {
                        // even in the best case this item cannot enter the top k
                        skip.insert(id);
                        continue;
                    }

                    if current > worst {
                        // better than the worst in top_k, insert
                        top_k.pop_first();
                        top_k.insert(current);
                    }
                }
            }
        }

        let matches: Vec<_> = top_k
            .into_iter()
            .sorted_by_key(|&candidate| (candidate.index, Reverse(candidate.score), candidate.id))
            .dedup_by(|a, b| a.index == b.index)
            .sorted_by_key(|&candidate| (Reverse(candidate.score), candidate.index))
            .take(k)
            .map(|candidate| {
                (
                    candidate.index,
                    candidate.score.into_inner(),
                    self.get_column_for_id(candidate.id),
                )
            })
            .collect();

        debug!(
            "Got top {} matches for query '{query}' in {:.2}ms",
            matches.len(),
            start.elapsed().as_secs_f32() * 1000.0
        );

        Ok(matches)
    }

    pub fn get_type(&self) -> &str {
        "prefix"
    }

    pub fn get_identifier(&self, id: usize) -> anyhow::Result<String> {
        self.data
            ._get_val(id, 0)
            .ok_or_else(|| anyhow!("invalid id"))
    }

    pub fn get_name(&self, id: usize) -> anyhow::Result<String> {
        self.data
            ._get_val(id, 1)
            .ok_or_else(|| anyhow!("invalid id"))
    }

    pub fn get_row(&self, id: usize) -> anyhow::Result<Vec<String>> {
        self.data._get_row(id).ok_or_else(|| anyhow!("invalid id"))
    }

    pub fn get_val(&self, id: usize, column: usize) -> anyhow::Result<String> {
        self.data
            ._get_val(id, column)
            .ok_or_else(|| anyhow!("invalid id or column"))
    }

    pub fn sub_index_by_ids(&self, ids: Vec<usize>) -> anyhow::Result<Self> {
        if !ids.iter().all(|&id| id < self.data.len()) {
            return Err(anyhow!("invalid ids"));
        }
        let ids: BTreeSet<_> = if let Some(sub_index) = self.sub_index.as_ref() {
            ids.into_iter()
                .filter(|id| sub_index.contains(id))
                .collect()
        } else {
            ids.into_iter().collect()
        };
        let mut index = self.clone();
        index.sub_index = Some(ids.into());
        Ok(index)
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }

    pub fn __iter__(&self) -> IndexIter {
        IndexIter::new(
            self.data.clone(),
            self.sub_index.as_ref().map(|s| s.iter().copied().collect()),
        )
    }
}

#[cfg(test)]
mod test {

    use std::fs::create_dir_all;

    use tempfile::tempdir;

    use super::*;

    fn test_data() -> Vec<&'static str> {
        vec!["id\tlabels", "0\tagar", "1\tagar agar"]
    }

    fn build_prefix_index(file: &str) -> PrefixIndex {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        let offsets_file = temp_dir
            .path()
            .join("test.offsets")
            .as_os_str()
            .to_str()
            .unwrap()
            .to_string();

        IndexData::build(file, &offsets_file).expect("Failed to build index data");
        let data = IndexData::load(file, &offsets_file).expect("Failed to load index data");

        let index_dir = temp_dir.path().join("index");
        create_dir_all(&index_dir).expect("Failed to create index directory");
        let index_dir = index_dir
            .as_os_str()
            .to_str()
            .expect("Invalid index directory path");

        PrefixIndex::build(data.clone(), index_dir).expect("Failed to build index");
        PrefixIndex::load(data, index_dir).expect("Failed to load index")
    }

    #[test]
    fn test_special_prefix_index() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_file = temp_dir.path().join("test.tsv");

        let mut file = File::create(&data_file).expect("Failed to create test data file");
        for line in test_data() {
            writeln!(file, "{line}").expect("Failed to write test data");
        }

        let index = build_prefix_index(data_file.to_str().expect("Invalid data file path"));

        let matches = index
            .find_matches("agar", 2)
            .expect("Failed to find matches");
        assert_eq!(matches, vec![(0, 1.0, 1), (1, 0.75, 1)]);

        let matches = index
            .find_matches("agar agar", 2)
            .expect("Failed to find matches");
        assert_eq!(matches, vec![(1, 2.0, 1), (0, 0.5, 1)]);
    }

    #[test]
    fn test_prefix_index() {
        let dir = env!("CARGO_MANIFEST_DIR");
        let data_file = format!("{dir}/test.tsv");

        let index = build_prefix_index(&data_file);

        let matches = index
            .find_matches("United States", 1)
            .expect("Failed to find matches");

        assert_eq!(matches[0], (0, 2.0, 1));

        // partial match
        let matches = index
            .find_matches("United State", 1)
            .expect("Failed to find matches");

        assert_eq!(matches[0], (0, 1.75, 1));

        // now with synonym
        let matches = index
            .find_matches("the U.S. of A", 1)
            .expect("Failed to find matches");

        assert_eq!(matches[0], (0, 4.0, 2));
        assert_eq!(index.get_val(0, 2).unwrap(), "the U.S. of A");

        // with partial match
        let matches = index
            .find_matches("the U.S. of Am", 1)
            .expect("Failed to find matches");

        assert_eq!(matches[0], (0, 3.75, 26));
        assert_eq!(index.get_val(0, 26).unwrap(), "the U.S. of America");

        let matches = index
            .find_matches("the U.S. of", 1)
            .expect("Failed to find matches");

        // one unmatched word in document
        assert_eq!(matches[0], (0, 3.0 - 0.25, 2));

        // now with no matching keywords
        let matches = index
            .find_matches("theunitedstates", 1)
            .expect("Failed to find matches");

        assert!(matches.is_empty());

        let matches = index
            .find_matches("the U.S.", 2)
            .expect("Failed to find matches");

        assert_eq!(matches[0], (0, 2.0, 22));

        // now with partially matching query
        let matches = index
            .find_matches("the U.S. of C", 1)
            .expect("Failed to find matches");

        // 3 exact matches, 1 keyword unmatched
        assert_eq!(matches[0], (0, 3.0 - 0.5, 2));

        // now with duplicate query keywords
        // the first 4 match exactly, but the second 4 decrease the score
        // by 0.5 each because they are not found again
        let matches = index
            .find_matches("the U.S. of A the U.S. of A", 1)
            .expect("Failed to find matches");

        assert_eq!(matches[0], (0, 4.0 - 4.0 * 0.5, 2));

        // now exclude with sub index
        let matches = index
            .sub_index_by_ids((1..index.data.len()).collect())
            .expect("Failed to create sub index")
            .find_matches("the U.S. of A", 10)
            .expect("Failed to find matches");

        assert!(matches.iter().all(|(id, ..)| *id != 0));
    }
}
