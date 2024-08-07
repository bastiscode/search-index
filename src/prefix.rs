use any_ascii::any_ascii;
use anyhow::anyhow;
use itertools::Itertools;
use memmap2::Mmap;
use pyo3::prelude::*;
use std::{
    cmp::{Ordering, Reverse},
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

use crate::data::IndexData;

#[pyclass]
#[derive(Clone)]
pub struct PrefixIndex {
    data: Arc<IndexData>,
    index: Arc<Mmap>,
    offsets: Arc<Mmap>,
    sub_index: Option<Arc<Vec<usize>>>,
}

const SIZE: usize = size_of::<u64>();

impl PrefixIndex {
    fn normalize(name: &str) -> String {
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
    fn get_ids(&self, offset: usize) -> Vec<usize> {
        let mut num_ids_bytes = [0; SIZE];
        num_ids_bytes.copy_from_slice(&self.index[offset..offset + SIZE]);
        let num_ids = u64::from_le_bytes(num_ids_bytes) as usize;
        self.index[offset + SIZE..offset + SIZE + num_ids * SIZE]
            .chunks_exact(SIZE)
            .filter_map(|bytes| {
                let mut id_bytes = [0; SIZE];
                id_bytes.copy_from_slice(bytes);
                let id = u64::from_le_bytes(id_bytes) as usize;
                self.sub_index.as_ref().map_or(Some(id), |sub_index| {
                    if sub_index.binary_search(&id).is_ok() {
                        Some(id)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    #[inline]
    fn get_keyword(&self, index: usize) -> (&[u8], usize) {
        let first = index * 2 * SIZE;
        let second = index * 2 * SIZE + SIZE;
        let third = index * 2 * SIZE + 2 * SIZE;
        let mut start_bytes = [0; SIZE];
        start_bytes.copy_from_slice(&self.offsets[first..second]);
        let start = u64::from_le_bytes(start_bytes) as usize;
        let mut end_bytes = [0; SIZE];
        end_bytes.copy_from_slice(&self.offsets[second..third]);
        let end = u64::from_le_bytes(end_bytes) as usize;
        (&self.index[start..end], end)
    }

    #[inline]
    fn get_ids_if_equal(&self, index: usize, keyword: &[u8]) -> Option<Vec<usize>> {
        let (index_keyword, ids_start) = self.get_keyword(index);
        if let Ordering::Equal = Self::prefix_cmp(index_keyword, keyword) {
            Some(self.get_ids(ids_start))
        } else {
            None
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.offsets.len() / (2 * SIZE)
    }

    fn get_keyword_matches(&self, keyword: &str) -> Vec<usize> {
        let mut lower = 0;
        let mut upper = self.size();
        while lower < upper {
            let mid = (lower + upper) / 2;
            let (mid_keyword, ids_start) = self.get_keyword(mid);
            match Self::prefix_cmp(mid_keyword, keyword.as_bytes()) {
                Ordering::Less => lower = mid + 1,
                Ordering::Greater => upper = mid,
                Ordering::Equal => {
                    // move to left and right and find all matches
                    let ids = self.get_ids(ids_start);
                    return ids
                        .into_iter()
                        .chain(
                            // left to mid
                            (0..mid)
                                .rev()
                                .map(|id| self.get_ids_if_equal(id, keyword.as_bytes()))
                                .take_while(|ids| ids.is_some())
                                .flat_map(|ids| ids.unwrap()),
                        )
                        .chain(
                            // mid to right
                            (mid + 1..self.size())
                                .map(|id| self.get_ids_if_equal(id, keyword.as_bytes()))
                                .take_while(|ids| ids.is_some())
                                .flat_map(|ids| ids.unwrap()),
                        )
                        .unique()
                        .collect();
                }
            }
        }
        vec![]
    }
}

#[pyclass]
pub struct PrefixIndexIter {
    data: Arc<IndexData>,
    sub_index: Option<Arc<Vec<usize>>>,
    index: usize,
}

#[pymethods]
impl PrefixIndexIter {
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

#[pymethods]
impl PrefixIndex {
    // implement functions from pyi interface
    #[staticmethod]
    #[pyo3(signature = (data_file, index_dir, use_synonyms = true))]
    pub fn build(data_file: &str, index_dir: &str, use_synonyms: bool) -> anyhow::Result<()> {
        let data = IndexData::new(data_file)?;
        let mut map = HashMap::new();
        for (i, row) in data.iter().enumerate() {
            let mut split = row.split('\t');
            let name = Self::normalize(
                split
                    .next()
                    .ok_or_else(|| anyhow!("name not found in row {i}: {row}"))?,
            );
            for keyword in name.split_whitespace().filter(|s| !s.is_empty()) {
                map.entry(keyword.to_string())
                    .or_insert_with(HashSet::new)
                    .insert(i);
            }
            if !use_synonyms {
                continue;
            }
            let syns = split
                .nth(1)
                .ok_or_else(|| anyhow!("synonyms not found in row {i}: {row}"))?;
            for syn in syns.split(';').map(Self::normalize) {
                for keyword in syn.split_whitespace().filter(|s| !s.is_empty()) {
                    map.entry(keyword.to_string())
                        .or_insert_with(HashSet::new)
                        .insert(i);
                }
            }
        }
        // write the map to different files on disk so we can load it memory mapped
        // first sort by key to have them in lexicographical order
        let index_dir = Path::new(index_dir);
        let mut index_file = BufWriter::new(File::create(index_dir.join("index.data"))?);
        let mut offset_file = BufWriter::new(File::create(index_dir.join("index.offsets"))?);
        let mut offset = 0;
        for (name, ids) in map.into_iter().sorted_by(|(a, _), (b, _)| a.cmp(b)) {
            if index_file.write(name.as_bytes())? < name.len() {
                return Err(anyhow!("failed to write name"));
            }
            let offset_bytes = u64::try_from(offset)?.to_le_bytes();
            offset += name.len();
            let next_offset_bytes = u64::try_from(offset)?.to_le_bytes();
            if offset_file.write(&offset_bytes)? < offset_bytes.len() {
                return Err(anyhow!("failed to offset"));
            }
            if offset_file.write(&next_offset_bytes)? < next_offset_bytes.len() {
                return Err(anyhow!("failed to write offset"));
            }
            let num_id_bytes = u64::try_from(ids.len())?.to_le_bytes();
            if index_file.write(&num_id_bytes)? < num_id_bytes.len() {
                return Err(anyhow!("failed to write number of ids"));
            }
            offset += num_id_bytes.len();
            for id in ids {
                let id_bytes = u64::try_from(id)?.to_le_bytes();
                if index_file.write(&id_bytes)? < id_bytes.len() {
                    return Err(anyhow!("failed to write id"));
                }
                offset += id_bytes.len();
            }
        }
        Ok(())
    }

    #[staticmethod]
    pub fn load(data_file: &str, index_dir: &str) -> anyhow::Result<Self> {
        let data = Arc::new(IndexData::new(data_file)?);
        let index_dir = Path::new(index_dir);
        let index = Arc::new(unsafe { Mmap::map(&File::open(index_dir.join("index.data"))?)? });
        let offsets =
            Arc::new(unsafe { Mmap::map(&File::open(index_dir.join("index.offsets"))?)? });
        Ok(Self {
            data,
            index,
            offsets,
            sub_index: None,
        })
    }

    pub fn find_matches(&self, query: &str) -> anyhow::Result<Vec<(usize, (usize, usize))>> {
        let norm = Self::normalize(query);
        let matches = norm
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .unique()
            .fold(HashMap::new(), |mut map, keyword| {
                for id in self.get_keyword_matches(keyword) {
                    *map.entry(id).or_insert(0) += 1;
                }
                map
            });
        let mut matches: Vec<_> = matches
            .into_iter()
            .map(|(id, count)| {
                // add score to sort key
                let col = self
                    .data
                    .get_val(id, 1)
                    .ok_or_else(|| anyhow!("invalid id"))?;
                let score = col.parse::<usize>()?;
                Ok((id, (count, score)))
            })
            .collect::<anyhow::Result<_>>()?;
        matches.sort_by_key(|&(_, (count, score))| (Reverse(count), Reverse(score)));
        Ok(matches)
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

    pub fn sub_index_by_ids(&self, mut ids: Vec<usize>) -> anyhow::Result<Self> {
        if !ids.iter().all(|&id| id < self.data.len()) {
            return Err(anyhow!("invalid ids"));
        }
        ids.sort();
        let sub_index = if let Some(sub_index) = self.sub_index.as_ref() {
            // perform list intersection
            let mut intersection = vec![];
            let mut i = 0;
            let mut j = 0;
            while i < sub_index.len() && j < ids.len() {
                match sub_index[i].cmp(&ids[j]) {
                    Ordering::Less => i += 1,
                    Ordering::Equal => {
                        intersection.push(sub_index[i]);
                        i += 1;
                        j += 1;
                    }
                    Ordering::Greater => j += 1,
                }
            }
            // add remaining elements
            intersection.extend(sub_index[i..].iter().copied());
            intersection.extend(ids[j..].iter().copied());
            intersection
        } else {
            ids
        };
        let mut index = self.clone();
        index.sub_index = Some(Arc::new(sub_index));
        Ok(index)
    }

    pub fn __len__(&self) -> usize {
        self.sub_index
            .as_ref()
            .map_or(self.data.len(), |sub_index| sub_index.len())
    }

    pub fn __iter__(&self) -> PrefixIndexIter {
        let data = self.data.clone();
        let sub_index = self.sub_index.clone();
        PrefixIndexIter {
            data,
            sub_index,
            index: 0,
        }
    }
}
