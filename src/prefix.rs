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

use crate::{
    data::IndexData,
    utils::{list_intersection, normalize, IndexIter},
};

#[pyclass]
#[derive(Clone)]
pub struct PrefixIndex {
    data: Arc<IndexData>,
    index: Arc<Mmap>,
    offsets: Arc<[usize]>,
    num_ids: Arc<[usize]>,
    sub_index: Option<Arc<[usize]>>,
}

const SIZE: usize = size_of::<u64>();

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
    fn parse_ids(&self, start: usize, end: usize) -> Vec<usize> {
        self.index[start..end]
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
    fn get_keyword(&self, index: usize) -> (&[u8], (usize, usize)) {
        let start = self.offsets[index];
        let next_start = self
            .offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| self.index.len());
        let num_ids = self.num_ids[index];
        let end = next_start - num_ids * SIZE;
        (&self.index[start..end], (end, next_start))
    }

    #[inline]
    fn get_ids_on_prefix_match(&self, index: usize, prefix: &[u8]) -> Option<Vec<usize>> {
        let (index_keyword, (start, end)) = self.get_keyword(index);
        if let Ordering::Equal = Self::prefix_cmp(index_keyword, prefix) {
            Some(self.parse_ids(start, end))
        } else {
            None
        }
    }

    fn size(&self) -> usize {
        self.offsets.len()
    }

    fn get_prefix_matches(&self, prefix: &str) -> Vec<usize> {
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
                        .map(|idx| self.get_ids_on_prefix_match(idx, prefix.as_bytes()))
                        .take_while(|ids| ids.is_some())
                        .flatten()
                        .flatten()
                        .chain(self.parse_ids(start, end))
                        .chain(
                            // mid to right
                            (mid + 1..self.size())
                                .map(|idx| self.get_ids_on_prefix_match(idx, prefix.as_bytes()))
                                .take_while(|ids| ids.is_some())
                                .flatten()
                                .flatten(),
                        )
                        .unique()
                        .collect();
                }
            }
        }
        vec![]
    }
}

pub type Ranking = (usize, usize);

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
            let name = normalize(
                split
                    .next()
                    .ok_or_else(|| anyhow!("name not found in row {i}: {row}"))?,
            );
            let mut keywords: Vec<_> = name
                .split_whitespace()
                .map(|keyword| keyword.to_string())
                .collect();
            for keyword in name.split_whitespace().filter(|s| !s.is_empty()) {
                map.entry(keyword.to_string())
                    .or_insert_with(HashSet::new)
                    .insert(i);
            }
            if use_synonyms {
                keywords.extend(
                    split
                        .nth(1)
                        .ok_or_else(|| anyhow!("synonyms not found in row {i}: {row}"))?
                        .split(";;;")
                        .map(normalize)
                        .flat_map(|syn| {
                            syn.split_whitespace()
                                .map(|s| s.to_string())
                                .collect::<Vec<_>>()
                        }),
                );
            }
            for keyword in keywords.into_iter().filter(|s| !s.is_empty()) {
                map.entry(keyword.to_string())
                    .or_insert_with(HashSet::new)
                    .insert(i);
            }
        }
        // write the map to different files on disk so we can load it memory mapped
        // first sort by key to have them in lexicographical order
        let index_dir = Path::new(index_dir);
        let mut index_file = BufWriter::new(File::create(index_dir.join("index.data"))?);
        let mut offset_file = BufWriter::new(File::create(index_dir.join("index.offsets"))?);
        let mut offset = 0;
        for (name, ids) in map.into_iter().sorted_by(|(a, _), (b, _)| a.cmp(b)) {
            index_file.write_all(name.as_bytes())?;
            let offset_bytes = u64::try_from(offset)?.to_le_bytes();
            offset_file.write_all(&offset_bytes)?;
            offset += name.len();
            let num_id_bytes = u64::try_from(ids.len())?.to_le_bytes();
            offset_file.write_all(&num_id_bytes)?;
            for id in ids {
                let id_bytes = u64::try_from(id)?.to_le_bytes();
                index_file.write_all(&id_bytes)?;
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
        let mut offsets = vec![];
        let mut num_ids = vec![];
        let offset_bytes = unsafe { Mmap::map(&File::open(index_dir.join("index.offsets"))?)? };
        for chunk in offset_bytes.chunks_exact(SIZE * 2) {
            let mut offset = [0; SIZE];
            let mut num = [0; SIZE];
            offset.copy_from_slice(&chunk[..SIZE]);
            num.copy_from_slice(&chunk[SIZE..]);
            offsets.push(u64::from_le_bytes(offset) as usize);
            num_ids.push(u64::from_le_bytes(num) as usize);
        }
        Ok(Self {
            data,
            index,
            offsets: offsets.into(),
            num_ids: num_ids.into(),
            sub_index: None,
        })
    }

    pub fn find_matches(&self, query: &str) -> anyhow::Result<Vec<(usize, Ranking)>> {
        let norm = normalize(query);
        let matches = norm
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .unique()
            .fold(HashMap::new(), |mut map, keyword| {
                for id in self.get_prefix_matches(keyword) {
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
                    .ok_or_else(|| anyhow!("failed to get score for id {id}"))?;
                let score = col.parse::<usize>()?;
                Ok((id, (count, score)))
            })
            .collect::<anyhow::Result<_>>()?;
        matches.sort_by_key(|&(id, (count, score))| (Reverse(count), Reverse(score), id));
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
