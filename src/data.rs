use std::fs::File;

use anyhow::anyhow;
use memmap2::Mmap;

pub(crate) struct IndexData {
    mmap: Mmap,
    offsets: Vec<usize>,
}

impl IndexData {
    pub fn new(file: &str) -> anyhow::Result<Self> {
        // iterate over the file and store the offest of each line
        let mmap = unsafe { Mmap::map(&File::open(file)?)? };
        let mut offsets = vec![];
        // trim final newlines
        let trim = mmap.iter().rev().take_while(|&&b| b == b'\n').count();
        let mut lines = mmap[..mmap.len() - trim].split(|&b| b == b'\n');
        // skip the first line (header)
        let mut offset = lines.next().map(|line| line.len() + 1).unwrap_or(0);
        for line in lines {
            // check that line is valid utf8
            if let Err(e) = std::str::from_utf8(line) {
                return Err(anyhow!("found invalid utf8: {}", e));
            };
            offsets.push(offset);
            offset += line.len() + 1;
        }
        Ok(IndexData { mmap, offsets })
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn get_row(&self, idx: usize) -> Option<&str> {
        let offset = self.offsets.get(idx)?;
        let line = self.mmap.get(*offset..)?;
        let len = line.iter().position(|&b| b == b'\n').unwrap_or(line.len());
        Some(unsafe { std::str::from_utf8_unchecked(&line[..len]) })
    }

    pub fn get_val(&self, idx: usize, column: usize) -> Option<&str> {
        self.get_row(idx)
            .and_then(|line| line.split('\t').nth(column))
    }

    pub fn iter(&self) -> IndexDataIter {
        IndexDataIter { data: self, idx: 0 }
    }
}

pub struct IndexDataIter<'a> {
    data: &'a IndexData,
    idx: usize,
}

impl<'a> Iterator for IndexDataIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        self.data.get_row(self.idx).map(|row| {
            self.idx += 1;
            row
        })
    }
}
