mod data;
mod prefix;
mod qgram;
mod utils;

use prefix::PrefixIndex;
use pyo3::prelude::*;
use qgram::QGramIndex;

#[pymodule]
fn _internal(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    let _ = m.add_class::<QGramIndex>();
    let _ = m.add_class::<PrefixIndex>();
    Ok(())
}
