mod data;
mod prefix;
mod qgram;
mod utils;

use prefix::PrefixIndex;
use pyo3::prelude::*;
use qgram::QGramIndex;
use qgram::pyied;
use utils::normalize;

#[pymodule]
fn _internal(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    let _ = m.add_class::<QGramIndex>();
    let _ = m.add_class::<PrefixIndex>();
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(pyied, m)?)?;
    Ok(())
}

