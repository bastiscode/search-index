mod data;
mod prefix;
mod qgram;
mod utils;

use data::IndexData;
use prefix::PrefixIndex;
use pyo3::prelude::*;
use qgram::{ied, ped, QGramIndex};
use utils::normalize;

#[pymodule]
fn _internal(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_class::<QGramIndex>();
    let _ = m.add_class::<PrefixIndex>();
    let _ = m.add_class::<IndexData>();
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(ied, m)?)?;
    m.add_function(wrap_pyfunction!(ped, m)?)?;
    Ok(())
}
