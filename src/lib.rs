mod data;
mod mapping;
mod prefix;
mod qgram;
mod utils;

pub use data::IndexData;
pub use mapping::Mapping;
pub use prefix::{PrefixIndex, Score};
use pyo3::prelude::*;
pub use qgram::QGramIndex;
use qgram::{ied, ped};
use utils::normalize;

#[pymodule]
fn _internal(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<QGramIndex>()?;
    m.add_class::<PrefixIndex>()?;
    m.add_class::<IndexData>()?;
    m.add_class::<Mapping>()?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(ied, m)?)?;
    m.add_function(wrap_pyfunction!(ped, m)?)?;
    Ok(())
}
