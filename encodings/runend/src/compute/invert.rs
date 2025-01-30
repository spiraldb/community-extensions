use vortex_array::compute::{invert, InvertFn};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl InvertFn<RunEndArray> for RunEndEncoding {
    fn invert(&self, array: &RunEndArray) -> VortexResult<Array> {
        RunEndArray::with_offset_and_length(
            array.ends(),
            invert(&array.values())?,
            array.len(),
            array.offset(),
        )
        .map(|a| a.into_array())
    }
}
