use vortex_array::compute::{FillNullFn, fill_null};
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{RunEndArray, RunEndEncoding};

impl FillNullFn<&RunEndArray> for RunEndEncoding {
    fn fill_null(&self, array: &RunEndArray, fill_value: Scalar) -> VortexResult<ArrayRef> {
        Ok(RunEndArray::with_offset_and_length(
            array.ends().clone(),
            fill_null(array.values(), fill_value)?,
            array.offset(),
            array.len(),
        )?
        .into_array())
    }
}
