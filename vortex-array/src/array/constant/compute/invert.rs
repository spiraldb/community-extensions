use vortex_error::VortexResult;

use crate::array::{ConstantArray, ConstantEncoding};
use crate::compute::InvertFn;
use crate::{ArrayData, IntoArrayData};

impl InvertFn<ConstantArray> for ConstantEncoding {
    fn invert(&self, array: &ConstantArray) -> VortexResult<ArrayData> {
        match array.scalar().as_bool().value() {
            None => Ok(array.clone().into_array()),
            Some(b) => Ok(ConstantArray::new(!b, array.len()).into_array()),
        }
    }
}
