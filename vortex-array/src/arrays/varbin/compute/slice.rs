use vortex_error::VortexResult;

use crate::arrays::VarBinEncoding;
use crate::arrays::varbin::VarBinArray;
use crate::compute::{SliceFn, slice};
use crate::{Array, ArrayRef};

impl SliceFn<&VarBinArray> for VarBinEncoding {
    fn slice(&self, array: &VarBinArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        VarBinArray::try_new(
            slice(array.offsets(), start, stop + 1)?,
            array.bytes().clone(),
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }
}
