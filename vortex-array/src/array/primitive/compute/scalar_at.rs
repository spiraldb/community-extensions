use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::ScalarAtFn;
use crate::variants::PrimitiveArrayTrait;

impl ScalarAtFn<PrimitiveArray> for PrimitiveEncoding {
    fn scalar_at(&self, array: &PrimitiveArray, index: usize) -> VortexResult<Scalar> {
        Ok(match_each_native_ptype!(array.ptype(), |$T| {
            Scalar::primitive(array.as_slice::<$T>()[index], array.dtype().nullability())
        }))
    }
}
