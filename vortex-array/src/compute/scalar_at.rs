use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::validity::ArrayValidity;
use crate::{ArrayDType, ArrayData};

/// Implementation of scalar_at for an encoding.
///
/// SAFETY: the index is guaranteed to be within the bounds of the [ArrayData].
pub trait ScalarAtFn<Array> {
    fn scalar_at(&self, array: &Array, index: usize) -> VortexResult<Scalar>;
}

impl<E: Encoding> ScalarAtFn<ArrayData> for E
where
    E: ScalarAtFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn scalar_at(&self, array: &ArrayData, index: usize) -> VortexResult<Scalar> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        ScalarAtFn::scalar_at(encoding, array_ref, index)
    }
}

pub fn scalar_at(array: impl AsRef<ArrayData>, index: usize) -> VortexResult<Scalar> {
    let array = array.as_ref();
    if index >= array.len() {
        vortex_bail!(OutOfBounds: index, 0, array.len());
    }

    if !array.is_valid(index)? {
        return Ok(Scalar::null(array.dtype().clone()));
    }

    let scalar = array
        .encoding()
        .scalar_at_fn()
        .map(|f| f.scalar_at(array, index))
        .unwrap_or_else(|| Err(vortex_err!(NotImplemented: "scalar_at", array.encoding().id())))?;

    debug_assert_eq!(
        scalar.dtype(),
        array.dtype(),
        "ScalarAt dtype mismatch {}",
        array.encoding().id()
    );

    Ok(scalar)
}
