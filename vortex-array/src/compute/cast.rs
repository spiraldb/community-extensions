use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::{Array, IntoArray, IntoCanonical};

pub trait CastFn<A> {
    fn cast(&self, array: &A, dtype: &DType) -> VortexResult<Array>;
}

impl<E: Encoding> CastFn<Array> for E
where
    E: CastFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn cast(&self, array: &Array, dtype: &DType) -> VortexResult<Array> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        CastFn::cast(encoding, array_ref, dtype)
    }
}

/// Attempt to cast an array to a desired DType.
///
/// Some array support the ability to narrow or upcast.
pub fn try_cast(array: impl AsRef<Array>, dtype: &DType) -> VortexResult<Array> {
    let array = array.as_ref();
    if array.dtype() == dtype {
        return Ok(array.clone());
    }

    let casted = try_cast_impl(array, dtype)?;

    debug_assert_eq!(
        casted.len(),
        array.len(),
        "Cast length mismatch {}",
        array.encoding()
    );
    debug_assert_eq!(
        casted.dtype(),
        dtype,
        "Cast dtype mismatch {}",
        array.encoding()
    );

    Ok(casted)
}

fn try_cast_impl(array: &Array, dtype: &DType) -> VortexResult<Array> {
    // TODO(ngates): check for null_count if dtype is non-nullable
    if let Some(f) = array.vtable().cast_fn() {
        return f.cast(array, dtype);
    }

    // Otherwise, we fall back to the canonical implementations.
    log::debug!(
        "Falling back to canonical cast for encoding {} and dtype {} to {}",
        array.encoding(),
        array.dtype(),
        dtype
    );
    let canonicalized = array.clone().into_canonical()?.into_array();
    if let Some(f) = canonicalized.vtable().cast_fn() {
        return f.cast(&canonicalized, dtype);
    }

    vortex_bail!(
        "No compute kernel to cast array from {} to {}",
        array.dtype(),
        dtype
    )
}
