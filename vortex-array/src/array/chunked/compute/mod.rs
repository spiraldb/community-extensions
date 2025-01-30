use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::chunked::ChunkedArray;
use crate::array::ChunkedEncoding;
use crate::compute::{
    try_cast, BinaryBooleanFn, BinaryNumericFn, CastFn, CompareFn, FillNullFn, FilterFn, InvertFn,
    ScalarAtFn, SliceFn, TakeFn,
};
use crate::vtable::ComputeVTable;
use crate::{Array, IntoArray};

mod binary_numeric;
mod boolean;
mod compare;
mod fill_null;
mod filter;
mod invert;
mod scalar_at;
mod slice;
mod take;

impl ComputeVTable for ChunkedEncoding {
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<Array>> {
        Some(self)
    }

    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<Array>> {
        Some(self)
    }

    fn cast_fn(&self) -> Option<&dyn CastFn<Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<Array>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }
}

impl CastFn<ChunkedArray> for ChunkedEncoding {
    fn cast(&self, array: &ChunkedArray, dtype: &DType) -> VortexResult<Array> {
        let mut cast_chunks = Vec::new();
        for chunk in array.chunks() {
            cast_chunks.push(try_cast(&chunk, dtype)?);
        }

        Ok(ChunkedArray::try_new(cast_chunks, dtype.clone())?.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::array::chunked::ChunkedArray;
    use crate::compute::try_cast;
    use crate::{IntoArray, IntoArrayVariant};

    #[test]
    fn test_cast_chunked() {
        let arr0 = buffer![0u32, 1].into_array();
        let arr1 = buffer![2u32, 3].into_array();

        let chunked = ChunkedArray::try_new(
            vec![arr0, arr1],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        // Two levels of chunking, just to be fancy.
        let root = ChunkedArray::try_new(
            vec![chunked],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        assert_eq!(
            try_cast(
                &root,
                &DType::Primitive(PType::U64, Nullability::NonNullable)
            )
            .unwrap()
            .into_primitive()
            .unwrap()
            .as_slice::<u64>(),
            &[0u64, 1, 2, 3],
        );
    }
}
