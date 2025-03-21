use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{
    BinaryNumericFn, FilterKernel, FilterKernelAdapter, InvertFn, KernelRef, ScalarAtFn,
    SearchSortedFn, SearchSortedUsizeFn, SliceFn, TakeFn,
};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayComputeImpl, ArrayRef};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::{SparseArray, SparseEncoding};

mod binary_numeric;
mod invert;
mod search_sorted;
mod slice;
mod take;

impl ArrayComputeImpl for SparseArray {
    const FILTER: Option<KernelRef> = FilterKernelAdapter(SparseEncoding).some();
}
impl ComputeVTable for SparseEncoding {
    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<&dyn Array>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<&dyn Array>> {
        Some(self)
    }

    fn search_sorted_usize_fn(&self) -> Option<&dyn SearchSortedUsizeFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}

impl ScalarAtFn<&SparseArray> for SparseEncoding {
    fn scalar_at(&self, array: &SparseArray, index: usize) -> VortexResult<Scalar> {
        Ok(array
            .patches()
            .get_patched(index)?
            .unwrap_or_else(|| array.fill_scalar().clone()))
    }
}

impl FilterKernel for SparseEncoding {
    fn filter(&self, array: &SparseArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let new_length = mask.true_count();

        let Some(new_patches) = array.patches().filter(mask)? else {
            return Ok(ConstantArray::new(array.fill_scalar().clone(), new_length).into_array());
        };

        Ok(
            SparseArray::try_new_from_patches(new_patches, array.fill_scalar().clone())?
                .into_array(),
        )
    }
}

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric;
    use vortex_array::compute::conformance::mask::test_mask;
    use vortex_array::compute::{filter, try_cast};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::SparseArray;

    #[fixture]
    fn array() -> ArrayRef {
        SparseArray::try_new(
            buffer![2u64, 9, 15].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            20,
            Scalar::null_typed::<i32>(),
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    fn test_filter(array: ArrayRef) {
        let mut predicate = vec![false, false, true];
        predicate.extend_from_slice(&[false; 17]);
        let mask = Mask::from_iter(predicate);

        let filtered_array = filter(&array, &mask).unwrap();
        let filtered_array = SparseArray::try_from(filtered_array).unwrap();

        assert_eq!(filtered_array.len(), 1);
        assert_eq!(filtered_array.patches().values().len(), 1);
        assert_eq!(filtered_array.patches().indices().len(), 1);
    }

    #[test]
    fn true_fill_value() {
        let mask = Mask::from_iter([false, true, false, true, false, true, true]);
        let array = SparseArray::try_new(
            buffer![0_u64, 3, 6].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
            7,
            Scalar::null_typed::<i32>(),
        )
        .unwrap()
        .into_array();

        let filtered_array = filter(&array, &mask).unwrap();
        let filtered_array = SparseArray::try_from(filtered_array).unwrap();

        assert_eq!(filtered_array.len(), 4);
        let primitive = filtered_array.patches().indices().to_primitive().unwrap();

        assert_eq!(primitive.as_slice::<u64>(), &[1, 3]);
    }

    #[rstest]
    fn test_sparse_binary_numeric(array: ArrayRef) {
        test_binary_numeric::<i32>(array)
    }

    #[test]
    fn test_mask_sparse_array() {
        let null_fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        test_mask(
            &SparseArray::try_new(
                buffer![1u64, 2, 4].into_array(),
                try_cast(
                    &buffer![100i32, 200, 300].into_array(),
                    null_fill_value.dtype(),
                )
                .unwrap(),
                5,
                null_fill_value,
            )
            .unwrap(),
        );

        let ten_fill_value = Scalar::from(10i32);
        test_mask(
            &SparseArray::try_new(
                buffer![1u64, 2, 4].into_array(),
                buffer![100i32, 200, 300].into_array(),
                5,
                ten_fill_value,
            )
            .unwrap(),
        )
    }
}
