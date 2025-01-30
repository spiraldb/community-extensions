mod binary_numeric;
mod compare;
mod like;

use vortex_array::compute::{
    filter, scalar_at, slice, take, BinaryNumericFn, CompareFn, FilterFn, LikeFn, ScalarAtFn,
    SliceFn, TakeFn,
};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::{DictArray, DictEncoding};

impl ComputeVTable for DictEncoding {
    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn like_fn(&self) -> Option<&dyn LikeFn<Array>> {
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

impl ScalarAtFn<DictArray> for DictEncoding {
    fn scalar_at(&self, array: &DictArray, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = scalar_at(array.codes(), index)?.as_ref().try_into()?;
        scalar_at(array.values(), dict_index)
    }
}

impl TakeFn<DictArray> for DictEncoding {
    fn take(&self, array: &DictArray, indices: &Array) -> VortexResult<Array> {
        // Dict
        //   codes: 0 0 1
        //   dict: a b c d e f g h
        let codes = take(array.codes(), indices)?;
        DictArray::try_new(codes, array.values()).map(|a| a.into_array())
    }
}

impl FilterFn<DictArray> for DictEncoding {
    fn filter(&self, array: &DictArray, mask: &Mask) -> VortexResult<Array> {
        let codes = filter(&array.codes(), mask)?;
        DictArray::try_new(codes, array.values()).map(|a| a.into_array())
    }
}

impl SliceFn<DictArray> for DictEncoding {
    // TODO(robert): Add function to trim the dictionary
    fn slice(&self, array: &DictArray, start: usize, stop: usize) -> VortexResult<Array> {
        DictArray::try_new(slice(array.codes(), start, stop)?, array.values())
            .map(|a| a.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::array::{ConstantArray, PrimitiveArray, VarBinViewArray};
    use vortex_array::compute::test_harness::test_binary_numeric;
    use vortex_array::compute::{compare, scalar_at, slice, Operator};
    use vortex_array::{Array, IntoArray, IntoArrayVariant};
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::dict_encode;

    #[test]
    fn canonicalise_nullable_primitive() {
        let reference =
            PrimitiveArray::from_option_iter([Some(42), Some(-9), None, Some(42), None, Some(-9)]);
        let dict = dict_encode(reference.as_ref()).unwrap();
        let flattened_dict = dict.into_array().into_primitive().unwrap();
        assert_eq!(flattened_dict.byte_buffer(), reference.byte_buffer());
    }

    #[test]
    fn canonicalise_nullable_varbin() {
        let reference = VarBinViewArray::from_iter(
            vec![Some("a"), Some("b"), None, Some("a"), None, Some("b")],
            DType::Utf8(Nullability::Nullable),
        );
        assert_eq!(reference.len(), 6);
        let dict = dict_encode(reference.as_ref()).unwrap();
        let flattened_dict = dict.into_array().into_varbinview().unwrap();
        assert_eq!(
            flattened_dict
                .with_iterator(|iter| iter
                    .map(|slice| slice.map(|s| s.to_vec()))
                    .collect::<Vec<_>>())
                .unwrap(),
            reference
                .with_iterator(|iter| iter
                    .map(|slice| slice.map(|s| s.to_vec()))
                    .collect::<Vec<_>>())
                .unwrap(),
        );
    }

    fn sliced_dict_array() -> Array {
        let reference = PrimitiveArray::from_option_iter([
            Some(42),
            Some(-9),
            None,
            Some(42),
            Some(1),
            Some(5),
        ]);
        let dict = dict_encode(reference.as_ref()).unwrap();
        slice(dict, 1, 4).unwrap()
    }

    #[test]
    fn compare_sliced_dict() {
        let sliced = sliced_dict_array();
        let compared = compare(sliced, ConstantArray::new(42, 3), Operator::Eq).unwrap();

        assert_eq!(
            scalar_at(&compared, 0).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            scalar_at(&compared, 1).unwrap(),
            Scalar::null(DType::Bool(Nullability::Nullable))
        );
        assert_eq!(
            scalar_at(compared, 2).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
    }

    #[test]
    fn test_dict_binary_numeric() {
        let array = sliced_dict_array();
        test_binary_numeric::<i32>(array)
    }
}
