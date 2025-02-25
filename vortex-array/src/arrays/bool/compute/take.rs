use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, BoolEncoding, ConstantArray, PrimitiveArray};
use crate::builders::ArrayBuilder;
use crate::compute::{TakeFn, fill_null};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, ToCanonical};

impl TakeFn<&BoolArray> for BoolEncoding {
    fn take(&self, array: &BoolArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices_nulls_zeroed = match indices.validity_mask()? {
            Mask::AllTrue(_) => indices.to_array(),
            Mask::AllFalse(_) => {
                return Ok(ConstantArray::new(
                    Scalar::null(array.dtype().as_nullable()),
                    indices.len(),
                )
                .into_array());
            }
            Mask::Values(_) => fill_null(indices, Scalar::from(0).cast(indices.dtype())?)?,
        };
        let indices_nulls_zeroed = indices_nulls_zeroed.to_primitive()?;
        let buffer = match_each_integer_ptype!(indices_nulls_zeroed.ptype(), |$I| {
            take_valid_indices::<$I>(array, &indices_nulls_zeroed)
        });

        Ok(BoolArray::new(buffer, array.validity().take(indices)?).into_array())
    }

    fn take_into(
        &self,
        array: &BoolArray,
        indices: &dyn Array,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        builder.extend_from_array(&self.take(array, indices)?)
    }
}

fn take_valid_indices<I: AsPrimitive<usize> + NativePType>(
    array: &BoolArray,
    indices: &PrimitiveArray,
) -> BooleanBuffer {
    // For boolean arrays that roughly fit into a single page (at least, on Linux), it's worth
    // the overhead to convert to a Vec<bool>.
    if array.len() <= 4096 {
        let bools = array.boolean_buffer().into_iter().collect_vec();
        take_byte_bool(bools, indices.as_slice::<I>())
    } else {
        take_bool(array.boolean_buffer(), indices.as_slice::<I>())
    }
}

fn take_byte_bool<I: AsPrimitive<usize>>(bools: Vec<bool>, indices: &[I]) -> BooleanBuffer {
    BooleanBuffer::collect_bool(indices.len(), |idx| {
        bools[unsafe { (*indices.get_unchecked(idx)).as_() }]
    })
}

fn take_bool<I: AsPrimitive<usize>>(bools: &BooleanBuffer, indices: &[I]) -> BooleanBuffer {
    BooleanBuffer::collect_bool(indices.len(), |idx| {
        // We can always take from the indices unchecked since collect_bool just iterates len.
        bools.value(unsafe { (*indices.get_unchecked(idx)).as_() })
    })
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::arrays::BoolArray;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::compute::{scalar_at, take};
    use crate::validity::Validity;
    use crate::{Array, ToCanonical};

    #[test]
    fn take_nullable() {
        let reference = BoolArray::from_iter(vec![
            Some(false),
            Some(true),
            Some(false),
            None,
            Some(false),
        ]);

        let b = take(&reference, &PrimitiveArray::from_iter([0, 3, 4]))
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(
            b.boolean_buffer(),
            BoolArray::from_iter([Some(false), None, Some(false)]).boolean_buffer()
        );

        let nullable_bool_dtype = DType::Bool(Nullability::Nullable);
        let all_invalid_indices = PrimitiveArray::from_option_iter([None::<u32>, None, None]);
        let b = take(&reference, &all_invalid_indices).unwrap();
        assert_eq!(b.dtype(), &nullable_bool_dtype);
        assert_eq!(
            scalar_at(&b, 0).unwrap(),
            Scalar::null(nullable_bool_dtype.clone())
        );
        assert_eq!(
            scalar_at(&b, 1).unwrap(),
            Scalar::null(nullable_bool_dtype.clone())
        );
        assert_eq!(scalar_at(&b, 2).unwrap(), Scalar::null(nullable_bool_dtype));
    }

    #[test]
    fn test_bool_array_take_with_null_out_of_bounds_indices() {
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).to_array()),
        );
        let actual = take(&values, &indices).unwrap();
        assert_eq!(scalar_at(&actual, 0).unwrap(), Scalar::from(Some(false)));
        // position 3 is null
        assert_eq!(scalar_at(&actual, 1).unwrap(), Scalar::null_typed::<bool>());
        // the third index is null
        assert_eq!(scalar_at(&actual, 2).unwrap(), Scalar::null_typed::<bool>());
    }

    #[test]
    fn test_non_null_bool_array_take_with_null_out_of_bounds_indices() {
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([true, true, false]).to_array()),
        );
        let actual = take(&values, &indices).unwrap();
        assert_eq!(scalar_at(&actual, 0).unwrap(), Scalar::from(Some(false)));
        assert_eq!(scalar_at(&actual, 1).unwrap(), Scalar::from(Some(true)));
        // the third index is null
        assert_eq!(scalar_at(&actual, 2).unwrap(), Scalar::null_typed::<bool>());
    }

    #[test]
    fn test_bool_array_take_all_null_indices() {
        let values = BoolArray::from_iter(vec![Some(false), Some(true), None, None, Some(false)]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).to_array()),
        );
        let actual = take(&values, &indices).unwrap();
        assert_eq!(scalar_at(&actual, 0).unwrap(), Scalar::null_typed::<bool>());
        assert_eq!(scalar_at(&actual, 1).unwrap(), Scalar::null_typed::<bool>());
        assert_eq!(scalar_at(&actual, 2).unwrap(), Scalar::null_typed::<bool>());
    }

    #[test]
    fn test_non_null_bool_array_take_all_null_indices() {
        let values = BoolArray::from_iter(vec![false, true, false, true, false]);
        let indices = PrimitiveArray::new(
            buffer![0, 3, 100],
            Validity::Array(BoolArray::from_iter([false, false, false]).to_array()),
        );
        let actual = take(&values, &indices).unwrap();
        assert_eq!(scalar_at(&actual, 0).unwrap(), Scalar::null_typed::<bool>());
        assert_eq!(scalar_at(&actual, 1).unwrap(), Scalar::null_typed::<bool>());
        assert_eq!(scalar_at(&actual, 2).unwrap(), Scalar::null_typed::<bool>());
    }
}
