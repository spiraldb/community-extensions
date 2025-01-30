use num_traits::AsPrimitive;
use vortex_array::compute::{FillForwardFn, ScalarAtFn, SliceFn, TakeFn};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_dtype::{match_each_integer_ptype, Nullability};
use vortex_error::{vortex_err, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use super::{ByteBoolArray, ByteBoolEncoding};

impl ComputeVTable for ByteBoolEncoding {
    fn fill_forward_fn(&self) -> Option<&dyn FillForwardFn<Array>> {
        None
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

impl ScalarAtFn<ByteBoolArray> for ByteBoolEncoding {
    fn scalar_at(&self, array: &ByteBoolArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            array.buffer()[index] == 1,
            array.dtype().nullability(),
        ))
    }
}

impl SliceFn<ByteBoolArray> for ByteBoolEncoding {
    fn slice(&self, array: &ByteBoolArray, start: usize, stop: usize) -> VortexResult<Array> {
        Ok(ByteBoolArray::try_new(
            array.buffer().slice(start..stop),
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

impl TakeFn<ByteBoolArray> for ByteBoolEncoding {
    fn take(&self, array: &ByteBoolArray, indices: &Array) -> VortexResult<Array> {
        let validity = array.logical_validity()?;
        let indices = indices.clone().into_primitive()?;
        let bools = array.as_slice();

        // FIXME(ngates): we should be operating over canonical validity, which doesn't
        //  have fallible is_valid function.
        let arr = match validity {
            Mask::AllTrue(_) => {
                let bools = match_each_integer_ptype!(indices.ptype(), |$I| {
                    indices.as_slice::<$I>()
                    .iter()
                    .map(|&idx| {
                        let idx: usize = idx.as_();
                        bools[idx]
                    })
                    .collect::<Vec<_>>()
                });

                ByteBoolArray::from(bools).into_array()
            }
            Mask::AllFalse(_) => ByteBoolArray::from(vec![None; indices.len()]).into_array(),
            Mask::Values(values) => {
                let bools = match_each_integer_ptype!(indices.ptype(), |$I| {
                    indices.as_slice::<$I>()
                    .iter()
                    .map(|&idx| {
                        let idx = idx.as_();
                        if values.value(idx) {
                            Some(bools[idx])
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<Option<_>>>()
                });

                ByteBoolArray::from(bools).into_array()
            }
        };

        Ok(arr)
    }
}

impl FillForwardFn<ByteBoolArray> for ByteBoolEncoding {
    fn fill_forward(&self, array: &ByteBoolArray) -> VortexResult<Array> {
        let validity = array.logical_validity()?;
        if array.dtype().nullability() == Nullability::NonNullable {
            return Ok(array.clone().into_array());
        }
        // all valid, but we need to convert to non-nullable
        if validity.all_true() {
            return Ok(
                ByteBoolArray::try_new(array.buffer().clone(), Validity::AllValid)?.into_array(),
            );
        }
        // all invalid => fill with default value (false)
        if validity.all_false() {
            return Ok(
                ByteBoolArray::try_from_vec(vec![false; array.len()], Validity::AllValid)?
                    .into_array(),
            );
        }

        let validity = validity
            .to_null_buffer()
            .ok_or_else(|| vortex_err!("Failed to convert array validity to null buffer"))?;

        let bools = array.as_slice();
        let mut last_value = bool::default();

        let filled = bools
            .iter()
            .zip(validity.inner().iter())
            .map(|(&v, is_valid)| {
                if is_valid {
                    last_value = v
                }

                last_value
            })
            .collect::<Vec<_>>();

        Ok(ByteBoolArray::try_from_vec(filled, Validity::AllValid)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::{compare, scalar_at, slice, Operator};

    use super::*;

    #[test]
    fn test_slice() {
        let original = vec![Some(true), Some(true), None, Some(false), None];
        let vortex_arr = ByteBoolArray::from(original);

        let sliced_arr = slice(vortex_arr.as_ref(), 1, 4).unwrap();
        let sliced_arr = ByteBoolArray::try_from(sliced_arr).unwrap();

        let s = scalar_at(sliced_arr.as_ref(), 0).unwrap();
        assert_eq!(s.as_bool().value(), Some(true));

        let s = scalar_at(sliced_arr.as_ref(), 1).unwrap();
        assert!(!sliced_arr.is_valid(1).unwrap());
        assert!(s.is_null());
        assert_eq!(s.as_bool().value(), None);

        let s = scalar_at(sliced_arr.as_ref(), 2).unwrap();
        assert_eq!(s.as_bool().value(), Some(false));
    }

    #[test]
    fn test_compare_all_equal() {
        let lhs = ByteBoolArray::from(vec![true; 5]);
        let rhs = ByteBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        for i in 0..arr.len() {
            let s = scalar_at(arr.as_ref(), i).unwrap();
            assert!(s.is_valid());
            assert_eq!(s.as_bool().value(), Some(true));
        }
    }

    #[test]
    fn test_compare_all_different() {
        let lhs = ByteBoolArray::from(vec![false; 5]);
        let rhs = ByteBoolArray::from(vec![true; 5]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        for i in 0..arr.len() {
            let s = scalar_at(&arr, i).unwrap();
            assert!(s.is_valid());
            assert_eq!(s.as_bool().value(), Some(false));
        }
    }

    #[test]
    fn test_compare_with_nulls() {
        let lhs = ByteBoolArray::from(vec![true; 5]);
        let rhs = ByteBoolArray::from(vec![Some(true), Some(true), Some(true), Some(false), None]);

        let arr = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        for i in 0..3 {
            let s = scalar_at(&arr, i).unwrap();
            assert!(s.is_valid());
            assert_eq!(s.as_bool().value(), Some(true));
        }

        let s = scalar_at(&arr, 3).unwrap();
        assert!(s.is_valid());
        assert_eq!(s.as_bool().value(), Some(false));

        let s = scalar_at(&arr, 4).unwrap();
        assert!(s.is_null());
    }
}
