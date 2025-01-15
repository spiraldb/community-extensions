use std::cmp::{max, min};
use std::fmt::{Display, Formatter};

use arrow_buffer::BooleanBuffer;
use vortex_array::array::{BoolArray, PrimitiveArray, SparseArray};
use vortex_array::compute::{and, filter, slice, try_cast, FilterMask};
use vortex_array::validity::{ArrayValidity, LogicalValidity, Validity};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::Buffer;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexResult, VortexUnwrap};

/// A RowMask captures a set of selected rows offset by a range.
///
/// i.e., row zero of the inner FilterMask represents the offset row of the RowMask.
#[derive(Debug, Clone)]
pub struct RowMask {
    mask: FilterMask,
    begin: usize,
    end: usize,
}

#[cfg(test)]
impl PartialEq for RowMask {
    fn eq(&self, other: &Self) -> bool {
        self.begin == other.begin
            && self.end == other.end
            && self.mask.boolean_buffer() == other.mask.boolean_buffer()
    }
}

impl Display for RowMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RowSelector [{}..{}]", self.begin, self.end)
    }
}

impl RowMask {
    pub fn try_new(mask: FilterMask, begin: usize, end: usize) -> VortexResult<Self> {
        if mask.len() != (end - begin) {
            vortex_bail!(
                "FilterMask must be the same length {} as the given range {}..{}",
                mask.len(),
                begin,
                end
            );
        }
        Ok(Self { mask, begin, end })
    }

    /// Construct a RowMask which is valid in the given range.
    pub fn new_valid_between(begin: usize, end: usize) -> Self {
        RowMask::try_new(
            FilterMask::from(BooleanBuffer::new_set(end - begin)),
            begin,
            end,
        )
        .vortex_unwrap()
    }

    /// Construct a RowMask which is invalid everywhere in the given range.
    pub fn new_invalid_between(begin: usize, end: usize) -> Self {
        RowMask::try_new(
            FilterMask::from(BooleanBuffer::new_unset(end - begin)),
            begin,
            end,
        )
        .vortex_unwrap()
    }

    /// Creates a RowMask from an array, only supported boolean and integer types.
    pub fn from_array(array: &ArrayData, begin: usize, end: usize) -> VortexResult<Self> {
        if array.dtype().is_int() {
            Self::from_index_array(array, begin, end)
        } else if array.dtype().is_boolean() {
            Self::from_mask_array(array, begin, end)
        } else {
            vortex_bail!(
                "RowMask can only be created from integer or boolean arrays, got {} instead.",
                array.dtype()
            );
        }
    }

    /// Construct a RowMask from a Boolean typed array.
    ///
    /// True-valued positions are kept by the returned mask.
    fn from_mask_array(array: &ArrayData, begin: usize, end: usize) -> VortexResult<Self> {
        match array.logical_validity() {
            LogicalValidity::AllValid(_) => {
                Self::try_new(FilterMask::try_from(array.clone())?, begin, end)
            }
            LogicalValidity::AllInvalid(_) => Ok(Self::new_invalid_between(begin, end)),
            LogicalValidity::Array(validity) => {
                let bitmask = and(array.clone(), validity)?;
                Self::try_new(FilterMask::try_from(bitmask)?, begin, end)
            }
        }
    }

    /// Construct a RowMask from an integral array.
    ///
    /// The array values are interpreted as indices and those indices are kept by the returned mask.
    fn from_index_array(array: &ArrayData, begin: usize, end: usize) -> VortexResult<Self> {
        let indices =
            try_cast(array, &DType::Primitive(PType::U64, NonNullable))?.into_primitive()?;

        // TODO(ngates): should from_indices take u64?
        let mask = FilterMask::from_indices(
            end - begin,
            indices
                .as_slice::<u64>()
                .iter()
                .map(|i| *i as usize)
                .collect(),
        );

        RowMask::try_new(mask, begin, end)
    }

    /// Combine the RowMask with bitmask values resulting in new RowMask containing only values true in the bitmask
    pub fn and_bitmask(&self, bitmask: ArrayData) -> VortexResult<Self> {
        // If we are a dense all true bitmap just take the bitmask array
        if self.mask.true_count() == self.len() {
            if bitmask.len() != self.len() {
                vortex_bail!(
                    "Bitmask length {} does not match our length {}",
                    bitmask.len(),
                    self.mask.len()
                );
            }
            Self::from_mask_array(&bitmask, self.begin, self.end)
        } else {
            // TODO(robert): Avoid densifying sparse values just to get true indices
            let sparse_mask =
                SparseArray::try_new(self.to_indices_array()?, bitmask, self.len(), false.into())?
                    .into_array()
                    .into_bool()?;
            Self::from_mask_array(sparse_mask.as_ref(), self.begin(), self.end())
        }
    }

    pub fn and_rowmask(&self, other: RowMask) -> VortexResult<Self> {
        if other.true_count() == other.len() {
            return Ok(self.clone());
        }

        // If both masks align perfectly
        if self.begin == other.begin && self.end == other.end {
            let this_buffer = self.mask.boolean_buffer();
            let other_buffer = other.mask.boolean_buffer();

            let unified = this_buffer & other_buffer;
            return RowMask::from_mask_array(
                BoolArray::from(unified).as_ref(),
                self.begin,
                self.end,
            );
        }

        // Disjoint row ranges
        if self.end <= other.begin || self.begin >= other.end {
            return Ok(RowMask::new_invalid_between(
                min(self.begin, other.begin),
                max(self.end, other.end),
            ));
        }

        let output_begin = min(self.begin, other.begin);
        let output_end = max(self.end, other.end);
        let output_len = output_end - output_begin;

        let output_mask = FilterMask::from_intersection_indices(
            output_len,
            self.mask
                .indices()
                .iter()
                .copied()
                .map(|v| v + self.begin - output_begin),
            other
                .mask
                .indices()
                .iter()
                .copied()
                .map(|v| v + other.begin - output_begin),
        );

        Self::try_new(output_mask, output_begin, output_end)
    }

    pub fn is_all_false(&self) -> bool {
        self.mask.true_count() == 0
    }

    pub fn begin(&self) -> usize {
        self.begin
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn len(&self) -> usize {
        self.mask.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Limit mask to [begin..end) range
    pub fn slice(&self, begin: usize, end: usize) -> VortexResult<Self> {
        let range_begin = max(self.begin, begin);
        let range_end = min(self.end, end);
        RowMask::try_new(
            if range_begin == self.begin && range_end == self.end {
                self.mask.clone()
            } else {
                FilterMask::from(
                    self.mask
                        .boolean_buffer()
                        .slice(range_begin - self.begin, range_end - range_begin),
                )
            },
            range_begin,
            range_end,
        )
    }

    /// Filter array with this `RowMask`.
    ///
    /// This function assumes that Array is no longer than the mask length and that the mask starts on same offset as the array,
    /// i.e. the beginning of the array corresponds to the beginning of the mask with begin = 0
    pub fn filter_array(&self, array: impl AsRef<ArrayData>) -> VortexResult<Option<ArrayData>> {
        let true_count = self.mask.true_count();
        if true_count == 0 {
            return Ok(None);
        }

        let array = array.as_ref();

        let sliced = if self.len() == array.len() {
            array
        } else {
            // TODO(ngates): I thought the point was the array only covers the valid row range of
            //  the mask?
            &slice(array, self.begin, self.end)?
        };

        if true_count == sliced.len() {
            return Ok(Some(sliced.clone()));
        }

        filter(sliced, &self.mask).map(Some)
    }

    #[allow(deprecated)]
    fn to_indices_array(&self) -> VortexResult<ArrayData> {
        Ok(PrimitiveArray::new(
            self.mask
                .indices()
                .iter()
                .map(|i| *i as u64)
                .collect::<Buffer<u64>>(),
            Validity::NonNullable,
        )
        .into_array())
    }

    pub fn shift(self, offset: usize) -> VortexResult<RowMask> {
        let valid_shift = self.begin >= offset;
        if !valid_shift {
            vortex_bail!(
                "Can shift RowMask by at most {}, tried to shift by {offset}",
                self.begin
            )
        }
        RowMask::try_new(self.mask, self.begin - offset, self.end - offset)
    }

    // Get the true count of the underlying mask.
    pub fn true_count(&self) -> usize {
        self.mask.true_count()
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use rstest::rstest;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::FilterMask;
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_buffer::{buffer, Buffer};
    use vortex_error::VortexUnwrap;

    use crate::read::mask::RowMask;

    #[rstest]
    #[case(
        RowMask::try_new(FilterMask::from_iter([true, true, true, false, false, false, false, false, true, true]), 0, 10).unwrap(), (0, 1),
        RowMask::try_new(FilterMask::from_iter([true]), 0, 1).unwrap())]
    #[case(
        RowMask::try_new(FilterMask::from_iter([false, false, false, false, false, true, true, true, true, true]), 0, 10).unwrap(), (2, 5),
        RowMask::try_new(FilterMask::from_iter([false, false, false]), 2, 5).unwrap()
    )]
    #[case(
        RowMask::try_new(FilterMask::from_iter([true, true, true, true, false, false, false, false, false, false]), 0, 10).unwrap(), (2, 5),
        RowMask::try_new(FilterMask::from_iter([true, true, false]), 2, 5).unwrap()
    )]
    #[case(
        RowMask::try_new(FilterMask::from_iter([true, true, true, false, false, true, true, false, false, false]), 0, 10).unwrap(), (2, 6),
        RowMask::try_new(FilterMask::from_iter([true, false, false, true]), 2, 6).unwrap())]
    #[case(
        RowMask::try_new(FilterMask::from_iter([false, false, false, false, false, true, true, true, true, true]), 0, 10).unwrap(), (7, 11),
        RowMask::try_new(FilterMask::from_iter([true, true, true]), 7, 10).unwrap())]
    #[case(
        RowMask::try_new(FilterMask::from_iter([false, true, true, true, true, true]), 3, 9).unwrap(), (0, 5),
        RowMask::try_new(FilterMask::from_iter([false, true]), 3, 5).unwrap())]
    #[cfg_attr(miri, ignore)]
    fn slice(#[case] first: RowMask, #[case] range: (usize, usize), #[case] expected: RowMask) {
        assert_eq!(first.slice(range.0, range.1).vortex_unwrap(), expected);
    }

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn test_new() {
        RowMask::try_new(FilterMask::from(BooleanBuffer::new_unset(10)), 5, 10).unwrap();
    }

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn shift_invalid() {
        RowMask::try_new(FilterMask::from_iter([true, true, true, true, true]), 5, 10)
            .unwrap()
            .shift(7)
            .unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shift() {
        assert_eq!(
            RowMask::try_new(FilterMask::from_iter([true, true, true, true, true]), 5, 10)
                .unwrap()
                .shift(5)
                .unwrap(),
            RowMask::try_new(FilterMask::from_iter([true, true, true, true, true]), 0, 5).unwrap()
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filter_array() {
        let mask = RowMask::try_new(
            FilterMask::from_iter([
                false, false, false, false, false, true, true, true, true, true,
            ]),
            0,
            10,
        )
        .unwrap();
        let array = Buffer::from_iter(0..20).into_array();
        let filtered = mask.filter_array(array).unwrap().unwrap();
        assert_eq!(
            filtered.into_primitive().unwrap().as_slice::<i32>(),
            (5..10).collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic]
    fn test_row_mask_type_validation() {
        let array = PrimitiveArray::new(buffer![1.0, 2.0], Validity::AllInvalid).into_array();
        RowMask::from_array(&array, 0, 2).unwrap();
    }

    #[test]
    fn test_and_rowmap_disjoint() {
        let a = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 3], Validity::AllValid).as_ref(),
            0,
            10,
        )
        .unwrap();
        let b = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 3], Validity::AllValid).as_ref(),
            15,
            20,
        )
        .unwrap();

        let output = a.and_rowmask(b).unwrap();

        assert_eq!(output.begin, 0);
        assert_eq!(output.end, 20);
        assert!(output.is_all_false());
    }

    #[test]
    fn test_and_rowmap_aligned() {
        let a = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 3], Validity::AllValid).as_ref(),
            0,
            10,
        )
        .unwrap();
        let b = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 7], Validity::AllValid).as_ref(),
            0,
            10,
        )
        .unwrap();

        let output = a.and_rowmask(b).unwrap();

        assert_eq!(output.begin, 0);
        assert_eq!(output.end, 10);
        assert_eq!(output.true_count(), 2);
    }

    #[test]
    fn test_and_rowmap_intersect() {
        let a = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 3], Validity::AllValid).as_ref(),
            0,
            10,
        )
        .unwrap();
        let b = RowMask::from_array(
            PrimitiveArray::new(buffer!(1, 2, 7), Validity::AllValid).as_ref(),
            5,
            15,
        )
        .unwrap();

        let output = a.and_rowmask(b).unwrap();

        assert_eq!(output.begin, 0);
        assert_eq!(output.end, 15);
        assert_eq!(output.true_count(), 0);
    }
}
