use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less};

use fastlanes::BitPacking;
use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_array::compute::{
    IndexOrd, Len, SearchResult, SearchSorted, SearchSortedFn, SearchSortedSide,
    SearchSortedUsizeFn,
};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_dtype::{match_each_unsigned_integer_ptype, DType, NativePType};
use vortex_error::{VortexError, VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use crate::{unpack_single_primitive, BitPackedArray, BitPackedEncoding};

impl SearchSortedFn<BitPackedArray> for BitPackedEncoding {
    fn search_sorted(
        &self,
        array: &BitPackedArray,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        // NOTE: it is a precondition of BitPackedArray that all values must be >= 0, thus it is
        //  always safe to promote to unsigned type without loss of ordering of the values.
        match_each_unsigned_integer_ptype!(array.ptype().to_unsigned(), |$P| {
            search_sorted_typed::<$P>(array, value, side)
        })
    }

    fn search_sorted_many(
        &self,
        array: &BitPackedArray,
        values: &[Scalar],
        side: SearchSortedSide,
    ) -> VortexResult<Vec<SearchResult>> {
        match_each_unsigned_integer_ptype!(array.ptype(), |$P| {
            let searcher = BitPackedSearch::<'_, $P>::new(array);

            values
                .iter()
                .map(|value| {
                    // Unwrap to native value
                    let unwrapped_value: $P = value.cast(array.dtype())?.try_into()?;
                    Ok(searcher.search_sorted(&unwrapped_value, side))
                })
                .try_collect()
        })
    }
}

impl SearchSortedUsizeFn<BitPackedArray> for BitPackedEncoding {
    fn search_sorted_usize(
        &self,
        array: &BitPackedArray,
        value: usize,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        match_each_unsigned_integer_ptype!(array.ptype().to_unsigned(), |$P| {
            // NOTE: conversion may truncate silently.
            if let Some(pvalue) = num_traits::cast::<usize, $P>(value) {
                search_sorted_native(array, pvalue, side)
            } else {
                // provided u64 is too large to fit in the provided PType, value must be off
                // the right end of the array.
                Ok(SearchResult::NotFound(array.len()))
            }
        })
    }

    fn search_sorted_usize_many(
        &self,
        array: &BitPackedArray,
        values: &[usize],
        side: SearchSortedSide,
    ) -> VortexResult<Vec<SearchResult>> {
        match_each_unsigned_integer_ptype!(array.ptype().to_unsigned(), |$P| {
            let searcher = BitPackedSearch::<'_, $P>::new(array);

            values
                .iter()
                .map(|&value| {
                    // NOTE: truncating cast
                    let cast_value: $P = value as $P;
                    Ok(searcher.search_sorted(&cast_value, side))
                })
                .try_collect()
        })
    }
}

fn search_sorted_typed<T>(
    array: &BitPackedArray,
    value: &Scalar,
    side: SearchSortedSide,
) -> VortexResult<SearchResult>
where
    T: NativePType
        + TryFrom<Scalar, Error = VortexError>
        + BitPacking
        + AsPrimitive<usize>
        + AsPrimitive<u64>,
{
    // NOTE: we use the unsigned variant of the BitPackedArray DType so that we can use it
    //  in the BitPackedSearch. We need a type that impls fastlanes::BitPack, and it is a
    //  precondition for BitPackedArray that all values must be non-negative, so promotion
    //  is cheap and safe.
    let Ok(unsigned_value) = value.cast(&DType::from(array.ptype().to_unsigned())) else {
        // If the value can't be casted to unsigned dtype then it can't exist in the array and would be smaller than any value present
        return Ok(SearchResult::NotFound(0));
    };
    let native_value: T = unsigned_value.try_into()?;
    search_sorted_native(array, native_value, side)
}

/// Native variant of search_sorted that operates over Rust unsigned integer types.
fn search_sorted_native<T>(
    array: &BitPackedArray,
    value: T,
    side: SearchSortedSide,
) -> VortexResult<SearchResult>
where
    T: NativePType + BitPacking + AsPrimitive<usize> + AsPrimitive<u64>,
{
    if let Some(patches) = array.patches() {
        // If patches exist they must be the last elements in the array, if the value we're looking for is greater than
        // max packed value just search the patches
        let usize_value: usize = value.as_();
        if usize_value > array.max_packed_value() {
            patches.search_sorted(usize_value, side)
        } else {
            Ok(BitPackedSearch::<'_, T>::new(array).search_sorted(&value, side))
        }
    } else {
        Ok(BitPackedSearch::<'_, T>::new(array).search_sorted(&value, side))
    }
}

/// This wrapper exists, so that you can't invoke SearchSorted::search_sorted directly on BitPackedArray as it omits searching patches
#[derive(Debug)]
struct BitPackedSearch<'a, T> {
    // NOTE: caching this here is important for performance, as each call to `as_slice`
    //  invokes a call to DType <> PType conversion
    packed_as_slice: &'a [T],
    offset: u16,
    length: usize,
    bit_width: u8,
    first_non_null_idx: usize,
    first_patch_index: usize,
}

impl<'a, T: BitPacking + NativePType> BitPackedSearch<'a, T> {
    pub fn new(array: &'a BitPackedArray) -> Self {
        let first_patch_index = array
            .patches()
            .map(|p| p.min_index())
            .transpose()
            .vortex_expect("Failed to get min patch index")
            .unwrap_or_else(|| array.len());
        let first_non_null_idx = match array.validity() {
            Validity::NonNullable | Validity::AllValid => 0,
            Validity::AllInvalid => array.len(),
            Validity::Array(varray) => {
                // In sorted order, nulls come before all the non-null values.
                varray
                    .statistics()
                    .compute_true_count()
                    .vortex_expect("Failed to compute true count")
            }
        };

        Self {
            packed_as_slice: array.packed_slice::<T>(),
            offset: array.offset(),
            length: array.len(),
            bit_width: array.bit_width(),
            first_non_null_idx,
            first_patch_index,
        }
    }
}

impl<T: BitPacking + NativePType> IndexOrd<T> for BitPackedSearch<'_, T> {
    fn index_cmp(&self, idx: usize, elem: &T) -> Option<Ordering> {
        if idx < self.first_non_null_idx {
            return Some(Less);
        }
        if idx >= self.first_patch_index {
            return Some(Greater);
        }

        // SAFETY: Used in search_sorted_by which ensures that idx is within bounds
        let val: T = unsafe {
            unpack_single_primitive(
                self.packed_as_slice,
                self.bit_width as usize,
                idx + self.offset as usize,
            )
        };
        Some(val.total_compare(*elem))
    }
}

impl<T> Len for BitPackedSearch<'_, T> {
    fn len(&self) -> usize {
        self.length
    }
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::{
        search_sorted, search_sorted_many, slice, SearchResult, SearchSortedSide,
    };
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::BitPackedArray;

    #[test]
    fn search_with_patches() {
        let bitpacked = BitPackedArray::encode(&buffer![1u32, 2, 3, 4, 5].into_array(), 2)
            .unwrap()
            .into_array();
        assert_eq!(
            search_sorted(&bitpacked, 4, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(3)
        );
        assert_eq!(
            search_sorted(&bitpacked, 5, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(4)
        );
        assert_eq!(
            search_sorted(&bitpacked, 6, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(5)
        );
        assert_eq!(
            search_sorted(&bitpacked, 0, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
    }

    #[test]
    fn search_sliced() {
        let bitpacked = slice(
            BitPackedArray::encode(PrimitiveArray::from_iter([1u32, 2, 3, 4, 5]).as_ref(), 2)
                .unwrap(),
            2,
            4,
        )
        .unwrap();
        assert_eq!(
            search_sorted(&bitpacked, 3, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(0)
        );
        assert_eq!(
            search_sorted(&bitpacked, 4, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(1)
        );
    }

    #[test]
    fn test_search_sorted_nulls() {
        let bitpacked = BitPackedArray::encode(
            PrimitiveArray::from_option_iter([None, None, Some(1u64)]).as_ref(),
            2,
        )
        .unwrap();

        let found = search_sorted(
            bitpacked.as_ref(),
            Scalar::primitive(1u64, Nullability::Nullable),
            SearchSortedSide::Left,
        )
        .unwrap();
        assert_eq!(found, SearchResult::Found(2));
    }

    #[test]
    fn test_search_sorted_nulls_not_found() {
        let bitpacked = BitPackedArray::encode(
            PrimitiveArray::from_option_iter([None, None, Some(0u8), Some(107u8)]).as_ref(),
            0,
        )
        .unwrap();

        let found = search_sorted(
            bitpacked.as_ref(),
            Scalar::primitive(127u8, Nullability::Nullable),
            SearchSortedSide::Left,
        )
        .unwrap();
        assert_eq!(found, SearchResult::NotFound(4));
    }

    #[test]
    fn test_search_sorted_many() {
        // Test search_sorted_many with an array that contains several null values.
        let bitpacked = BitPackedArray::encode(
            PrimitiveArray::from_option_iter([
                None,
                None,
                None,
                None,
                Some(1u64),
                Some(2u64),
                Some(3u64),
            ])
            .as_ref(),
            3,
        )
        .unwrap();

        let results = search_sorted_many(
            bitpacked.as_ref(),
            &[3u64, 2u64, 1u64],
            SearchSortedSide::Left,
        )
        .unwrap();

        assert_eq!(
            results,
            vec![
                SearchResult::Found(6),
                SearchResult::Found(5),
                SearchResult::Found(4),
            ]
        );
    }

    #[test]
    fn test_missing_signed() {
        let bitpacked = BitPackedArray::encode(&buffer![1i32, 2, 3, 4, 5].into_array(), 2)
            .unwrap()
            .into_array();
        assert_eq!(
            search_sorted(&bitpacked, -4, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
    }
}
