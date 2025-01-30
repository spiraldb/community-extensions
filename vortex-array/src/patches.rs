use std::cmp::Ordering;
use std::fmt::Debug;

use itertools::Itertools as _;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use vortex_buffer::BufferMut;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{match_each_integer_ptype, DType, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_mask::{AllOr, Mask};
use vortex_scalar::Scalar;

use crate::aliases::hash_map::HashMap;
use crate::array::PrimitiveArray;
use crate::compute::{
    filter, scalar_at, search_sorted, search_sorted_usize, search_sorted_usize_many, slice,
    sub_scalar, take, SearchResult, SearchSortedSide,
};
use crate::stats::Stat;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray, IntoArrayVariant};

#[derive(
    Copy,
    Clone,
    Debug,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
    rkyv::bytecheck::CheckBytes,
)]
#[bytecheck(crate = rkyv::bytecheck)]
#[repr(C)]
pub struct PatchesMetadata {
    len: usize,
    indices_ptype: PType,
}

impl PatchesMetadata {
    pub fn new(len: usize, indices_ptype: PType) -> Self {
        Self { len, indices_ptype }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn indices_dtype(&self) -> DType {
        assert!(
            self.indices_ptype.is_unsigned_int(),
            "Patch indices must be unsigned integers"
        );
        DType::Primitive(self.indices_ptype, NonNullable)
    }
}

/// A helper for working with patched arrays.
#[derive(Debug, Clone)]
pub struct Patches {
    array_len: usize,
    indices: Array,
    values: Array,
}

impl Patches {
    pub fn new(array_len: usize, indices: Array, values: Array) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Patch indices and values must have the same length"
        );
        assert!(
            indices.dtype().is_unsigned_int(),
            "Patch indices must be unsigned integers"
        );
        assert!(
            indices.len() <= array_len,
            "Patch indices must be shorter than the array length"
        );
        assert!(!indices.is_empty(), "Patch indices must not be empty");
        if let Some(max) = indices.statistics().get_as::<u64>(Stat::Max) {
            assert!(
                max < array_len as u64,
                "Patch indices {} are longer than the array length {}",
                max,
                array_len
            );
        }
        Self {
            array_len,
            indices,
            values,
        }
    }

    pub fn into_parts(self) -> (usize, Array, Array) {
        (self.array_len, self.indices, self.values)
    }

    pub fn array_len(&self) -> usize {
        self.array_len
    }

    pub fn num_patches(&self) -> usize {
        self.indices.len()
    }

    pub fn dtype(&self) -> &DType {
        self.values.dtype()
    }

    pub fn indices(&self) -> &Array {
        &self.indices
    }

    pub fn into_indices(self) -> Array {
        self.indices
    }

    pub fn values(&self) -> &Array {
        &self.values
    }

    pub fn into_values(self) -> Array {
        self.values
    }

    pub fn indices_ptype(&self) -> PType {
        PType::try_from(self.indices.dtype()).vortex_expect("primitive indices")
    }

    pub fn to_metadata(&self, len: usize, dtype: &DType) -> VortexResult<PatchesMetadata> {
        if self.indices.len() > len {
            vortex_bail!(
                "Patch indices {} are longer than the array length {}",
                self.indices.len(),
                len
            );
        }
        if self.values.dtype() != dtype {
            vortex_bail!(
                "Patch values dtype {} does not match array dtype {}",
                self.values.dtype(),
                dtype
            );
        }
        Ok(PatchesMetadata {
            len: self.indices.len(),
            indices_ptype: PType::try_from(self.indices.dtype()).vortex_expect("primitive indices"),
        })
    }

    /// Get the patched value at a given index if it exists.
    pub fn get_patched(&self, index: usize) -> VortexResult<Option<Scalar>> {
        if let Some(patch_idx) = self.search_index(index)?.to_found() {
            scalar_at(self.values(), patch_idx).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Return the insertion point of [index] in the [Self::indices].
    fn search_index(&self, index: usize) -> VortexResult<SearchResult> {
        search_sorted_usize(&self.indices, index, SearchSortedSide::Left)
    }

    /// Return the search_sorted result for the given target re-mapped into the original indices.
    pub fn search_sorted<T: Into<Scalar>>(
        &self,
        target: T,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        search_sorted(self.values(), target.into(), side).and_then(|sr| {
            let sidx = sr.to_offsets_index(self.indices().len());
            let index = usize::try_from(&scalar_at(self.indices(), sidx)?)?;
            Ok(match sr {
                // If we reached the end of patched values when searching then the result is one after the last patch index
                SearchResult::Found(i) => SearchResult::Found(if i == self.indices().len() {
                    index + 1
                } else {
                    index
                }),
                // If the result is NotFound we should return index that's one after the nearest not found index for the corresponding value
                SearchResult::NotFound(i) => {
                    SearchResult::NotFound(if i == 0 { index } else { index + 1 })
                }
            })
        })
    }

    /// Returns the minimum patch index
    pub fn min_index(&self) -> VortexResult<usize> {
        usize::try_from(&scalar_at(self.indices(), 0)?)
    }

    /// Returns the maximum patch index
    pub fn max_index(&self) -> VortexResult<usize> {
        usize::try_from(&scalar_at(self.indices(), self.indices().len() - 1)?)
    }

    /// Filter the patches by a mask, resulting in new patches for the filtered array.
    pub fn filter(&self, mask: &Mask) -> VortexResult<Option<Self>> {
        match mask.indices() {
            AllOr::All => Ok(Some(self.clone())),
            AllOr::None => Ok(None),
            AllOr::Some(mask_indices) => {
                let flat_indices = self.indices().clone().into_primitive()?;
                match_each_integer_ptype!(flat_indices.ptype(), |$I| {
                    filter_patches_with_mask(
                        flat_indices.as_slice::<$I>(),
                        self.values(),
                        mask_indices,
                    )
                })
            }
        }
    }

    /// Slice the patches by a range of the patched array.
    pub fn slice(&self, start: usize, stop: usize) -> VortexResult<Option<Self>> {
        let patch_start = self.search_index(start)?.to_index();
        let patch_stop = self.search_index(stop)?.to_index();

        if patch_start == patch_stop {
            return Ok(None);
        }

        // Slice out the values
        let values = slice(self.values(), patch_start, patch_stop)?;

        // Subtract the start value from the indices
        let indices = slice(self.indices(), patch_start, patch_stop)?;
        let indices = sub_scalar(&indices, Scalar::from(start).cast(indices.dtype())?)?;

        Ok(Some(Self::new(stop - start, indices, values)))
    }

    // https://docs.google.com/spreadsheets/d/1D9vBZ1QJ6mwcIvV5wIL0hjGgVchcEnAyhvitqWu2ugU
    const PREFER_MAP_WHEN_PATCHES_OVER_INDICES_LESS_THAN: f64 = 5.0;

    fn is_map_faster_than_search(&self, take_indices: &PrimitiveArray) -> bool {
        (self.num_patches() as f64 / take_indices.len() as f64)
            < Self::PREFER_MAP_WHEN_PATCHES_OVER_INDICES_LESS_THAN
    }

    /// Take the indices from the patches.
    pub fn take(&self, take_indices: &Array) -> VortexResult<Option<Self>> {
        if take_indices.is_empty() {
            return Ok(None);
        }
        let take_indices = take_indices.clone().into_primitive()?;
        if self.is_map_faster_than_search(&take_indices) {
            self.take_map(take_indices)
        } else {
            self.take_search(take_indices)
        }
    }

    pub fn take_search(&self, take_indices: PrimitiveArray) -> VortexResult<Option<Self>> {
        let new_length = take_indices.len();

        let take_indices = match_each_integer_ptype!(take_indices.ptype(), |$P| {
            take_indices
                .as_slice::<$P>()
                .iter()
                .copied()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()?
        });

        let (values_indices, new_indices): (BufferMut<u64>, BufferMut<u64>) =
            search_sorted_usize_many(self.indices(), &take_indices, SearchSortedSide::Left)?
                .iter()
                .enumerate()
                .filter_map(|(idx_in_take, search_result)| {
                    search_result
                        .to_found()
                        .map(|patch_idx| (patch_idx as u64, idx_in_take as u64))
                })
                .unzip();

        if new_indices.is_empty() {
            return Ok(None);
        }

        let new_indices = new_indices.into_array();
        let values_indices = values_indices.into_array();
        let new_values = take(self.values(), values_indices)?;

        Ok(Some(Self::new(new_length, new_indices, new_values)))
    }

    pub fn take_map(&self, take_indices: PrimitiveArray) -> VortexResult<Option<Self>> {
        let indices = self.indices.clone().into_primitive()?;
        match_each_integer_ptype!(self.indices_ptype(), |$INDICES| {
            let indices = indices
                .as_slice::<$INDICES>();
            match_each_integer_ptype!(take_indices.ptype(), |$TAKE_INDICES| {
                let take_indices = take_indices
                    .as_slice::<$TAKE_INDICES>();

                let new_length = take_indices.len();
                let sparse_index_to_value_index: HashMap<$INDICES, usize> = indices
                    .iter()
                    .enumerate()
                    .map(|(value_index, sparse_index)| (*sparse_index, value_index))
                    .collect();
                let min_index = self.min_index()?;
                let max_index = self.max_index()?;
                let (new_sparse_indices, value_indices): (BufferMut<u64>, BufferMut<u64>) =
                    take_indices
                    .iter()
                    .map(|x| usize::try_from(*x))
                    .process_results(|iter| {
                        iter
                           .enumerate()
                           .filter(|(_, ti)| *ti >= min_index && *ti <= max_index)
                           .filter_map(|(new_sparse_index, take_sparse_index)| {
                               sparse_index_to_value_index
                                   .get(&<$INDICES>::try_from(take_sparse_index).ok().vortex_expect(
                                       "take_sparse_index is between min and max index",
                                   ))
                                   .map(|value_index| (new_sparse_index as u64, *value_index as u64))
                           })
                           .unzip()
                    })?;

                if new_sparse_indices.is_empty() {
                    return Ok(None);
                }

                Ok(Some(Patches::new(
                    new_length,
                    new_sparse_indices.into_array(),
                    take(self.values(), value_indices.into_array())?,
                )))
            })
        })
    }

    pub fn map_values<F>(self, f: F) -> VortexResult<Self>
    where
        F: FnOnce(Array) -> VortexResult<Array>,
    {
        let values = f(self.values)?;
        if self.indices.len() != values.len() {
            vortex_bail!(
                "map_values must preserve length: expected {} received {}",
                self.indices.len(),
                values.len()
            )
        }
        Ok(Self::new(self.array_len, self.indices, values))
    }

    pub fn map_values_opt<F>(self, f: F) -> VortexResult<Option<Self>>
    where
        F: FnOnce(Array) -> Option<Array>,
    {
        let Some(values) = f(self.values) else {
            return Ok(None);
        };
        if self.indices.len() == values.len() {
            vortex_bail!(
                "map_values must preserve length: expected {} received {}",
                self.indices.len(),
                values.len()
            )
        }
        Ok(Some(Self::new(self.array_len, self.indices, values)))
    }
}

/// Filter patches with the provided mask (in flattened space).
///
/// The filter mask may contain indices that are non-patched. The return value of this function
/// is a new set of `Patches` with the indices relative to the provided `mask` rank, and the
/// patch values.
fn filter_patches_with_mask<T: ToPrimitive + Copy + Ord>(
    patch_indices: &[T],
    patch_values: &Array,
    mask_indices: &[usize],
) -> VortexResult<Option<Patches>> {
    let true_count = mask_indices.len();
    let mut new_patch_indices = BufferMut::<u64>::with_capacity(true_count);
    let mut new_mask_indices = Vec::with_capacity(true_count);

    // Attempt to move the window by `STRIDE` elements on each iteration. This assumes that
    // the patches are relatively sparse compared to the overall mask, and so many indices in the
    // mask will end up being skipped.
    const STRIDE: usize = 4;

    let mut mask_idx = 0usize;
    let mut true_idx = 0usize;

    while mask_idx < patch_indices.len() && true_idx < true_count {
        // NOTE: we are searching for overlaps between sorted, unaligned indices in `patch_indices`
        //  and `mask_indices`. We assume that Patches are sparse relative to the global space of
        //  the mask (which covers both patch and non-patch values of the parent array), and so to
        //  quickly jump through regions with no overlap, we attempt to move our pointers by STRIDE
        //  elements on each iteration. If we cannot rule out overlap due to min/max values, we
        //  fallback to performing a two-way iterator merge.
        if (mask_idx + STRIDE) < patch_indices.len() && (true_idx + STRIDE) < mask_indices.len() {
            // Load a vector of each into our registers.
            let left_min = patch_indices[mask_idx].to_usize().vortex_expect("left_min");
            let left_max = patch_indices[mask_idx + STRIDE]
                .to_usize()
                .vortex_expect("left_max");
            let right_min = mask_indices[true_idx];
            let right_max = mask_indices[true_idx + STRIDE];

            if left_min > right_max {
                // Advance right side
                true_idx += STRIDE;
                continue;
            } else if right_min > left_max {
                mask_idx += STRIDE;
                continue;
            } else {
                // Fallthrough to direct comparison path.
            }
        }

        // Two-way sorted iterator merge:

        let left = patch_indices[mask_idx].to_usize().vortex_expect("left");
        let right = mask_indices[true_idx];

        match left.cmp(&right) {
            Ordering::Less => {
                mask_idx += 1;
            }
            Ordering::Greater => {
                true_idx += 1;
            }
            Ordering::Equal => {
                // Save the mask index as well as the positional index.
                new_mask_indices.push(mask_idx);
                new_patch_indices.push(true_idx as u64);

                mask_idx += 1;
                true_idx += 1;
            }
        }
    }

    if new_mask_indices.is_empty() {
        return Ok(None);
    }

    let new_patch_indices = new_patch_indices.into_array();
    let new_patch_values = filter(
        patch_values,
        &Mask::from_indices(patch_values.len(), new_mask_indices),
    )?;

    Ok(Some(Patches::new(
        true_count,
        new_patch_indices,
        new_patch_values,
    )))
}

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::array::PrimitiveArray;
    use crate::compute::{SearchResult, SearchSortedSide};
    use crate::patches::Patches;
    use crate::validity::Validity;
    use crate::{IntoArray, IntoArrayVariant};

    #[test]
    fn test_filter() {
        let patches = Patches::new(
            100,
            buffer![10u32, 11, 20].into_array(),
            buffer![100, 110, 200].into_array(),
        );

        let filtered = patches
            .filter(&Mask::from_indices(100, vec![10, 20, 30]))
            .unwrap()
            .unwrap();

        let indices = filtered.indices().clone().into_primitive().unwrap();
        let values = filtered.values().clone().into_primitive().unwrap();
        assert_eq!(indices.as_slice::<u64>(), &[0, 1]);
        assert_eq!(values.as_slice::<i32>(), &[100, 200]);
    }

    #[fixture]
    fn patches() -> Patches {
        Patches::new(
            20,
            buffer![2u64, 9, 15].into_array(),
            PrimitiveArray::new(buffer![33_i32, 44, 55], Validity::AllValid).into_array(),
        )
    }

    #[rstest]
    fn search_larger_than(patches: Patches) {
        let res = patches.search_sorted(66, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_less_than(patches: Patches) {
        let res = patches.search_sorted(22, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::NotFound(2));
    }

    #[rstest]
    fn search_found(patches: Patches) {
        let res = patches.search_sorted(44, SearchSortedSide::Left).unwrap();
        assert_eq!(res, SearchResult::Found(9));
    }

    #[rstest]
    fn search_not_found_right(patches: Patches) {
        let res = patches.search_sorted(56, SearchSortedSide::Right).unwrap();
        assert_eq!(res, SearchResult::NotFound(16));
    }

    #[rstest]
    fn search_sliced(patches: Patches) {
        let sliced = patches.slice(7, 20).unwrap().unwrap();
        assert_eq!(
            sliced.search_sorted(22, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(2)
        );
    }

    #[test]
    fn search_right() {
        let patches = Patches::new(
            2,
            buffer![0u64].into_array(),
            PrimitiveArray::new(buffer![0u8], Validity::AllValid).into_array(),
        );

        assert_eq!(
            patches.search_sorted(0, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(1)
        );
        assert_eq!(
            patches.search_sorted(1, SearchSortedSide::Right).unwrap(),
            SearchResult::NotFound(1)
        );
    }
}
