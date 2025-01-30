//! Array validity and nullability behavior, used by arrays and compute functions.

use std::fmt::{Debug, Display};
use std::ops::BitAnd;

use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, NullBuffer};
use serde::{Deserialize, Serialize};
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect as _, VortexResult};
use vortex_mask::{Mask, MaskValues};

use crate::array::{BoolArray, ConstantArray};
use crate::compute::{filter, scalar_at, slice, take};
use crate::patches::Patches;
use crate::{Array, IntoArray, IntoArrayVariant};

impl Array {
    /// Return whether the element at the given index is valid (true) or null (false).
    pub fn is_valid(&self, index: usize) -> VortexResult<bool> {
        self.vtable().is_valid(self, index)
    }

    /// Return the number of null elements in the array.
    pub fn null_count(&self) -> VortexResult<usize> {
        self.vtable().null_count(self)
    }

    /// Return the logical validity of the array if nullable, and None if non-nullable.
    pub fn logical_validity(&self) -> VortexResult<Mask> {
        self.vtable().logical_validity(self)
    }
}

#[derive(
    Copy,
    Clone,
    Debug,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Portable,
    rkyv::Serialize,
    rkyv::Deserialize,
    rkyv::bytecheck::CheckBytes,
)]
#[rkyv(as = ValidityMetadata)]
#[bytecheck(crate = rkyv::bytecheck)]
#[repr(u8)]
pub enum ValidityMetadata {
    NonNullable,
    AllValid,
    AllInvalid,
    Array,
}

impl Display for ValidityMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ValidityMetadata {
    pub fn to_validity<F>(&self, array_fn: F) -> Validity
    where
        F: FnOnce() -> Array,
    {
        match self {
            Self::NonNullable => Validity::NonNullable,
            Self::AllValid => Validity::AllValid,
            Self::AllInvalid => Validity::AllInvalid,
            Self::Array => Validity::Array(array_fn()),
        }
    }
}

/// Validity information for an array
#[derive(Clone, Debug)]
pub enum Validity {
    /// Items *can't* be null
    NonNullable,
    /// All items are valid
    AllValid,
    /// All items are null
    AllInvalid,
    /// Specified items are null
    Array(Array),
}

impl Validity {
    /// The [`DType`] of the underlying validity array (if it exists).
    pub const DTYPE: DType = DType::Bool(Nullability::NonNullable);

    pub fn to_metadata(&self, length: usize) -> VortexResult<ValidityMetadata> {
        match self {
            Self::NonNullable => Ok(ValidityMetadata::NonNullable),
            Self::AllValid => Ok(ValidityMetadata::AllValid),
            Self::AllInvalid => Ok(ValidityMetadata::AllInvalid),
            Self::Array(a) => {
                // We force the caller to validate the length here.
                let validity_len = a.len();
                if validity_len != length {
                    vortex_bail!(
                        "Validity array length {} doesn't match array length {}",
                        validity_len,
                        length
                    )
                }
                Ok(ValidityMetadata::Array)
            }
        }
    }

    pub fn null_count(&self, length: usize) -> VortexResult<usize> {
        match self {
            Self::NonNullable | Self::AllValid => Ok(0),
            Self::AllInvalid => Ok(length),
            Self::Array(a) => {
                let validity_len = a.len();
                if validity_len != length {
                    vortex_bail!(
                        "Validity array length {} doesn't match array length {}",
                        validity_len,
                        length
                    )
                }
                let true_count = a.statistics().compute_true_count().ok_or_else(|| {
                    vortex_err!("Failed to compute true count from validity array")
                })?;
                Ok(length - true_count)
            }
        }
    }

    /// If Validity is [`Validity::Array`], returns the array, otherwise returns `None`.
    pub fn into_array(self) -> Option<Array> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// If Validity is [`Validity::Array`], returns a reference to the array array, otherwise returns `None`.
    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn nullability(&self) -> Nullability {
        match self {
            Self::NonNullable => Nullability::NonNullable,
            _ => Nullability::Nullable,
        }
    }

    /// Returns whether the `index` item is valid.
    #[inline]
    pub fn is_valid(&self, index: usize) -> VortexResult<bool> {
        Ok(match self {
            Self::NonNullable | Self::AllValid => true,
            Self::AllInvalid => false,
            Self::Array(a) => {
                let scalar = scalar_at(a, index)?;
                scalar
                    .as_bool()
                    .value()
                    .vortex_expect("Validity must be non-nullable")
            }
        })
    }

    #[inline]
    pub fn is_null(&self, index: usize) -> VortexResult<bool> {
        Ok(!self.is_valid(index)?)
    }

    pub fn slice(&self, start: usize, stop: usize) -> VortexResult<Self> {
        match self {
            Self::Array(a) => Ok(Self::Array(slice(a, start, stop)?)),
            _ => Ok(self.clone()),
        }
    }

    pub fn take(&self, indices: &Array) -> VortexResult<Self> {
        match self {
            Self::NonNullable => Ok(Self::NonNullable),
            Self::AllValid => Ok(Self::AllValid),
            Self::AllInvalid => Ok(Self::AllInvalid),
            Self::Array(a) => Ok(Self::Array(take(a, indices)?)),
        }
    }

    /// Take the validity buffer at the provided indices.
    ///
    /// # Safety
    ///
    /// It is assumed the caller has checked that all indices are <= the length of this validity
    /// buffer.
    ///
    /// Failure to do so may result in UB.
    pub unsafe fn take_unchecked(&self, indices: &Array) -> VortexResult<Self> {
        match self {
            Self::NonNullable => Ok(Self::NonNullable),
            Self::AllValid => Ok(Self::AllValid),
            Self::AllInvalid => Ok(Self::AllInvalid),
            Self::Array(a) => {
                let taken = if let Some(take_fn) = a.vtable().take_fn() {
                    unsafe { take_fn.take_unchecked(a, indices) }
                } else {
                    take(a, indices)
                };

                taken.map(Self::Array)
            }
        }
    }

    pub fn filter(&self, mask: &Mask) -> VortexResult<Self> {
        // NOTE(ngates): we take the mask as a reference to avoid the caller cloning unnecessarily
        //  if we happen to be NonNullable, AllValid, or AllInvalid.
        match self {
            v @ (Validity::NonNullable | Validity::AllValid | Validity::AllInvalid) => {
                Ok(v.clone())
            }
            Validity::Array(arr) => Ok(Validity::Array(filter(arr, mask)?)),
        }
    }

    pub fn to_logical(&self, length: usize) -> VortexResult<Mask> {
        Ok(match self {
            Self::NonNullable | Self::AllValid => Mask::AllTrue(length),
            Self::AllInvalid => Mask::AllFalse(length),
            Self::Array(a) => Mask::try_from(a.clone())?,
        })
    }

    /// Logically & two Validity values of the same length
    pub fn and(self, rhs: Validity) -> VortexResult<Validity> {
        let validity = match (self, rhs) {
            // Should be pretty clear
            (Validity::NonNullable, Validity::NonNullable) => Validity::NonNullable,
            // Any `AllInvalid` makes the output all invalid values
            (Validity::AllInvalid, _) | (_, Validity::AllInvalid) => Validity::AllInvalid,
            // All truthy values on one side, which makes no effect on an `Array` variant
            (Validity::Array(a), Validity::AllValid)
            | (Validity::Array(a), Validity::NonNullable)
            | (Validity::NonNullable, Validity::Array(a))
            | (Validity::AllValid, Validity::Array(a)) => Validity::Array(a),
            // Both sides are all valid
            (Validity::NonNullable, Validity::AllValid)
            | (Validity::AllValid, Validity::NonNullable)
            | (Validity::AllValid, Validity::AllValid) => Validity::AllValid,
            // Here we actually have to do some work
            (Validity::Array(lhs), Validity::Array(rhs)) => {
                let lhs = BoolArray::try_from(lhs)?;
                let rhs = BoolArray::try_from(rhs)?;

                let lhs = lhs.boolean_buffer();
                let rhs = rhs.boolean_buffer();

                Validity::from(lhs.bitand(&rhs))
            }
        };

        Ok(validity)
    }

    pub fn patch(self, len: usize, indices: &Array, patches: Validity) -> VortexResult<Self> {
        match (&self, &patches) {
            (Validity::NonNullable, Validity::NonNullable) => return Ok(Validity::NonNullable),
            (Validity::NonNullable, _) => {
                vortex_bail!("Can't patch a non-nullable validity with nullable validity")
            }
            (_, Validity::NonNullable) => {
                vortex_bail!("Can't patch a nullable validity with non-nullable validity")
            }
            (Validity::AllValid, Validity::AllValid) => return Ok(Validity::AllValid),
            (Validity::AllInvalid, Validity::AllInvalid) => return Ok(Validity::AllInvalid),
            _ => {}
        };

        let own_nullability = if self == Validity::NonNullable {
            Nullability::NonNullable
        } else {
            Nullability::Nullable
        };

        let source = match self {
            Validity::NonNullable => BoolArray::from(BooleanBuffer::new_set(len)),
            Validity::AllValid => BoolArray::from(BooleanBuffer::new_set(len)),
            Validity::AllInvalid => BoolArray::from(BooleanBuffer::new_unset(len)),
            Validity::Array(a) => a.into_bool()?,
        };

        let patch_values = match patches {
            Validity::NonNullable => BoolArray::from(BooleanBuffer::new_set(indices.len())),
            Validity::AllValid => BoolArray::from(BooleanBuffer::new_set(indices.len())),
            Validity::AllInvalid => BoolArray::from(BooleanBuffer::new_unset(indices.len())),
            Validity::Array(a) => a.into_bool()?,
        };

        let patches = Patches::new(len, indices.clone(), patch_values.into_array());

        Ok(Self::from_array(
            source.patch(patches)?.into_array(),
            own_nullability,
        ))
    }

    /// Convert into a nullable variant
    pub fn into_nullable(self) -> Validity {
        match self {
            Self::NonNullable => Self::AllValid,
            _ => self,
        }
    }

    /// Create Validity from boolean array with given nullability of the array.
    ///
    /// Note: You want to pass the nullability of parent array and not the nullability of the validity array itself
    ///     as that is always nonnullable
    pub fn from_array(value: Array, nullability: Nullability) -> Self {
        if !matches!(value.dtype(), DType::Bool(Nullability::NonNullable)) {
            vortex_panic!("Expected a non-nullable boolean array")
        }
        match nullability {
            Nullability::NonNullable => Self::NonNullable,
            Nullability::Nullable => Self::Array(value),
        }
    }
}

impl PartialEq for Validity {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NonNullable, Self::NonNullable) => true,
            (Self::AllValid, Self::AllValid) => true,
            (Self::AllInvalid, Self::AllInvalid) => true,
            (Self::Array(a), Self::Array(b)) => {
                let a_buffer = a
                    .clone()
                    .into_bool()
                    .vortex_expect("Failed to get Validity Array as BoolArray")
                    .boolean_buffer();
                let b_buffer = b
                    .clone()
                    .into_bool()
                    .vortex_expect("Failed to get Validity Array as BoolArray")
                    .boolean_buffer();
                a_buffer == b_buffer
            }
            _ => false,
        }
    }
}

impl From<BooleanBuffer> for Validity {
    fn from(value: BooleanBuffer) -> Self {
        if value.count_set_bits() == value.len() {
            Self::AllValid
        } else if value.count_set_bits() == 0 {
            Self::AllInvalid
        } else {
            Self::Array(BoolArray::from(value).into_array())
        }
    }
}

impl From<NullBuffer> for Validity {
    fn from(value: NullBuffer) -> Self {
        value.into_inner().into()
    }
}

impl FromIterator<Mask> for Validity {
    fn from_iter<T: IntoIterator<Item = Mask>>(iter: T) -> Self {
        let validities: Vec<Mask> = iter.into_iter().collect();

        // If they're all valid, then return a single validity.
        if validities.iter().all(|v| v.all_true()) {
            return Self::AllValid;
        }
        // If they're all invalid, then return a single invalidity.
        if validities.iter().all(|v| v.all_false()) {
            return Self::AllInvalid;
        }

        // Else, construct the boolean buffer
        let mut buffer = BooleanBufferBuilder::new(validities.iter().map(|v| v.len()).sum());
        for validity in validities {
            match validity {
                Mask::AllTrue(count) => buffer.append_n(count, true),
                Mask::AllFalse(count) => buffer.append_n(count, false),
                Mask::Values(values) => {
                    buffer.append_buffer(values.boolean_buffer());
                }
            };
        }
        let bool_array = BoolArray::from(buffer.finish());
        Self::Array(bool_array.into_array())
    }
}

impl FromIterator<bool> for Validity {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Validity::from(BooleanBuffer::from_iter(iter))
    }
}

impl From<Nullability> for Validity {
    fn from(value: Nullability) -> Self {
        match value {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        }
    }
}

impl Validity {
    pub fn from_mask(mask: Mask, nullability: Nullability) -> Self {
        assert!(
            nullability == Nullability::Nullable || matches!(mask, Mask::AllTrue(_)),
            "NonNullable validity must be AllValid",
        );
        match mask {
            Mask::AllTrue(_) => match nullability {
                Nullability::NonNullable => Validity::NonNullable,
                Nullability::Nullable => Validity::AllValid,
            },
            Mask::AllFalse(_) => Validity::AllInvalid,
            Mask::Values(values) => Validity::Array(values.into_array()),
        }
    }
}

impl IntoArray for Mask {
    fn into_array(self) -> Array {
        match self {
            Self::AllTrue(len) => ConstantArray::new(true, len).into_array(),
            Self::AllFalse(len) => ConstantArray::new(false, len).into_array(),
            Self::Values(a) => a.into_array(),
        }
    }
}

impl IntoArray for &MaskValues {
    fn into_array(self) -> Array {
        BoolArray::new(self.boolean_buffer().clone(), Nullability::NonNullable).into_array()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::{buffer, Buffer};
    use vortex_dtype::Nullability;
    use vortex_mask::Mask;

    use crate::array::{BoolArray, PrimitiveArray};
    use crate::validity::Validity;
    use crate::IntoArray;

    #[rstest]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::AllValid, Validity::AllValid)]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::AllInvalid, Validity::Array(BoolArray::from_iter([true, true, false, true, false]).into_array())
    )]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::Array(BoolArray::from_iter([true, false]).into_array()), Validity::Array(BoolArray::from_iter([true, true, true, true, false]).into_array())
    )]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::AllValid, Validity::Array(BoolArray::from_iter([false, false, true, false, true]).into_array())
    )]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::AllInvalid, Validity::AllInvalid)]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::Array(BoolArray::from_iter([true, false]).into_array()), Validity::Array(BoolArray::from_iter([false, false, true, false, false]).into_array())
    )]
    #[case(Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()), 5, &[2, 4], Validity::AllValid, Validity::Array(BoolArray::from_iter([false, true, true, true, true]).into_array())
    )]
    #[case(Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()), 5, &[2, 4], Validity::AllInvalid, Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array())
    )]
    #[case(Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()), 5, &[2, 4], Validity::Array(BoolArray::from_iter([true, false]).into_array()), Validity::Array(BoolArray::from_iter([false, true, true, true, false]).into_array())
    )]
    fn patch_validity(
        #[case] validity: Validity,
        #[case] len: usize,
        #[case] positions: &[u64],
        #[case] patches: Validity,
        #[case] expected: Validity,
    ) {
        let indices =
            PrimitiveArray::new(Buffer::copy_from(positions), Validity::NonNullable).into_array();
        assert_eq!(validity.patch(len, &indices, patches).unwrap(), expected);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_patch() {
        Validity::NonNullable
            .patch(2, &buffer![4].into_array(), Validity::AllInvalid)
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn into_validity_nullable() {
        Validity::from_mask(Mask::AllFalse(10), Nullability::NonNullable);
    }

    #[test]
    #[should_panic]
    fn into_validity_nullable_array() {
        Validity::from_mask(Mask::from_iter(vec![true, false]), Nullability::NonNullable);
    }
}
