use std::fmt::{Debug, Display};

use arrow_buffer::BooleanBuffer;
use serde::{Deserialize, Serialize};
use vortex_array::array::BoolArray;
use vortex_array::encoding::ids;
use vortex_array::stats::StatsSet;
use vortex_array::validate::ValidateVTable;
use vortex_array::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use vortex_array::variants::{BoolArrayTrait, VariantsVTable};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{impl_encoding, ArrayLen, Canonical, IntoCanonical};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult};

impl_encoding!("vortex.bytebool", ids::BYTE_BOOL, ByteBool);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ByteBoolMetadata {
    validity: ValidityMetadata,
}

impl Display for ByteBoolMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ByteBoolArray {
    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(0, &Validity::DTYPE, self.len())
                .vortex_expect("ByteBoolArray: accessing validity child")
        })
    }

    pub fn try_new(buffer: ByteBuffer, validity: Validity) -> VortexResult<Self> {
        let length = buffer.len();
        Self::try_from_parts(
            DType::Bool(validity.nullability()),
            length,
            ByteBoolMetadata {
                validity: validity.to_metadata(length)?,
            },
            Some([buffer.into_byte_buffer()].into()),
            validity.into_array().map(|v| [v].into()),
            StatsSet::default(),
        )
    }

    // TODO(ngates): deprecate construction from vec
    pub fn try_from_vec<V: Into<Validity>>(data: Vec<bool>, validity: V) -> VortexResult<Self> {
        let validity = validity.into();
        // SAFETY: we are transmuting a Vec<bool> into a Vec<u8>
        let data: Vec<u8> = unsafe { std::mem::transmute(data) };
        Self::try_new(ByteBuffer::from(data), validity)
    }

    pub fn buffer(&self) -> &ByteBuffer {
        self.as_ref()
            .byte_buffer(0)
            .vortex_expect("ByteBoolArray is missing the underlying buffer")
    }

    pub fn as_slice(&self) -> &[bool] {
        // Safety: The internal buffer contains byte-sized bools
        unsafe { std::mem::transmute(self.buffer().as_slice()) }
    }
}

impl ValidateVTable<ByteBoolArray> for ByteBoolEncoding {}

impl VariantsVTable<ByteBoolArray> for ByteBoolEncoding {
    fn as_bool_array<'a>(&self, array: &'a ByteBoolArray) -> Option<&'a dyn BoolArrayTrait> {
        Some(array)
    }
}

impl BoolArrayTrait for ByteBoolArray {}

impl From<Vec<bool>> for ByteBoolArray {
    fn from(value: Vec<bool>) -> Self {
        Self::try_from_vec(value, Validity::AllValid)
            .vortex_expect("Failed to create ByteBoolArray from Vec<bool>")
    }
}

impl From<Vec<Option<bool>>> for ByteBoolArray {
    fn from(value: Vec<Option<bool>>) -> Self {
        let validity = Validity::from_iter(value.iter().map(|v| v.is_some()));

        // This doesn't reallocate, and the compiler even vectorizes it
        let data = value.into_iter().map(Option::unwrap_or_default).collect();

        Self::try_from_vec(data, validity)
            .vortex_expect("Failed to create ByteBoolArray from nullable bools")
    }
}

impl IntoCanonical for ByteBoolArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        let boolean_buffer = BooleanBuffer::from(self.as_slice());
        let validity = self.validity();

        Ok(Canonical::Bool(BoolArray::try_new(
            boolean_buffer,
            validity,
        )?))
    }
}

impl ValidityVTable<ByteBoolArray> for ByteBoolEncoding {
    fn is_valid(&self, array: &ByteBoolArray, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &ByteBoolArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<ByteBoolArray> for ByteBoolEncoding {
    fn accept(&self, array: &ByteBoolArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(array.buffer())?;
        visitor.visit_validity(&array.validity())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::test_harness::check_metadata;
    use vortex_array::validity::ArrayValidity;

    use super::*;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_bytebool_metadata() {
        check_metadata(
            "bytebool.metadata",
            ByteBoolMetadata {
                validity: ValidityMetadata::AllValid,
            },
        );
    }

    #[test]
    fn test_validity_construction() {
        let v = vec![true, false];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(arr.is_valid(idx));
        }

        let v = vec![Some(true), None, Some(false)];
        let arr = ByteBoolArray::from(v);
        assert!(arr.is_valid(0));
        assert!(!arr.is_valid(1));
        assert!(arr.is_valid(2));
        assert_eq!(arr.len(), 3);

        let v: Vec<Option<bool>> = vec![None, None];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(!arr.is_valid(idx));
        }
        assert_eq!(arr.len(), 2);
    }
}
