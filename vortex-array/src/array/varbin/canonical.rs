use arrow_array::ArrayRef;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::varbin::arrow::varbin_to_arrow;
use crate::array::varbin::VarBinArray;
use crate::array::VarBinViewArray;
use crate::arrow::FromArrowArray;
use crate::{ArrayDType, ArrayData, Canonical, IntoCanonical};

impl IntoCanonical for VarBinArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        let nullable = self.dtype().is_nullable();
        let array_ref = varbin_to_arrow(&self)?;
        let array = match self.dtype() {
            DType::Utf8(_) => arrow_cast::cast(array_ref.as_ref(), &DataType::Utf8View)?,
            DType::Binary(_) => arrow_cast::cast(array_ref.as_ref(), &DataType::BinaryView)?,

            _ => unreachable!("VarBinArray must have Utf8 or Binary dtype"),
        };

        VarBinViewArray::try_from(ArrayData::from_arrow(array, nullable)).map(Canonical::VarBinView)
    }

    fn into_arrow(self) -> VortexResult<ArrayRef> {
        // Specialized implementation of `into_arrow` for VarBin since it has a direct
        // Arrow representation.
        varbin_to_arrow(&self)
    }

    fn into_arrow_with_data_type(self, data_type: &DataType) -> VortexResult<ArrayRef> {
        let array_ref = self.into_arrow()?;

        Ok(if array_ref.data_type() != data_type {
            arrow_cast::cast(array_ref.as_ref(), data_type)?
        } else {
            array_ref
        })
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::array::varbin::builder::VarBinBuilder;
    use crate::validity::ArrayValidity;
    use crate::{ArrayDType, IntoCanonical};

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn test_canonical_varbin(#[case] dtype: DType) {
        let mut varbin = VarBinBuilder::<i32>::with_capacity(10);
        varbin.push_null();
        varbin.push_null();
        // inlined value
        varbin.push_value("123456789012".as_bytes());
        // non-inlinable value
        varbin.push_value("1234567890123".as_bytes());
        let varbin = varbin.finish(dtype.clone());

        let canonical = varbin.into_canonical().unwrap().into_varbinview().unwrap();
        assert_eq!(canonical.dtype(), &dtype);

        assert!(!canonical.is_valid(0).unwrap());
        assert!(!canonical.is_valid(1).unwrap());

        // First value is inlined (12 bytes)
        assert!(canonical.views()[2].is_inlined());
        assert_eq!(canonical.bytes_at(2).as_slice(), "123456789012".as_bytes());

        // Second value is not inlined (13 bytes)
        assert!(!canonical.views()[3].is_inlined());
        assert_eq!(canonical.bytes_at(3).as_slice(), "1234567890123".as_bytes());
    }
}
