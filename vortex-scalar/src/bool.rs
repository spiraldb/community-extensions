use std::cmp::Ordering;

use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect as _, VortexResult};

use crate::{InnerScalarValue, Scalar, ScalarValue};

#[derive(Debug, Hash)]
pub struct BoolScalar<'a> {
    dtype: &'a DType,
    value: Option<bool>,
}

impl PartialEq for BoolScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for BoolScalar<'_> {}

impl PartialOrd for BoolScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.dtype != other.dtype {
            return None;
        }
        self.value.partial_cmp(&other.value)
    }
}

impl<'a> BoolScalar<'a> {
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    pub fn value(&self) -> Option<bool> {
        self.value
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if !matches!(dtype, DType::Bool(..)) {
            vortex_bail!("Can't cast bool to {}", dtype)
        }
        Ok(Scalar::bool(
            self.value.vortex_expect("nullness handled in Scalar::cast"),
            dtype.nullability(),
        ))
    }

    pub fn invert(self) -> BoolScalar<'a> {
        BoolScalar {
            dtype: self.dtype,
            value: self.value.map(|v| !v),
        }
    }

    pub fn into_scalar(self) -> Scalar {
        Scalar {
            dtype: self.dtype.clone(),
            value: self
                .value
                .map(|x| ScalarValue(InnerScalarValue::Bool(x)))
                .unwrap_or_else(|| ScalarValue(InnerScalarValue::Null)),
        }
    }
}

impl Scalar {
    pub fn bool(value: bool, nullability: Nullability) -> Self {
        Self {
            dtype: DType::Bool(nullability),
            value: ScalarValue(InnerScalarValue::Bool(value)),
        }
    }
}

impl<'a> TryFrom<&'a Scalar> for BoolScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        if !matches!(value.dtype(), DType::Bool(_)) {
            vortex_bail!("Expected bool scalar, found {}", value.dtype())
        }
        Ok(Self {
            dtype: value.dtype(),
            value: value.value.as_bool()?,
        })
    }
}

impl TryFrom<&Scalar> for bool {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> VortexResult<Self> {
        <Option<bool>>::try_from(value)?
            .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
    }
}

impl TryFrom<&Scalar> for Option<bool> {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> VortexResult<Self> {
        Ok(BoolScalar::try_from(value)?.value())
    }
}

impl TryFrom<Scalar> for bool {
    type Error = VortexError;

    fn try_from(value: Scalar) -> VortexResult<Self> {
        Self::try_from(&value)
    }
}

impl TryFrom<Scalar> for Option<bool> {
    type Error = VortexError;

    fn try_from(value: Scalar) -> VortexResult<Self> {
        Self::try_from(&value)
    }
}

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        Self {
            dtype: DType::Bool(NonNullable),
            value: value.into(),
        }
    }
}

impl From<bool> for ScalarValue {
    fn from(value: bool) -> Self {
        ScalarValue(InnerScalarValue::Bool(value))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn into_from() {
        let scalar: Scalar = false.into();
        assert!(!bool::try_from(&scalar).unwrap());
    }
}
