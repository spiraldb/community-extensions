use arrow_buffer::NullBufferBuilder;
use num_traits::{AsPrimitive, PrimInt};
use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType};
use vortex_error::{VortexExpect as _, vortex_panic};

use crate::Array;
use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::varbin::VarBinArray;
use crate::validity::Validity;

pub struct VarBinBuilder<O: NativePType> {
    offsets: BufferMut<O>,
    data: BufferMut<u8>,
    validity: NullBufferBuilder,
}

impl<O: NativePType + PrimInt> Default for VarBinBuilder<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: NativePType + PrimInt> VarBinBuilder<O> {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(len: usize) -> Self {
        let mut offsets = BufferMut::with_capacity(len + 1);
        offsets.push(O::zero());
        Self {
            offsets,
            data: BufferMut::empty(),
            validity: NullBufferBuilder::new(len),
        }
    }

    #[inline]
    pub fn append(&mut self, value: Option<&[u8]>) {
        match value {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }

    #[inline]
    pub fn append_value(&mut self, value: impl AsRef<[u8]>) {
        let slice = value.as_ref();
        self.offsets
            .push(O::from(self.data.len() + slice.len()).unwrap_or_else(|| {
                vortex_panic!(
                    "Failed to convert sum of {} and {} to offset of type {}",
                    self.data.len(),
                    slice.len(),
                    std::any::type_name::<O>()
                )
            }));
        self.data.extend_from_slice(slice);
        self.validity.append_non_null();
    }

    #[inline]
    pub fn append_null(&mut self) {
        self.offsets.push(self.offsets[self.offsets.len() - 1]);
        self.validity.append_null();
    }

    #[inline]
    pub fn append_n_nulls(&mut self, n: usize) {
        self.offsets.push_n(self.offsets[self.offsets.len() - 1], n);
        self.validity.append_n_nulls(n);
    }

    #[inline]
    pub fn append_values(&mut self, values: &[u8], end_offsets: impl Iterator<Item = O>, num: usize)
    where
        O: 'static,
        usize: AsPrimitive<O>,
    {
        self.offsets
            .extend(end_offsets.map(|offset| offset + self.data.len().as_()));
        self.data.extend_from_slice(values);
        self.validity.append_n_non_nulls(num);
    }

    pub fn finish(mut self, dtype: DType) -> VarBinArray {
        let offsets = PrimitiveArray::new(self.offsets.freeze(), Validity::NonNullable);
        let nulls = self.validity.finish();

        let validity = if dtype.is_nullable() {
            nulls.map(Validity::from).unwrap_or(Validity::AllValid)
        } else {
            assert!(nulls.is_none(), "dtype and validity mismatch");
            Validity::NonNullable
        };

        VarBinArray::try_new(offsets.into_array(), self.data.freeze(), dtype, validity)
            .vortex_expect("Unexpected error while building VarBinArray")
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::array::Array;
    use crate::arrays::varbin::builder::VarBinBuilder;
    use crate::compute::scalar_at;

    #[test]
    fn test_builder() {
        let mut builder = VarBinBuilder::<i32>::with_capacity(0);
        builder.append(Some(b"hello"));
        builder.append(None);
        builder.append(Some(b"world"));
        let array = builder.finish(DType::Utf8(Nullable));

        assert_eq!(array.len(), 3);
        assert_eq!(array.dtype().nullability(), Nullable);
        assert_eq!(
            scalar_at(&array, 0).unwrap(),
            Scalar::utf8("hello".to_string(), Nullable)
        );
        assert!(scalar_at(&array, 1).unwrap().is_null());
    }
}
