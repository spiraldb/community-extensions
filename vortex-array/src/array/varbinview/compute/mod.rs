use std::ops::Deref;

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::{match_each_integer_ptype, PType};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::varbin::varbin_scalar;
use crate::array::varbinview::{VarBinViewArray, VIEW_SIZE_BYTES};
use crate::array::{PrimitiveArray, VarBinViewEncoding};
use crate::compute::{slice, ComputeVTable, ScalarAtFn, SliceFn, TakeFn};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};

impl ComputeVTable for VarBinViewEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<VarBinViewArray> for VarBinViewEncoding {
    fn scalar_at(&self, array: &VarBinViewArray, index: usize) -> VortexResult<Scalar> {
        array
            .bytes_at(index)
            .map(|bytes| varbin_scalar(ByteBuffer::from(bytes), array.dtype()))
    }
}

impl SliceFn<VarBinViewArray> for VarBinViewEncoding {
    fn slice(&self, array: &VarBinViewArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(VarBinViewArray::try_new(
            slice(
                array.views(),
                start * VIEW_SIZE_BYTES,
                stop * VIEW_SIZE_BYTES,
            )?,
            (0..array.metadata().buffer_lens.len())
                .map(|i| array.buffer(i))
                .collect::<Vec<_>>(),
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }
}

/// Take involves creating a new array that references the old array, just with the given set of views.
impl TakeFn<VarBinViewArray> for VarBinViewEncoding {
    fn take(&self, array: &VarBinViewArray, indices: &ArrayData) -> VortexResult<ArrayData> {
        // Compute the new validity
        let validity = array.validity().take(indices)?;

        // Convert our views array into an Arrow u128 Buffer (16 bytes per view)
        let views_buffer =
            Buffer::<u128>::from_byte_buffer(array.views().into_primitive()?.into_byte_buffer());

        let indices = indices.clone().into_primitive()?;

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |$I| {
            take_views(views_buffer, indices.as_slice::<$I>())
        });

        // Cast views back to u8
        let views_array = PrimitiveArray::from_byte_buffer(
            ByteBuffer::from_bytes_aligned(views_buffer.into_inner(), Alignment::of::<u128>()),
            PType::U8,
            Validity::NonNullable,
        );

        Ok(VarBinViewArray::try_new(
            views_array.into_array(),
            array.buffers().collect_vec(),
            array.dtype().clone(),
            validity,
        )?
        .into_array())
    }

    unsafe fn take_unchecked(
        &self,
        array: &VarBinViewArray,
        indices: &ArrayData,
    ) -> VortexResult<ArrayData> {
        // Compute the new validity
        let validity = array.validity().take(indices)?;

        // Convert our views array into an Arrow u128 Buffer (16 bytes per view)
        let views_buffer =
            Buffer::<u128>::from_byte_buffer(array.views().into_primitive()?.into_byte_buffer());

        let indices = indices.clone().into_primitive()?;

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |$I| {
            take_views_unchecked(views_buffer, indices.as_slice::<$I>())
        });

        // Cast views back to u8
        let views_array = PrimitiveArray::from_byte_buffer(
            views_buffer.into_byte_buffer(),
            PType::U8,
            Validity::NonNullable,
        );

        Ok(VarBinViewArray::try_new(
            views_array.into_array(),
            array.buffers().collect_vec(),
            array.dtype().clone(),
            validity,
        )?
        .into_array())
    }
}

fn take_views<I: AsPrimitive<usize>>(views: Buffer<u128>, indices: &[I]) -> Buffer<u128> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    Buffer::<u128>::from_iter(indices.iter().map(|i| views_ref[i.as_()]))
}

fn take_views_unchecked<I: AsPrimitive<usize>>(views: Buffer<u128>, indices: &[I]) -> Buffer<u128> {
    // NOTE(ngates): this deref is not actually trivial, so we run it once.
    let views_ref = views.deref();
    Buffer::<u128>::from_iter(
        indices
            .iter()
            .map(|i| unsafe { *views_ref.get_unchecked(i.as_()) }),
    )
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::accessor::ArrayAccessor;
    use crate::array::VarBinViewArray;
    use crate::compute::take;
    use crate::{ArrayDType, IntoArrayData, IntoArrayVariant};

    #[test]
    fn take_nullable() {
        let arr = VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            None,
            Some("six"),
        ]);

        let taken = take(arr, buffer![0, 3].into_array()).unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken
                .into_varbinview()
                .unwrap()
                .with_iterator(|it| it
                    .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                    .collect::<Vec<_>>())
                .unwrap(),
            [Some("one".to_string()), Some("four".to_string())]
        );
    }
}
