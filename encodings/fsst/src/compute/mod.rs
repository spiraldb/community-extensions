mod compare;

use vortex_array::array::varbin_scalar;
use vortex_array::compute::{
    filter, scalar_at, slice, take, CompareFn, FilterFn, ScalarAtFn, SliceFn, TakeFn,
};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, IntoArray};
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::{FSSTArray, FSSTEncoding};

impl ComputeVTable for FSSTEncoding {
    fn compare_fn(&self) -> Option<&dyn CompareFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
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

impl SliceFn<FSSTArray> for FSSTEncoding {
    fn slice(&self, array: &FSSTArray, start: usize, stop: usize) -> VortexResult<Array> {
        // Slicing an FSST array leaves the symbol table unmodified,
        // only slicing the `codes` array.
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols(),
            array.symbol_lengths(),
            slice(array.codes(), start, stop)?,
            slice(array.uncompressed_lengths(), start, stop)?,
        )?
        .into_array())
    }
}

impl TakeFn<FSSTArray> for FSSTEncoding {
    // Take on an FSSTArray is a simple take on the codes array.
    fn take(&self, array: &FSSTArray, indices: &Array) -> VortexResult<Array> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols(),
            array.symbol_lengths(),
            take(array.codes(), indices)?,
            take(array.uncompressed_lengths(), indices)?,
        )?
        .into_array())
    }
}

impl ScalarAtFn<FSSTArray> for FSSTEncoding {
    fn scalar_at(&self, array: &FSSTArray, index: usize) -> VortexResult<Scalar> {
        let compressed = scalar_at(array.codes(), index)?;
        let binary_datum = compressed
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("expected null to already be handled"))?;

        array.with_decompressor(|decompressor| {
            let decoded_buffer = ByteBuffer::from(decompressor.decompress(binary_datum.as_slice()));
            Ok(varbin_scalar(decoded_buffer, array.dtype()))
        })
    }
}

impl FilterFn<FSSTArray> for FSSTEncoding {
    // Filtering an FSSTArray filters the codes array, leaving the symbols array untouched
    fn filter(&self, array: &FSSTArray, mask: &Mask) -> VortexResult<Array> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols(),
            array.symbol_lengths(),
            filter(&array.codes(), mask)?,
            filter(&array.uncompressed_lengths(), mask)?,
        )?
        .into_array())
    }
}
