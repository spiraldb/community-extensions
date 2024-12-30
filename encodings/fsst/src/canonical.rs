use arrow_array::builder::make_view;
use vortex_array::array::{VarBinArray, VarBinViewArray};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{
    ArrayDType, ArrayData, Canonical, IntoArrayData, IntoArrayVariant, IntoCanonical,
};
use vortex_buffer::Buffer;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::FSSTArray;

impl IntoCanonical for FSSTArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        self.with_decompressor(|decompressor| {
            // FSSTArray has two child arrays:
            //
            //  1. A VarBinArray, which holds the string heap of the compressed codes.
            //  2. An uncompressed_lengths primitive array, storing the length of each original
            //     string element.
            //
            // To speed up canonicalization, we can decompress the entire string-heap in a single
            // call. We then turn our uncompressed_lengths into an offsets buffer
            // necessary for a VarBinViewArray and construct the canonical array.

            let compressed_bytes = VarBinArray::try_from(self.codes())?
                .sliced_bytes()?
                .into_primitive()?;

            // Bulk-decompress the entire array.
            // TODO(ngates): return non-vec to avoid this copy
            //   See: https://github.com/spiraldb/fsst/issues/61
            let uncompressed_bytes = decompressor.decompress(compressed_bytes.as_slice::<u8>());

            let uncompressed_lens_array = self
                .uncompressed_lengths()
                .into_canonical()?
                .into_primitive()?;

            // Directly create the binary views.
            let views: Buffer<u128> = match_each_integer_ptype!(uncompressed_lens_array.ptype(), |$P| {
                uncompressed_lens_array.as_slice::<$P>()
                    .iter()
                    .map(|&len| len as usize)
                    .scan(0, |offset, len| {
                        let str_start = *offset;
                        let str_end = *offset + len;

                        *offset += len;

                        Some(make_view(
                            &uncompressed_bytes[str_start..str_end],
                            0u32,
                            str_start as u32,
                        ))
                    })
                    .collect()
            });

            let views_array: ArrayData = Buffer::<u8>::from_byte_buffer(views.into_byte_buffer()).into_array();
            // TODO(ngates): return non-vec to avoid this copy
            //   See: https://github.com/spiraldb/fsst/issues/61
            let uncompressed_bytes_array = Buffer::copy_from(uncompressed_bytes).into_array();

            VarBinViewArray::try_new(
                views_array,
                vec![uncompressed_bytes_array],
                self.dtype().clone(),
                self.validity(),
            )
            .map(Canonical::VarBinView)
        })
    }
}
