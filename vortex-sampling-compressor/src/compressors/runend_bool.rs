use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::BoolEncoding;
use vortex_array::encoding::{Encoding, EncodingRef};
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_error::VortexResult;
use vortex_runend_bool::compress::runend_bool_encode_slice;
use vortex_runend_bool::{RunEndBoolArray, RunEndBoolEncoding};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::downscale::downscale_integer_array;
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct RunEndBoolCompressor;

impl EncodingCompressor for RunEndBoolCompressor {
    fn id(&self) -> &str {
        RunEndBoolEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::RUN_END_BOOL_COST
    }

    fn can_compress(&self, array: &ArrayData) -> Option<&dyn EncodingCompressor> {
        // Only support bool arrays
        if !array.is_encoding(BoolEncoding::ID) {
            return None;
        }

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &ArrayData,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let bool_array = array.clone().into_bool()?;
        let (ends, start) = runend_bool_encode_slice(&bool_array.boolean_buffer());
        let ends = downscale_integer_array(ends.into_array())?;

        let compressed_ends = ctx
            .auxiliary("ends")
            .compress(&ends, like.as_ref().and_then(|l| l.child(0)))?;

        Ok(CompressedArray::compressed(
            RunEndBoolArray::try_new(compressed_ends.array, start, bool_array.validity())?
                .into_array(),
            Some(CompressionTree::new(self, vec![compressed_ends.path])),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        HashSet::from([&RunEndBoolEncoding as EncodingRef])
    }
}
