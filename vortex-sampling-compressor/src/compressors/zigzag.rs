use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::PrimitiveArray;
use vortex_array::stats::Stat;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, Encoding, EncodingId, IntoArray};
use vortex_error::VortexResult;
use vortex_zigzag::{zigzag_encode, ZigZagArray, ZigZagEncoding};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct ZigZagCompressor;

impl EncodingCompressor for ZigZagCompressor {
    fn id(&self) -> &str {
        ZigZagEncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::ZIGZAG_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = PrimitiveArray::maybe_from(array)?;

        // Only supports signed integers
        if !parray.ptype().is_signed_int() {
            return None;
        }

        // Only compress if the array has negative values
        // TODO(ngates): also check that Stat::Max is less than half the max value of the type
        parray
            .statistics()
            .compute_as::<i64>(Stat::Min)
            .filter(|&min| min < 0)
            .map(|_| self as &dyn EncodingCompressor)
    }

    fn compress<'a>(
        &'a self,
        array: &Array,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let encoded = zigzag_encode(PrimitiveArray::try_from(array.clone())?)?;
        let compressed =
            ctx.compress(&encoded.encoded(), like.as_ref().and_then(|l| l.child(0)))?;
        Ok(CompressedArray::compressed(
            ZigZagArray::try_new(compressed.array)?.into_array(),
            Some(CompressionTree::new(self, vec![compressed.path])),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([ZigZagEncoding::ID])
    }
}
