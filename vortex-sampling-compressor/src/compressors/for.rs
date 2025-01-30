use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::PrimitiveArray;
use vortex_array::stats::trailing_zeros;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, Encoding, EncodingId, IntoArray, IntoArrayVariant};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_fastlanes::{for_compress, FoRArray, FoREncoding};

use crate::compressors::{CompressedArray, CompressionTree, EncodingCompressor};
use crate::{constants, SamplingCompressor};

#[derive(Debug)]
pub struct FoRCompressor;

impl EncodingCompressor for FoRCompressor {
    fn id(&self) -> &str {
        FoREncoding::ID.as_ref()
    }

    fn cost(&self) -> u8 {
        constants::FOR_COST
    }

    fn can_compress(&self, array: &Array) -> Option<&dyn EncodingCompressor> {
        // Only support primitive arrays
        let parray = PrimitiveArray::maybe_from(array)?;

        // Only supports integers
        if !parray.ptype().is_int() {
            return None;
        }

        // For all-null, cannot encode.
        if parray.logical_validity().ok()?.all_false() {
            return None;
        }

        // Nothing for us to do if the min is already zero and tz == 0
        let shift = trailing_zeros(array);
        match_each_integer_ptype!(parray.ptype(), |$P| {
            let min: $P = parray.statistics().compute_min()?;
            if min == 0 && shift == 0 {
                return None;
            }
        });

        Some(self)
    }

    fn compress<'a>(
        &'a self,
        array: &Array,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        let compressed = for_compress(array.clone().into_primitive()?)?;

        let compressed_child = ctx.named("for_encoded").excluding(self).compress(
            &compressed.encoded(),
            like.as_ref().and_then(|l| l.child(0)),
        )?;
        Ok(CompressedArray::compressed(
            FoRArray::try_new(
                compressed_child.array,
                compressed.reference_scalar(),
                compressed.shift(),
            )
            .map(|a| a.into_array())?,
            Some(CompressionTree::new(self, vec![compressed_child.path])),
            array,
        ))
    }

    fn used_encodings(&self) -> HashSet<EncodingId> {
        HashSet::from([FoREncoding::ID])
    }
}
