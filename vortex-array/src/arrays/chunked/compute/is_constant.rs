use vortex_error::{VortexExpect, VortexResult};

use crate::Array;
use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{IsConstantFn, IsConstantOpts, is_constant_opts, scalar_at};

impl IsConstantFn<&ChunkedArray> for ChunkedEncoding {
    fn is_constant(
        &self,
        array: &ChunkedArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        let mut chunks = array.chunks().iter().skip_while(|c| c.is_empty());

        let first_chunk = chunks.next().vortex_expect("Must have at least one value");

        if !is_constant_opts(first_chunk, opts)? {
            return Ok(Some(false));
        }

        let first_value = scalar_at(first_chunk, 0)?.into_nullable();

        for chunk in chunks {
            if chunk.is_empty() {
                continue;
            }

            if !is_constant_opts(chunk, opts)? {
                return Ok(Some(false));
            }

            if first_value != scalar_at(chunk, 0)?.into_nullable() {
                return Ok(Some(false));
            }
        }

        Ok(Some(true))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{Buffer, buffer};
    use vortex_dtype::{DType, Nullability, PType};

    use crate::arrays::ChunkedArray;
    use crate::{Array, IntoArray};

    #[test]
    fn empty_chunk_is_constant() {
        let chunked = ChunkedArray::try_new(
            vec![
                Buffer::<u8>::empty().into_array(),
                Buffer::<u8>::empty().into_array(),
                buffer![255u8, 255].into_array(),
                Buffer::<u8>::empty().into_array(),
                buffer![255u8, 255].into_array(),
            ],
            DType::Primitive(PType::U8, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        assert!(chunked.statistics().compute_is_constant().unwrap());
    }
}
