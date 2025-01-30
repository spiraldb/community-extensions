use arrow_buffer::ArrowNativeType;
use fastlanes::BitPacking;
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::{filter, FilterFn};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{match_each_unsigned_integer_ptype, NativePType};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use super::chunked_indices;
use crate::bitpacking::compute::take::UNPACK_CHUNK_THRESHOLD;
use crate::{BitPackedArray, BitPackedEncoding};

impl FilterFn<BitPackedArray> for BitPackedEncoding {
    fn filter(&self, array: &BitPackedArray, mask: &Mask) -> VortexResult<Array> {
        let primitive = match_each_unsigned_integer_ptype!(array.ptype().to_unsigned(), |$I| {
            filter_primitive::<$I>(array, mask)
        });
        Ok(primitive?.into_array())
    }
}

/// Specialized filter kernel for primitive bit-packed arrays.
///
/// Because the FastLanes bit-packing kernels are only implemented for unsigned types, the provided
/// `T` should be promoted to the unsigned variant for any target bit width.
/// For example, if the array is bit-packed `i16`, this function called be called with `T = u16`.
///
/// All bit-packing operations will use the unsigned kernels, but the logical type of `array`
/// dictates the final `PType` of the result.
fn filter_primitive<T: NativePType + BitPacking + ArrowNativeType>(
    array: &BitPackedArray,
    mask: &Mask,
) -> VortexResult<PrimitiveArray> {
    let validity = array.validity().filter(mask)?;

    let patches = array
        .patches()
        .map(|patches| patches.filter(mask))
        .transpose()?
        .flatten();

    // Short-circuit if the selectivity is high enough.
    if mask.density() > 0.8 {
        return filter(array.clone().into_primitive()?.as_ref(), mask)
            .and_then(|a| a.into_primitive());
    }

    let values: Buffer<T> = filter_indices(
        array,
        mask.true_count(),
        mask.values()
            .vortex_expect("AllTrue and AllFalse handled by filter fn")
            .indices()
            .iter()
            .copied(),
    );

    let mut values = PrimitiveArray::new(values, validity).reinterpret_cast(array.ptype());
    if let Some(patches) = patches {
        values = values.patch(patches)?;
    }
    Ok(values)
}

fn filter_indices<T: NativePType + BitPacking + ArrowNativeType>(
    array: &BitPackedArray,
    indices_len: usize,
    indices: impl Iterator<Item = usize>,
) -> Buffer<T> {
    let offset = array.offset() as usize;
    let bit_width = array.bit_width() as usize;
    let mut values = BufferMut::with_capacity(indices_len);

    // Some re-usable memory to store per-chunk indices.
    let mut unpacked = [T::zero(); 1024];
    let packed_bytes = array.packed_slice::<T>();

    // Group the indices by the FastLanes chunk they belong to.
    let chunk_size = 128 * bit_width / size_of::<T>();

    chunked_indices(indices, offset, |chunk_idx, indices_within_chunk| {
        let packed = &packed_bytes[chunk_idx * chunk_size..][..chunk_size];

        if indices_within_chunk.len() == 1024 {
            // Unpack the entire chunk.
            unsafe {
                let values_len = values.len();
                values.set_len(values_len + 1024);
                BitPacking::unchecked_unpack(
                    bit_width,
                    packed,
                    &mut values.as_mut_slice()[values_len..],
                );
            }
        } else if indices_within_chunk.len() > UNPACK_CHUNK_THRESHOLD {
            // Unpack into a temporary chunk and then copy the values.
            unsafe { BitPacking::unchecked_unpack(bit_width, packed, &mut unpacked) }
            values.extend(
                indices_within_chunk
                    .iter()
                    .map(|&idx| unsafe { *unpacked.get_unchecked(idx) }),
            );
        } else {
            // Otherwise, unpack each element individually.
            values.extend(indices_within_chunk.iter().map(|&idx| unsafe {
                BitPacking::unchecked_unpack_single(bit_width, packed, idx)
            }));
        }
    });

    values.freeze()
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::{filter, slice};
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayVariant;
    use vortex_buffer::Buffer;
    use vortex_mask::Mask;

    use crate::BitPackedArray;

    #[test]
    fn take_indices() {
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();

        let mask = Mask::from_indices(bitpacked.len(), vec![0, 125, 2047, 2049, 2151, 2790]);

        let primitive_result = filter(bitpacked.as_ref(), &mask)
            .unwrap()
            .into_primitive()
            .unwrap();
        let res_bytes = primitive_result.as_slice::<u8>();
        assert_eq!(res_bytes, &[0, 62, 31, 33, 9, 18]);
    }

    #[test]
    fn take_sliced_indices() {
        // Create a u8 array modulo 63.
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let sliced = slice(bitpacked.as_ref(), 128, 2050).unwrap();

        let mask = Mask::from_indices(sliced.len(), vec![1919, 1921]);

        let primitive_result = filter(&sliced, &mask).unwrap().into_primitive().unwrap();
        let res_bytes = primitive_result.as_slice::<u8>();
        assert_eq!(res_bytes, &[31, 33]);
    }

    #[test]
    fn filter_bitpacked() {
        let unpacked = PrimitiveArray::from_iter((0..4096).map(|i| (i % 63) as u8));
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 6).unwrap();
        let filtered = filter(
            bitpacked.as_ref(),
            &Mask::from_indices(4096, (0..1024).collect()),
        )
        .unwrap();
        assert_eq!(
            filtered.into_primitive().unwrap().as_slice::<u8>(),
            (0..1024).map(|i| (i % 63) as u8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn filter_bitpacked_signed() {
        let values: Buffer<i64> = (0..500).collect();
        let unpacked = PrimitiveArray::new(values.clone(), Validity::NonNullable);
        let bitpacked = BitPackedArray::encode(unpacked.as_ref(), 9).unwrap();
        let filtered = filter(
            bitpacked.as_ref(),
            &Mask::from_indices(values.len(), (0..250).collect()),
        )
        .unwrap()
        .into_primitive()
        .unwrap();

        assert_eq!(filtered.as_slice::<i64>(), &values[0..250]);
    }
}
