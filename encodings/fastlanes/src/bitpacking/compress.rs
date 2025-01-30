use arrow_buffer::ArrowNativeType;
use fastlanes::BitPacking;
use vortex_array::array::PrimitiveArray;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::IntoArray;
use vortex_buffer::{buffer, Buffer, BufferMut, ByteBuffer};
use vortex_dtype::{
    match_each_integer_ptype, match_each_unsigned_integer_ptype, NativePType, PType,
};
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::BitPackedArray;

pub fn bitpack_encode(array: PrimitiveArray, bit_width: u8) -> VortexResult<BitPackedArray> {
    let bit_width_freq = array
        .statistics()
        .compute_bit_width_freq()
        .ok_or_else(|| vortex_err!(ComputeError: "missing bit width frequency"))?;

    // Check array contains no negative values.
    if array.ptype().is_signed_int() {
        let has_negative_values = match_each_integer_ptype!(array.ptype(), |$P| {
            array.statistics().compute_min::<Option<$P>>().unwrap_or_default().unwrap_or_default() < 0
        });
        if has_negative_values {
            vortex_bail!("cannot bitpack_encode array containing negative integers")
        }
    }

    let num_exceptions = count_exceptions(bit_width, &bit_width_freq);

    if bit_width >= array.ptype().bit_width() as u8 {
        // Nothing we can do
        vortex_bail!("Cannot pack -- specified bit width is greater than or equal to raw bit width")
    }

    // SAFETY: we check that array only contains non-negative values.
    let packed = unsafe { bitpack_unchecked(&array, bit_width)? };
    let patches = (num_exceptions > 0)
        .then(|| gather_patches(&array, bit_width, num_exceptions))
        .flatten();

    // SAFETY: values already checked to be non-negative.
    unsafe {
        BitPackedArray::new_unchecked(
            packed,
            array.ptype(),
            array.validity(),
            patches,
            bit_width,
            array.len(),
        )
    }
}

/// Bitpack an array into the specified bit-width without checking statistics.
///
/// # Safety
///
/// It is the caller's responsibility to ensure that all values in the array can lossless pack
/// into the specified bit-width.
///
/// Failure to do so will result in data loss.
pub unsafe fn bitpack_encode_unchecked(
    array: PrimitiveArray,
    bit_width: u8,
) -> VortexResult<BitPackedArray> {
    // SAFETY: non-negativity of input checked by caller.
    unsafe {
        let packed = bitpack_unchecked(&array, bit_width)?;

        BitPackedArray::new_unchecked(
            packed,
            array.ptype(),
            array.validity(),
            None,
            bit_width,
            array.len(),
        )
    }
}

/// Bitpack a [PrimitiveArray] to the given width.
///
/// On success, returns a [Buffer] containing the packed data.
///
/// # Safety
///
/// Internally this function will promote the provided array to its unsigned equivalent. This will
/// violate ordering guarantees if the array contains any negative values.
///
/// It is the caller's responsibility to ensure that `parray` is non-negative before calling
/// this function.
pub unsafe fn bitpack_unchecked(
    parray: &PrimitiveArray,
    bit_width: u8,
) -> VortexResult<ByteBuffer> {
    let parray = parray.reinterpret_cast(parray.ptype().to_unsigned());
    let packed = match_each_unsigned_integer_ptype!(parray.ptype(), |$P| {
        bitpack_primitive(parray.as_slice::<$P>(), bit_width)
    });
    Ok(packed)
}

/// Bitpack a slice of primitives down to the given width.
///
/// See `bitpack` for more caller information.
pub fn bitpack_primitive<T: NativePType + BitPacking + ArrowNativeType>(
    array: &[T],
    bit_width: u8,
) -> ByteBuffer {
    if bit_width == 0 {
        return ByteBuffer::empty();
    }
    let bit_width = bit_width as usize;

    // How many fastlanes vectors we will process.
    let num_chunks = (array.len() + 1023) / 1024;
    let num_full_chunks = array.len() / 1024;
    let packed_len = 128 * bit_width / size_of::<T>();
    // packed_len says how many values of size T we're going to include.
    // 1024 * bit_width / 8 == the number of bytes we're going to get.
    // then we divide by the size of T to get the number of elements.

    // Allocate a result byte array.
    let mut output = BufferMut::<T>::with_capacity(num_chunks * packed_len);

    // Loop over all but the last chunk.
    (0..num_full_chunks).for_each(|i| {
        let start_elem = i * 1024;
        let output_len = output.len();
        unsafe {
            output.set_len(output_len + packed_len);
            BitPacking::unchecked_pack(
                bit_width,
                &array[start_elem..][..1024],
                &mut output[output_len..][..packed_len],
            );
        };
    });

    // Pad the last chunk with zeros to a full 1024 elements.
    if num_chunks != num_full_chunks {
        let last_chunk_size = array.len() % 1024;
        let mut last_chunk: [T; 1024] = [T::zero(); 1024];
        last_chunk[..last_chunk_size].copy_from_slice(&array[array.len() - last_chunk_size..]);

        let output_len = output.len();
        unsafe {
            output.set_len(output_len + packed_len);
            BitPacking::unchecked_pack(
                bit_width,
                &last_chunk,
                &mut output[output_len..][..packed_len],
            );
        };
    }

    output.freeze().into_byte_buffer()
}

pub fn gather_patches(
    parray: &PrimitiveArray,
    bit_width: u8,
    num_exceptions_hint: usize,
) -> Option<Patches> {
    let patch_validity = match parray.validity() {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    match_each_integer_ptype!(parray.ptype(), |$T| {
        let mut indices: BufferMut<u64> = BufferMut::with_capacity(num_exceptions_hint);
        let mut values: BufferMut<$T> = BufferMut::with_capacity(num_exceptions_hint);
        for (i, v) in parray.as_slice::<$T>().iter().enumerate() {
            if (v.leading_zeros() as usize) < parray.ptype().bit_width() - bit_width as usize && parray.is_valid(i).vortex_expect("validity") {
                indices.push(i as u64);
                values.push(*v);
            }
        }
        (!indices.is_empty()).then(|| Patches::new(
            parray.len(),
            indices.into_array(),
            PrimitiveArray::new(values, patch_validity).into_array(),
        ))
    })
}

pub fn unpack(array: BitPackedArray) -> VortexResult<PrimitiveArray> {
    let bit_width = array.bit_width() as usize;
    let length = array.len();
    let offset = array.offset() as usize;
    let ptype = array.ptype();
    let mut unpacked = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |$P| {
        PrimitiveArray::new(
            unpack_primitive::<$P>(array.packed_slice::<$P>(), bit_width, offset, length),
            array.validity(),
        )
    });

    // Cast to signed if necessary
    if ptype.is_signed_int() {
        unpacked = unpacked.reinterpret_cast(ptype);
    }

    if let Some(patches) = array.patches() {
        unpacked.patch(patches)
    } else {
        Ok(unpacked)
    }
}

pub fn unpack_primitive<T: NativePType + BitPacking>(
    packed: &[T],
    bit_width: usize,
    offset: usize,
    length: usize,
) -> Buffer<T> {
    if bit_width == 0 {
        return buffer![T::zero(); length];
    }

    // How many fastlanes vectors we will process.
    // Packed array might not start at 0 when the array is sliced. Offset is guaranteed to be < 1024.
    let num_chunks = (offset + length + 1023) / 1024;
    let elems_per_chunk = 128 * bit_width / size_of::<T>();
    assert_eq!(
        packed.len(),
        num_chunks * elems_per_chunk,
        "Invalid packed length: got {}, expected {}",
        packed.len(),
        num_chunks * elems_per_chunk
    );

    // Allocate a result vector.
    // TODO(ngates): do we want to use fastlanes alignment for this buffer?
    let mut output = BufferMut::with_capacity(num_chunks * 1024 - offset);

    // Handle first chunk if offset is non 0. We have to decode the chunk and skip first offset elements
    let first_full_chunk = if offset != 0 {
        let chunk: &[T] = &packed[0..elems_per_chunk];
        let mut decoded = [T::zero(); 1024];
        unsafe { BitPacking::unchecked_unpack(bit_width, chunk, &mut decoded) };
        output.extend_from_slice(&decoded[offset..]);
        1
    } else {
        0
    };

    // Loop over all the chunks.
    (first_full_chunk..num_chunks).for_each(|i| {
        let chunk: &[T] = &packed[i * elems_per_chunk..][0..elems_per_chunk];
        unsafe {
            let output_len = output.len();
            output.set_len(output_len + 1024);
            BitPacking::unchecked_unpack(bit_width, chunk, &mut output[output_len..][0..1024]);
        }
    });

    // The final chunk may have had padding
    output.truncate(length);

    assert_eq!(
        output.len(),
        length,
        "Expected unpacked array to be of length {} but got {}",
        length,
        output.len()
    );
    output.freeze()
}

pub fn unpack_single(array: &BitPackedArray, index: usize) -> VortexResult<Scalar> {
    let bit_width = array.bit_width() as usize;
    let ptype = array.ptype();
    // let packed = array.packed().into_primitive()?;
    let index_in_encoded = index + array.offset() as usize;
    let scalar: Scalar = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |$P| unsafe {
        unpack_single_primitive::<$P>(array.packed_slice::<$P>(), bit_width, index_in_encoded).into()
    });
    // Cast to fix signedness and nullability
    scalar.cast(array.dtype())
}

/// # Safety
///
/// The caller must ensure the following invariants hold:
/// * `packed.len() == (length + 1023) / 1024 * 128 * bit_width`
/// * `index_to_decode < length`
///
/// Where `length` is the length of the array/slice backed by `packed`
/// (but is not provided to this function).
pub unsafe fn unpack_single_primitive<T: NativePType + BitPacking>(
    packed: &[T],
    bit_width: usize,
    index_to_decode: usize,
) -> T {
    let chunk_index = index_to_decode / 1024;
    let index_in_chunk = index_to_decode % 1024;
    let elems_per_chunk: usize = 128 * bit_width / size_of::<T>();

    let packed_chunk = &packed[chunk_index * elems_per_chunk..][0..elems_per_chunk];
    unsafe { BitPacking::unchecked_unpack_single(bit_width, packed_chunk, index_in_chunk) }
}

pub fn find_min_patchless_bit_width(array: &PrimitiveArray) -> VortexResult<u8> {
    let bit_width_freq = array
        .statistics()
        .compute_bit_width_freq()
        .ok_or_else(|| vortex_err!(ComputeError: "Failed to compute bit width frequency"))?;

    min_patchless_bit_width(&bit_width_freq)
}

fn min_patchless_bit_width(bit_width_freq: &[usize]) -> VortexResult<u8> {
    if bit_width_freq.is_empty() {
        vortex_bail!("Empty bit width frequency!");
    }
    Ok(bit_width_freq
        .iter()
        .enumerate()
        .filter_map(|(bw, count)| (*count > 0).then_some(bw as u8))
        .max()
        .unwrap_or_default())
}

pub fn find_best_bit_width(array: &PrimitiveArray) -> VortexResult<u8> {
    let bit_width_freq = array
        .statistics()
        .compute_bit_width_freq()
        .ok_or_else(|| vortex_err!(ComputeError: "Failed to compute bit width frequency"))?;

    best_bit_width(&bit_width_freq, bytes_per_exception(array.ptype()))
}

/// Assuming exceptions cost 1 value + 1 u32 index, figure out the best bit-width to use.
/// We could try to be clever, but we can never really predict how the exceptions will compress.
#[allow(clippy::cast_possible_truncation)]
fn best_bit_width(bit_width_freq: &[usize], bytes_per_exception: usize) -> VortexResult<u8> {
    if bit_width_freq.len() > u8::MAX as usize {
        vortex_bail!("Too many bit widths");
    }

    let len: usize = bit_width_freq.iter().sum();
    let mut num_packed = 0;
    let mut best_cost = len * bytes_per_exception;
    let mut best_width = 0;
    for (bit_width, freq) in bit_width_freq.iter().enumerate() {
        let packed_cost = ((bit_width * len) + 7) / 8; // round up to bytes

        num_packed += *freq;
        let exceptions_cost = (len - num_packed) * bytes_per_exception;

        let cost = exceptions_cost + packed_cost;
        if cost < best_cost {
            best_cost = cost;
            best_width = bit_width;
        }
    }

    Ok(best_width as u8)
}

fn bytes_per_exception(ptype: PType) -> usize {
    ptype.byte_width() + 4
}

pub fn count_exceptions(bit_width: u8, bit_width_freq: &[usize]) -> usize {
    if bit_width_freq.len() <= bit_width as usize {
        return 0;
    }
    bit_width_freq[bit_width as usize + 1..].iter().sum()
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod test {
    use vortex_array::IntoArrayVariant;
    use vortex_error::VortexError;

    use super::*;

    #[test]
    fn test_best_bit_width() {
        // 10 1-bit values, 20 2-bit, etc.
        let freq = vec![0, 10, 20, 15, 1, 0, 0, 0];
        // 3-bits => (46 * 3) + (8 * 1 * 5) => 178 bits => 23 bytes and zero exceptions
        assert_eq!(
            best_bit_width(&freq, bytes_per_exception(PType::U8)).unwrap(),
            3
        );
        assert_eq!(min_patchless_bit_width(&freq).unwrap(), 4)
    }

    #[test]
    fn null_patches() {
        let valid_values = (0..24).map(|v| v < 1 << 4).collect::<Vec<_>>();
        let values = PrimitiveArray::new(
            (0u32..24).collect::<Buffer<_>>(),
            Validity::from_iter(valid_values),
        );
        assert!(values.ptype().is_unsigned_int());
        let compressed = BitPackedArray::encode(values.as_ref(), 4).unwrap();
        assert!(compressed.patches().is_none());
        assert_eq!(
            (0..(1 << 4)).collect::<Vec<_>>(),
            compressed
                .logical_validity()
                .unwrap()
                .to_null_buffer()
                .unwrap()
                .into_inner()
                .set_indices()
                .collect::<Vec<_>>()
        )
    }

    #[test]
    fn test_compression_roundtrip_fast() {
        compression_roundtrip(125);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // This test is too slow on miri
    fn test_compression_roundtrip() {
        compression_roundtrip(1024);
        compression_roundtrip(10_000);
        compression_roundtrip(10_240);
    }

    fn compression_roundtrip(n: usize) {
        let values = PrimitiveArray::from_iter((0..n).map(|i| (i % 2047) as u16));
        let compressed = BitPackedArray::encode(values.as_ref(), 11).unwrap();
        let decompressed = compressed.clone().into_primitive().unwrap();
        assert_eq!(decompressed.as_slice::<u16>(), values.as_slice::<u16>());

        values
            .as_slice::<u16>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                let scalar: u16 = unpack_single(&compressed, i).unwrap().try_into().unwrap();
                assert_eq!(scalar, *v);
            });
    }

    #[test]
    fn compress_signed_fails() {
        let values: Buffer<i64> = (-500..500).collect();
        let array = PrimitiveArray::new(values, Validity::AllValid);
        assert!(array.ptype().is_signed_int());

        let err = BitPackedArray::encode(array.as_ref(), 1024u32.ilog2() as u8).unwrap_err();
        assert!(matches!(err, VortexError::InvalidArgument(_, _)));
    }
}
