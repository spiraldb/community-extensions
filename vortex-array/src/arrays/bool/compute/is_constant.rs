use vortex_error::VortexResult;

use crate::arrays::{BoolArray, BoolEncoding};
use crate::compute::{IsConstantFn, IsConstantOpts};

impl IsConstantFn<&BoolArray> for BoolEncoding {
    fn is_constant(&self, array: &BoolArray, _opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        let buffer = array.boolean_buffer();

        // Safety:
        // We must have at least one value at this point
        let first_value = unsafe { buffer.value_unchecked(0) };
        let value_block = if first_value { u64::MAX } else { 0_u64 };

        let bit_chunks = buffer.bit_chunks();
        let packed = bit_chunks.iter().all(|chunk| chunk == value_block);
        let reminder = bit_chunks.remainder_bits().count_ones() as usize
            == bit_chunks.remainder_len() * (first_value as usize);

        // We iterate on blocks of u64
        Ok(Some(packed & reminder))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(vec![true], Some(true))]
    #[case(vec![false; 65], Some(true))]
    #[case({
        let mut v = vec![true; 64];
        v.push(false);
        v
    }, Some(false))]
    fn test_is_constant(#[case] input: Vec<bool>, #[case] expected: Option<bool>) {
        let array = BoolArray::from_iter(input);

        let output = BoolEncoding
            .is_constant(&array, &IsConstantOpts::default())
            .unwrap();
        assert_eq!(output, expected);
    }
}
