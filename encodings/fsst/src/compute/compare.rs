use fsst::Symbol;
use vortex_array::array::ConstantArray;
use vortex_array::compute::{compare, CompareFn, Operator};
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::{FSSTArray, FSSTEncoding};

impl CompareFn<FSSTArray> for FSSTEncoding {
    fn compare(
        &self,
        lhs: &FSSTArray,
        rhs: &Array,
        operator: Operator,
    ) -> VortexResult<Option<Array>> {
        match (rhs.as_constant(), operator) {
            (Some(constant), _) if constant.is_null() => {
                // All comparisons to null must return null
                Ok(Some(
                    ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), lhs.len())
                        .into_array(),
                ))
            }
            (Some(constant), Operator::Eq | Operator::NotEq) => compare_fsst_constant(
                lhs,
                &ConstantArray::new(constant, lhs.len()),
                operator == Operator::Eq,
            )
            .map(Some),
            // Otherwise, fall back to the default comparison behavior.
            _ => Ok(None),
        }
    }
}

/// Specialized compare function implementation used when performing equals or not equals against
/// a constant.
fn compare_fsst_constant(
    left: &FSSTArray,
    right: &ConstantArray,
    equal: bool,
) -> VortexResult<Array> {
    let symbols = left.symbols().into_primitive()?;
    let symbols_u64 = symbols.as_slice::<u64>();

    let symbol_lens = left.symbol_lengths().into_primitive()?;
    let symbol_lens_u8 = symbol_lens.as_slice::<u8>();

    let mut compressor = fsst::CompressorBuilder::new();
    for (symbol, symbol_len) in symbols_u64.iter().zip(symbol_lens_u8.iter()) {
        compressor.insert(Symbol::from_slice(&symbol.to_le_bytes()), *symbol_len as _);
    }
    let compressor = compressor.build();

    let encoded_scalar = match left.dtype() {
        DType::Utf8(_) => {
            let value = right
                .scalar()
                .as_utf8()
                .value()
                .vortex_expect("Expected non-null scalar");
            ByteBuffer::from(compressor.compress(value.as_bytes()))
        }
        DType::Binary(_) => {
            let value = right
                .scalar()
                .as_binary()
                .value()
                .vortex_expect("Expected non-null scalar");
            ByteBuffer::from(compressor.compress(value.as_slice()))
        }
        _ => unreachable!("FSSTArray can only have string or binary data type"),
    };

    let rhs = ConstantArray::new(encoded_scalar, left.len());
    compare(
        left.codes(),
        rhs,
        if equal { Operator::Eq } else { Operator::NotEq },
    )
}

#[cfg(test)]
mod tests {
    use vortex_array::array::{ConstantArray, VarBinArray};
    use vortex_array::compute::{compare, scalar_at, Operator};
    use vortex_array::{IntoArray, IntoArrayVariant};
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::{fsst_compress, fsst_train_compressor};

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_fsst() {
        let lhs = VarBinArray::from_iter(
            [
                Some("hello"),
                None,
                Some("world"),
                None,
                Some("this is a very long string"),
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let compressor = fsst_train_compressor(&lhs).unwrap();
        let lhs = fsst_compress(&lhs, &compressor).unwrap();

        let rhs = ConstantArray::new("world", lhs.len()).into_array();

        // Ensure fastpath for Eq exists, and returns correct answer
        let equals: Vec<bool> = compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .into_iter()
            .collect();

        assert_eq!(equals, vec![false, false, true, false, false]);

        // Ensure fastpath for Eq exists, and returns correct answer
        let not_equals: Vec<bool> = compare(&lhs, &rhs, Operator::NotEq)
            .unwrap()
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .into_iter()
            .collect();

        assert_eq!(not_equals, vec![true, true, false, true, true]);

        // Ensure null constants are handled correctly.
        let null_rhs =
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), lhs.len());
        let equals_null = compare(&lhs, null_rhs.as_ref(), Operator::Eq).unwrap();
        for idx in 0..lhs.len() {
            assert!(scalar_at(&equals_null, idx).unwrap().is_null());
        }

        let noteq_null = compare(&lhs, null_rhs.as_ref(), Operator::NotEq).unwrap();
        for idx in 0..lhs.len() {
            assert!(scalar_at(&noteq_null, idx).unwrap().is_null());
        }
    }
}
