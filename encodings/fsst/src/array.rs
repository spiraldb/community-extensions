use fsst::{Decompressor, Symbol};
use serde::{Deserialize, Serialize};
use vortex_array::array::{VarBinArray, VarBinEncoding};
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_array::variants::{BinaryArrayTrait, Utf8ArrayTrait};
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{
    StatisticsVTable, ValidateVTable, ValidityVTable, VariantsVTable, VisitorVTable,
};
use vortex_array::{encoding_ids, impl_encoding, Array, Encoding, IntoArrayVariant, SerdeMetadata};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_mask::Mask;

impl_encoding!(
    "vortex.fsst",
    encoding_ids::FSST,
    FSST,
    SerdeMetadata<FSSTMetadata>
);

static SYMBOLS_DTYPE: DType = DType::Primitive(PType::U64, Nullability::NonNullable);
static SYMBOL_LENS_DTYPE: DType = DType::Primitive(PType::U8, Nullability::NonNullable);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FSSTMetadata {
    symbols_len: usize,
    codes_nullability: Nullability,
    uncompressed_lengths_ptype: PType,
}

impl FSSTArray {
    /// Build an FSST array from a set of `symbols` and `codes`.
    ///
    /// Symbols are 8-bytes and can represent short strings, each of which is assigned
    /// a code.
    ///
    /// The `codes` array is a Binary array where each binary datum is a sequence of 8-bit codes.
    /// Each code corresponds either to a symbol, or to the "escape code",
    /// which tells the decoder to emit the following byte without doing a table lookup.
    pub fn try_new(
        dtype: DType,
        symbols: Array,
        symbol_lengths: Array,
        codes: Array,
        uncompressed_lengths: Array,
    ) -> VortexResult<Self> {
        // Check: symbols must be a u64 array
        if symbols.dtype() != &SYMBOLS_DTYPE {
            vortex_bail!(InvalidArgument: "symbols array must be of type u64")
        }

        if symbol_lengths.dtype() != &SYMBOL_LENS_DTYPE {
            vortex_bail!(InvalidArgument: "symbol_lengths array must be of type u8")
        }

        // Check: symbols must not have length > MAX_CODE
        if symbols.len() > 255 {
            vortex_bail!(InvalidArgument: "symbols array must have length <= 255");
        }

        if symbols.len() != symbol_lengths.len() {
            vortex_bail!(InvalidArgument: "symbols and symbol_lengths arrays must have same length");
        }

        if uncompressed_lengths.len() != codes.len() {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must be same len as codes");
        }

        if !uncompressed_lengths.dtype().is_int() || uncompressed_lengths.dtype().is_nullable() {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must have integer type and cannot be nullable, found {}", uncompressed_lengths.dtype());
        }

        if codes.encoding() != VarBinEncoding::ID {
            vortex_bail!(
                InvalidArgument: "codes must have varbin encoding, was {}",
                codes.encoding()
            );
        }

        // Check: strings must be a Binary array.
        if !matches!(codes.dtype(), DType::Binary(_)) {
            vortex_bail!(InvalidArgument: "codes array must be DType::Binary type");
        }

        let symbols_len = symbols.len();
        let len = codes.len();
        let uncompressed_lengths_ptype = PType::try_from(uncompressed_lengths.dtype())?;
        let codes_nullability = codes.dtype().nullability();
        let children = [symbols, symbol_lengths, codes, uncompressed_lengths].into();

        Self::try_from_parts(
            dtype,
            len,
            SerdeMetadata(FSSTMetadata {
                symbols_len,
                codes_nullability,
                uncompressed_lengths_ptype,
            }),
            None,
            Some(children),
            StatsSet::default(),
        )
    }

    /// Access the symbol table array
    pub fn symbols(&self) -> Array {
        self.as_ref()
            .child(0, &SYMBOLS_DTYPE, self.metadata().symbols_len)
            .vortex_expect("FSSTArray symbols child")
    }

    /// Access the symbol table array
    pub fn symbol_lengths(&self) -> Array {
        self.as_ref()
            .child(1, &SYMBOL_LENS_DTYPE, self.metadata().symbols_len)
            .vortex_expect("FSSTArray symbol_lengths child")
    }

    /// Access the codes array
    pub fn codes(&self) -> Array {
        self.as_ref()
            .child(2, &self.codes_dtype(), self.len())
            .vortex_expect("FSSTArray codes child")
    }

    /// Get the DType of the codes array
    #[inline]
    pub fn codes_dtype(&self) -> DType {
        DType::Binary(self.metadata().codes_nullability)
    }

    /// Get the uncompressed length for each element in the array.
    pub fn uncompressed_lengths(&self) -> Array {
        self.as_ref()
            .child(3, &self.uncompressed_lengths_dtype(), self.len())
            .vortex_expect("FSST uncompressed_lengths child")
    }

    /// Get the DType of the uncompressed lengths array
    #[inline]
    pub fn uncompressed_lengths_dtype(&self) -> DType {
        DType::Primitive(
            self.metadata().uncompressed_lengths_ptype,
            Nullability::NonNullable,
        )
    }

    /// Get the validity for this array.
    pub fn validity(&self) -> Validity {
        VarBinArray::try_from(self.codes())
            .vortex_expect("FSSTArray must have a codes child array")
            .validity()
    }

    /// Build a [`Decompressor`][fsst::Decompressor] that can be used to decompress values from
    /// this array, and pass it to the given function.
    ///
    /// This is private to the crate to avoid leaking `fsst-rs` types as part of the public API.
    pub(crate) fn with_decompressor<F, R>(&self, apply: F) -> VortexResult<R>
    where
        F: FnOnce(Decompressor) -> VortexResult<R>,
    {
        // canonicalize the symbols child array, so we can view it contiguously
        let symbols_array = self
            .symbols()
            .into_primitive()
            .map_err(|err| err.with_context("Symbols must be a Primitive Array"))?;
        let symbols = symbols_array.as_slice::<u64>();

        let symbol_lengths_array = self
            .symbol_lengths()
            .into_primitive()
            .map_err(|err| err.with_context("Symbol lengths must be a Primitive Array"))?;
        let symbol_lengths = symbol_lengths_array.as_slice::<u8>();

        // Transmute the 64-bit symbol values into fsst `Symbol`s.
        // SAFETY: Symbol is guaranteed to be 8 bytes, guaranteed by the compiler.
        let symbols = unsafe { std::mem::transmute::<&[u64], &[Symbol]>(symbols) };

        // Build a new decompressor that uses these symbols.
        let decompressor = Decompressor::new(symbols, symbol_lengths);
        apply(decompressor)
    }
}

impl VisitorVTable<FSSTArray> for FSSTEncoding {
    fn accept(&self, array: &FSSTArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("symbols", &array.symbols())?;
        visitor.visit_child("symbol_lengths", &array.symbol_lengths())?;
        visitor.visit_child("codes", &array.codes())?;
        visitor.visit_child("uncompressed_lengths", &array.uncompressed_lengths())
    }
}

impl StatisticsVTable<FSSTArray> for FSSTEncoding {}

impl ValidityVTable<FSSTArray> for FSSTEncoding {
    fn is_valid(&self, array: &FSSTArray, index: usize) -> VortexResult<bool> {
        array.codes().is_valid(index)
    }

    fn logical_validity(&self, array: &FSSTArray) -> VortexResult<Mask> {
        array.codes().logical_validity()
    }
}

impl VariantsVTable<FSSTArray> for FSSTEncoding {
    fn as_utf8_array<'a>(&self, array: &'a FSSTArray) -> Option<&'a dyn Utf8ArrayTrait> {
        Some(array)
    }

    fn as_binary_array<'a>(&self, array: &'a FSSTArray) -> Option<&'a dyn BinaryArrayTrait> {
        Some(array)
    }
}

impl Utf8ArrayTrait for FSSTArray {}

impl BinaryArrayTrait for FSSTArray {}

impl ValidateVTable<FSSTArray> for FSSTEncoding {}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_array::SerdeMetadata;
    use vortex_dtype::{Nullability, PType};

    use crate::FSSTMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fsst_metadata() {
        check_metadata(
            "fsst.metadata",
            SerdeMetadata(FSSTMetadata {
                symbols_len: usize::MAX,
                codes_nullability: Nullability::Nullable,
                uncompressed_lengths_ptype: PType::U64,
            }),
        );
    }
}
