//! Encodings that enable zero-copy sharing of data with Arrow.

use std::ops::Deref;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::array::{
    BoolArray, ExtensionArray, ListArray, NullArray, PrimitiveArray, StructArray, VarBinViewArray,
};
use crate::arrow::IntoArrowArray;
use crate::builders::builder_with_capacity;
use crate::compute::{preferred_arrow_data_type, to_arrow};
use crate::{Array, IntoArray};

/// The set of canonical array encodings, also the set of encodings that can be transferred to
/// Arrow with zero-copy.
///
/// Note that a canonical form is not recursive, i.e. a StructArray may contain non-canonical
/// child arrays, which may themselves need to be [canonicalized](IntoCanonical).
///
/// # Logical vs. Physical encodings
///
/// Vortex separates logical and physical types, however this creates ambiguity with Arrow, there is
/// no separation. Thus, if you receive an Arrow array, compress it using Vortex, and then
/// decompress it later to pass to a compute kernel, there are multiple suitable Arrow array
/// variants to hold the data.
///
/// To disambiguate, we choose a canonical physical encoding for every Vortex [`DType`], which
/// will correspond to an arrow-rs [`arrow_schema::DataType`].
///
/// # Views support
///
/// Binary and String views, also known as "German strings" are a better encoding format for
/// nearly all use-cases. Variable-length binary views are part of the Apache Arrow spec, and are
/// fully supported by the Datafusion query engine. We use them as our canonical string encoding
/// for all `Utf8` and `Binary` typed arrays in Vortex.
///
#[derive(Debug, Clone)]
pub enum Canonical {
    Null(NullArray),
    Bool(BoolArray),
    Primitive(PrimitiveArray),
    Struct(StructArray),
    // TODO(joe): maybe this should be a ListView, however this will be annoying in spiral
    List(ListArray),
    VarBinView(VarBinViewArray),
    Extension(ExtensionArray),
}

impl Deref for Canonical {
    type Target = Array;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl Canonical {
    // Create an empty canonical array of the given dtype.
    pub fn empty(dtype: &DType) -> Canonical {
        Self::try_empty(dtype).vortex_expect("Cannot fail to build an empty array")
    }

    pub fn try_empty(dtype: &DType) -> VortexResult<Canonical> {
        builder_with_capacity(dtype, 0).finish()?.into_canonical()
    }
}

// Unwrap canonical type back down to specialized type.
impl Canonical {
    pub fn into_null(self) -> VortexResult<NullArray> {
        match self {
            Canonical::Null(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap NullArray from {:?}", &self),
        }
    }

    pub fn into_bool(self) -> VortexResult<BoolArray> {
        match self {
            Canonical::Bool(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap BoolArray from {:?}", &self),
        }
    }

    pub fn into_primitive(self) -> VortexResult<PrimitiveArray> {
        match self {
            Canonical::Primitive(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap PrimitiveArray from {:?}", &self),
        }
    }

    pub fn into_struct(self) -> VortexResult<StructArray> {
        match self {
            Canonical::Struct(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap StructArray from {:?}", &self),
        }
    }

    pub fn into_list(self) -> VortexResult<ListArray> {
        match self {
            Canonical::List(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap StructArray from {:?}", &self),
        }
    }

    pub fn into_varbinview(self) -> VortexResult<VarBinViewArray> {
        match self {
            Canonical::VarBinView(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap VarBinViewArray from {:?}", &self),
        }
    }

    pub fn into_extension(self) -> VortexResult<ExtensionArray> {
        match self {
            Canonical::Extension(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap ExtensionArray from {:?}", &self),
        }
    }
}

/// Canonicalize an [`Array`] into one of the [`Canonical`] array forms.
///
/// # Invariants
///
/// The DType of the array will be unchanged by canonicalization.
pub trait IntoCanonical {
    /// Canonicalize the array.
    fn into_canonical(self) -> VortexResult<Canonical>;
}

impl<A: IntoArray> IntoCanonical for A {
    fn into_canonical(self) -> VortexResult<Canonical> {
        self.into_array().into_canonical()
    }
}

/// Trait for types that can be converted from an owned type into an owned array variant.
///
/// # Canonicalization
///
/// This trait has a blanket implementation for all types implementing [IntoCanonical].
pub trait IntoArrayVariant {
    fn into_null(self) -> VortexResult<NullArray>;

    fn into_bool(self) -> VortexResult<BoolArray>;

    fn into_primitive(self) -> VortexResult<PrimitiveArray>;

    fn into_struct(self) -> VortexResult<StructArray>;

    fn into_list(self) -> VortexResult<ListArray>;

    fn into_varbinview(self) -> VortexResult<VarBinViewArray>;

    fn into_extension(self) -> VortexResult<ExtensionArray>;
}

impl<T> IntoArrayVariant for T
where
    T: IntoCanonical,
{
    fn into_null(self) -> VortexResult<NullArray> {
        self.into_canonical()?.into_null()
    }

    fn into_bool(self) -> VortexResult<BoolArray> {
        self.into_canonical()?.into_bool()
    }

    fn into_primitive(self) -> VortexResult<PrimitiveArray> {
        self.into_canonical()?.into_primitive()
    }

    fn into_struct(self) -> VortexResult<StructArray> {
        self.into_canonical()?.into_struct()
    }

    fn into_list(self) -> VortexResult<ListArray> {
        self.into_canonical()?.into_list()
    }

    fn into_varbinview(self) -> VortexResult<VarBinViewArray> {
        self.into_canonical()?.into_varbinview()
    }

    fn into_extension(self) -> VortexResult<ExtensionArray> {
        self.into_canonical()?.into_extension()
    }
}

impl IntoCanonical for Array {
    /// Canonicalize an [`Array`] into one of the [`Canonical`] array forms.
    ///
    /// # Invariants
    ///
    /// The DType of the array will be unchanged by canonicalization.
    fn into_canonical(self) -> VortexResult<Canonical> {
        // We only care to know when we canonicalize something non-trivial.
        if !self.is_canonical() && self.len() > 1 {
            log::trace!("Canonicalizing array with encoding {:?}", self.vtable());
        }

        #[cfg(feature = "canonical_counter")]
        self.inc_canonical_counter();

        let canonical = self.vtable().into_canonical(self.clone())?;
        canonical.as_ref().inherit_statistics(self.statistics());

        Ok(canonical)
    }
}

impl IntoArrowArray for Array {
    /// Convert this [`Array`] into an Arrow [`ArrayRef`] by using the array's preferred
    /// Arrow [`DataType`].
    fn into_arrow_preferred(self) -> VortexResult<ArrayRef> {
        let data_type = preferred_arrow_data_type(&self)?;
        self.into_arrow(&data_type)
    }

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrayRef> {
        to_arrow(self, data_type)
    }
}

/// This conversion is always "free" and should not touch underlying data. All it does is create an
/// owned pointer to the underlying concrete array type.
///
/// This combined with the above [IntoCanonical] impl for [Array] allows simple two-way conversions
/// between arbitrary Vortex encodings and canonical Arrow-compatible encodings.
impl From<Canonical> for Array {
    fn from(value: Canonical) -> Self {
        match value {
            Canonical::Null(a) => a.into_array(),
            Canonical::Bool(a) => a.into_array(),
            Canonical::Primitive(a) => a.into_array(),
            Canonical::Struct(a) => a.into_array(),
            Canonical::List(a) => a.into_array(),
            Canonical::VarBinView(a) => a.into_array(),
            Canonical::Extension(a) => a.into_array(),
        }
    }
}

impl AsRef<Array> for Canonical {
    fn as_ref(&self) -> &Array {
        match self {
            Canonical::Null(a) => a.as_ref(),
            Canonical::Bool(a) => a.as_ref(),
            Canonical::Primitive(a) => a.as_ref(),
            Canonical::Struct(a) => a.as_ref(),
            Canonical::List(a) => a.as_ref(),
            Canonical::VarBinView(a) => a.as_ref(),
            Canonical::Extension(a) => a.as_ref(),
        }
    }
}

impl IntoArray for Canonical {
    fn into_array(self) -> Array {
        match self {
            Canonical::Null(a) => a.into_array(),
            Canonical::Bool(a) => a.into_array(),
            Canonical::Primitive(a) => a.into_array(),
            Canonical::Struct(a) => a.into_array(),
            Canonical::List(a) => a.into_array(),
            Canonical::VarBinView(a) => a.into_array(),
            Canonical::Extension(a) => a.into_array(),
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, Int64Type, UInt64Type};
    use arrow_array::{
        Array as ArrowArray, ArrayRef, ListArray as ArrowListArray,
        PrimitiveArray as ArrowPrimitiveArray, StringArray, StringViewArray,
        StructArray as ArrowStructArray,
    };
    use arrow_buffer::{NullBufferBuilder, OffsetBuffer};
    use arrow_schema::{DataType, Field};
    use vortex_buffer::buffer;

    use crate::array::{ConstantArray, StructArray};
    use crate::arrow::{FromArrowArray, IntoArrowArray};
    use crate::{Array, IntoArray};

    #[test]
    fn test_canonicalize_nested_struct() {
        // Create a struct array with multiple internal components.
        let nested_struct_array = StructArray::from_fields(&[
            ("a", buffer![1u64].into_array()),
            (
                "b",
                StructArray::from_fields(&[(
                    "inner_a",
                    // The nested struct contains a ConstantArray representing the primitive array
                    //   [100i64]
                    // ConstantArray is not a canonical type, so converting `into_arrow()` should
                    // map this to the nearest canonical type (PrimitiveArray).
                    ConstantArray::new(100i64, 1).into_array(),
                )])
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap();

        let arrow_struct = nested_struct_array
            .into_array()
            .into_arrow_preferred()
            .unwrap()
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .cloned()
            .unwrap();

        assert!(arrow_struct
            .column(0)
            .as_any()
            .downcast_ref::<ArrowPrimitiveArray<UInt64Type>>()
            .is_some());

        let inner_struct = arrow_struct
            .column(1)
            .clone()
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .cloned()
            .unwrap();

        let inner_a = inner_struct
            .column(0)
            .as_any()
            .downcast_ref::<ArrowPrimitiveArray<Int64Type>>();
        assert!(inner_a.is_some());

        assert_eq!(
            inner_a.cloned().unwrap(),
            ArrowPrimitiveArray::from_iter([100i64]),
        );
    }

    #[test]
    fn roundtrip_struct() {
        let mut nulls = NullBufferBuilder::new(6);
        nulls.append_n_non_nulls(4);
        nulls.append_null();
        nulls.append_non_null();
        let names = Arc::new(StringViewArray::from_iter(vec![
            Some("Joseph"),
            None,
            Some("Angela"),
            Some("Mikhail"),
            None,
            None,
        ]));
        let ages = Arc::new(ArrowPrimitiveArray::<Int32Type>::from(vec![
            Some(25),
            Some(31),
            None,
            Some(57),
            None,
            None,
        ]));

        let arrow_struct = ArrowStructArray::new(
            vec![
                Arc::new(Field::new("name", DataType::Utf8View, true)),
                Arc::new(Field::new("age", DataType::Int32, true)),
            ]
            .into(),
            vec![names, ages],
            nulls.finish(),
        );

        let vortex_struct = Array::from_arrow(&arrow_struct, true);

        assert_eq!(
            &arrow_struct,
            vortex_struct.into_arrow_preferred().unwrap().as_struct()
        );
    }

    #[test]
    fn roundtrip_list() {
        let names = Arc::new(StringArray::from_iter(vec![
            Some("Joseph"),
            Some("Angela"),
            Some("Mikhail"),
        ]));

        let arrow_list = ArrowListArray::new(
            Arc::new(Field::new_list_field(DataType::Utf8, true)),
            OffsetBuffer::from_lengths(vec![0, 2, 1]),
            names,
            None,
        );
        let list_data_type = arrow_list.data_type();

        let vortex_list = Array::from_arrow(&arrow_list, true);

        let rt_arrow_list = vortex_list.into_arrow(list_data_type).unwrap();

        assert_eq!(
            (Arc::new(arrow_list.clone()) as ArrayRef).as_ref(),
            rt_arrow_list.as_ref()
        );
    }
}
