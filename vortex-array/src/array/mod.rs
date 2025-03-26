mod canonical;
mod compute;
mod convert;
mod implementation;
mod statistics;
mod validity;
mod variants;
mod visitor;

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

pub use canonical::*;
pub use compute::*;
pub use convert::*;
pub use implementation::*;
pub use statistics::*;
pub use validity::*;
pub use variants::*;
pub use visitor::*;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::arrays::{
    BoolEncoding, ExtensionEncoding, ListEncoding, NullEncoding, PrimitiveEncoding, StructEncoding,
    VarBinEncoding, VarBinViewEncoding,
};
use crate::builders::ArrayBuilder;
use crate::compute::{ComputeFn, KernelRef};
use crate::stats::StatsSetRef;
use crate::vtable::{EncodingVTable, VTableRef};
use crate::{Canonical, EncodingId};

/// The base trait for all Vortex arrays.
///
/// Users should invoke functions on this trait. Implementations should implement the corresponding
/// function on the `_Impl` traits, e.g. [`ArrayValidityImpl`]. The functions here dispatch to the
/// implementations, while validating pre- and post-conditions.
pub trait Array: Send + Sync + Debug + ArrayStatistics + ArrayVariants + ArrayVisitor {
    /// Returns the array as a reference to a generic [`Any`] trait object.
    fn as_any(&self) -> &dyn Any;

    /// Returns the array as an [`Arc`] reference to a generic [`Any`] trait object.
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Returns the array as an [`ArrayRef`].
    fn to_array(&self) -> ArrayRef;

    /// Converts the array into an [`ArrayRef`].
    fn into_array(self) -> ArrayRef
    where
        Self: Sized;

    /// Returns the length of the array.
    fn len(&self) -> usize;

    /// Returns whether the array is empty (has zero rows).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the logical Vortex [`DType`] of the array.
    fn dtype(&self) -> &DType;

    /// Returns the encoding of the array.
    fn encoding(&self) -> EncodingId;

    /// Returns the encoding VTable.
    fn vtable(&self) -> VTableRef;

    /// Attempts to find a kernel for the given compute invocation.
    fn find_kernel(&self, compute_fn: &dyn ComputeFn) -> Option<KernelRef>;

    /// Returns whether the array is of the given encoding.
    fn is_encoding(&self, encoding: EncodingId) -> bool {
        self.encoding() == encoding
    }

    /// Returns whether this array is an arrow encoding.
    // TODO(ngates): this shouldn't live here.
    fn is_arrow(&self) -> bool {
        self.is_encoding(NullEncoding.id())
            || self.is_encoding(BoolEncoding.id())
            || self.is_encoding(PrimitiveEncoding.id())
            || self.is_encoding(VarBinEncoding.id())
            || self.is_encoding(VarBinViewEncoding.id())
    }

    /// Whether the array is of a canonical encoding.
    // TODO(ngates): this shouldn't live here.
    fn is_canonical(&self) -> bool {
        self.is_encoding(NullEncoding.id())
            || self.is_encoding(BoolEncoding.id())
            || self.is_encoding(PrimitiveEncoding.id())
            || self.is_encoding(StructEncoding.id())
            || self.is_encoding(ListEncoding.id())
            || self.is_encoding(VarBinViewEncoding.id())
            || self.is_encoding(ExtensionEncoding.id())
    }

    /// Returns whether the item at `index` is valid.
    fn is_valid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether the item at `index` is invalid.
    fn is_invalid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether all items in the array are valid.
    ///
    /// This is usually cheaper than computing a precise `valid_count`.
    fn all_valid(&self) -> VortexResult<bool>;

    /// Returns whether the array is all invalid.
    ///
    /// This is usually cheaper than computing a precise `invalid_count`.
    fn all_invalid(&self) -> VortexResult<bool>;

    /// Returns the number of valid elements in the array.
    fn valid_count(&self) -> VortexResult<usize>;

    /// Returns the number of invalid elements in the array.
    fn invalid_count(&self) -> VortexResult<usize>;

    /// Returns the canonical validity mask for the array.
    fn validity_mask(&self) -> VortexResult<Mask>;

    /// Returns the canonical representation of the array.
    fn to_canonical(&self) -> VortexResult<Canonical>;

    /// Writes the array into the canonical builder.
    ///
    /// The [`DType`] of the builder must match that of the array.
    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()>;

    /// Returns the statistics of the array.
    // TODO(ngates): change how this works. It's weird.
    fn statistics(&self) -> StatsSetRef<'_>;

    /// Replaces the children of the array with the given array references.
    fn with_children(&self, children: &[ArrayRef]) -> VortexResult<ArrayRef>;
}

impl Array for Arc<dyn Array> {
    fn as_any(&self) -> &dyn Any {
        self.as_ref().as_any()
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_array(&self) -> ArrayRef {
        self.clone()
    }

    fn into_array(self) -> ArrayRef {
        self
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn dtype(&self) -> &DType {
        self.as_ref().dtype()
    }

    fn encoding(&self) -> EncodingId {
        self.as_ref().encoding()
    }

    fn vtable(&self) -> VTableRef {
        self.as_ref().vtable()
    }

    fn find_kernel(&self, compute_fn: &dyn ComputeFn) -> Option<KernelRef> {
        self.as_ref().find_kernel(compute_fn)
    }

    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        self.as_ref().is_valid(index)
    }

    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        self.as_ref().is_invalid(index)
    }

    fn all_valid(&self) -> VortexResult<bool> {
        self.as_ref().all_valid()
    }

    fn all_invalid(&self) -> VortexResult<bool> {
        self.as_ref().all_invalid()
    }

    fn valid_count(&self) -> VortexResult<usize> {
        self.as_ref().valid_count()
    }

    fn invalid_count(&self) -> VortexResult<usize> {
        self.as_ref().invalid_count()
    }

    fn validity_mask(&self) -> VortexResult<Mask> {
        self.as_ref().validity_mask()
    }

    fn to_canonical(&self) -> VortexResult<Canonical> {
        self.as_ref().to_canonical()
    }

    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        self.as_ref().append_to_builder(builder)
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        self.as_ref().statistics()
    }

    fn with_children(&self, children: &[ArrayRef]) -> VortexResult<ArrayRef> {
        self.as_ref().with_children(children)
    }
}

/// A reference counted pointer to a dynamic [`Array`] trait object.
pub type ArrayRef = Arc<dyn Array>;

impl ToOwned for dyn Array {
    type Owned = ArrayRef;

    fn to_owned(&self) -> Self::Owned {
        self.to_array()
    }
}

impl<A: Array + Clone + 'static> TryFromArrayRef for A {
    fn try_from_array(array: ArrayRef) -> Result<Self, ArrayRef> {
        let fallback = array.clone();
        if let Ok(array) = array.as_any_arc().downcast::<A>() {
            // manually drop the fallback value so `Arc::unwrap_or_clone` doesn't always have to clone
            drop(fallback);
            Ok(Arc::unwrap_or_clone(array))
        } else {
            Err(fallback)
        }
    }
}

impl<A: Array + Clone + 'static> TryFromArrayRef for Arc<A> {
    fn try_from_array(array: ArrayRef) -> Result<Self, ArrayRef> {
        let fallback = array.clone();
        array.as_any_arc().downcast::<A>().map_err(|_| fallback)
    }
}

pub trait ArrayExt: Array {
    /// Returns the array downcast to the given `A`.
    fn as_<A: Array + 'static>(&self) -> &A {
        self.as_any()
            .downcast_ref::<A>()
            .vortex_expect("Failed to downcast")
    }

    /// Returns the array downcast to the given `A`.
    fn as_opt<A: Array + 'static>(&self) -> Option<&A> {
        self.as_any().downcast_ref::<A>()
    }

    /// Is self an array with encoding `A`.
    fn is<A: Array + 'static>(&self) -> bool {
        self.as_opt::<A>().is_some()
    }
}

impl<A: Array + ?Sized> ArrayExt for A {}

impl Display for dyn Array {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({}, len={})",
            self.encoding(),
            self.dtype(),
            self.len()
        )
    }
}

#[macro_export]
macro_rules! try_from_array_ref {
    ($Array:ty) => {
        impl TryFrom<$crate::ArrayRef> for $Array {
            type Error = vortex_error::VortexError;

            fn try_from(value: $crate::ArrayRef) -> Result<Self, Self::Error> {
                Ok(::std::sync::Arc::unwrap_or_clone(
                    value.as_any_arc().downcast::<Self>().map_err(|_| {
                        vortex_error::vortex_err!(
                            "Cannot downcast to {}",
                            std::any::type_name::<Self>()
                        )
                    })?,
                ))
            }
        }
    };
}
