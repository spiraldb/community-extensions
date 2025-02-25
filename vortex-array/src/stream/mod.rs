use std::pin::Pin;

pub use adapter::*;
pub use ext::*;
use futures_util::{Stream, stream};
pub use take_rows::*;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::iter::ArrayIteratorArrayExt;
use crate::{Array, ArrayRef};

mod adapter;
mod ext;
mod take_rows;

/// A stream of array chunks along with a DType.
///
/// Can be thought of as equivalent to Arrow's RecordBatchReader.
pub trait ArrayStream: Stream<Item = VortexResult<ArrayRef>> {
    fn dtype(&self) -> &DType;
}

/// Trait for a [`Stream`] of [`ArrayRef`]s that can be passed between threads.
pub type SendableArrayStream = Pin<Box<dyn ArrayStream + Send>>;

pub trait ArrayStreamArrayExt: Array {
    /// Create an [`ArrayStream`] over the array.
    fn to_array_stream(&self) -> impl ArrayStream + 'static {
        ArrayStreamAdapter::new(self.dtype().clone(), stream::iter(self.to_array_iterator()))
    }
}

impl<A: ?Sized + Array> ArrayStreamArrayExt for A {}
