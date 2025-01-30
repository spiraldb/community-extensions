use std::sync::RwLock;

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::stats::StatsSet;
use crate::vtable::VTableRef;
use crate::Array;

/// Owned [`Array`] with serialized metadata, backed by heap-allocated memory.
#[derive(Debug)]
pub(super) struct OwnedArray {
    pub(super) encoding: VTableRef,
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) metadata: Option<ByteBuffer>,
    pub(super) buffers: Option<Box<[ByteBuffer]>>,
    pub(super) children: Option<Box<[Array]>>,
    pub(super) stats_set: RwLock<StatsSet>,
    #[cfg(feature = "canonical_counter")]
    pub(super) canonical_counter: std::sync::atomic::AtomicUsize,
}

impl OwnedArray {
    pub fn byte_buffer(&self, index: usize) -> Option<&ByteBuffer> {
        self.buffers.as_ref().and_then(|b| b.get(index))
    }

    // We want to allow these panics because they are indicative of implementation error.
    #[allow(clippy::panic_in_result_fn)]
    pub fn child(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<&Array> {
        match self.children.as_ref().and_then(|c| c.get(index)) {
            None => vortex_bail!(
                "Array::child({}): child {index} not found",
                self.encoding.id().as_ref()
            ),
            Some(child) => {
                assert_eq!(
                    child.dtype(),
                    dtype,
                    "child {index} requested with incorrect dtype for encoding {}",
                    self.encoding.id().as_ref(),
                );
                assert_eq!(
                    child.len(),
                    len,
                    "child {index} requested with incorrect length for encoding {}",
                    self.encoding.id().as_ref(),
                );
                Ok(child)
            }
        }
    }

    pub fn nchildren(&self) -> usize {
        self.children.as_ref().map_or(0, |c| c.len())
    }
}
