pub mod inline;

#[cfg(feature = "tokio")]
pub mod tokio;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use vortex_array::Array;
use vortex_error::VortexResult;

/// An execution driver is used to drive the execution of the scan operation.
///
/// It is passed a stream of futures that (typically) process a single split of the file.
/// Drivers are able to control the concurrency of the execution with [`futures::stream::buffered`],
/// as well as _where_ the futures are executed by spawning them onto a specific runtime or thread
/// pool.
///
/// Note that the futures encapsulate heavy CPU code such as filtering and decompression. To
/// offload keep I/O work separate, please see the [`crate::io::IoDriver`] trait.
pub trait ExecDriver: Send + Sync {
    fn drive(
        &self,
        stream: BoxStream<'static, BoxFuture<'static, VortexResult<Option<Array>>>>,
    ) -> BoxStream<'static, VortexResult<Array>>;
}
