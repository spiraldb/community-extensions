//! Core traits and implementations for asynchronous IO.
//!
//! Vortex implements an IPC streaming format as well as a file format, both of which
//! run on top of a variety of storage systems that can be accessed from multiple async
//! runtimes.
//!
//! This crate provides core traits for positioned and streaming IO, and via feature
//! flags implements the core traits for several common async runtimes and backing stores.

pub use dispatcher::*;
pub use io_buf::*;
pub use limit::*;
#[cfg(feature = "object_store")]
pub use object_store::*;
pub use read::*;
#[cfg(feature = "tokio")]
pub use tokio::*;
pub use write::*;

#[cfg(feature = "compio")]
mod compio;
mod dispatcher;
mod io_buf;
mod limit;
#[cfg(feature = "object_store")]
mod object_store;
mod read;
#[cfg(feature = "tokio")]
mod tokio;
mod write;

/// Required alignment for all custom buffer allocations.
pub const ALIGNMENT: usize = 64;
