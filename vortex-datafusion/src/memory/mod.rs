//! In-memory implementation of a Vortex table provider.
mod exec;
mod plans;
mod provider;
mod statistics;
mod stream;

pub use provider::VortexMemTable;
