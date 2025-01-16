//! Provides types that can be used by I/O frameworks to work with byte buffer-shaped data.

use std::ops::Range;

use bytes::Bytes;
use vortex_buffer::{Buffer, ConstByteBuffer};

/// Trait for types that can provide a readonly byte buffer interface to I/O frameworks.
///
/// # Safety
/// The type must support contiguous raw memory access via pointer, such as `Vec` or `[u8]`.
pub unsafe trait IoBuf: Unpin + 'static {
    /// Returns a raw pointer to the vector’s buffer.
    fn read_ptr(&self) -> *const u8;

    /// Number of initialized bytes.
    fn bytes_init(&self) -> usize;

    /// Access the buffer as a byte slice
    fn as_slice(&self) -> &[u8];

    /// Access the buffer as a byte slice with begin and end indices
    #[inline]
    fn slice_owned(self, range: Range<usize>) -> OwnedSlice<Self>
    where
        Self: Sized,
    {
        OwnedSlice {
            buf: self,
            begin: range.start,
            end: range.end,
        }
    }
}

/// An owned view into a contiguous sequence of bytes.
pub struct OwnedSlice<T> {
    buf: T,
    begin: usize,
    end: usize,
}

impl<T> OwnedSlice<T> {
    /// Unwrap the slice into its underlying type.
    pub fn into_inner(self) -> T {
        self.buf
    }
}

unsafe impl IoBuf for &'static [u8] {
    #[inline]
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    #[inline]
    fn bytes_init(&self) -> usize {
        self.len()
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        self
    }
}

unsafe impl<const N: usize> IoBuf for [u8; N] {
    #[inline]
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    #[inline]
    fn bytes_init(&self) -> usize {
        N
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

unsafe impl IoBuf for Vec<u8> {
    #[inline]
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    #[inline]
    fn bytes_init(&self) -> usize {
        self.len()
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

unsafe impl<T: IoBuf> IoBuf for OwnedSlice<T> {
    #[inline]
    fn read_ptr(&self) -> *const u8 {
        unsafe { self.buf.read_ptr().add(self.begin) }
    }

    #[inline]
    fn bytes_init(&self) -> usize {
        self.end - self.begin
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.read_ptr(), self.bytes_init()) }
    }
}

unsafe impl IoBuf for Bytes {
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len()
    }

    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

unsafe impl<const A: usize> IoBuf for ConstByteBuffer<A> {
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len()
    }

    fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }
}

unsafe impl<T: Unpin + 'static> IoBuf for Buffer<T> {
    fn read_ptr(&self) -> *const u8 {
        self.as_ptr().cast()
    }

    fn bytes_init(&self) -> usize {
        self.len() * size_of::<T>()
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.read_ptr(), self.bytes_init()) }
    }
}
