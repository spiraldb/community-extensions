use std::fmt::Display;
use std::ops::{Add, Div, Mul, Rem, Sub};

use num_traits::{CheckedAdd, CheckedSub, ConstZero, One, Zero};

/// Signed 256-bit integer type.
///
/// This one of the physical representations of `DecimalScalar` values and can be safely converted
/// back and forth with Arrow's [`i256`][arrow_buffer::i256].
#[repr(transparent)]
#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct i256(arrow_buffer::i256);

impl i256 {
    pub const ZERO: Self = Self(arrow_buffer::i256::ZERO);
    pub const ONE: Self = Self(arrow_buffer::i256::ONE);

    /// Construct a new `i256` from an unsigned `lower` bits and a signed `upper` bits.
    pub fn from_parts(lower: u128, upper: i128) -> Self {
        Self(arrow_buffer::i256::from_parts(lower, upper))
    }

    /// Create an `i256` value from a signed 128-bit value.
    pub fn from_i128(i: i128) -> Self {
        Self(arrow_buffer::i256::from_i128(i))
    }

    /// Create an integer value from its representation as a byte array in little-endian.
    pub fn from_le_bytes(bytes: [u8; 32]) -> Self {
        Self(arrow_buffer::i256::from_le_bytes(bytes))
    }

    /// Return the memory representation of this integer as a byte array in little-endian byte order.
    pub fn to_le_bytes(&self) -> [u8; 32] {
        self.0.to_le_bytes()
    }

    /// Split the 256-bit signed integer value into an unsigned lower bits and a signed upper bits.
    ///
    /// This versions gives us ownership of the value.
    pub fn into_parts(self) -> (u128, i128) {
        self.0.to_parts()
    }

    /// Split the 256-bit signed integer value into an unsigned lower bits and a signed upper bits.
    pub fn to_parts(&self) -> (u128, i128) {
        self.0.to_parts()
    }

    pub fn wrapping_pow(&self, exp: u32) -> Self {
        Self(self.0.wrapping_pow(exp))
    }
}

impl From<i256> for arrow_buffer::i256 {
    fn from(i: i256) -> Self {
        i.0
    }
}

impl From<arrow_buffer::i256> for i256 {
    fn from(i: arrow_buffer::i256) -> Self {
        Self(i)
    }
}

impl Display for i256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for i256 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}

impl Sub for i256 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}

impl Mul<Self> for i256 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl Div<Self> for i256 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.div(rhs.0))
    }
}

impl Rem<Self> for i256 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0.rem(rhs.0))
    }
}

impl Zero for i256 {
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl ConstZero for i256 {
    const ZERO: Self = Self(arrow_buffer::i256::ZERO);
}

impl One for i256 {
    fn one() -> Self {
        Self(arrow_buffer::i256::ONE)
    }
}

impl CheckedAdd for i256 {
    fn checked_add(&self, v: &Self) -> Option<Self> {
        self.0.checked_add(v.0).map(Self)
    }
}

impl CheckedSub for i256 {
    fn checked_sub(&self, v: &Self) -> Option<Self> {
        self.0.checked_sub(v.0).map(Self)
    }
}
