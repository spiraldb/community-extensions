//! All the built-in encoding schemes and arrays.

#[cfg(test)]
mod assertions;

mod bool;
mod chunked;
mod constant;
mod datetime;
mod extension;
mod list;
mod null;
mod primitive;
mod struct_;
mod varbin;
mod varbinview;

pub mod from;

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
//#[cfg(test)]
//mod test_compatibility;

pub use self::bool::*;
pub use self::chunked::*;
pub use self::constant::*;
pub use self::datetime::*;
pub use self::extension::*;
pub use self::list::*;
pub use self::null::*;
pub use self::primitive::*;
pub use self::struct_::*;
pub use self::varbin::*;
pub use self::varbinview::*;
