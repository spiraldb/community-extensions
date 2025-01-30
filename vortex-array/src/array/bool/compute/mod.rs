use crate::array::BoolEncoding;
use crate::compute::{
    BinaryBooleanFn, FillForwardFn, FillNullFn, FilterFn, InvertFn, ScalarAtFn, SliceFn, TakeFn,
    ToArrowFn,
};
use crate::vtable::ComputeVTable;
use crate::Array;

mod fill_forward;
mod fill_null;
pub mod filter;
mod flatten;
mod invert;
mod scalar_at;
mod slice;
mod take;
mod to_arrow;

impl ComputeVTable for BoolEncoding {
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<Array>> {
        // We only implement this when other is a constant value, otherwise we fall back to the
        // default implementation that canonicalizes to Arrow.
        // TODO(ngates): implement this for constants.
        // other.is_constant().then_some(self)
        None
    }

    fn fill_forward_fn(&self) -> Option<&dyn FillForwardFn<Array>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<Array>> {
        Some(self)
    }
}
