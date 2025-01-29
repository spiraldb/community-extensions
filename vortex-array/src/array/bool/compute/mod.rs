use crate::array::BoolEncoding;
use crate::compute::{
    BinaryBooleanFn, FillForwardFn, FillNullFn, FilterFn, InvertFn, ScalarAtFn, SliceFn, TakeFn,
};
use crate::vtable::ComputeVTable;
use crate::ArrayData;

mod fill_forward;
mod fill_null;
pub mod filter;
mod flatten;
mod invert;
mod scalar_at;
mod slice;
mod take;

impl ComputeVTable for BoolEncoding {
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<ArrayData>> {
        // We only implement this when other is a constant value, otherwise we fall back to the
        // default implementation that canonicalizes to Arrow.
        // TODO(ngates): implement this for constants.
        // other.is_constant().then_some(self)
        None
    }

    fn fill_forward_fn(&self) -> Option<&dyn FillForwardFn<ArrayData>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<ArrayData>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}
