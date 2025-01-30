use vortex_array::array::ConstantArray;
use vortex_array::compute::{compare, take, CompareFn, Operator};
use vortex_array::Array;
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl CompareFn<DictArray> for DictEncoding {
    fn compare(
        &self,
        lhs: &DictArray,
        rhs: &Array,
        operator: Operator,
    ) -> VortexResult<Option<Array>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(const_scalar) = rhs.as_constant() {
            // Ensure the other is the same length as the dictionary
            let compare_result = compare(
                lhs.values(),
                ConstantArray::new(const_scalar, lhs.values().len()),
                operator,
            )?;
            return take(compare_result, lhs.codes()).map(Some);
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        Ok(None)
    }
}
