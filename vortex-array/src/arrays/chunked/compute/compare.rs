use vortex_error::VortexResult;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::builders::{ArrayBuilder, BoolBuilder};
use crate::compute::{CompareFn, Operator, compare, slice};
use crate::{Array, ArrayRef};

impl CompareFn<&ChunkedArray> for ChunkedEncoding {
    fn compare(
        &self,
        lhs: &ChunkedArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut idx = 0;

        let mut bool_builder = BoolBuilder::with_capacity(
            // nullable <= non-nullable
            (lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into(),
            lhs.len(),
        );

        for chunk in lhs.non_empty_chunks() {
            let sliced = slice(rhs, idx, idx + chunk.len())?;
            let cmp_result = compare(chunk, &sliced, operator)?;

            bool_builder.extend_from_array(&cmp_result)?;
            idx += chunk.len();
        }

        Ok(Some(bool_builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::PrimitiveArray;

    #[test]
    fn empty_compare() {
        let base = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        let chunked =
            ChunkedArray::try_new(vec![base.clone(), base.clone()], base.dtype().clone()).unwrap();
        let chunked_empty = ChunkedArray::try_new(vec![], base.dtype().clone()).unwrap();

        let r = compare(&chunked, &chunked_empty, Operator::Eq).unwrap();

        assert!(r.is_empty());
    }
}
