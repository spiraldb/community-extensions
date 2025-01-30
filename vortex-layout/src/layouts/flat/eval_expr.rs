use async_trait::async_trait;
use flatbuffers::root;
use futures::future::try_join_all;
use vortex_array::compute::{filter, slice};
use vortex_array::parts::ArrayParts;
use vortex_array::Array;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_flatbuffers::{array as fba, FlatBuffer};
use vortex_scan::RowMask;

use crate::layouts::flat::reader::FlatReader;
use crate::reader::LayoutReaderExt;
use crate::{ExprEvaluator, LayoutReader};

#[async_trait]
impl ExprEvaluator for FlatReader {
    async fn evaluate_expr(self: &Self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array> {
        assert!(row_mask.true_count() > 0);

        // Fetch all the array buffers.
        let mut buffers = try_join_all(
            self.layout()
                .segments()
                .map(|segment_id| self.segments().get(segment_id)),
        )
        .await?;

        // Pop the array flatbuffer.
        let flatbuffer = FlatBuffer::try_from(
            buffers
                .pop()
                .ok_or_else(|| vortex_err!("Flat message missing"))?,
        )?;

        let row_count = usize::try_from(self.layout().row_count())
            .vortex_expect("FlatLayout row count does not fit within usize");

        let array_parts = ArrayParts::new(
            row_count,
            root::<fba::Array>(flatbuffer.as_ref()).vortex_expect("Invalid fba::Array flatbuffer"),
            flatbuffer.clone(),
            buffers,
        );

        // Decode into an Array.
        let array = array_parts.decode(self.ctx(), self.dtype().clone())?;
        assert_eq!(
            array.len() as u64,
            self.row_count(),
            "FlatLayout array length mismatch {} != {}",
            array.len(),
            self.row_count()
        );

        // TODO(ngates): what's the best order to apply the filter mask / expression?

        // Filter the array based on the row mask.
        let begin = usize::try_from(row_mask.begin())
            .vortex_expect("RowMask begin must fit within FlatLayout size");
        let array = slice(array, begin, begin + row_mask.len())?;
        let array = filter(&array, row_mask.filter_mask())?;

        // Then apply the expression
        expr.evaluate(&array)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_buffer::BooleanBuffer;
    use futures::executor::block_on;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_expr::{gt, ident, lit, Identity};
    use vortex_scan::RowMask;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;

    #[test]
    fn flat_identity() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutWriter::new(array.dtype().clone(), Default::default())
                .push_one(&mut segments, array.clone().into_array())
                .unwrap();

            let result = layout
                .reader(Arc::new(segments), Default::default())
                .unwrap()
                .evaluate_expr(
                    RowMask::new_valid_between(0, layout.row_count()),
                    Identity::new_expr(),
                )
                .await
                .unwrap()
                .into_primitive()
                .unwrap();

            assert_eq!(array.as_slice::<i32>(), result.as_slice::<i32>());
        })
    }

    #[test]
    fn flat_expr() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutWriter::new(array.dtype().clone(), Default::default())
                .push_one(&mut segments, array.into_array())
                .unwrap();

            let expr = gt(Identity::new_expr(), lit(3i32));
            let result = layout
                .reader(Arc::new(segments), Default::default())
                .unwrap()
                .evaluate_expr(RowMask::new_valid_between(0, layout.row_count()), expr)
                .await
                .unwrap()
                .into_bool()
                .unwrap();

            assert_eq!(
                BooleanBuffer::from_iter([false, false, false, true, true]),
                result.boolean_buffer()
            );
        })
    }

    #[test]
    fn flat_unaligned_row_mask() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutWriter::new(array.dtype().clone(), Default::default())
                .push_one(&mut segments, array.clone().into_array())
                .unwrap();

            let result = layout
                .reader(Arc::new(segments), Default::default())
                .unwrap()
                .evaluate_expr(RowMask::new_valid_between(2, 4), ident())
                .await
                .unwrap()
                .into_primitive()
                .unwrap();

            assert_eq!(result.as_slice::<i32>(), &[3, 4],);
        })
    }
}
