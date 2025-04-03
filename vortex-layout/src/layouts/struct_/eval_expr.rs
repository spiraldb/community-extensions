use std::ops::{BitAnd, Range};
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use futures::stream::FuturesOrdered;
use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult};
use vortex_expr::ExprRef;
use vortex_expr::transform::partition::PartitionedExpr;
use vortex_mask::Mask;

use crate::layouts::struct_::reader::StructReader;
use crate::{
    ArrayEvaluation, ExprEvaluator, Layout, LayoutReader, MaskEvaluation, NoOpPruningEvaluation,
    PruningEvaluation,
};

impl ExprEvaluator for StructReader {
    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        // Partition the expression into expressions that can be evaluated over individual fields
        let partitioned = self.partition_expr(expr.clone());

        if partitioned.partition_names.len() == 1 {
            return self
                .child(&partitioned.partition_names[0])?
                .pruning_evaluation(row_range, &partitioned.partitions[0]);
        }

        // TODO(ngates): if all partitions are boolean, we can use a pruning evaluation. Otherwise
        //  there's not much we can do? Maybe... it's complicated...
        Ok(Box::new(NoOpPruningEvaluation))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        // Partition the expression into expressions that can be evaluated over individual fields
        let partitioned = self.partition_expr(expr.clone());

        // Short-circuit if there is only one partition
        if partitioned.partition_names.len() == 1 {
            return self
                .child(&partitioned.partition_names[0])?
                .filter_evaluation(row_range, &partitioned.partitions[0]);
        }

        // TODO(ngates): for any partition that returns a boolean, we can use a mask evaluation.

        // Construct evaluations for each child.
        let field_evals: Vec<_> = partitioned
            .partition_names
            .iter()
            .zip_eq(partitioned.partitions.iter())
            .zip_eq(partitioned.partition_dtypes.iter())
            .map(|((name, expr), dtype)| {
                let reader = self.child(name)?;
                Ok::<_, VortexError>(if matches!(dtype, DType::Bool(_)) {
                    // If the partition evaluates to a boolean, we can evaluate it as a mask which
                    // can often be more efficient since nulls are turned into `false` early on,
                    // and layouts can perform predicate pruning / indexing.
                    FieldEval::Mask(reader.filter_evaluation(row_range, expr)?)
                } else {
                    // Otherwise, we evaluate the projection as an array, and combine the results
                    // at the end.
                    FieldEval::Array(reader.projection_evaluation(row_range, expr)?)
                })
            })
            .try_collect()?;

        Ok(Box::new(StructMaskEvaluation {
            partitioned,
            field_evals,
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        // Partition the expression into expressions that can be evaluated over individual fields
        let partitioned = self.partition_expr(expr.clone());

        // Short-circuit if there is only one partition
        if partitioned.partition_names.len() == 1 {
            return self
                .child(&partitioned.partition_names[0])?
                .projection_evaluation(row_range, &partitioned.partitions[0]);
        }

        // Construct evaluations for each child.
        let field_evals: Vec<_> = partitioned
            .partition_names
            .iter()
            .zip_eq(partitioned.partitions.iter())
            .map(|(name, expr)| self.child(name)?.projection_evaluation(row_range, expr))
            .try_collect()?;

        Ok(Box::new(StructArrayEvaluation {
            layout: self.layout().clone(),
            partitioned,
            field_evals,
        }))
    }
}

struct StructMaskEvaluation {
    partitioned: Arc<PartitionedExpr>,
    field_evals: Vec<FieldEval>,
}

enum FieldEval {
    Mask(Box<dyn MaskEvaluation>),
    Array(Box<dyn ArrayEvaluation>),
}

#[async_trait]
impl MaskEvaluation for StructMaskEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        // TODO(ngates): ideally we'd spawn these so the CPU can be utilized more effectively.
        let field_arrays: Vec<_> = FuturesOrdered::from_iter(self.field_evals.iter().map(|eval| {
            let mask = mask.clone();
            async move {
                match eval {
                    FieldEval::Mask(eval) => Ok(eval.invoke(mask.clone()).await?.into_array()),
                    FieldEval::Array(eval) => eval.invoke(Mask::new_true(mask.len())).await,
                }
            }
        }))
        .try_collect()
        .await?;

        let root_scope = StructArray::try_new(
            self.partitioned.partition_names.clone(),
            field_arrays,
            mask.len(),
            Validity::NonNullable,
        )?
        .into_array();

        let root_mask = Mask::try_from(self.partitioned.root.evaluate(&root_scope)?.as_ref())?;
        let mask = mask.bitand(&root_mask);

        Ok(mask)
    }
}

struct StructArrayEvaluation {
    layout: Layout,
    partitioned: Arc<PartitionedExpr>,
    field_evals: Vec<Box<dyn ArrayEvaluation>>,
}

#[async_trait]
impl ArrayEvaluation for StructArrayEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        log::debug!(
            "Struct array evaluation {} - {} (mask = {})",
            self.layout.name(),
            self.partitioned,
            mask.density()
        );

        let field_arrays: Vec<_> = FuturesOrdered::from_iter(
            self.field_evals
                .iter()
                .map(|eval| eval.invoke(mask.clone())),
        )
        .try_collect()
        .await?;

        let root_scope = StructArray::try_new(
            self.partitioned.partition_names.clone(),
            field_arrays,
            mask.true_count(),
            Validity::NonNullable,
        )?
        .into_array();

        self.partitioned.root.evaluate(&root_scope)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use rstest::{fixture, rstest};
    use vortex_array::arrays::StructArray;
    use vortex_array::{Array, ArrayContext, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability, StructDType};
    use vortex_error::VortexUnwrap;
    use vortex_expr::{get_item, gt, ident, pack};
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::layouts::struct_::writer::StructLayoutWriter;
    use crate::segments::{SegmentSource, TestSegments};
    use crate::writer::LayoutWriterExt;
    use crate::{ExprEvaluator, Layout};

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn struct_layout() -> (ArrayContext, Arc<dyn SegmentSource>, Layout) {
        let ctx = ArrayContext::empty();
        let mut segments = TestSegments::default();

        let layout = StructLayoutWriter::try_new(
            DType::Struct(
                Arc::new(StructDType::new(
                    vec!["a".into(), "b".into(), "c".into()].into(),
                    vec![I32.into(), I32.into(), I32.into()],
                )),
                Nullability::NonNullable,
            ),
            vec![
                Box::new(FlatLayoutWriter::new(
                    ctx.clone(),
                    I32.into(),
                    Default::default(),
                )),
                Box::new(FlatLayoutWriter::new(
                    ctx.clone(),
                    I32.into(),
                    Default::default(),
                )),
                Box::new(FlatLayoutWriter::new(
                    ctx.clone(),
                    I32.into(),
                    Default::default(),
                )),
            ],
        )
        .vortex_unwrap()
        .push_all(
            &mut segments,
            [Ok(StructArray::from_fields(
                [
                    ("a", buffer![7, 2, 3].into_array()),
                    ("b", buffer![4, 5, 6].into_array()),
                    ("c", buffer![4, 5, 6].into_array()),
                ]
                .as_slice(),
            )
            .unwrap()
            .into_array())],
        )
        .unwrap();
        (ctx, Arc::new(segments), layout)
    }

    #[rstest]
    fn test_struct_layout(
        #[from(struct_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn SegmentSource>,
            Layout,
        ),
    ) {
        let reader = layout.reader(&segments, &ctx).unwrap();
        let expr = gt(get_item("a", ident()), get_item("b", ident()));
        let result = block_on(
            reader
                .projection_evaluation(&(0..3), &expr)
                .unwrap()
                .invoke(Mask::new_true(3)),
        )
        .unwrap();
        assert_eq!(
            vec![true, false, false],
            result
                .to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn test_struct_layout_row_mask(
        #[from(struct_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn SegmentSource>,
            Layout,
        ),
    ) {
        let reader = layout.reader(&segments, &ctx).unwrap();
        let expr = gt(get_item("a", ident()), get_item("b", ident()));
        let result = block_on(
            reader
                .projection_evaluation(&(0..3), &expr)
                .unwrap()
                .invoke(Mask::from_iter([true, true, false])),
        )
        .unwrap();

        assert_eq!(result.len(), 2);

        assert_eq!(
            vec![true, false],
            result
                .to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn test_struct_layout_select(
        #[from(struct_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn SegmentSource>,
            Layout,
        ),
    ) {
        let reader = layout.reader(&segments, &ctx).unwrap();
        let expr = pack([("a", get_item("a", ident())), ("b", get_item("b", ident()))]);
        let result = block_on(
            reader
                .projection_evaluation(&(0..3), &expr)
                .unwrap()
                // Take rows 0 and 1, skip row 2, and anything after that
                .invoke(Mask::from_iter([true, true, false])),
        )
        .unwrap();

        assert_eq!(result.len(), 2);

        assert_eq!(
            result
                .as_struct_typed()
                .unwrap()
                .maybe_null_field_by_name("a")
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [7, 2].as_slice()
        );

        assert_eq!(
            result
                .as_struct_typed()
                .unwrap()
                .maybe_null_field_by_name("b")
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [4, 5].as_slice()
        );
    }
}
