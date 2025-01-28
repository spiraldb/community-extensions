#![cfg(feature = "datafusion")]

use std::sync::Arc;

use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::{expressions, PhysicalExpr};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::{get_item, ident, lit, BinaryExpr, ExprRef, Like, Operator};

// TODO(joe): Don't return an error when we have an unsupported node, bubble up "TRUE" as in keep
//  for that node, up to any `and` or `or` node.
pub fn convert_expr_to_vortex(physical_expr: Arc<dyn PhysicalExpr>) -> VortexResult<ExprRef> {
    if let Some(binary_expr) = physical_expr
        .as_any()
        .downcast_ref::<expressions::BinaryExpr>()
    {
        let left = convert_expr_to_vortex(binary_expr.left().clone())?;
        let right = convert_expr_to_vortex(binary_expr.right().clone())?;
        let operator = *binary_expr.op();

        return Ok(BinaryExpr::new_expr(left, operator.try_into()?, right));
    }

    if let Some(col_expr) = physical_expr.as_any().downcast_ref::<expressions::Column>() {
        return Ok(get_item(col_expr.name().to_owned(), ident()));
    }

    if let Some(like) = physical_expr
        .as_any()
        .downcast_ref::<expressions::LikeExpr>()
    {
        let child = convert_expr_to_vortex(like.expr().clone())?;
        let pattern = convert_expr_to_vortex(like.pattern().clone())?;
        return Ok(Like::new_expr(
            child,
            pattern,
            like.negated(),
            like.case_insensitive(),
        ));
    }

    if let Some(literal) = physical_expr
        .as_any()
        .downcast_ref::<expressions::Literal>()
    {
        let value = Scalar::from(literal.value().clone());
        return Ok(lit(value));
    }

    vortex_bail!(
        "Couldn't convert DataFusion physical {physical_expr} expression to a vortex expression"
    )
}

impl TryFrom<DFOperator> for Operator {
    type Error = VortexError;

    fn try_from(value: DFOperator) -> Result<Self, Self::Error> {
        match value {
            DFOperator::Eq => Ok(Operator::Eq),
            DFOperator::NotEq => Ok(Operator::NotEq),
            DFOperator::Lt => Ok(Operator::Lt),
            DFOperator::LtEq => Ok(Operator::Lte),
            DFOperator::Gt => Ok(Operator::Gt),
            DFOperator::GtEq => Ok(Operator::Gte),
            DFOperator::And => Ok(Operator::And),
            DFOperator::Or => Ok(Operator::Or),
            DFOperator::IsDistinctFrom
            | DFOperator::IsNotDistinctFrom
            | DFOperator::RegexMatch
            | DFOperator::RegexIMatch
            | DFOperator::RegexNotMatch
            | DFOperator::RegexNotIMatch
            | DFOperator::LikeMatch
            | DFOperator::ILikeMatch
            | DFOperator::NotLikeMatch
            | DFOperator::NotILikeMatch
            | DFOperator::BitwiseAnd
            | DFOperator::BitwiseOr
            | DFOperator::BitwiseXor
            | DFOperator::BitwiseShiftRight
            | DFOperator::BitwiseShiftLeft
            | DFOperator::StringConcat
            | DFOperator::AtArrow
            | DFOperator::ArrowAt
            | DFOperator::Plus
            | DFOperator::Minus
            | DFOperator::Multiply
            | DFOperator::Divide
            | DFOperator::Modulo => Err(vortex_err!("Unsupported datafusion operator {value}")),
        }
    }
}
