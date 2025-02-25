use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools as _;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{DType, FieldName, FieldNames, Nullability, StructDType};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail, vortex_err};

use crate::{ExprRef, VortexExpr};

/// Pack zero or more expressions into a structure with named fields.
///
/// # Examples
///
/// ```
/// use vortex_array::IntoArray;
/// use vortex_array::compute::scalar_at;
/// use vortex_buffer::buffer;
/// use vortex_expr::{Pack, Identity, VortexExpr};
/// use vortex_scalar::Scalar;
///
/// let example = Pack::try_new_expr(
///     ["x".into(), "x copy".into(), "second x copy".into()].into(),
///     vec![Identity::new_expr(), Identity::new_expr(), Identity::new_expr()],
/// ).unwrap();
/// let packed = example.evaluate(&buffer![100, 110, 200].into_array()).unwrap();
/// let x_copy = packed
///     .as_struct_typed()
///     .unwrap()
///     .maybe_null_field_by_name("x copy")
///     .unwrap();
/// assert_eq!(scalar_at(&x_copy, 0).unwrap(), Scalar::from(100));
/// assert_eq!(scalar_at(&x_copy, 1).unwrap(), Scalar::from(110));
/// assert_eq!(scalar_at(&x_copy, 2).unwrap(), Scalar::from(200));
/// ```
///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pack {
    names: FieldNames,
    values: Vec<ExprRef>,
}

impl Pack {
    pub fn try_new_expr(names: FieldNames, values: Vec<ExprRef>) -> VortexResult<Arc<Self>> {
        if names.len() != values.len() {
            vortex_bail!("length mismatch {} {}", names.len(), values.len());
        }
        Ok(Arc::new(Pack { names, values }))
    }

    pub fn names(&self) -> &FieldNames {
        &self.names
    }

    pub fn field(&self, field_name: &FieldName) -> VortexResult<ExprRef> {
        let idx = self
            .names
            .iter()
            .position(|name| name == field_name)
            .ok_or_else(|| {
                vortex_err!(
                    "Cannot find field {} in pack fields {:?}",
                    field_name,
                    self.names
                )
            })?;

        self.values
            .get(idx)
            .cloned()
            .ok_or_else(|| vortex_err!("field index out of bounds: {}", idx))
    }
}

pub fn pack(elements: impl IntoIterator<Item = (impl Into<FieldName>, ExprRef)>) -> ExprRef {
    let (names, values): (Vec<_>, Vec<_>) = elements
        .into_iter()
        .map(|(name, value)| (name.into(), value))
        .unzip();
    Pack::try_new_expr(names.into(), values)
        .vortex_expect("pack names and values have the same length")
}

impl Display for Pack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("{")?;
        self.names
            .iter()
            .zip(&self.values)
            .format_with(", ", |(name, expr), f| f(&format_args!("{name}: {expr}")))
            .fmt(f)?;
        f.write_str("}")
    }
}

impl VortexExpr for Pack {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &dyn Array) -> VortexResult<ArrayRef> {
        let len = batch.len();
        let value_arrays = self
            .values
            .iter()
            .map(|value_expr| value_expr.evaluate(batch))
            .process_results(|it| it.collect::<Vec<_>>())?;
        Ok(
            StructArray::try_new(self.names.clone(), value_arrays, len, Validity::NonNullable)?
                .into_array(),
        )
    }

    fn children(&self) -> Vec<&ExprRef> {
        self.values.iter().collect()
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), self.values.len());
        Self::try_new_expr(self.names.clone(), children)
            .vortex_expect("children are known to have the same length as names")
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        let value_dtypes = self
            .values
            .iter()
            .map(|value_expr| value_expr.return_dtype(scope_dtype))
            .process_results(|it| it.collect())?;
        Ok(DType::Struct(
            Arc::new(StructDType::new(self.names.clone(), value_dtypes)),
            Nullability::NonNullable,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::FieldNames;
    use vortex_error::{VortexResult, vortex_bail, vortex_err};

    use crate::{Pack, VortexExpr, col};

    fn test_array() -> StructArray {
        StructArray::from_fields(&[
            ("a", buffer![0, 1, 2].into_array()),
            ("b", buffer![4, 5, 6].into_array()),
        ])
        .unwrap()
    }

    fn primitive_field(array: &dyn Array, field_path: &[&str]) -> VortexResult<PrimitiveArray> {
        let mut field_path = field_path.iter();

        let Some(field) = field_path.next() else {
            vortex_bail!("empty field path");
        };

        let mut array = array
            .as_struct_typed()
            .ok_or_else(|| vortex_err!("expected a struct"))?
            .maybe_null_field_by_name(field)?;

        for field in field_path {
            array = array
                .as_struct_typed()
                .ok_or_else(|| vortex_err!("expected a struct"))?
                .maybe_null_field_by_name(field)?;
        }
        Ok(array.to_primitive().unwrap())
    }

    #[test]
    pub fn test_empty_pack() {
        let expr = Pack::try_new_expr(Arc::new([]), Vec::new()).unwrap();

        let test_array = test_array().into_array();
        let actual_array = expr.evaluate(&test_array).unwrap();
        assert_eq!(actual_array.len(), test_array.len());
        assert!(actual_array.as_struct_typed().unwrap().nfields() == 0);
    }

    #[test]
    pub fn test_simple_pack() {
        let expr = Pack::try_new_expr(
            ["one".into(), "two".into(), "three".into()].into(),
            vec![col("a"), col("b"), col("a")],
        )
        .unwrap();

        let actual_array = expr.evaluate(&test_array()).unwrap();
        let expected_names: FieldNames = ["one".into(), "two".into(), "three".into()].into();
        assert_eq!(
            actual_array.as_struct_typed().unwrap().names(),
            &expected_names
        );

        assert_eq!(
            primitive_field(&actual_array, &["one"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
        assert_eq!(
            primitive_field(&actual_array, &["two"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(&actual_array, &["three"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
    }

    #[test]
    pub fn test_nested_pack() {
        let expr = Pack::try_new_expr(
            ["one".into(), "two".into(), "three".into()].into(),
            vec![
                col("a"),
                Pack::try_new_expr(
                    ["two_one".into(), "two_two".into()].into(),
                    vec![col("b"), col("b")],
                )
                .unwrap(),
                col("a"),
            ],
        )
        .unwrap();

        let actual_array = expr.evaluate(&test_array()).unwrap();
        let expected_names: FieldNames = ["one".into(), "two".into(), "three".into()].into();
        assert_eq!(
            actual_array.as_struct_typed().unwrap().names(),
            &expected_names
        );

        assert_eq!(
            primitive_field(&actual_array, &["one"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
        assert_eq!(
            primitive_field(&actual_array, &["two", "two_one"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(&actual_array, &["two", "two_two"])
                .unwrap()
                .as_slice::<i32>(),
            [4, 5, 6]
        );
        assert_eq!(
            primitive_field(&actual_array, &["three"])
                .unwrap()
                .as_slice::<i32>(),
            [0, 1, 2]
        );
    }
}
