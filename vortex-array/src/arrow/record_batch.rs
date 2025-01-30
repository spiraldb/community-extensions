use arrow_array::cast::AsArray;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use vortex_error::{vortex_err, VortexError, VortexResult};

use crate::array::StructArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::validity::Validity;
use crate::{Array, IntoArray, IntoArrayVariant};

impl TryFrom<RecordBatch> for Array {
    type Error = VortexError;

    fn try_from(value: RecordBatch) -> VortexResult<Self> {
        Ok(StructArray::try_new(
            value
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str().into())
                .collect(),
            value
                .columns()
                .iter()
                .zip(value.schema().fields())
                .map(|(array, field)| Array::from_arrow(array.clone(), field.is_nullable()))
                .collect(),
            value.num_rows(),
            Validity::NonNullable, // Must match FromArrowType<SchemaRef> for DType
        )?
        .into_array())
    }
}

impl TryFrom<Array> for RecordBatch {
    type Error = VortexError;

    fn try_from(value: Array) -> VortexResult<Self> {
        let struct_arr = value.into_struct().map_err(|err| {
            vortex_err!("RecordBatch can only be constructed from a Vortex StructArray: {err}")
        })?;

        struct_arr.into_record_batch()
    }
}

impl StructArray {
    pub fn into_record_batch(self) -> VortexResult<RecordBatch> {
        let array_ref = self.into_array().into_arrow_preferred()?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }

    pub fn into_record_batch_with_schema(self, schema: &Schema) -> VortexResult<RecordBatch> {
        let data_type = DataType::Struct(schema.fields.clone());
        let array_ref = self.into_array().into_arrow(&data_type)?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }
}
