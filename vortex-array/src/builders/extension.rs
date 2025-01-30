use std::any::Any;
use std::sync::Arc;

use vortex_dtype::{DType, ExtDType};
use vortex_error::VortexResult;
use vortex_scalar::ExtScalar;

use crate::array::ExtensionArray;
use crate::builders::{builder_with_capacity, ArrayBuilder, ArrayBuilderExt};
use crate::{Array, IntoArray};

pub struct ExtensionBuilder {
    storage: Box<dyn ArrayBuilder>,
    dtype: DType,
}

impl ExtensionBuilder {
    pub fn new(ext_dtype: Arc<ExtDType>) -> Self {
        Self::with_capacity(ext_dtype, 1024)
    }

    pub fn with_capacity(ext_dtype: Arc<ExtDType>, capacity: usize) -> Self {
        Self {
            storage: builder_with_capacity(ext_dtype.storage_dtype(), capacity),
            dtype: DType::Extension(ext_dtype),
        }
    }

    pub fn append_value(&mut self, value: ExtScalar) -> VortexResult<()> {
        self.storage.append_scalar(&value.storage())
    }

    pub fn append_option(&mut self, value: Option<ExtScalar>) -> VortexResult<()> {
        match value {
            Some(value) => self.append_value(value),
            None => {
                self.append_nulls(1);
                Ok(())
            }
        }
    }

    fn ext_dtype(&self) -> Arc<ExtDType> {
        if let DType::Extension(ext_dtype) = &self.dtype {
            ext_dtype.clone()
        } else {
            unreachable!()
        }
    }
}

impl ArrayBuilder for ExtensionBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.storage.append_zeros(n)
    }

    fn append_nulls(&mut self, n: usize) {
        self.storage.append_nulls(n)
    }

    fn finish(&mut self) -> VortexResult<Array> {
        let storage = self.storage.finish()?;
        Ok(ExtensionArray::new(self.ext_dtype(), storage).into_array())
    }
}
