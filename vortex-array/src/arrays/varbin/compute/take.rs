use num_traits::PrimInt;
use vortex_dtype::{DType, NativePType, match_each_integer_ptype};
use vortex_error::{VortexResult, vortex_err, vortex_panic};
use vortex_mask::Mask;

use crate::arrays::VarBinEncoding;
use crate::arrays::varbin::VarBinArray;
use crate::arrays::varbin::builder::VarBinBuilder;
use crate::compute::TakeFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, ToCanonical};

impl TakeFn<&VarBinArray> for VarBinEncoding {
    fn take(&self, array: &VarBinArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let offsets = array.offsets().to_primitive()?;
        let data = array.bytes();
        let indices = indices.to_primitive()?;
        match_each_integer_ptype!(offsets.ptype(), |$O| {
            match_each_integer_ptype!(indices.ptype(), |$I| {
                Ok(take(
                    array.dtype().clone(),
                    offsets.as_slice::<$O>(),
                    data.as_slice(),
                    indices.as_slice::<$I>(),
                    array.validity_mask()?,
                    indices.validity_mask()?,
                )?.into_array())
            })
        })
    }
}

fn take<I: NativePType, O: NativePType + PrimInt>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    indices: &[I],
    validity_mask: Mask,
    indices_validity_mask: Mask,
) -> VortexResult<VarBinArray> {
    if !validity_mask.all_true() || !indices_validity_mask.all_true() {
        return Ok(take_nullable(
            dtype,
            offsets,
            data,
            indices,
            validity_mask,
            indices_validity_mask,
        ));
    }

    let mut builder = VarBinBuilder::<O>::with_capacity(indices.len());
    for &idx in indices {
        let idx = idx
            .to_usize()
            .ok_or_else(|| vortex_err!("Failed to convert index to usize: {}", idx))?;
        let start = offsets[idx]
            .to_usize()
            .ok_or_else(|| vortex_err!("Failed to convert offset to usize: {}", offsets[idx]))?;
        let stop = offsets[idx + 1].to_usize().ok_or_else(|| {
            vortex_err!("Failed to convert offset to usize: {}", offsets[idx + 1])
        })?;
        builder.append_value(&data[start..stop]);
    }
    Ok(builder.finish(dtype))
}

fn take_nullable<I: NativePType, O: NativePType + PrimInt>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    indices: &[I],
    data_validity: Mask,
    indices_validity: Mask,
) -> VarBinArray {
    let mut builder = VarBinBuilder::<O>::with_capacity(indices.len());
    for (idx, data_idx) in indices.iter().enumerate() {
        if !indices_validity.value(idx) {
            builder.append_null();
            continue;
        }
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));
        if data_validity.value(data_idx) {
            let start = offsets[data_idx].to_usize().unwrap_or_else(|| {
                vortex_panic!("Failed to convert offset to usize: {}", offsets[data_idx])
            });
            let stop = offsets[data_idx + 1].to_usize().unwrap_or_else(|| {
                vortex_panic!(
                    "Failed to convert offset to usize: {}",
                    offsets[data_idx + 1]
                )
            });
            builder.append_value(&data[start..stop]);
        } else {
            builder.append_null();
        }
    }
    builder.finish(dtype)
}
