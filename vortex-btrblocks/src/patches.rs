use vortex_array::Array;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::scalar_at;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;

use crate::downscale::downscale_integer_array;

/// Compresses the given patches by downscaling integers and checking for constant values.
pub fn compress_patches(patches: &Patches) -> VortexResult<Patches> {
    // Downscale the patch indices.
    let indices = downscale_integer_array(patches.indices().clone())?;

    // Check if the values are constant.
    let values = patches.values();
    let values = if values
        .statistics()
        .compute_is_constant()
        .unwrap_or_default()
    {
        ConstantArray::new(scalar_at(values, 0)?, values.len()).into_array()
    } else {
        values.clone()
    };

    Ok(Patches::new(
        patches.array_len(),
        patches.offset(),
        indices,
        values,
    ))
}
