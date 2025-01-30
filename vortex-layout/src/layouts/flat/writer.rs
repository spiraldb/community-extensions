use vortex_array::parts::ArrayPartsFlatBuffer;
use vortex_array::stats::{Stat, STATS_TO_WRITE};
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_flatbuffers::WriteFlatBufferExt;

use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentWriter;
use crate::strategies::LayoutWriter;
use crate::{Layout, LayoutVTableRef};

pub struct FlatLayoutOptions {
    /// Stats to preserve when writing arrays
    pub array_stats: Vec<Stat>,
}

impl Default for FlatLayoutOptions {
    fn default() -> Self {
        Self {
            array_stats: STATS_TO_WRITE.to_vec(),
        }
    }
}

/// Writer for the flat layout.
pub struct FlatLayoutWriter {
    options: FlatLayoutOptions,
    dtype: DType,
    layout: Option<Layout>,
}

impl FlatLayoutWriter {
    pub fn new(dtype: DType, options: FlatLayoutOptions) -> Self {
        Self {
            options,
            dtype,
            layout: None,
        }
    }
}

fn retain_only_stats(array: &Array, stats: &[Stat]) {
    array.statistics().retain_only(stats);
    for child in array.children() {
        retain_only_stats(&child, stats)
    }
}

impl LayoutWriter for FlatLayoutWriter {
    fn push_chunk(&mut self, segments: &mut dyn SegmentWriter, chunk: Array) -> VortexResult<()> {
        if self.layout.is_some() {
            vortex_bail!("FlatLayoutStrategy::push_batch called after finish");
        }
        let row_count = chunk.len() as u64;
        retain_only_stats(&chunk, &self.options.array_stats);

        // We store each Array buffer in its own segment.
        let mut segment_ids = vec![];
        for child in chunk.depth_first_traversal() {
            for buffer in child.byte_buffers() {
                // TODO(ngates): decide a way of splitting buffers if they exceed u32 size.
                //  We could write empty segments either side of buffers to concatenate?
                //  Or we could use Layout::metadata to store this information.
                segment_ids.push(segments.put(buffer));
            }
        }

        // ...followed by a FlatBuffer describing the array layout.
        let flatbuffer = ArrayPartsFlatBuffer::new(&chunk).write_flatbuffer_bytes();
        segment_ids.push(segments.put(flatbuffer.into_inner()));

        self.layout = Some(Layout::new_owned(
            LayoutVTableRef::from_static(&FlatLayout),
            self.dtype.clone(),
            row_count,
            Some(segment_ids),
            None,
            None,
        ));
        Ok(())
    }

    fn finish(&mut self, _segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        self.layout
            .take()
            .ok_or_else(|| vortex_err!("FlatLayoutStrategy::finish called without push_batch"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::stats::Stat;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_expr::ident;
    use vortex_scan::RowMask;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;

    #[test]
    fn flat_stats() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            assert!(array.statistics().compute_bit_width_freq().is_some());
            assert!(array.statistics().compute_trailing_zero_freq().is_some());
            let layout = FlatLayoutWriter::new(array.dtype().clone(), Default::default())
                .push_one(&mut segments, array.into_array())
                .unwrap();

            let result = layout
                .reader(Arc::new(segments), Default::default())
                .unwrap()
                .evaluate_expr(RowMask::new_valid_between(0, layout.row_count()), ident())
                .await
                .unwrap();

            assert!(result.statistics().get(Stat::BitWidthFreq).is_none());
            assert!(result.statistics().get(Stat::TrailingZeroFreq).is_none());
        })
    }
}
