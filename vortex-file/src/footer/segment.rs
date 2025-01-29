use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_flatbuffers::footer as fb;

/// The location of a segment within a Vortex file.
#[derive(Clone, Debug)]
pub struct Segment {
    pub offset: u64,
    pub length: u32,
    pub alignment: Alignment,
}

impl From<&Segment> for fb::Segment {
    fn from(value: &Segment) -> Self {
        fb::Segment::new(value.offset, value.length, value.alignment.exponent(), 0, 0)
    }
}

impl TryFrom<&fb::Segment> for Segment {
    type Error = VortexError;

    fn try_from(value: &fb::Segment) -> Result<Self, Self::Error> {
        Ok(Self {
            offset: value.offset(),
            length: value.length(),
            alignment: Alignment::from_exponent(value.alignment_exponent()),
        })
    }
}
