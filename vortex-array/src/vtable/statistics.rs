use vortex_error::{VortexExpect, VortexResult};

use crate::Array;
use crate::encoding::Encoding;
use crate::stats::{Stat, StatsSet};

/// Encoding VTable for computing array statistics.
pub trait StatisticsVTable<A> {
    /// Compute the requested statistic. Can return additional stats.
    fn compute_statistics(&self, _array: A, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

impl<'a, E: Encoding> StatisticsVTable<&'a dyn Array> for E
where
    E: StatisticsVTable<&'a E::Array>,
{
    fn compute_statistics(&self, array: &'a dyn Array, stat: Stat) -> VortexResult<StatsSet> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        StatisticsVTable::compute_statistics(self, array_ref, stat)
    }
}
