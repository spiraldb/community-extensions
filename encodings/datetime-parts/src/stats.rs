use vortex_array::Array;
use vortex_array::stats::{Precision, Stat, StatsSet};
use vortex_array::vtable::StatisticsVTable;
use vortex_error::VortexResult;
use vortex_scalar::ScalarValue;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl StatisticsVTable<&DateTimePartsArray> for DateTimePartsEncoding {
    fn compute_statistics(&self, array: &DateTimePartsArray, stat: Stat) -> VortexResult<StatsSet> {
        let maybe_stat = match stat {
            Stat::NullCount => Some(ScalarValue::from(array.invalid_count()?)),
            Stat::IsConstant => Some(ScalarValue::from(
                array.days().is_constant()
                    && array.seconds().is_constant()
                    && array.subseconds().is_constant(),
            )),
            _ => None,
        };

        let mut stats = StatsSet::default();
        if let Some(value) = maybe_stat {
            stats.set(stat, Precision::exact(value));
        }
        Ok(stats)
    }
}
