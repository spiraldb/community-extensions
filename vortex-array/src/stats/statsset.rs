use enum_iterator::all;
use itertools::{EitherOrBoth, Itertools};
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexError};
use vortex_scalar::Scalar;

use crate::stats::Stat;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatsSet {
    values: Vec<(Stat, Scalar)>,
}

impl StatsSet {
    /// Create new StatSet without validating uniqueness of all the entries
    ///
    /// # Safety
    ///
    /// This method will not panic or trigger UB, but may lead to duplicate stats being stored.
    pub fn new_unchecked(values: Vec<(Stat, Scalar)>) -> Self {
        Self { values }
    }

    /// Specialized constructor for the case where the StatsSet represents
    /// an array consisting entirely of [null](vortex_dtype::DType::Null) values.
    pub fn nulls(len: usize, dtype: &DType) -> Self {
        let mut stats = Self::new_unchecked(vec![
            (Stat::Min, Scalar::null(dtype.clone())),
            (Stat::Max, Scalar::null(dtype.clone())),
            (Stat::RunCount, 1.into()),
            (Stat::NullCount, len.into()),
        ]);

        if len > 0 {
            stats.set(Stat::IsConstant, true);
            stats.set(Stat::IsSorted, true);
            stats.set(Stat::IsStrictSorted, len < 2);
        }

        // Add any DType-specific stats.
        match dtype {
            DType::Bool(_) => {
                stats.set(Stat::TrueCount, 0);
            }
            DType::Primitive(ptype, _) => {
                ptype.byte_width();
                stats.set(Stat::BitWidthFreq, vec![0u64; ptype.byte_width() * 8 + 1]);
                stats.set(
                    Stat::TrailingZeroFreq,
                    vec![ptype.byte_width() as u64 * 8; ptype.byte_width() * 8 + 1],
                );
            }
            _ => {}
        }

        stats
    }

    pub fn constant(scalar: &Scalar, length: usize) -> Self {
        let mut stats = Self::default();
        if length > 0 {
            stats.set(Stat::IsConstant, true);
            stats.set(Stat::IsSorted, true);
            stats.set(Stat::IsStrictSorted, length <= 1);
        }

        let run_count = if length == 0 { 0u64 } else { 1 };
        stats.set(Stat::RunCount, run_count);

        let null_count = if scalar.is_null() { length as u64 } else { 0 };
        stats.set(Stat::NullCount, null_count);

        if let Some(bool_scalar) = scalar.as_bool_opt() {
            let true_count = bool_scalar
                .value()
                .map(|b| if b { length as u64 } else { 0 })
                .unwrap_or(0);
            stats.set(Stat::TrueCount, true_count);
        }

        stats.set(Stat::Min, scalar.clone());
        stats.set(Stat::Max, scalar.clone());

        stats
    }

    pub fn bools_with_true_and_null_count(
        true_count: usize,
        null_count: usize,
        len: usize,
    ) -> Self {
        StatsSet::new_unchecked(vec![
            (Stat::TrueCount, true_count.into()),
            (Stat::NullCount, null_count.into()),
            (Stat::Min, (true_count == len).into()),
            (Stat::Max, (true_count > 0).into()),
            (
                Stat::IsConstant,
                ((true_count == 0 && null_count == 0) || true_count == len).into(),
            ),
        ])
    }

    pub fn of<S: Into<Scalar>>(stat: Stat, value: S) -> Self {
        Self::new_unchecked(vec![(stat, value.into())])
    }
}

// Getters and setters for individual stats.
impl StatsSet {
    /// Count of stored stats with known values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Predicate equivalent to a [len][Self::len] of zero.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, stat: Stat) -> Option<&Scalar> {
        self.values.iter().find(|(s, _)| *s == stat).map(|(_, v)| v)
    }

    pub fn get_as<T: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<T> {
        self.get(stat).map(|v| {
            T::try_from(v).unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to get stat {} as {}",
                    stat,
                    std::any::type_name::<T>()
                )
            })
        })
    }

    /// Set the stat `stat` to `value`.
    pub fn set<S: Into<Scalar>>(&mut self, stat: Stat, value: S) {
        if let Some(existing) = self.values.iter_mut().find(|(s, _)| *s == stat) {
            *existing = (stat, value.into());
        } else {
            self.values.push((stat, value.into()));
        }
    }

    /// Clear the stat `stat` from the set.
    pub fn clear(&mut self, stat: Stat) {
        self.values.retain(|(s, _)| *s != stat);
    }

    pub fn retain_only(&mut self, stats: &[Stat]) {
        self.values.retain(|(s, _)| stats.contains(s))
    }

    /// Iterate over the statistic names and values in-place.
    ///
    /// See [Iterator].
    pub fn iter(&self) -> impl Iterator<Item = &(Stat, Scalar)> {
        self.values.iter()
    }
}

// StatSetIntoIter just exists to protect current implementation from exposure on the public API.

/// Owned iterator over the stats.
///
/// See [IntoIterator].
pub struct StatsSetIntoIter(std::vec::IntoIter<(Stat, Scalar)>);

impl Iterator for StatsSetIntoIter {
    type Item = (Stat, Scalar);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl IntoIterator for StatsSet {
    type Item = (Stat, Scalar);
    type IntoIter = StatsSetIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        StatsSetIntoIter(self.values.into_iter())
    }
}

impl Extend<(Stat, Scalar)> for StatsSet {
    #[inline]
    fn extend<T: IntoIterator<Item = (Stat, Scalar)>>(&mut self, iter: T) {
        iter.into_iter().for_each(|(stat, scalar)| {
            self.set(stat, scalar);
        });
    }
}

// Merge helpers
impl StatsSet {
    /// Merge stats set `other` into `self`, with the semantic assumption that `other`
    /// contains stats from an array that is *appended* to the array represented by `self`.
    pub fn merge_ordered(mut self, other: &Self) -> Self {
        for s in all::<Stat>() {
            match s {
                Stat::BitWidthFreq => self.merge_bit_width_freq(other),
                Stat::TrailingZeroFreq => self.merge_trailing_zero_freq(other),
                Stat::IsConstant => self.merge_is_constant(other),
                Stat::IsSorted => self.merge_is_sorted(other),
                Stat::IsStrictSorted => self.merge_is_strict_sorted(other),
                Stat::Max => self.merge_max(other),
                Stat::Min => self.merge_min(other),
                Stat::RunCount => self.merge_run_count(other),
                Stat::TrueCount => self.merge_true_count(other),
                Stat::NullCount => self.merge_null_count(other),
                Stat::UncompressedSizeInBytes => self.merge_uncompressed_size_in_bytes(other),
            }
        }

        self
    }

    /// Merge stats set `other` into `self`, with no assumption on ordering.
    /// Stats that are not commutative (e.g., is_sorted) are dropped from the result.
    pub fn merge_unordered(mut self, other: &Self) -> Self {
        for s in all::<Stat>() {
            if !s.is_commutative() {
                self.clear(s);
                continue;
            }

            match s {
                Stat::BitWidthFreq => self.merge_bit_width_freq(other),
                Stat::TrailingZeroFreq => self.merge_trailing_zero_freq(other),
                Stat::IsConstant => self.merge_is_constant(other),
                Stat::Max => self.merge_max(other),
                Stat::Min => self.merge_min(other),
                Stat::TrueCount => self.merge_true_count(other),
                Stat::NullCount => self.merge_null_count(other),
                Stat::UncompressedSizeInBytes => self.merge_uncompressed_size_in_bytes(other),
                _ => vortex_panic!("Unrecognized commutative stat {}", s),
            }
        }

        self
    }

    fn merge_min(&mut self, other: &Self) {
        match (self.get(Stat::Min), other.get(Stat::Min)) {
            (Some(m1), Some(m2)) => {
                if m2 < m1 {
                    self.set(Stat::Min, m2.clone());
                }
            }
            _ => self.clear(Stat::Min),
        }
    }

    fn merge_max(&mut self, other: &Self) {
        match (self.get(Stat::Max), other.get(Stat::Max)) {
            (Some(m1), Some(m2)) => {
                if m2 > m1 {
                    self.set(Stat::Max, m2.clone());
                }
            }
            _ => self.clear(Stat::Max),
        }
    }

    fn merge_is_constant(&mut self, other: &Self) {
        if let Some(is_constant) = self.get_as(Stat::IsConstant) {
            if let Some(other_is_constant) = other.get_as(Stat::IsConstant) {
                if is_constant && other_is_constant && self.get(Stat::Min) == other.get(Stat::Min) {
                    return;
                }
            }
            self.set(Stat::IsConstant, false);
        }
    }

    fn merge_is_sorted(&mut self, other: &Self) {
        self.merge_sortedness_stat(other, Stat::IsSorted, |own, other| own <= other)
    }

    fn merge_is_strict_sorted(&mut self, other: &Self) {
        self.merge_sortedness_stat(other, Stat::IsStrictSorted, |own, other| own < other)
    }

    fn merge_sortedness_stat<F: Fn(Option<&Scalar>, Option<&Scalar>) -> bool>(
        &mut self,
        other: &Self,
        stat: Stat,
        cmp: F,
    ) {
        if let Some(is_sorted) = self.get_as(stat) {
            if let Some(other_is_sorted) = other.get_as(stat) {
                if !(self.get(Stat::Max).is_some() && other.get(Stat::Min).is_some()) {
                    self.clear(stat);
                } else if is_sorted
                    && other_is_sorted
                    && cmp(self.get(Stat::Max), other.get(Stat::Min))
                {
                    return;
                } else {
                    self.set(stat, false);
                }
            } else {
                self.clear(stat)
            }
        }
    }

    fn merge_true_count(&mut self, other: &Self) {
        self.merge_sum_stat(other, Stat::TrueCount)
    }

    fn merge_null_count(&mut self, other: &Self) {
        self.merge_sum_stat(other, Stat::NullCount)
    }

    fn merge_uncompressed_size_in_bytes(&mut self, other: &Self) {
        self.merge_sum_stat(other, Stat::UncompressedSizeInBytes)
    }

    fn merge_sum_stat(&mut self, other: &Self, stat: Stat) {
        match (self.get_as::<usize>(stat), other.get_as::<usize>(stat)) {
            (Some(nc1), Some(nc2)) => {
                self.set(stat, nc1 + nc2);
            }
            _ => self.clear(stat),
        }
    }

    fn merge_bit_width_freq(&mut self, other: &Self) {
        self.merge_freq_stat(other, Stat::BitWidthFreq)
    }

    fn merge_trailing_zero_freq(&mut self, other: &Self) {
        self.merge_freq_stat(other, Stat::TrailingZeroFreq)
    }

    fn merge_freq_stat(&mut self, other: &Self, stat: Stat) {
        match (
            self.get_as::<Vec<usize>>(stat),
            other.get_as::<Vec<usize>>(stat),
        ) {
            (Some(f1), Some(f2)) => {
                let combined_freq = f1
                    .iter()
                    .zip_longest(f2.iter())
                    .map(|pair| match pair {
                        EitherOrBoth::Both(a, b) => a + b,
                        EitherOrBoth::Left(a) => *a,
                        EitherOrBoth::Right(b) => *b,
                    })
                    .collect_vec();
                self.set(stat, combined_freq);
            }
            _ => self.clear(stat),
        }
    }

    /// Merged run count is an upper bound where we assume run is interrupted at the boundary
    fn merge_run_count(&mut self, other: &Self) {
        match (
            self.get_as::<usize>(Stat::RunCount),
            other.get_as::<usize>(Stat::RunCount),
        ) {
            (Some(r1), Some(r2)) => {
                self.set(Stat::RunCount, r1 + r2 + 1);
            }
            _ => self.clear(Stat::RunCount),
        }
    }
}

#[cfg(test)]
mod test {
    use enum_iterator::all;
    use itertools::Itertools;

    use crate::array::PrimitiveArray;
    use crate::stats::{ArrayStatistics as _, Stat, StatsSet};
    use crate::IntoArrayData as _;

    #[test]
    fn test_iter() {
        let set = StatsSet::new_unchecked(vec![(Stat::Max, 100.into()), (Stat::Min, 42.into())]);
        assert_eq!(
            set.iter().cloned().collect_vec(),
            vec![(Stat::Max, 100.into()), (Stat::Min, 42.into())]
        );
    }

    #[test]
    fn into_iter() {
        let set = StatsSet::new_unchecked(vec![(Stat::Max, 100.into()), (Stat::Min, 42.into())]);
        assert_eq!(
            set.into_iter().collect_vec(),
            vec![(Stat::Max, 100.into()), (Stat::Min, 42.into())]
        );
    }

    #[test]
    fn merge_into_min() {
        let first = StatsSet::of(Stat::Min, 42).merge_ordered(&StatsSet::default());
        assert_eq!(first.get(Stat::Min), None);
    }

    #[test]
    fn merge_from_min() {
        let first = StatsSet::default().merge_ordered(&StatsSet::of(Stat::Min, 42));
        assert_eq!(first.get(Stat::Min), None);
    }

    #[test]
    fn merge_mins() {
        let first = StatsSet::of(Stat::Min, 37).merge_ordered(&StatsSet::of(Stat::Min, 42));
        assert_eq!(first.get(Stat::Min).cloned(), Some(37.into()));
    }

    #[test]
    fn merge_into_max() {
        let first = StatsSet::of(Stat::Max, 42).merge_ordered(&StatsSet::default());
        assert_eq!(first.get(Stat::Max), None);
    }

    #[test]
    fn merge_from_max() {
        let first = StatsSet::default().merge_ordered(&StatsSet::of(Stat::Max, 42));
        assert_eq!(first.get(Stat::Max), None);
    }

    #[test]
    fn merge_maxes() {
        let first = StatsSet::of(Stat::Max, 37).merge_ordered(&StatsSet::of(Stat::Max, 42));
        assert_eq!(first.get(Stat::Max).cloned(), Some(42.into()));
    }

    #[test]
    fn merge_into_scalar() {
        let first = StatsSet::of(Stat::TrueCount, 42).merge_ordered(&StatsSet::default());
        assert_eq!(first.get(Stat::TrueCount), None);
    }

    #[test]
    fn merge_from_scalar() {
        let first = StatsSet::default().merge_ordered(&StatsSet::of(Stat::TrueCount, 42));
        assert_eq!(first.get(Stat::TrueCount), None);
    }

    #[test]
    fn merge_scalars() {
        let first =
            StatsSet::of(Stat::TrueCount, 37).merge_ordered(&StatsSet::of(Stat::TrueCount, 42));
        assert_eq!(first.get(Stat::TrueCount).cloned(), Some(79u64.into()));
    }

    #[test]
    fn merge_into_freq() {
        let vec = (0usize..255).collect_vec();
        let first = StatsSet::of(Stat::BitWidthFreq, vec).merge_ordered(&StatsSet::default());
        assert_eq!(first.get(Stat::BitWidthFreq), None);
    }

    #[test]
    fn merge_from_freq() {
        let vec = (0usize..255).collect_vec();
        let first = StatsSet::default().merge_ordered(&StatsSet::of(Stat::BitWidthFreq, vec));
        assert_eq!(first.get(Stat::BitWidthFreq), None);
    }

    #[test]
    fn merge_freqs() {
        let vec_in = vec![5u64; 256];
        let vec_out = vec![10u64; 256];
        let first = StatsSet::of(Stat::BitWidthFreq, vec_in.clone())
            .merge_ordered(&StatsSet::of(Stat::BitWidthFreq, vec_in));
        assert_eq!(first.get(Stat::BitWidthFreq).cloned(), Some(vec_out.into()));
    }

    #[test]
    fn merge_into_sortedness() {
        let first = StatsSet::of(Stat::IsStrictSorted, true).merge_ordered(&StatsSet::default());
        assert_eq!(first.get(Stat::IsStrictSorted), None);
    }

    #[test]
    fn merge_from_sortedness() {
        let first = StatsSet::default().merge_ordered(&StatsSet::of(Stat::IsStrictSorted, true));
        assert_eq!(first.get(Stat::IsStrictSorted), None);
    }

    #[test]
    fn merge_sortedness() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Max, 1);
        let mut second = StatsSet::of(Stat::IsStrictSorted, true);
        second.set(Stat::Min, 2);
        first = first.merge_ordered(&second);
        assert_eq!(first.get(Stat::IsStrictSorted).cloned(), Some(true.into()));
    }

    #[test]
    fn merge_sortedness_out_of_order() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Min, 1);
        let mut second = StatsSet::of(Stat::IsStrictSorted, true);
        second.set(Stat::Max, 2);
        second = second.merge_ordered(&first);
        assert_eq!(
            second.get(Stat::IsStrictSorted).cloned(),
            Some(false.into())
        );
    }

    #[test]
    fn merge_sortedness_only_one_sorted() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Max, 1);
        let mut second = StatsSet::of(Stat::IsStrictSorted, false);
        second.set(Stat::Min, 2);
        first.merge_ordered(&second);
        assert_eq!(
            second.get(Stat::IsStrictSorted).cloned(),
            Some(false.into())
        );
    }

    #[test]
    fn merge_sortedness_missing_min() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Max, 1);
        let second = StatsSet::of(Stat::IsStrictSorted, true);
        first = first.merge_ordered(&second);
        assert_eq!(first.get(Stat::IsStrictSorted).cloned(), None);
    }

    #[test]
    fn merge_unordered() {
        let array =
            PrimitiveArray::from_option_iter([Some(1), None, Some(2), Some(42), Some(10000), None])
                .into_array();
        let all_stats = all::<Stat>()
            .filter(|s| !matches!(s, Stat::TrueCount))
            .collect_vec();
        array.statistics().compute_all(&all_stats).unwrap();

        let stats = array.statistics().to_set();
        for stat in &all_stats {
            assert!(stats.get(*stat).is_some(), "Stat {} is missing", stat);
        }

        let merged = stats.clone().merge_unordered(&stats);
        for stat in &all_stats {
            assert_eq!(
                merged.get(*stat).is_some(),
                stat.is_commutative(),
                "Stat {} remains after merge_unordered despite not being commutative, or was removed despite being commutative",
                stat
            )
        }

        assert_eq!(merged.get(Stat::Min), stats.get(Stat::Min));
        assert_eq!(merged.get(Stat::Max), stats.get(Stat::Max));
        assert_eq!(
            merged
                .get(Stat::NullCount)
                .unwrap()
                .as_primitive()
                .typed_value::<u64>()
                .unwrap(),
            2 * stats
                .get(Stat::NullCount)
                .unwrap()
                .as_primitive()
                .typed_value::<u64>()
                .unwrap()
        );
    }
}
