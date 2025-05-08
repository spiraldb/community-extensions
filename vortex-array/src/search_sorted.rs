use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::fmt::{Debug, Display, Formatter};
use std::hint;

use vortex_scalar::Scalar;

use crate::Array;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SearchSortedSide {
    Left,
    Right,
}

impl Display for SearchSortedSide {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchSortedSide::Left => write!(f, "left"),
            SearchSortedSide::Right => write!(f, "right"),
        }
    }
}

/// Result of performing search_sorted on an Array
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SearchResult {
    /// Result for a found element was found in the array and another one could be inserted at the given position
    /// in the sorted order
    Found(usize),

    /// Result for an element not found, but that could be inserted at the given position
    /// in the sorted order.
    NotFound(usize),
}

impl SearchResult {
    /// Convert search result to an index only if the value have been found
    pub fn to_found(self) -> Option<usize> {
        match self {
            Self::Found(i) => Some(i),
            Self::NotFound(_) => None,
        }
    }

    /// Extract index out of search result regardless of whether the value have been found or not
    pub fn to_index(self) -> usize {
        match self {
            Self::Found(i) => i,
            Self::NotFound(i) => i,
        }
    }

    /// Convert search result into an index suitable for searching array of offset indices, i.e. first element starts at 0.
    ///
    /// For example for a ChunkedArray with chunk offsets array [0, 3, 8, 10] you can use this method to
    /// obtain index suitable for indexing into it after performing a search
    pub fn to_offsets_index(self, len: usize, side: SearchSortedSide) -> usize {
        match self {
            SearchResult::Found(i) => {
                if side == SearchSortedSide::Right || i == len {
                    i.saturating_sub(1)
                } else {
                    i
                }
            }
            SearchResult::NotFound(i) => i.saturating_sub(1),
        }
    }

    /// Convert search result into an index suitable for searching array of end indices without 0 offset,
    /// i.e. first element implicitly covers 0..0th-element range.
    ///
    /// For example for a RunEndArray with ends array [3, 8, 10], you can use this method to obtain index suitable for
    /// indexing into it after performing a search
    pub fn to_ends_index(self, len: usize) -> usize {
        let idx = self.to_index();
        if idx == len { idx - 1 } else { idx }
    }
}

impl Display for SearchResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchResult::Found(i) => write!(f, "Found({i})"),
            SearchResult::NotFound(i) => write!(f, "NotFound({i})"),
        }
    }
}

#[allow(clippy::len_without_is_empty)]
pub trait IndexOrd<V> {
    /// PartialOrd of the value at index `idx` with `elem`.
    /// For example, if self\[idx\] > elem, return Some(Greater).
    fn index_cmp(&self, idx: usize, elem: &V) -> Option<Ordering>;

    fn index_lt(&self, idx: usize, elem: &V) -> bool {
        matches!(self.index_cmp(idx, elem), Some(Less))
    }

    fn index_le(&self, idx: usize, elem: &V) -> bool {
        matches!(self.index_cmp(idx, elem), Some(Less | Equal))
    }

    fn index_gt(&self, idx: usize, elem: &V) -> bool {
        matches!(self.index_cmp(idx, elem), Some(Greater))
    }

    fn index_ge(&self, idx: usize, elem: &V) -> bool {
        matches!(self.index_cmp(idx, elem), Some(Greater | Equal))
    }

    /// Get the length of the underlying ordered collection
    fn index_len(&self) -> usize;
}

/// Searches for value assuming the array is sorted.
///
/// Returned indices satisfy following condition if the search for value was to be inserted into the array at found positions
///
/// |side |result satisfies|
/// |-----|----------------|
/// |left |array\[i-1\] < value <= array\[i\]|
/// |right|array\[i-1\] <= value < array\[i\]|
pub trait SearchSorted<T> {
    fn search_sorted_many<I: IntoIterator<Item = T>>(
        &self,
        values: I,
        side: SearchSortedSide,
    ) -> impl Iterator<Item = SearchResult>
    where
        Self: IndexOrd<T>,
    {
        values
            .into_iter()
            .map(move |value| self.search_sorted(&value, side))
    }

    fn search_sorted(&self, value: &T, side: SearchSortedSide) -> SearchResult
    where
        Self: IndexOrd<T>,
    {
        match side {
            SearchSortedSide::Left => self.search_sorted_by(
                |idx| self.index_cmp(idx, value).unwrap_or(Less),
                |idx| {
                    if self.index_lt(idx, value) {
                        Less
                    } else {
                        Greater
                    }
                },
                side,
            ),
            SearchSortedSide::Right => self.search_sorted_by(
                |idx| self.index_cmp(idx, value).unwrap_or(Less),
                |idx| {
                    if self.index_le(idx, value) {
                        Less
                    } else {
                        Greater
                    }
                },
                side,
            ),
        }
    }

    /// find function is used to find the element if it exists, if element exists side_find will be
    /// used to find desired index amongst equal values
    fn search_sorted_by<F: FnMut(usize) -> Ordering, N: FnMut(usize) -> Ordering>(
        &self,
        find: F,
        side_find: N,
        side: SearchSortedSide,
    ) -> SearchResult;
}

// Default implementation for types that implement IndexOrd.
impl<S, T> SearchSorted<T> for S
where
    S: IndexOrd<T> + ?Sized,
{
    fn search_sorted_by<F: FnMut(usize) -> Ordering, N: FnMut(usize) -> Ordering>(
        &self,
        find: F,
        side_find: N,
        side: SearchSortedSide,
    ) -> SearchResult {
        match search_sorted_side_idx(find, 0, self.index_len()) {
            SearchResult::Found(found) => {
                let idx_search = match side {
                    SearchSortedSide::Left => search_sorted_side_idx(side_find, 0, found),
                    SearchSortedSide::Right => {
                        search_sorted_side_idx(side_find, found, self.index_len())
                    }
                };
                match idx_search {
                    SearchResult::NotFound(i) => SearchResult::Found(i),
                    _ => unreachable!(
                        "searching amongst equal values should never return Found result"
                    ),
                }
            }
            s => s,
        }
    }
}

// Code adapted from Rust standard library slice::binary_search_by
fn search_sorted_side_idx<F: FnMut(usize) -> Ordering>(
    mut find: F,
    from: usize,
    to: usize,
) -> SearchResult {
    let mut size = to - from;
    if size == 0 {
        return SearchResult::NotFound(0);
    }
    let mut base = from;

    // This loop intentionally doesn't have an early exit if the comparison
    // returns Equal. We want the number of loop iterations to depend *only*
    // on the size of the input slice so that the CPU can reliably predict
    // the loop count.
    while size > 1 {
        let half = size / 2;
        let mid = base + half;

        // SAFETY: the call is made safe by the following inconstants:
        // - `mid >= 0`: by definition
        // - `mid < size`: `mid = size / 2 + size / 4 + size / 8 ...`
        let cmp = find(mid);

        // Binary search interacts poorly with branch prediction, so force
        // the compiler to use conditional moves if supported by the target
        // architecture.
        base = if cmp == Greater { base } else { mid };

        // This is imprecise in the case where `size` is odd and the
        // comparison returns Greater: the mid element still gets included
        // by `size` even though it's known to be larger than the element
        // being searched for.
        //
        // This is fine though: we gain more performance by keeping the
        // loop iteration count invariant (and thus predictable) than we
        // lose from considering one additional element.
        size -= half;
    }

    // SAFETY: base is always in [0, size) because base <= mid.
    let cmp = find(base);
    if cmp == Equal {
        // SAFETY: same as the call to `find` above.
        unsafe { hint::assert_unchecked(base < to) };
        SearchResult::Found(base)
    } else {
        let result = base + (cmp == Less) as usize;
        // SAFETY: same as the call to `find` above.
        // Note that this is `<=`, unlike the assert in the `Found` path.
        unsafe { hint::assert_unchecked(result <= to) };
        SearchResult::NotFound(result)
    }
}

impl IndexOrd<Scalar> for dyn Array + '_ {
    fn index_cmp(&self, idx: usize, elem: &Scalar) -> Option<Ordering> {
        let scalar_a = self.scalar_at(idx).ok()?;
        scalar_a.partial_cmp(elem)
    }

    fn index_len(&self) -> usize {
        Self::len(self)
    }
}

impl<T: PartialOrd> IndexOrd<T> for [T] {
    fn index_cmp(&self, idx: usize, elem: &T) -> Option<Ordering> {
        // SAFETY: Used in search_sorted_by same as the standard library. The search_sorted ensures idx is in bounds
        unsafe { self.get_unchecked(idx) }.partial_cmp(elem)
    }

    fn index_len(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod test {
    use crate::search_sorted::{SearchResult, SearchSorted, SearchSortedSide};

    #[test]
    fn left_side_equal() {
        let arr = [0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9];
        let res = arr.search_sorted(&2, SearchSortedSide::Left);
        assert_eq!(arr[res.to_index()], 2);
        assert_eq!(res, SearchResult::Found(2));
    }

    #[test]
    fn right_side_equal() {
        let arr = [0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9];
        let res = arr.search_sorted(&2, SearchSortedSide::Right);
        assert_eq!(arr[res.to_index() - 1], 2);
        assert_eq!(res, SearchResult::Found(6));
    }

    #[test]
    fn left_side_equal_beginning() {
        let arr = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let res = arr.search_sorted(&0, SearchSortedSide::Left);
        assert_eq!(arr[res.to_index()], 0);
        assert_eq!(res, SearchResult::Found(0));
    }

    #[test]
    fn right_side_equal_beginning() {
        let arr = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let res = arr.search_sorted(&0, SearchSortedSide::Right);
        assert_eq!(arr[res.to_index() - 1], 0);
        assert_eq!(res, SearchResult::Found(4));
    }

    #[test]
    fn left_side_equal_end() {
        let arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9];
        let res = arr.search_sorted(&9, SearchSortedSide::Left);
        assert_eq!(arr[res.to_index()], 9);
        assert_eq!(res, SearchResult::Found(9));
    }

    #[test]
    fn right_side_equal_end() {
        let arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9];
        let res = arr.search_sorted(&9, SearchSortedSide::Right);
        assert_eq!(arr[res.to_index() - 1], 9);
        assert_eq!(res, SearchResult::Found(13));
    }
}
