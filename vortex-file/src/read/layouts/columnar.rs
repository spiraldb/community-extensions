use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::array::StructArray;
use vortex_array::stats::ArrayStatistics;
use vortex_array::validity::Validity;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::field::Field;
use vortex_dtype::{FieldName, FieldNames};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_expr::{Column, Select, VortexExpr};
use vortex_flatbuffers::footer;

use crate::read::cache::{LazyDType, RelativeLayoutCache};
use crate::read::expr_project::expr_project;
use crate::read::mask::RowMask;
use crate::{
    Layout, LayoutDeserializer, LayoutId, LayoutReader, PollRead, Prune, RowFilter, Scan,
    COLUMNAR_LAYOUT_ID,
};

#[derive(Debug)]
pub struct ColumnarLayout;

impl Layout for ColumnarLayout {
    fn id(&self) -> LayoutId {
        COLUMNAR_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: footer::Layout,
        dtype: Arc<LazyDType>,
        scan: Scan,
        layout_serde: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(Arc::new(
            ColumnarLayoutBuilder {
                layout,
                scan,
                dtype,
                layout_serde,
                message_cache,
            }
            .build()?,
        ))
    }
}

struct ColumnarLayoutBuilder<'a> {
    layout: footer::Layout<'a>,
    scan: Scan,
    dtype: Arc<LazyDType>,
    layout_serde: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
}

impl ColumnarLayoutBuilder<'_> {
    fn build(&self) -> VortexResult<ColumnarLayoutReader> {
        let (refs, lazy_dtype) = self.fields_with_dtypes()?;
        let fb_children = self.layout.children().unwrap_or_default();

        let mut unhandled_names = Vec::new();
        let mut unhandled_children = Vec::new();
        let mut handled_children = Vec::new();
        let mut handled_names = Vec::new();

        for (field, name) in refs.into_iter().zip_eq(lazy_dtype.names()?.iter()) {
            let resolved_child = lazy_dtype.resolve_field(&field)?;
            let child_dtype = lazy_dtype.field(&field)?;
            let child_layout = fb_children.get(resolved_child);
            let projected_expr = self
                .scan
                .expr
                .as_ref()
                .and_then(|e| expr_project(e, &[field]));

            let handled =
                self.scan.expr.is_none() || (self.scan.expr.is_some() && projected_expr.is_some());

            let child = self.layout_serde.read_layout(
                child_layout,
                Scan::from(projected_expr),
                child_dtype,
                self.message_cache.relative(resolved_child as u16),
            )?;

            if handled {
                handled_children.push(child);
                handled_names.push(name.clone());
            } else {
                unhandled_children.push(child);
                unhandled_names.push(name.clone());
            }
        }

        if !unhandled_names.is_empty() {
            let prf = self
                .scan
                .expr
                .as_ref()
                .and_then(|e| {
                    expr_project(
                        e,
                        &unhandled_names
                            .iter()
                            .map(|n| Field::from(n.as_ref()))
                            .collect::<Vec<_>>(),
                    )
                })
                .ok_or_else(|| {
                    vortex_err!(
                        "Must be able to project {:?} filter into unhandled space {}",
                        self.scan.expr.as_ref(),
                        unhandled_names.iter().format(",")
                    )
                })?;

            handled_children.push(Arc::new(ColumnarLayoutReader::new(
                unhandled_names.into(),
                unhandled_children,
                Some(prf),
                false,
            )));
            handled_names.push("__unhandled".into());
        }

        let top_level_expr = self
            .scan
            .expr
            .as_ref()
            .map(|e| e.as_any().downcast_ref::<RowFilter>().is_some())
            .unwrap_or(false)
            .then(|| {
                RowFilter::from_conjunction_expr(
                    handled_names
                        .iter()
                        .map(|f| Column::new_expr(Field::from(&**f)))
                        .collect(),
                )
            });
        let shortcircuit_siblings = top_level_expr.is_some();
        Ok(ColumnarLayoutReader::new(
            handled_names.into(),
            handled_children,
            top_level_expr,
            shortcircuit_siblings,
        ))
    }

    /// Get fields referenced by scan expression along with their dtype
    fn fields_with_dtypes(&self) -> VortexResult<(Vec<Field>, Arc<LazyDType>)> {
        let fb_children = self.layout.children().unwrap_or_default();
        let field_refs = self.scan_fields();
        let lazy_dtype = field_refs
            .as_ref()
            .map(|e| self.dtype.project(e))
            // TODO(ngates): what is this unwrap for? Why would we default to our own dtype?
            .unwrap_or_else(|| Ok(self.dtype.clone()))?;

        Ok((
            field_refs.unwrap_or_else(|| (0..fb_children.len()).map(Field::from).collect()),
            lazy_dtype,
        ))
    }

    /// Get fields referenced by scan expression preserving order if we're using select to project
    fn scan_fields(&self) -> Option<Vec<Field>> {
        self.scan.expr.as_ref().map(|e| {
            if let Some(se) = e.as_any().downcast_ref::<Select>() {
                match se {
                    Select::Include(i) => i.clone(),
                    Select::Exclude(_) => vortex_panic!("Select::Exclude is not supported"),
                }
            } else {
                e.references().into_iter().cloned().collect::<Vec<_>>()
            }
        })
    }
}

type InProgressRanges = RwLock<HashMap<(usize, usize), Vec<Option<ArrayData>>>>;
type InProgressPrunes = RwLock<HashMap<(usize, usize), Vec<Option<Prune>>>>;

/// In memory representation of Columnar NestedLayout.
///
/// Each child represents a column
#[derive(Debug)]
pub struct ColumnarLayoutReader {
    names: FieldNames,
    children: Vec<Arc<dyn LayoutReader>>,
    expr: Option<Arc<dyn VortexExpr>>,
    // TODO(robert): This is a hack/optimization that tells us if we're reducing results with AND or not
    shortcircuit_siblings: bool,
    in_progress_ranges: InProgressRanges,
    in_progress_metadata: RwLock<HashMap<FieldName, Option<ArrayData>>>,
    in_progress_prunes: InProgressPrunes,
}

impl ColumnarLayoutReader {
    pub fn new(
        names: FieldNames,
        children: Vec<Arc<dyn LayoutReader>>,
        expr: Option<Arc<dyn VortexExpr>>,
        shortcircuit_siblings: bool,
    ) -> Self {
        assert_eq!(
            names.len(),
            children.len(),
            "Names and children must be of same length"
        );
        Self {
            names,
            children,
            expr,
            shortcircuit_siblings,
            in_progress_ranges: RwLock::new(HashMap::new()),
            in_progress_metadata: RwLock::new(HashMap::new()),
            in_progress_prunes: RwLock::new(HashMap::new()),
        }
    }
}

impl LayoutReader for ColumnarLayoutReader {
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        for child in &self.children {
            child.add_splits(row_offset, splits)?;
        }
        Ok(())
    }

    fn poll_read(&self, selection: &RowMask) -> VortexResult<Option<PollRead<ArrayData>>> {
        let mut in_progress_guard = self
            .in_progress_ranges
            .write()
            .unwrap_or_else(|poison| vortex_panic!("Failed to write to message cache: {poison}"));
        let selection_range = (selection.begin(), selection.end());
        let in_progress_selection = in_progress_guard
            .entry(selection_range)
            .or_insert_with(|| vec![None; self.children.len()]);
        let mut messages = Vec::new();
        for (i, child_array) in in_progress_selection
            .iter_mut()
            .enumerate()
            .filter(|(_, a)| a.is_none())
        {
            match self.children[i].poll_read(selection)? {
                Some(rr) => match rr {
                    PollRead::ReadMore(message) => {
                        messages.extend(message);
                    }
                    PollRead::Value(arr) => {
                        if self.shortcircuit_siblings
                            && arr
                                .statistics()
                                .compute_true_count()
                                .map(|true_count| true_count == 0)
                                .unwrap_or(false)
                        {
                            in_progress_guard.remove(&selection_range);
                            return Ok(None);
                        }
                        *child_array = Some(arr)
                    }
                },
                None => {
                    debug_assert!(
                        in_progress_selection.iter().all(Option::is_none),
                        "Expected layout {}({i}) to produce an array but it was empty",
                        self.names[i]
                    );
                    return Ok(None);
                }
            }
        }

        if messages.is_empty() {
            let child_arrays = in_progress_guard
                .remove(&selection_range)
                .ok_or_else(|| vortex_err!("There were no arrays and no messages"))?
                .into_iter()
                .enumerate()
                .map(|(i, a)| a.ok_or_else(|| vortex_err!("Missing child array at index {i}")))
                .collect::<VortexResult<Vec<_>>>()?;
            let len = child_arrays
                .first()
                .map(|l| l.len())
                .unwrap_or_else(|| selection.true_count());
            let array =
                StructArray::try_new(self.names.clone(), child_arrays, len, Validity::NonNullable)?
                    .into_array();

            self.expr
                .as_ref()
                .map(|e| e.evaluate(&array))
                .unwrap_or_else(|| Ok(array))
                .map(PollRead::Value)
                .map(Some)
        } else {
            Ok(Some(PollRead::ReadMore(messages)))
        }
    }

    fn poll_metadata(&self) -> VortexResult<Option<PollRead<Vec<Option<ArrayData>>>>> {
        let mut in_progress_metadata = self
            .in_progress_metadata
            .write()
            .unwrap_or_else(|e| vortex_panic!("lock is poisoned: {e}"));
        let mut messages = Vec::default();

        for (name, child_reader) in self.names.iter().zip(self.children.iter()) {
            if let Some(child_metadata) = child_reader.poll_metadata()? {
                match child_metadata {
                    PollRead::Value(data) => {
                        if data.len() != 1 {
                            vortex_bail!("expected exactly one metadata array per-child");
                        }
                        in_progress_metadata.insert(name.clone(), data[0].clone());
                    }
                    PollRead::ReadMore(rm) => {
                        messages.extend(rm);
                    }
                }
            } else {
                in_progress_metadata.insert(name.clone(), None);
            }
        }

        // We're done reading
        if messages.is_empty() {
            let child_arrays = self
                .names
                .iter()
                .map(|name| in_progress_metadata[name].clone()) // TODO(Adam): Some columns might not have statistics
                .collect::<Vec<_>>();

            Ok(Some(PollRead::Value(child_arrays)))
        } else {
            Ok(Some(PollRead::ReadMore(messages)))
        }
    }

    fn poll_prune(&self, begin: usize, end: usize) -> VortexResult<PollRead<Prune>> {
        let mut in_progress_guard = self
            .in_progress_prunes
            .write()
            .unwrap_or_else(|poison| vortex_panic!("Failed to write to message cache: {poison}"));
        let selection_range = (begin, end);
        let column_prunes = in_progress_guard
            .entry(selection_range)
            .or_insert_with(|| vec![None; self.children.len()]);
        let mut messages = Vec::new();

        // Check all children to see if they can be pruned.
        for (i, child_prune_state) in column_prunes
            .iter_mut()
            .enumerate()
            .filter(|(_, a)| a.is_none())
        {
            match self.children[i].poll_prune(begin, end)? {
                PollRead::ReadMore(message) => messages.extend(message),
                PollRead::Value(is_pruned) => {
                    // If we are short-circuiting and one of the child prunes have been
                    // pruned, we can immediately prune the range.
                    if is_pruned == Prune::CanPrune && self.shortcircuit_siblings {
                        // If any of the children prune, we short-circuit the entire pruning operation.
                        // We clear the in-progress pruning state and immediately prune this entire
                        // row range.
                        in_progress_guard.remove(&selection_range).ok_or_else(|| {
                            vortex_err!("There were no pruning results and no messages")
                        })?;
                        return Ok(PollRead::Value(Prune::CanPrune));
                    } else {
                        // Update the state.
                        *child_prune_state = Some(is_pruned)
                    }
                }
            }
        }

        if !messages.is_empty() {
            // Read more
            Ok(PollRead::ReadMore(messages))
        } else {
            // If any children were pruned, we can short-circuit pruning of the child arrays.
            let any_can_prune = column_prunes.iter().any(|col_prune| {
                col_prune.vortex_expect("prune state must be initialized") == Prune::CanPrune
            });
            Ok(PollRead::Value(if any_can_prune {
                Prune::CanPrune
            } else {
                Prune::CannotPrune
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::{Arc, RwLock};

    use bytes::Bytes;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::array::{ChunkedArray, StructArray, VarBinArray};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayDType, IntoArrayData, IntoArrayVariant};
    use vortex_buffer::Buffer;
    use vortex_dtype::field::Field;
    use vortex_dtype::{DType, Nullability};
    use vortex_expr::{BinaryExpr, Column, Literal, Operator};

    use crate::read::builder::initial_read::read_initial_bytes;
    use crate::read::cache::RelativeLayoutCache;
    use crate::read::layouts::test_read::{filter_read_layout, read_layout};
    use crate::{
        LayoutDeserializer, LayoutMessageCache, LayoutReader, RowFilter, Scan, VortexFileWriter,
    };

    async fn layout_and_bytes(
        cache: Arc<RwLock<LayoutMessageCache>>,
        scan: Scan,
    ) -> (Arc<dyn LayoutReader>, Arc<dyn LayoutReader>, Bytes, usize) {
        let int_array = Buffer::from_iter(0..100).into_array();
        let int2_array = Buffer::from_iter(100..200).into_array();
        let int_dtype = int_array.dtype().clone();
        let chunked = ChunkedArray::try_new(
            iter::repeat_n(int_array.clone(), 2)
                .chain(iter::repeat_n(int2_array, 2))
                .chain(iter::once(int_array))
                .collect(),
            int_dtype,
        )
        .unwrap()
        .into_array();
        let str_array = VarBinArray::from_vec(
            iter::repeat_n("test text", 200)
                .chain(iter::repeat_n("it", 200))
                .chain(iter::repeat_n("test text", 100))
                .collect(),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let len = chunked.len();
        let struct_arr = StructArray::try_new(
            vec!["ints".into(), "strs".into()].into(),
            vec![chunked, str_array],
            len,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let mut writer = VortexFileWriter::new(Vec::new());
        writer = writer.write_array_columns(struct_arr).await.unwrap();
        let written = Bytes::from(writer.finalize().await.unwrap());

        let initial_read = read_initial_bytes(&written, written.len() as u64)
            .await
            .unwrap();
        let layout_serde = LayoutDeserializer::default();

        let dtype = Arc::new(initial_read.lazy_dtype());
        (
            layout_serde
                .read_layout(
                    initial_read.fb_layout(),
                    scan,
                    dtype.clone(),
                    RelativeLayoutCache::new(cache.clone()),
                )
                .unwrap(),
            layout_serde
                .read_layout(
                    initial_read.fb_layout(),
                    Scan::empty(),
                    dtype,
                    RelativeLayoutCache::new(cache.clone()),
                )
                .unwrap(),
            written,
            len,
        )
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (filter_layout, project_layout, buf, length) = layout_and_bytes(
            cache.clone(),
            Scan::new(RowFilter::new_expr(BinaryExpr::new_expr(
                Column::new_expr(Field::from("ints")),
                Operator::Gt,
                Literal::new_expr(10.into()),
            ))),
        )
        .await;
        let arr = filter_read_layout(
            filter_layout.as_ref(),
            project_layout.as_ref(),
            cache,
            &buf,
            length,
        )
        .pop_front();

        assert!(arr.is_some());
        let prim_arr = arr
            .as_ref()
            .unwrap()
            .as_struct_array()
            .unwrap()
            .field(0)
            .unwrap()
            .into_primitive()
            .unwrap();
        let str_arr = arr
            .as_ref()
            .unwrap()
            .as_struct_array()
            .unwrap()
            .field(1)
            .unwrap()
            .into_varbinview()
            .unwrap();
        assert_eq!(prim_arr.as_slice::<i32>(), &(11..100).collect::<Vec<_>>());
        assert_eq!(
            str_arr
                .with_iterator(|iter| iter
                    .flatten()
                    .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                    .collect::<Vec<_>>())
                .unwrap(),
            iter::repeat_n("test text", 89).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range_no_filter() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (_, project_layout, buf, length) = layout_and_bytes(cache.clone(), Scan::empty()).await;
        let arr = read_layout(project_layout.as_ref(), cache, &buf, length).pop_front();

        assert!(arr.is_some());
        let prim_arr = arr
            .as_ref()
            .unwrap()
            .as_struct_array()
            .unwrap()
            .field(0)
            .unwrap()
            .into_primitive()
            .unwrap();
        let str_arr = arr
            .as_ref()
            .unwrap()
            .as_struct_array()
            .unwrap()
            .field(1)
            .unwrap()
            .into_varbinview()
            .unwrap();
        assert_eq!(prim_arr.as_slice::<i32>(), (0..100).collect::<Vec<_>>());
        assert_eq!(
            str_arr
                .with_iterator(|iter| iter
                    .flatten()
                    .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                    .collect::<Vec<_>>())
                .unwrap(),
            iter::repeat_n("test text", 100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn short_circuit() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (filter_layout, project_layout, buf, length) = layout_and_bytes(
            cache.clone(),
            Scan::new(RowFilter::new_expr(BinaryExpr::new_expr(
                BinaryExpr::new_expr(
                    Column::new_expr(Field::from("strs")),
                    Operator::Eq,
                    Literal::new_expr("it".into()),
                ),
                Operator::And,
                BinaryExpr::new_expr(
                    Column::new_expr(Field::from("ints")),
                    Operator::Lt,
                    Literal::new_expr(150.into()),
                ),
            ))),
        )
        .await;
        let arr = filter_read_layout(
            filter_layout.as_ref(),
            project_layout.as_ref(),
            cache,
            &buf,
            length,
        );

        assert_eq!(arr.len(), 2);
        let prim_arr_chunk0 = arr[0]
            .as_struct_array()
            .unwrap()
            .field(0)
            .unwrap()
            .into_primitive()
            .unwrap();
        let str_arr_chunk0 = arr[0]
            .as_struct_array()
            .unwrap()
            .field(1)
            .unwrap()
            .into_varbinview()
            .unwrap();
        assert_eq!(
            prim_arr_chunk0.as_slice::<i32>(),
            (100..150).collect::<Vec<_>>()
        );
        assert_eq!(
            str_arr_chunk0
                .with_iterator(|iter| iter
                    .flatten()
                    .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                    .collect::<Vec<_>>())
                .unwrap(),
            iter::repeat_n("it", 50).collect::<Vec<_>>()
        );
        let prim_arr_chunk1 = arr[1]
            .as_struct_array()
            .unwrap()
            .field(0)
            .unwrap()
            .into_primitive()
            .unwrap();
        let str_arr_chunk1 = arr[1]
            .as_struct_array()
            .unwrap()
            .field(1)
            .unwrap()
            .into_varbinview()
            .unwrap();
        assert_eq!(
            prim_arr_chunk1.as_slice::<i32>(),
            (100..150).collect::<Vec<_>>()
        );
        assert_eq!(
            str_arr_chunk1
                .with_iterator(|iter| iter
                    .flatten()
                    .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                    .collect::<Vec<_>>())
                .unwrap(),
            iter::repeat_n("it", 50).collect::<Vec<_>>()
        );
    }
}
