#![allow(clippy::cast_possible_truncation)]
use std::iter;
use std::pin::pin;
use std::sync::Arc;

use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, pin_mut};
use itertools::Itertools;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{
    ChunkedArray, ListArray, PrimitiveArray, StructArray, VarBinArray, VarBinViewArray,
};
use vortex_array::compute::scalar_at;
use vortex_array::stream::ArrayStreamArrayExt;
use vortex_array::validity::Validity;
use vortex_array::variants::{PrimitiveArrayTrait, StructArrayTrait};
use vortex_array::{Array, ArrayVariants, IntoArray, ToCanonical};
use vortex_buffer::{Buffer, ByteBufferMut, buffer};
use vortex_dtype::PType::I32;
use vortex_dtype::{DType, Nullability, PType, StructDType};
use vortex_error::{VortexResult, vortex_panic};
use vortex_expr::{and, eq, get_item, gt, gt_eq, ident, lit, lt, lt_eq, or, select};

use crate::{
    InMemoryVortexFile, V1_FOOTER_FBS_SIZE, VERSION, VortexFile, VortexOpenOptions,
    VortexWriteOptions,
};

#[test]
fn test_eof_values() {
    // this test exists as a reminder to think about whether we should increment the version
    // when we change the footer
    assert_eq!(VERSION, 1);
    assert_eq!(V1_FOOTER_FBS_SIZE, 32);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_read_simple() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), st.to_array_stream())
        .await
        .unwrap();

    let stream = VortexOpenOptions::in_memory(buf)
        .open()
        .await
        .unwrap()
        .scan()
        .into_array_stream()
        .unwrap();
    pin_mut!(stream);

    let mut row_count = 0;

    while let Some(array) = stream.next().await {
        let array = array.unwrap();
        row_count += array.len();
    }

    assert_eq!(row_count, 8);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_read_simple_with_spawn() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ])
    .into_array();

    let lists = ChunkedArray::from_iter([
        ListArray::from_iter_slow::<i16, _>(
            vec![vec![11, 12], vec![21, 22], vec![31, 32], vec![41, 42]],
            Arc::new(I32.into()),
        )
        .unwrap(),
        ListArray::from_iter_slow::<i8, _>(
            vec![vec![51, 52], vec![61, 62], vec![71, 72], vec![81, 82]],
            Arc::new(I32.into()),
        )
        .unwrap(),
    ])
    .into_array();

    let st =
        StructArray::from_fields(&[("strings", strings), ("numbers", numbers), ("lists", lists)])
            .unwrap();

    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), st.to_array_stream())
        .await
        .unwrap();

    assert!(!buf.is_empty());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_read_projection() {
    let strings_expected = ["ab", "foo", "bar", "baz", "ab", "foo", "bar", "baz"];
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(strings_expected[..4].to_vec()).into_array(),
        VarBinArray::from(strings_expected[4..].to_vec()).into_array(),
    ])
    .into_array();
    let strings_dtype = strings.dtype().clone();

    let numbers_expected = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let numbers = ChunkedArray::from_iter([
        Buffer::copy_from(&numbers_expected[..4]).into_array(),
        Buffer::copy_from(&numbers_expected[4..]).into_array(),
    ])
    .into_array();
    let numbers_dtype = numbers.dtype().clone();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();

    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), st.to_array_stream())
        .await
        .unwrap();

    let file = VortexOpenOptions::in_memory(buf).open().await.unwrap();
    let array = file
        .scan()
        .with_projection(select(["strings".into()], ident()))
        .into_array()
        .await
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            Arc::new(StructDType::new(
                vec!["strings".into()].into(),
                vec![strings_dtype.clone()]
            )),
            Nullability::NonNullable,
        )
    );

    let actual = array
        .as_struct_typed()
        .unwrap()
        .maybe_null_field_by_idx(0)
        .unwrap()
        .to_varbinview()
        .unwrap()
        .with_iterator(|x| {
            x.map(|x| unsafe { String::from_utf8_unchecked(x.unwrap().to_vec()) })
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(actual, strings_expected);

    let array = file
        .scan()
        .with_projection(select(["numbers".into()], ident()))
        .into_array()
        .await
        .unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Struct(
            Arc::new(StructDType::new(
                vec!["numbers".into()].into(),
                vec![numbers_dtype.clone()]
            )),
            Nullability::NonNullable,
        )
    );

    let primitive_array = array
        .as_struct_typed()
        .unwrap()
        .maybe_null_field_by_idx(0)
        .unwrap()
        .to_primitive()
        .unwrap();
    let actual = primitive_array.as_slice::<u32>();
    assert_eq!(actual, numbers_expected);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn unequal_batches() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "bob"]).into_array(),
        VarBinArray::from(vec!["baz", "ab", "foo", "bar", "baz", "alice"]).into_array(),
    ])
    .into_array();

    let numbers = ChunkedArray::from_iter([
        buffer![1u32, 2, 3, 4, 5].into_array(),
        buffer![6u32, 7, 8, 9, 10].into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), st.to_array_stream())
        .await
        .unwrap();

    let mut stream = pin!(
        VortexOpenOptions::in_memory(buf)
            .open()
            .await
            .unwrap()
            .scan()
            .into_array_stream()
            .unwrap()
    );

    let mut item_count = 0;

    while let Some(array) = stream.next().await {
        let array = array.unwrap();
        item_count += array.len();

        let numbers = array
            .as_struct_typed()
            .unwrap()
            .maybe_null_field_by_name("numbers");

        if let Ok(numbers) = numbers {
            let numbers = numbers.to_primitive().unwrap();
            assert_eq!(numbers.ptype(), PType::U32);
        } else {
            vortex_panic!("Expected column doesn't exist")
        }
    }
    assert_eq!(item_count, 10);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn write_chunked() {
    let strings = VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array();
    let string_dtype = strings.dtype().clone();
    let strings_chunked = ChunkedArray::try_new(iter::repeat_n(strings, 4).collect(), string_dtype)
        .unwrap()
        .into_array();
    let numbers = buffer![1u32, 2, 3, 4].into_array();
    let numbers_dtype = numbers.dtype().clone();
    let numbers_chunked =
        ChunkedArray::try_new(iter::repeat_n(numbers, 4).collect(), numbers_dtype)
            .unwrap()
            .into_array();
    let st = StructArray::try_new(
        ["strings".into(), "numbers".into()].into(),
        vec![strings_chunked, numbers_chunked],
        16,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let st_dtype = st.dtype().clone();

    let chunked_st = ChunkedArray::try_new(iter::repeat_n(st, 3).collect(), st_dtype)
        .unwrap()
        .into_array();
    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), chunked_st.to_array_stream())
        .await
        .unwrap();

    let mut stream = pin!(
        VortexOpenOptions::in_memory(buf)
            .open()
            .await
            .unwrap()
            .scan()
            .into_array_stream()
            .unwrap()
    );
    let mut array_len: usize = 0;
    while let Some(array) = stream.next().await {
        array_len += array.unwrap().len();
    }
    assert_eq!(array_len, 48);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_string() {
    let names_orig = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let ages_orig =
        PrimitiveArray::from_option_iter([Some(25), Some(31), None, Some(57), None]).into_array();
    let st = StructArray::try_new(
        ["name".into(), "age".into()].into(),
        vec![names_orig, ages_orig],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), st.to_array_stream())
        .await
        .unwrap();

    let result = VortexOpenOptions::in_memory(buf)
        .open()
        .await
        .unwrap()
        .scan()
        .with_filter(eq(get_item("name", ident()), lit("Joseph")))
        .into_array_stream()
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    let names = result[0]
        .as_struct_typed()
        .unwrap()
        .maybe_null_field_by_idx(0)
        .unwrap();
    assert_eq!(
        names
            .to_varbinview()
            .unwrap()
            .with_iterator(|iter| iter
                .flatten()
                .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                .collect::<Vec<_>>())
            .unwrap(),
        vec!["Joseph".to_string()]
    );
    let ages = result[0]
        .as_struct_typed()
        .unwrap()
        .maybe_null_field_by_idx(1)
        .unwrap();
    assert_eq!(ages.to_primitive().unwrap().as_slice::<i32>(), vec![25]);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_or() {
    let names = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    );
    let ages = PrimitiveArray::from_option_iter([Some(25), Some(31), None, Some(57), None]);
    let st = StructArray::try_new(
        ["name".into(), "age".into()].into(),
        vec![names.into_array(), ages.into_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), st.to_array_stream())
        .await
        .unwrap();

    let result = VortexOpenOptions::in_memory(buf)
        .open()
        .await
        .unwrap()
        .scan()
        .with_filter(or(
            eq(get_item("name", ident()), lit("Angela")),
            and(
                gt_eq(get_item("age", ident()), lit(20)),
                lt_eq(get_item("age", ident()), lit(30)),
            ),
        ))
        .into_array_stream()
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    let names = result[0]
        .as_struct_typed()
        .unwrap()
        .maybe_null_field_by_idx(0)
        .unwrap();
    assert_eq!(
        names
            .to_varbinview()
            .unwrap()
            .with_iterator(|iter| iter
                .flatten()
                .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                .collect::<Vec<_>>())
            .unwrap(),
        vec!["Joseph".to_string(), "Angela".to_string()]
    );
    let ages = result[0]
        .as_struct_typed()
        .unwrap()
        .maybe_null_field_by_idx(1)
        .unwrap();
    assert_eq!(
        ages.to_primitive()
            .unwrap()
            .with_iterator(|iter| iter.map(|x| x.cloned()).collect::<Vec<_>>())
            .unwrap(),
        vec![Some(25), None]
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_and() {
    let names = VarBinArray::from_iter(
        vec![Some("Joseph"), None, Some("Angela"), Some("Mikhail"), None],
        DType::Utf8(Nullability::Nullable),
    );
    let ages = PrimitiveArray::from_option_iter([Some(25), Some(31), None, Some(57), None]);
    let st = StructArray::try_new(
        ["name".into(), "age".into()].into(),
        vec![names.into_array(), ages.into_array()],
        5,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();

    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), st.to_array_stream())
        .await
        .unwrap();

    let result = VortexOpenOptions::in_memory(buf)
        .open()
        .await
        .unwrap()
        .scan()
        .with_filter(and(
            gt(get_item("age", ident()), lit(21)),
            lt_eq(get_item("age", ident()), lit(33)),
        ))
        .into_array_stream()
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    let names = result[0]
        .as_struct_typed()
        .unwrap()
        .maybe_null_field_by_idx(0)
        .unwrap();
    assert_eq!(
        names
            .to_varbinview()
            .unwrap()
            .with_iterator(|iter| iter
                .map(|s| s.map(|st| unsafe { String::from_utf8_unchecked(st.to_vec()) }))
                .collect::<Vec<_>>())
            .unwrap(),
        vec![Some("Joseph".to_string()), None]
    );
    let ages = result[0]
        .as_struct_typed()
        .unwrap()
        .maybe_null_field_by_idx(1)
        .unwrap();
    assert_eq!(ages.to_primitive().unwrap().as_slice::<i32>(), vec![25, 31]);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_with_indices_simple() {
    let expected_numbers_split: Vec<Buffer<i16>> = (0..5).map(|_| (0_i16..100).collect()).collect();
    let expected_array = StructArray::from_fields(&[(
        "numbers",
        ChunkedArray::from_iter(
            expected_numbers_split
                .iter()
                .cloned()
                .map(IntoArray::into_array),
        )
        .into_array(),
    )])
    .unwrap();
    let expected_numbers: Vec<i16> = expected_numbers_split.into_iter().flatten().collect();

    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), expected_array.to_array_stream())
        .await
        .unwrap();

    let file = VortexOpenOptions::in_memory(buf).open().await.unwrap();

    // test no indices
    let actual_kept_array = file
        .scan()
        .with_row_indices(Buffer::<u64>::empty())
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap();

    assert_eq!(actual_kept_array.len(), 0);

    // test a few indices
    let kept_indices = [0_u64, 3, 99, 100, 101, 399, 400, 401, 499];

    let actual_kept_array = file
        .scan()
        .with_row_indices(Buffer::from_iter(kept_indices))
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap();
    let actual_kept_numbers_array = actual_kept_array
        .maybe_null_field_by_idx(0)
        .unwrap()
        .to_primitive()
        .unwrap();

    let expected_kept_numbers: Vec<i16> = kept_indices
        .iter()
        .map(|&x| expected_numbers[x as usize])
        .collect();
    let actual_kept_numbers = actual_kept_numbers_array.as_slice::<i16>();

    assert_eq!(expected_kept_numbers, actual_kept_numbers);

    // test all indices
    let actual_array = file
        .scan()
        .with_row_indices((0u64..500).collect::<Buffer<_>>())
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap();
    let actual_numbers_array = actual_array
        .maybe_null_field_by_idx(0)
        .unwrap()
        .to_primitive()
        .unwrap();
    let actual_numbers = actual_numbers_array.as_slice::<i16>();

    assert_eq!(expected_numbers, actual_numbers);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_with_indices_on_two_columns() {
    let strings_expected = ["ab", "foo", "bar", "baz", "ab", "foo", "bar", "baz"];
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(strings_expected[..4].to_vec()).into_array(),
        VarBinArray::from(strings_expected[4..].to_vec()).into_array(),
    ])
    .into_array();

    let numbers_expected = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let numbers = ChunkedArray::from_iter([
        Buffer::copy_from(&numbers_expected[..4]).into_array(),
        Buffer::copy_from(&numbers_expected[4..]).into_array(),
    ])
    .into_array();

    let st = StructArray::from_fields(&[("strings", strings), ("numbers", numbers)]).unwrap();
    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), st.to_array_stream())
        .await
        .unwrap();

    let file = VortexOpenOptions::in_memory(buf).open().await.unwrap();

    let kept_indices = [0_u64, 3, 7];
    let array = file
        .scan()
        .with_row_indices(Buffer::from_iter(kept_indices))
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap()
        .to_struct()
        .unwrap();

    let strings_actual = array
        .maybe_null_field_by_idx(0)
        .unwrap()
        .to_varbinview()
        .unwrap()
        .with_iterator(|x| {
            x.map(|x| unsafe { String::from_utf8_unchecked(x.unwrap().to_vec()) })
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(
        strings_actual,
        kept_indices
            .iter()
            .map(|&x| strings_expected[x as usize])
            .collect::<Vec<_>>()
    );

    let numbers_actual_array = array
        .maybe_null_field_by_idx(1)
        .unwrap()
        .to_primitive()
        .unwrap();
    let numbers_actual = numbers_actual_array.as_slice::<u32>();
    assert_eq!(
        numbers_actual,
        kept_indices
            .iter()
            .map(|&x| numbers_expected[x as usize])
            .collect::<Vec<u32>>()
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_with_indices_and_with_row_filter_simple() {
    let expected_numbers_split: Vec<Buffer<i16>> = (0..5).map(|_| (0_i16..100).collect()).collect();
    let expected_array = StructArray::from_fields(&[(
        "numbers",
        ChunkedArray::from_iter(
            expected_numbers_split
                .iter()
                .cloned()
                .map(IntoArray::into_array),
        )
        .into_array(),
    )])
    .unwrap();
    let expected_numbers: Vec<i16> = expected_numbers_split.into_iter().flatten().collect();

    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), expected_array.to_array_stream())
        .await
        .unwrap();

    let file = VortexOpenOptions::in_memory(buf).open().await.unwrap();

    let actual_kept_array = file
        .scan()
        .with_filter(gt(get_item("numbers", ident()), lit(50_i16)))
        .with_row_indices(Buffer::empty())
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap();

    assert_eq!(actual_kept_array.len(), 0);

    // test a few indices
    let kept_indices = [0u64, 3, 99, 100, 101, 399, 400, 401, 499];

    let actual_kept_array = file
        .scan()
        .with_filter(gt(get_item("numbers", ident()), lit(50_i16)))
        .with_row_indices(Buffer::from_iter(kept_indices))
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap();

    let actual_kept_numbers_array = actual_kept_array
        .maybe_null_field_by_idx(0)
        .unwrap()
        .to_primitive()
        .unwrap();

    let expected_kept_numbers: Buffer<i16> = kept_indices
        .iter()
        .map(|&x| expected_numbers[x as usize])
        .filter(|&x| x > 50)
        .collect();
    let actual_kept_numbers = actual_kept_numbers_array.as_slice::<i16>();

    assert_eq!(expected_kept_numbers.as_slice(), actual_kept_numbers);

    // test all indices
    let actual_array = file
        .scan()
        .with_filter(gt(get_item("numbers", ident()), lit(50_i16)))
        .with_row_indices((0..500).collect::<Buffer<_>>())
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap();

    let actual_numbers_array = actual_array
        .maybe_null_field_by_idx(0)
        .unwrap()
        .to_primitive()
        .unwrap();
    let actual_numbers = actual_numbers_array.as_slice::<i16>();

    assert_eq!(
        expected_numbers
            .iter()
            .filter(|&&x| x > 50)
            .cloned()
            .collect::<Vec<_>>(),
        actual_numbers
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filter_string_chunked() {
    let name_chunk1 =
        VarBinViewArray::from_iter_nullable_str([Some("Joseph"), Some("James"), Some("Angela")])
            .into_array();
    let age_chunk1 = PrimitiveArray::from_option_iter([Some(25_i32), Some(31), None]).into_array();
    let name_chunk2 = VarBinViewArray::from_iter_nullable_str([
        Some("Pharrell".to_owned()),
        Some("Khalil".to_owned()),
        Some("Mikhail".to_owned()),
        None,
    ])
    .into_array();
    let age_chunk2 =
        PrimitiveArray::from_option_iter([Some(57_i32), Some(18), None, Some(32)]).into_array();

    let chunk1 = StructArray::from_fields(&[("name", name_chunk1), ("age", age_chunk1)])
        .unwrap()
        .into_array();
    let chunk2 = StructArray::from_fields(&[("name", name_chunk2), ("age", age_chunk2)])
        .unwrap()
        .into_array();
    let dtype = chunk1.dtype().clone();

    let array = ChunkedArray::try_new(vec![chunk1, chunk2], dtype)
        .unwrap()
        .into_array();

    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), array.to_array_stream())
        .await
        .unwrap();

    let file = VortexOpenOptions::in_memory(buf).open().await.unwrap();

    let actual_array = file
        .scan()
        .with_filter(eq(get_item("name", ident()), lit("Joseph")))
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap();

    assert_eq!(actual_array.len(), 1);
    let names = actual_array.maybe_null_field_by_idx(0).unwrap();
    assert_eq!(
        names
            .to_varbinview()
            .unwrap()
            .with_iterator(|iter| iter
                .flatten()
                .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                .collect::<Vec<_>>())
            .unwrap(),
        vec!["Joseph".to_string()]
    );
    let ages = actual_array.maybe_null_field_by_idx(1).unwrap();
    assert_eq!(ages.to_primitive().unwrap().as_slice::<i32>(), vec![25]);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_pruning_with_or() {
    let letter_chunk1 = VarBinViewArray::from_iter_nullable_str([
        Some("A".to_owned()),
        Some("B".to_owned()),
        Some("D".to_owned()),
    ])
    .into_array();
    let number_chunk1 =
        PrimitiveArray::from_option_iter([Some(25_i32), Some(31), None]).into_array();
    let letter_chunk2 = VarBinViewArray::from_iter_nullable_str([
        Some("G".to_owned()),
        Some("I".to_owned()),
        Some("J".to_owned()),
        None,
    ])
    .into_array();
    let number_chunk2 =
        PrimitiveArray::from_option_iter([Some(4_i32), Some(18), None, Some(21)]).into_array();
    let letter_chunk3 = VarBinViewArray::from_iter_nullable_str([
        Some("L".to_owned()),
        None,
        Some("O".to_owned()),
        Some("P".to_owned()),
    ])
    .into_array();
    let number_chunk3 =
        PrimitiveArray::from_option_iter([Some(10_i32), Some(15), None, Some(22)]).into_array();
    let letter_chunk4 = VarBinViewArray::from_iter_nullable_str([
        Some("X".to_owned()),
        Some("Y".to_owned()),
        Some("Z".to_owned()),
    ])
    .into_array();
    let number_chunk4 =
        PrimitiveArray::from_option_iter([Some(66_i32), Some(77), Some(88)]).into_array();

    let chunk1 = StructArray::from_fields(&[("letter", letter_chunk1), ("number", number_chunk1)])
        .unwrap()
        .into_array();
    let chunk2 = StructArray::from_fields(&[("letter", letter_chunk2), ("number", number_chunk2)])
        .unwrap()
        .into_array();
    let chunk3 = StructArray::from_fields(&[("letter", letter_chunk3), ("number", number_chunk3)])
        .unwrap()
        .into_array();
    let chunk4 = StructArray::from_fields(&[("letter", letter_chunk4), ("number", number_chunk4)])
        .unwrap()
        .into_array();
    let dtype = chunk1.dtype().clone();

    let array = ChunkedArray::try_new(vec![chunk1, chunk2, chunk3, chunk4], dtype)
        .unwrap()
        .into_array();

    let buf = VortexWriteOptions::default()
        .write(ByteBufferMut::empty(), array.to_array_stream())
        .await
        .unwrap();

    let file = VortexOpenOptions::in_memory(buf).open().await.unwrap();

    let actual_array = file
        .scan()
        .with_filter(or(
            lt_eq(get_item("letter", ident()), lit("J")),
            lt(get_item("number", ident()), lit(25)),
        ))
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap();

    assert_eq!(actual_array.len(), 10);
    let letters = actual_array.maybe_null_field_by_idx(0).unwrap();
    assert_eq!(
        letters
            .to_varbinview()
            .unwrap()
            .with_iterator(|iter| iter
                .map(|opt| opt.map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) }))
                .collect::<Vec<_>>())
            .unwrap(),
        vec![
            Some("A".to_string()),
            Some("B".to_string()),
            Some("D".to_string()),
            Some("G".to_string()),
            Some("I".to_string()),
            Some("J".to_string()),
            None,
            Some("L".to_string()),
            None,
            Some("P".to_string())
        ]
    );
    let numbers = actual_array.maybe_null_field_by_idx(1).unwrap();
    assert_eq!(
        (0..numbers.len())
            .map(|index| -> Option<i32> {
                scalar_at(&numbers, index)
                    .unwrap()
                    .as_primitive()
                    .typed_value::<i32>()
            })
            .collect::<Vec<_>>(),
        vec![
            Some(25),
            Some(31),
            None,
            Some(4),
            Some(18),
            None,
            Some(21),
            Some(10),
            Some(15),
            Some(22)
        ]
    );
}

#[tokio::test]
async fn test_repeated_projection() {
    let strings = ChunkedArray::from_iter([
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
        VarBinArray::from(vec!["ab", "foo", "bar", "baz"]).into_array(),
    ])
    .into_array();

    let single_column_array = StructArray::from_fields(&[("strings", strings.clone())])
        .unwrap()
        .into_array();

    let expected = StructArray::from_fields(&[("strings", strings.clone()), ("strings", strings)])
        .unwrap()
        .into_array();

    let buf = VortexWriteOptions::default()
        .write(
            ByteBufferMut::empty(),
            single_column_array.to_array_stream(),
        )
        .await
        .unwrap();

    let file = VortexOpenOptions::in_memory(buf).open().await.unwrap();

    let actual = file
        .scan()
        .with_projection(select(["strings".into(), "strings".into()], ident()))
        .into_array()
        .await
        .unwrap()
        .to_struct()
        .unwrap();

    assert_eq!(
        (0..actual.len())
            .map(|index| scalar_at(&actual, index).unwrap())
            .collect_vec(),
        (0..expected.len())
            .map(|index| scalar_at(&expected, index).unwrap())
            .collect_vec()
    );
}

async fn chunked_file() -> VortexResult<VortexFile<InMemoryVortexFile>> {
    let array = ChunkedArray::from_iter([
        buffer![0, 1, 2].into_array(),
        buffer![3, 4, 5].into_array(),
        buffer![6, 7, 8].into_array(),
    ])
    .into_array();

    let buffer: Bytes = VortexWriteOptions::default()
        .write(vec![], array.to_array_stream())
        .await?
        .into();
    VortexOpenOptions::in_memory(buffer).open().await
}

#[tokio::test]
async fn basic_file_roundtrip() -> VortexResult<()> {
    let vxf = chunked_file().await?;
    let result = vxf.scan().into_array().await?.to_primitive()?;

    assert_eq!(result.as_slice::<i32>(), &[0, 1, 2, 3, 4, 5, 6, 7, 8]);

    Ok(())
}

#[tokio::test]
async fn file_excluding_dtype() -> VortexResult<()> {
    let array = ChunkedArray::from_iter([
        buffer![0, 1, 2].into_array(),
        buffer![3, 4, 5].into_array(),
        buffer![6, 7, 8].into_array(),
    ])
    .into_array();
    let dtype = array.dtype().clone();

    let buffer: Bytes = VortexWriteOptions::default()
        .exclude_dtype()
        .write(vec![], array.to_array_stream())
        .await?
        .into();

    // Fail to open without DType.
    let vxf = VortexOpenOptions::in_memory(buffer.clone()).open().await;
    assert!(vxf.is_err(), "Opening without DType should fail");

    let vxf = VortexOpenOptions::in_memory(buffer)
        .with_dtype(dtype.clone())
        .open()
        .await?;
    assert_eq!(vxf.dtype(), &dtype);
    assert_eq!(vxf.row_count(), 9);

    Ok(())
}

#[tokio::test]
async fn file_take() -> VortexResult<()> {
    let vxf = chunked_file().await?;
    let result = vxf
        .scan()
        .with_row_indices(buffer![0, 1, 8])
        .into_array()
        .await?
        .to_primitive()?;

    assert_eq!(result.as_slice::<i32>(), &[0, 1, 8]);

    Ok(())
}
