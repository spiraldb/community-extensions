#![feature(exit_status_error)]

use std::env::temp_dir;
use std::fs::{create_dir_all, File};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use arrow_array::{RecordBatch, RecordBatchReader};
use blob::SlowObjectStoreRegistry;
use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::cache::cache_unit::{DefaultFileStatisticsCache, DefaultListFilesCache};
use datafusion::execution::object_store::DefaultObjectStoreRegistry;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_physical_plan::{collect, ExecutionPlan};
use itertools::Itertools;
use log::LevelFilter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
use vortex::array::ChunkedArray;
use vortex::arrow::FromArrowType;
use vortex::compress::CompressionStrategy;
use vortex::dtype::DType;
use vortex::encodings::fastlanes::DeltaEncoding;
use vortex::sampling_compressor::SamplingCompressor;
use vortex::{ArrayData, Context, ContextRef, IntoArrayData};

use crate::data_downloads::FileType;
use crate::reader::BATCH_SIZE;
use crate::taxi_data::taxi_data_parquet;

pub mod blob;
pub mod clickbench;
pub mod data_downloads;
pub mod display;
pub mod parquet_utils;
pub mod public_bi_data;
pub mod reader;
pub mod taxi_data;
pub mod tpch;
pub mod vortex_utils;

// Sizes match default compressor configuration
const TARGET_BLOCK_BYTESIZE: usize = 16 * (1 << 20);
const TARGET_BLOCK_SIZE: usize = 64 * (1 << 10);

pub static CTX: LazyLock<ContextRef> = LazyLock::new(|| {
    Arc::new(
        Context::default()
            .with_encodings(SamplingCompressor::default().used_encodings())
            .with_encoding(&DeltaEncoding),
    )
});

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Format {
    Csv,
    Arrow,
    Parquet,
    InMemoryVortex,
    OnDiskVortex { enable_compression: bool },
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Format::Csv => write!(f, "csv"),
            Format::Arrow => write!(f, "arrow"),
            Format::Parquet => write!(f, "parquet"),
            Format::InMemoryVortex => {
                write!(f, "in_memory_vortex")
            }
            Format::OnDiskVortex { enable_compression } => {
                write!(f, "on_disk_vortex(compressed={enable_compression})")
            }
        }
    }
}

impl Format {
    pub fn name(&self) -> String {
        match self {
            Format::Csv => "csv".to_string(),
            Format::Arrow => "arrow".to_string(),
            Format::Parquet => "parquet".to_string(),
            Format::InMemoryVortex => "vortex-in-memory".to_string(),
            Format::OnDiskVortex { enable_compression } => if *enable_compression {
                "vortex-file-compressed"
            } else {
                "vortex-file-uncompressed"
            }
            .to_string(),
        }
    }
}

/// Creates a file if it doesn't already exist.
/// NB: Does NOT modify the given path to ensure that it resides in the data directory.
pub fn idempotent<T, E, P: IdempotentPath + ?Sized>(
    path: &P,
    f: impl FnOnce(&Path) -> Result<T, E>,
) -> Result<PathBuf, E> {
    let data_path = path.to_data_path();
    if !data_path.exists() {
        let temp_location = path.to_temp_path();
        let temp_path = temp_location.as_path();
        f(temp_path)?;
        std::fs::rename(temp_path, &data_path).unwrap();
    }
    Ok(data_path)
}

pub async fn idempotent_async<T, E, F, P>(
    path: &P,
    f: impl FnOnce(PathBuf) -> F,
) -> Result<PathBuf, E>
where
    F: Future<Output = Result<T, E>>,
    P: IdempotentPath + ?Sized,
{
    let data_path = path.to_data_path();
    if !data_path.exists() {
        let temp_location = path.to_temp_path();
        f(temp_location.clone()).await?;
        std::fs::rename(temp_location.as_path(), &data_path).unwrap();
    }
    Ok(data_path)
}

pub trait IdempotentPath {
    fn to_data_path(&self) -> PathBuf;
    fn to_temp_path(&self) -> PathBuf;
}

impl IdempotentPath for str {
    fn to_data_path(&self) -> PathBuf {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join(self);
        if !path.parent().unwrap().exists() {
            create_dir_all(path.parent().unwrap()).unwrap();
        }
        path
    }

    fn to_temp_path(&self) -> PathBuf {
        let temp_dir = temp_dir().join(uuid::Uuid::new_v4().to_string());
        if !temp_dir.exists() {
            create_dir_all(temp_dir.clone()).unwrap();
        }
        temp_dir.join(self)
    }
}

impl IdempotentPath for PathBuf {
    fn to_data_path(&self) -> PathBuf {
        if !self.parent().unwrap().exists() {
            create_dir_all(self.parent().unwrap()).unwrap();
        }
        self.to_path_buf()
    }

    fn to_temp_path(&self) -> PathBuf {
        let temp_dir = std::env::temp_dir().join(uuid::Uuid::new_v4().to_string());
        if !temp_dir.exists() {
            create_dir_all(temp_dir.clone()).unwrap();
        }
        temp_dir.join(self.file_name().unwrap())
    }
}

pub fn setup_logger(level: LevelFilter) {
    TermLogger::init(
        level,
        Config::default(),
        TerminalMode::Stderr,
        ColorChoice::Auto,
    )
    .unwrap();
}

pub fn fetch_taxi_data() -> ArrayData {
    let file = File::open(taxi_data_parquet()).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.with_batch_size(BATCH_SIZE).build().unwrap();

    let schema = reader.schema();
    ChunkedArray::try_new(
        reader
            .into_iter()
            .map(|batch_result| batch_result.unwrap())
            .map(ArrayData::try_from)
            .map(Result::unwrap)
            .collect_vec(),
        DType::from_arrow(schema),
    )
    .unwrap()
    .into_array()
}

pub fn compress_taxi_data() -> ArrayData {
    CompressionStrategy::compress(&SamplingCompressor::default(), &fetch_taxi_data()).unwrap()
}

pub struct CompressionRunStats {
    schema: DType,
    total_compressed_size: Option<u64>,
    compressed_sizes: Vec<u64>,
    file_type: FileType,
    file_name: String,
}

impl CompressionRunStats {
    pub fn to_results(&self, dataset_name: String) -> Vec<CompressionRunResults> {
        let DType::Struct(st, _) = &self.schema else {
            unreachable!()
        };

        self.compressed_sizes
            .iter()
            .zip_eq(st.names().iter().zip_eq(st.dtypes()))
            .map(
                |(&size, (column_name, column_type))| CompressionRunResults {
                    dataset_name: dataset_name.clone(),
                    file_name: self.file_name.clone(),
                    file_type: self.file_type.to_string(),
                    column_name: (**column_name).to_string(),
                    column_type: column_type.to_string(),
                    compressed_size: size,
                    total_compressed_size: self.total_compressed_size,
                },
            )
            .collect::<Vec<_>>()
    }
}

pub struct CompressionRunResults {
    pub dataset_name: String,
    pub file_name: String,
    pub file_type: String,
    pub column_name: String,
    pub column_type: String,
    pub compressed_size: u64,
    pub total_compressed_size: Option<u64>,
}

pub async fn execute_query(ctx: &SessionContext, query: &str) -> anyhow::Result<Vec<RecordBatch>> {
    let plan = ctx.sql(query).await?;
    let (state, plan) = plan.into_parts();
    let physical_plan = state.create_physical_plan(&plan).await?;
    let result = collect(physical_plan.clone(), state.task_ctx()).await?;
    Ok(result)
}

pub async fn physical_plan(
    ctx: &SessionContext,
    query: &str,
) -> anyhow::Result<Arc<dyn ExecutionPlan>> {
    let plan = ctx.sql(query).await?;
    let (state, plan) = plan.into_parts();
    Ok(state.create_physical_plan(&plan).await?)
}

#[derive(Clone, Debug)]
pub struct Measurement {
    pub query_idx: usize,
    pub time: Duration,
    pub format: Format,
    pub dataset: String,
}

#[derive(Serialize)]
pub struct JsonValue {
    pub name: String,
    pub unit: String,
    pub value: u128,
    pub commit_id: String,
}

pub static GIT_COMMIT_ID: LazyLock<String> = LazyLock::new(|| {
    String::from_utf8(
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .trim()
    .to_string()
});

impl Measurement {
    pub fn to_json(&self) -> JsonValue {
        let name = format!(
            "{dataset}_q{query_idx:02}/{format}",
            dataset = self.dataset,
            format = self.format.name(),
            query_idx = self.query_idx
        );

        JsonValue {
            name,
            unit: "ns".to_string(),
            value: self.time.as_nanos(),
            commit_id: GIT_COMMIT_ID.to_string(),
        }
    }
}

pub fn get_session_with_cache(emulate_object_store: bool) -> SessionContext {
    let registry = if emulate_object_store {
        Arc::new(SlowObjectStoreRegistry::default()) as _
    } else {
        Arc::new(DefaultObjectStoreRegistry::new()) as _
    };

    let file_static_cache = Arc::new(DefaultFileStatisticsCache::default());
    let list_file_cache = Arc::new(DefaultListFilesCache::default());

    let cache_config = CacheManagerConfig::default()
        .with_files_statistics_cache(Some(file_static_cache))
        .with_list_files_cache(Some(list_file_cache));

    let rt = RuntimeEnvBuilder::new()
        .with_cache_manager(cache_config)
        .with_object_store_registry(registry)
        .build_arc()
        .expect("could not build runtime environment");

    SessionContext::new_with_config_rt(SessionConfig::default(), rt)
}

#[cfg(test)]
mod test {
    use std::fs::File;
    use std::ops::Deref;
    use std::sync::Arc;

    use arrow_array::{ArrayRef as ArrowArrayRef, StructArray as ArrowStructArray};
    use log::LevelFilter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use vortex::arrow::{FromArrowArray, IntoArrowArray};
    use vortex::compress::CompressionStrategy;
    use vortex::sampling_compressor::SamplingCompressor;
    use vortex::ArrayData;

    use crate::taxi_data::taxi_data_parquet;
    use crate::{compress_taxi_data, setup_logger};

    #[ignore]
    #[test]
    fn compression_ratio() {
        setup_logger(LevelFilter::Debug);
        _ = compress_taxi_data();
    }

    #[ignore]
    #[test]
    fn round_trip_arrow() {
        let file = File::open(taxi_data_parquet()).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.with_limit(1).build().unwrap();

        for record_batch in reader.map(|batch_result| batch_result.unwrap()) {
            let struct_arrow: ArrowStructArray = record_batch.into();
            let arrow_array: ArrowArrayRef = Arc::new(struct_arrow);
            let vortex_array = ArrayData::from_arrow(arrow_array.clone(), false);
            let vortex_as_arrow = vortex_array.into_arrow_preferred().unwrap();
            assert_eq!(vortex_as_arrow.deref(), arrow_array.deref());
        }
    }

    // Ignoring since Struct arrays don't currently support equality.
    // https://github.com/apache/arrow-rs/issues/5199
    #[ignore]
    #[test]
    fn round_trip_arrow_compressed() {
        let file = File::open(taxi_data_parquet()).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.with_limit(1).build().unwrap();
        let compressor: &dyn CompressionStrategy = &SamplingCompressor::default();

        for record_batch in reader.map(|batch_result| batch_result.unwrap()) {
            let struct_arrow: ArrowStructArray = record_batch.into();
            let arrow_array: ArrowArrayRef = Arc::new(struct_arrow);
            let vortex_array = ArrayData::from_arrow(arrow_array.clone(), false);

            let compressed = compressor.compress(&vortex_array).unwrap();
            let compressed_as_arrow = compressed.into_arrow_preferred().unwrap();
            assert_eq!(compressed_as_arrow.deref(), arrow_array.deref());
        }
    }
}
