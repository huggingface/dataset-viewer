from pathlib import Path
from typing import Any, Literal, Optional

import polars as pl
from datasets.features.features import Features, FeatureType, Translation, TranslationVariableLanguages, Value, _visit
from huggingface_hub.repocard_data import DatasetCardData

from libcommon.constants import ROW_IDX_COLUMN
from libcommon.parquet_utils import (
    PARTIAL_PREFIX,
    is_list_pa_type,
    parquet_export_is_partial,
)
from libcommon.statistics_utils import (
    STRING_DTYPES,
    AudioColumn,
    ImageColumn,
    ListColumn,
    StringColumn,
)

DISABLED_DUCKDB_REF_BRANCH_DATASET_NAME_PATTERN = "*NoDuckdbRef*"

DATASET_TYPE = "dataset"
DEFAULT_STEMMER = "none"  # Exact word matches
DUCKDB_DEFAULT_INDEX_FILENAME = "index.duckdb"
DUCKDB_DEFAULT_PARTIAL_INDEX_FILENAME = "partial-index.duckdb"
CREATE_INDEX_COMMAND = (
    f"PRAGMA create_fts_index('data', '{ROW_IDX_COLUMN}', {{columns}}, stemmer='{{stemmer}}', overwrite=1);"
)
CREATE_TABLE_COMMAND_FROM_LIST_OF_PARQUET_FILES = (
    "CREATE OR REPLACE TABLE data AS SELECT {columns} FROM read_parquet({source});"
)
CREATE_TABLE_JOIN_WITH_TRANSFORMED_DATA_COMMAND_FROM_LIST_OF_PARQUET_FILES = """
    CREATE OR REPLACE TABLE data AS 
    SELECT {columns}, transformed_df.* FROM read_parquet({source}) 
    POSITIONAL JOIN transformed_df;
"""
CREATE_SEQUENCE_COMMAND = "CREATE OR REPLACE SEQUENCE serial START 0 MINVALUE 0;"
ALTER_TABLE_BY_ADDING_SEQUENCE_COLUMN = (
    f"ALTER TABLE data ADD COLUMN {ROW_IDX_COLUMN} BIGINT DEFAULT nextval('serial');"
)
CREATE_INDEX_ID_COLUMN_COMMANDS = CREATE_SEQUENCE_COMMAND + ALTER_TABLE_BY_ADDING_SEQUENCE_COLUMN
INSTALL_AND_LOAD_EXTENSION_COMMAND = "INSTALL 'fts'; LOAD 'fts';"
SET_EXTENSIONS_DIRECTORY_COMMAND = "SET extension_directory='{directory}';"
REPO_TYPE = "dataset"
# Only some languages are supported, see: https://duckdb.org/docs/extensions/full_text_search.html#pragma-create_fts_index
STEMMER_MAPPING = {
    # Stemmer : ["value ISO 639-1", "value ISO 639-2/3"]
    "arabic": ["ar", "ara"],
    "basque": ["eu", "eus"],
    "catalan": ["ca", "cat"],
    "danish": ["da", "dan"],
    "dutch": ["nl", "nld"],
    "english": ["en", "eng"],
    "finnish": ["fi", "fin"],
    "french": ["fr", "fra"],
    "german": ["de", "deu"],
    "greek": ["el", "ell"],
    "hindi": ["hi", "hin"],
    "hungarian": ["hu", "hun"],
    "indonesian": ["id", "ind"],
    "irish": ["ga", "gle"],
    "italian": ["it", "ita"],
    "lithuanian": ["lt", "lit"],
    "nepali": ["ne", "nep"],
    "norwegian": ["no", "nor"],
    "portuguese": ["pt", "por"],
    "romanian": ["ro", "ron"],
    "russian": ["ru", "rus"],
    "serbian": ["sr", "srp"],
    "spanish": ["es", "spa"],
    "swedish": ["sv", "swe"],
    "tamil": ["ta", "tam"],
    "turkish": ["tr", "tur"],
}

LengthDtype = Literal["string", "list"]


def get_indexable_columns(features: Features) -> list[str]:
    indexable_columns: list[str] = []
    for column, feature in features.items():
        indexable = False

        def check_indexable(feature: FeatureType) -> None:
            nonlocal indexable
            if isinstance(feature, Value) and feature.dtype in STRING_DTYPES:
                indexable = True
            elif isinstance(feature, (Translation, TranslationVariableLanguages)):
                indexable = True

        _visit(feature, check_indexable)
        if indexable:
            indexable_columns.append(column)
    return indexable_columns


def get_monolingual_stemmer(card_data: Optional[DatasetCardData]) -> str:
    if card_data is None:
        return DEFAULT_STEMMER
    all_languages = card_data["language"]
    if isinstance(all_languages, list) and len(all_languages) == 1:
        first_language = all_languages[0]
    elif isinstance(all_languages, str):
        first_language = all_languages
    else:
        return DEFAULT_STEMMER

    return next((language for language, codes in STEMMER_MAPPING.items() if first_language in codes), DEFAULT_STEMMER)


def compute_length_column(
    parquet_paths: list[Path],
    column_name: str,
    target_df: Optional[pl.DataFrame],
    dtype: LengthDtype,
) -> pl.DataFrame:
    column_class = ListColumn if dtype == "list" else StringColumn
    df = pl.read_parquet(parquet_paths, columns=[column_name])
    lengths_column_name = f"{column_name}.length"
    lengths_df: pl.DataFrame = column_class.compute_transformed_data(
        df, column_name, transformed_column_name=lengths_column_name
    )
    if target_df is None:
        return lengths_df.select(pl.col(lengths_column_name))

    target_df.insert_column(target_df.shape[1], lengths_df[lengths_column_name])
    return target_df


def compute_audio_duration_column(
    parquet_paths: list[Path],
    column_name: str,
    target_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    duration_column_name = f"{column_name}.duration"
    durations = AudioColumn.compute_transformed_data(parquet_paths, column_name, AudioColumn.get_duration)
    duration_df = pl.from_dict({duration_column_name: durations})
    if target_df is None:
        return duration_df
    target_df.insert_column(target_df.shape[1], duration_df[duration_column_name])
    return target_df


def compute_image_width_height_column(
    parquet_paths: list[Path],
    column_name: str,
    target_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    shapes = ImageColumn.compute_transformed_data(parquet_paths, column_name, ImageColumn.get_shape)
    widths, heights = list(zip(*shapes))
    width_column_name, height_column_name = f"{column_name}.width", f"{column_name}.height"
    shapes_df = pl.from_dict({width_column_name: widths, height_column_name: heights})
    if target_df is None:
        return shapes_df
    target_df.insert_column(target_df.shape[1], shapes_df[width_column_name])
    target_df.insert_column(target_df.shape[1], shapes_df[height_column_name])
    return target_df


def compute_transformed_data(parquet_paths: list[Path], features: dict[str, Any]) -> Optional[pl.DataFrame]:
    transformed_df = None
    for feature_name, feature in features.items():
        if isinstance(feature, list) or (
            isinstance(feature, dict) and feature.get("_type") in ("LargeList", "Sequence")
        ):
            first_parquet_file = parquet_paths[0]
            if is_list_pa_type(first_parquet_file, feature_name):
                transformed_df = compute_length_column(parquet_paths, feature_name, transformed_df, dtype="list")

        elif isinstance(feature, dict):
            if feature.get("_type") == "Value" and feature.get("dtype") in STRING_DTYPES:
                transformed_df = compute_length_column(parquet_paths, feature_name, transformed_df, dtype="string")

            elif feature.get("_type") == "Audio":
                transformed_df = compute_audio_duration_column(parquet_paths, feature_name, transformed_df)

            elif feature.get("_type") == "Image":
                transformed_df = compute_image_width_height_column(parquet_paths, feature_name, transformed_df)

    return transformed_df


def duckdb_index_is_partial(duckdb_index_url: str) -> bool:
    """
    Check if the DuckDB index is on the full dataset or if it's partial.
    It could be partial for two reasons:

    1. if the Parquet export that was used to build it is partial
    2. if it's a partial index of the Parquet export

    Args:
        duckdb_index_url (`str`): The URL of the DuckDB index file.

    Returns:
        `bool`: True is the DuckDB index is partial,
            or False if it's an index of the full dataset.
    """
    _, duckdb_index_file_name = duckdb_index_url.rsplit("/", 1)
    return parquet_export_is_partial(duckdb_index_url) or duckdb_index_file_name.startswith(PARTIAL_PREFIX)
