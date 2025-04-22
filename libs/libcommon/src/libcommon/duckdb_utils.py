import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Optional

import duckdb
import polars as pl
from datasets.features.features import Features, FeatureType, Translation, TranslationVariableLanguages, Value, _visit
from huggingface_hub.repocard_data import DatasetCardData
from tqdm.contrib.concurrent import thread_map

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

DISABLED_DUCKDB_REF_BRANCH_DATASET_NAME_PATTERNS = [
    "vevotx/*",
    "openai/*",
    "EleutherAI/*",
    "HuggingFaceFW/*",
    "TIGER-Lab/*",
    "Rapidata/*",  # images
    "MrDragonFox/*",  # audios
    "*NoDuckdbRef*",
]

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


def create_index(
    database: str,
    input_table: str,
    columns: list[str],
    stemmer: str,
    input_id: str,
    fts_schema: str,
    extensions_directory: Optional[str] = None,
) -> None:
    placeholders: dict[str, str] = {
        "database": database,
        "input_table": input_table,
        "stemmer": stemmer,
        "input_id": input_id,
        "fts_schema": fts_schema,
    }

    def _sql(con: duckdb.DuckDBPyConnection, query: str) -> duckdb.DuckDBPyRelation:
        query = dedent(query)
        for key, value in placeholders.items():
            query = query.replace(f"%{key}%", value)
        out = con.sql(query)
        return out

    with tempfile.TemporaryDirectory(suffix=".duckdb") as tmp_dir:
        with duckdb.connect(":memory:") as con:
            # configure duckdb extensions
            if extensions_directory is not None:
                con.execute(SET_EXTENSIONS_DIRECTORY_COMMAND.format(directory=extensions_directory))
            con.execute(INSTALL_AND_LOAD_EXTENSION_COMMAND)

            # init
            _sql(con, "ATTACH '%database%' as db;")
            _sql(con, "USE db;")

            # ingest data
            _sql(
                con,
                "CREATE TABLE IF NOT EXISTS %input_table% AS SELECT *, row_number() OVER () AS %input_id% FROM read_parquet('data/split_0/partial-split_0/*.parquet');",
            )
            count = _sql(con, "SELECT count(*) FROM %input_table%;").fetchone()[0]

            # create fts schema
            _sql(con, "DROP SCHEMA IF EXISTS %fts_schema% CASCADE;")
            _sql(con, "CREATE SCHEMA %fts_schema%;")

            # define stopwords
            _sql(con, "CREATE TABLE %fts_schema%.stopwords (sw VARCHAR);")
            _sql(
                con,
                "INSERT INTO %fts_schema%.stopwords VALUES ('a'), ('a''s'), ('able'), ('about'), ('above'), ('according'), ('accordingly'), ('across'), ('actually'), ('after'), ('afterwards'), ('again'), ('against'), ('ain''t'), ('all'), ('allow'), ('allows'), ('almost'), ('alone'), ('along'), ('already'), ('also'), ('although'), ('always'), ('am'), ('among'), ('amongst'), ('an'), ('and'), ('another'), ('any'), ('anybody'), ('anyhow'), ('anyone'), ('anything'), ('anyway'), ('anyways'), ('anywhere'), ('apart'), ('appear'), ('appreciate'), ('appropriate'), ('are'), ('aren''t'), ('around'), ('as'), ('aside'), ('ask'), ('asking'), ('associated'), ('at'), ('available'), ('away'), ('awfully'), ('b'), ('be'), ('became'), ('because'), ('become'), ('becomes'), ('becoming'), ('been'), ('before'), ('beforehand'), ('behind'), ('being'), ('believe'), ('below'), ('beside'), ('besides'), ('best'), ('better'), ('between'), ('beyond'), ('both'), ('brief'), ('but'), ('by'), ('c'), ('c''mon'), ('c''s'), ('came'), ('can'), ('can''t'), ('cannot'), ('cant'), ('cause'), ('causes'), ('certain'), ('certainly'), ('changes'), ('clearly'), ('co'), ('com'), ('come'), ('comes'), ('concerning'), ('consequently'), ('consider'), ('considering'), ('contain'), ('containing'), ('contains'), ('corresponding'), ('could'), ('couldn''t'), ('course'), ('currently'), ('d'), ('definitely'), ('described'), ('despite'), ('did'), ('didn''t'), ('different'), ('do'), ('does'), ('doesn''t'), ('doing'), ('don''t'), ('done'), ('down'), ('downwards'), ('during'), ('e'), ('each'), ('edu'), ('eg'), ('eight'), ('either'), ('else'), ('elsewhere'), ('enough'), ('entirely'), ('especially'), ('et'), ('etc'), ('even'), ('ever'), ('every'), ('everybody'), ('everyone'), ('everything'), ('everywhere'), ('ex'), ('exactly'), ('example'), ('except'), ('f'), ('far'), ('few'), ('fifth'), ('first'), ('five'), ('followed'), ('following'), ('follows'), ('for'), ('former'), ('formerly'), ('forth'), ('four'), ('from'), ('further'), ('furthermore'), ('g'), ('get'), ('gets'), ('getting'), ('given'), ('gives'), ('go'), ('goes'), ('going'), ('gone'), ('got'), ('gotten'), ('greetings'), ('h'), ('had'), ('hadn''t'), ('happens'), ('hardly'), ('has'), ('hasn''t'), ('have'), ('haven''t'), ('having'), ('he'), ('he''s'), ('hello'), ('help'), ('hence'), ('her'), ('here'), ('here''s'), ('hereafter'), ('hereby'), ('herein'), ('hereupon'), ('hers'), ('herself'), ('hi'), ('him'), ('himself'), ('his'), ('hither'), ('hopefully'), ('how'), ('howbeit'), ('however'), ('i'), ('i''d'), ('i''ll'), ('i''m'), ('i''ve'), ('ie'), ('if'), ('ignored'), ('immediate'), ('in'), ('inasmuch'), ('inc'), ('indeed'), ('indicate'), ('indicated'), ('indicates'), ('inner'), ('insofar'), ('instead'), ('into'), ('inward'), ('is'), ('isn''t'), ('it'), ('it''d'), ('it''ll'), ('it''s'), ('its'), ('itself'), ('j'), ('just'), ('k'), ('keep'), ('keeps'), ('kept'), ('know'), ('knows'), ('known'), ('l'), ('last'), ('lately'), ('later'), ('latter'), ('latterly'), ('least'), ('less'), ('lest'), ('let'), ('let''s'), ('like'), ('liked'), ('likely'), ('little'), ('look'), ('looking'), ('looks'), ('ltd'), ('m'), ('mainly'), ('many'), ('may'), ('maybe'), ('me'), ('mean'), ('meanwhile'), ('merely'), ('might'), ('more'), ('moreover'), ('most'), ('mostly'), ('much'), ('must'), ('my'), ('myself'), ('n'), ('name'), ('namely'), ('nd'), ('near'), ('nearly'), ('necessary'), ('need'), ('needs'), ('neither'), ('never'), ('nevertheless'), ('new'), ('next'), ('nine'), ('no'), ('nobody'), ('non'), ('none'), ('noone'), ('nor'), ('normally'), ('not'), ('nothing'), ('novel'), ('now'), ('nowhere'), ('o'), ('obviously'), ('of'), ('off'), ('often'), ('oh'), ('ok'), ('okay'), ('old'), ('on'), ('once'), ('one'), ('ones'), ('only'), ('onto'), ('or'), ('other'), ('others'), ('otherwise'), ('ought'), ('our'), ('ours'), ('ourselves'), ('out'), ('outside'), ('over'), ('overall'), ('own');",
            )
            _sql(
                con,
                "INSERT INTO %fts_schema%.stopwords VALUES ('p'), ('particular'), ('particularly'), ('per'), ('perhaps'), ('placed'), ('please'), ('plus'), ('possible'), ('presumably'), ('probably'), ('provides'), ('q'), ('que'), ('quite'), ('qv'), ('r'), ('rather'), ('rd'), ('re'), ('really'), ('reasonably'), ('regarding'), ('regardless'), ('regards'), ('relatively'), ('respectively'), ('right'), ('s'), ('said'), ('same'), ('saw'), ('say'), ('saying'), ('says'), ('second'), ('secondly'), ('see'), ('seeing'), ('seem'), ('seemed'), ('seeming'), ('seems'), ('seen'), ('self'), ('selves'), ('sensible'), ('sent'), ('serious'), ('seriously'), ('seven'), ('several'), ('shall'), ('she'), ('should'), ('shouldn''t'), ('since'), ('six'), ('so'), ('some'), ('somebody'), ('somehow'), ('someone'), ('something'), ('sometime'), ('sometimes'), ('somewhat'), ('somewhere'), ('soon'), ('sorry'), ('specified'), ('specify'), ('specifying'), ('still'), ('sub'), ('such'), ('sup'), ('sure'), ('t'), ('t''s'), ('take'), ('taken'), ('tell'), ('tends'), ('th'), ('than'), ('thank'), ('thanks'), ('thanx'), ('that'), ('that''s'), ('thats'), ('the'), ('their'), ('theirs'), ('them'), ('themselves'), ('then'), ('thence'), ('there'), ('there''s'), ('thereafter'), ('thereby'), ('therefore'), ('therein'), ('theres'), ('thereupon'), ('these'), ('they'), ('they''d'), ('they''ll'), ('they''re'), ('they''ve'), ('think'), ('third'), ('this'), ('thorough'), ('thoroughly'), ('those'), ('though'), ('three'), ('through'), ('throughout'), ('thru'), ('thus'), ('to'), ('together'), ('too'), ('took'), ('toward'), ('towards'), ('tried'), ('tries'), ('truly'), ('try'), ('trying'), ('twice'), ('two'), ('u'), ('un'), ('under'), ('unfortunately'), ('unless'), ('unlikely'), ('until'), ('unto'), ('up'), ('upon'), ('us'), ('use'), ('used'), ('useful'), ('uses'), ('using'), ('usually'), ('uucp'), ('v'), ('value'), ('various'), ('very'), ('via'), ('viz'), ('vs'), ('w'), ('want'), ('wants'), ('was'), ('wasn''t'), ('way'), ('we'), ('we''d'), ('we''ll'), ('we''re'), ('we''ve'), ('welcome'), ('well'), ('went'), ('were'), ('weren''t'), ('what'), ('what''s'), ('whatever'), ('when'), ('whence'), ('whenever'), ('where'), ('where''s'), ('whereafter'), ('whereas'), ('whereby'), ('wherein'), ('whereupon'), ('wherever'), ('whether'), ('which'), ('while'), ('whither'), ('who'), ('who''s'), ('whoever'), ('whole'), ('whom'), ('whose'), ('why'), ('will'), ('willing'), ('wish'), ('with'), ('within'), ('without'), ('won''t'), ('wonder'), ('would'), ('would'), ('wouldn''t'), ('x'), ('y'), ('yes'), ('yet'), ('you'), ('you''d'), ('you''ll'), ('you''re'), ('you''ve'), ('your'), ('yours'), ('yourself'), ('yourselves'), ('z'), ('zero');",
            )

            # define tokenize macro
            _sql(
                con,
                "CREATE MACRO %fts_schema%.tokenize(s) AS string_split_regex(regexp_replace(lower(strip_accents(s::VARCHAR)), '[^a-z]', ' ', 'g'), '\s+');",
            )

            # create fields table
            field_values = ", ".join(f"({i}, '{field}')" for i, field in enumerate(columns))
            _sql(
                con,
                """
                CREATE TABLE %fts_schema%.docs AS (
                    SELECT rowid AS docid,
                        "%input_id%" AS name
                    FROM %input_table%
                );
            """,
            )
            _sql(con, "CREATE TABLE %fts_schema%.fields (fieldid BIGINT, field VARCHAR);")
            _sql(
                con,
                f"INSERT INTO %fts_schema%.fields VALUES {field_values};",
            )
            _sql(con, "CHECKPOINT;")

        # tokenize in parallel (see https://github.com/duckdb/duckdb-fts/issues/7)
        num_jobs = min(16, count // 4)
        batch_size = 1 + count // num_jobs
        commands = [
            (
                (
                    SET_EXTENSIONS_DIRECTORY_COMMAND.format(directory=extensions_directory)
                    if extensions_directory is not None
                    else ""
                )
                + INSTALL_AND_LOAD_EXTENSION_COMMAND
                + (
                    "ATTACH '%database%' as db (READ_ONLY);"
                    "USE db;"
                    f"ATTACH '{tmp_dir}/tmp_{rank}_{i}.duckdb' as tmp_{rank}_{i};"
                    f"""
                    CREATE TABLE tmp_{rank}_{i}.tokenized AS (
                        SELECT unnest(%fts_schema%.tokenize(fts_ii."{column}")) AS w,
                            {rank * batch_size} + row_number() OVER () - 1 AS docid,
                            {i} AS fieldid
                        FROM (
                            SELECT * FROM %input_table% LIMIT {batch_size} OFFSET {rank * batch_size}
                        ) AS fts_ii
                    );
                    CHECKPOINT;
                    """
                )
            )
            for rank in range(num_jobs)
            for i, column in enumerate(columns)
        ]

        def _parallel_sql(command: str) -> None:
            with duckdb.connect(":memory:") as rank_con:
                _sql(rank_con, command)

        thread_map(_parallel_sql, commands, desc="Tokenize")

        # # NON-PARALEL VERSION HERE FOR DOCUMENTATION:
        #
        # for i, column in enumerate(columns):
        #     _sql(con, f"""
        #         CREATE TABLE tmp.tokenized_{i} AS (
        #             SELECT unnest(%fts_schema%.tokenize(fts_ii."{column}")) AS w,
        #                 rowid AS docid,
        #                 {i} AS fieldid
        #             FROM %input_table% AS fts_ii
        #         )
        #     """)
        # union_fields_query = " UNION ALL ".join(f"SELECT * FROM tmp.tokenized_{i}" for i in range(len(columns)))

        with duckdb.connect(":memory:") as con:
            # configure duckdb extensions
            if extensions_directory is not None:
                con.execute(SET_EXTENSIONS_DIRECTORY_COMMAND.format(directory=extensions_directory))
            con.execute(INSTALL_AND_LOAD_EXTENSION_COMMAND)

            # init
            _sql(con, f"ATTACH '{tmp_dir}/tmp.duckdb' as tmp;")
            _sql(con, "ATTACH '%database%' as db;")
            _sql(con, "USE db;")
            _sql(
                con,
                ";".join(
                    f"ATTACH '{tmp_dir}/tmp_{rank}_{i}.duckdb' as tmp_{rank}_{i} (READ_ONLY);"
                    for rank in range(num_jobs)
                    for i in range(len(columns))
                ),
            )

            # merge tokenizations
            union_fields_query = " UNION ALL ".join(
                f"SELECT * FROM tmp_{rank}_{i}.tokenized" for rank in range(num_jobs) for i in range(len(columns))
            )
            _sql(con, f"CREATE TABLE tmp.tokenized AS {union_fields_query}")

            # step and stop
            _sql(
                con,
                """
                CREATE TABLE tmp.stemmed_stopped AS (
                    SELECT stem(t.w, '%stemmer%') AS term,
                        t.docid AS docid,
                        t.fieldid AS fieldid
                    FROM tmp.tokenized AS t
                    WHERE t.w NOT NULL
                    AND len(t.w) > 0
                    AND t.w NOT IN (SELECT sw FROM %fts_schema%.stopwords)
                )
            """,
            )

            # create terms table
            _sql(
                con,
                """
                CREATE TABLE %fts_schema%.terms AS (
                    SELECT ss.term,
                        ss.docid,
                        ss.fieldid
                    FROM tmp.stemmed_stopped AS ss
                )
            """,
            )

            # add doc lengths
            _sql(con, "ALTER TABLE %fts_schema%.docs ADD len BIGINT;")
            _sql(
                con,
                """
                UPDATE %fts_schema%.docs d
                SET len = (
                    SELECT count(term)
                    FROM %fts_schema%.terms AS t
                    WHERE t.docid = d.docid
                );
            """,
            )

            # create dictionary
            _sql(
                con,
                """
                CREATE TABLE tmp.distinct_terms AS (
                    SELECT DISTINCT term
                    FROM %fts_schema%.terms
                    ORDER BY docid, term
                )
            """,
            )
            _sql(
                con,
                """
                CREATE TABLE %fts_schema%.dict AS (
                    SELECT row_number() OVER () - 1 AS termid,
                    dt.term
                    FROM tmp.distinct_terms AS dt
                )
            """,
            )
            _sql(con, "ALTER TABLE %fts_schema%.terms ADD termid BIGINT;")
            _sql(
                con,
                """
                UPDATE %fts_schema%.terms t
                SET termid = (
                    SELECT termid
                    FROM %fts_schema%.dict d
                    WHERE t.term = d.term
                );
            """,
            )
            _sql(con, "ALTER TABLE %fts_schema%.terms DROP term;")

            # compute df
            _sql(con, "ALTER TABLE %fts_schema%.dict ADD df BIGINT;")
            _sql(
                con,
                """
                UPDATE %fts_schema%.dict d
                SET df = (
                    SELECT count(distinct docid)
                    FROM %fts_schema%.terms t
                    WHERE d.termid = t.termid
                    GROUP BY termid
                );
            """,
            )

            # compute stats
            _sql(
                con,
                """
                CREATE TABLE %fts_schema%.stats AS (
                    SELECT COUNT(docs.docid) AS num_docs,
                        SUM(docs.len) / COUNT(docs.len) AS avgdl
                    FROM %fts_schema%.docs AS docs
                );
            """,
            )

            # define match_bm25
            _sql(
                con,
                """
                CREATE MACRO %fts_schema%.match_bm25(docname, query_string, fields := NULL, k := 1.2, b := 0.75, conjunctive := false) AS (
                    WITH tokens AS (
                        SELECT DISTINCT stem(unnest(%fts_schema%.tokenize(query_string)), '%stemmer%') AS t
                    ),
                    fieldids AS (
                        SELECT fieldid
                        FROM %fts_schema%.fields
                        WHERE CASE WHEN fields IS NULL THEN 1 ELSE field IN (SELECT * FROM (SELECT UNNEST(string_split(fields, ','))) AS fsq) END
                    ),
                    qtermids AS (
                        SELECT termid
                        FROM %fts_schema%.dict AS dict,
                            tokens
                        WHERE dict.term = tokens.t
                    ),
                    qterms AS (
                        SELECT termid,
                            docid
                        FROM %fts_schema%.terms AS terms
                        WHERE CASE WHEN fields IS NULL THEN 1 ELSE fieldid IN (SELECT * FROM fieldids) END
                        AND termid IN (SELECT qtermids.termid FROM qtermids)
                    ),
                    term_tf AS (
                        SELECT termid,
                                docid,
                            COUNT(*) AS tf
                        FROM qterms
                        GROUP BY docid,
                                termid
                    ),
                    cdocs AS (
                        SELECT docid
                        FROM qterms
                        GROUP BY docid
                        HAVING CASE WHEN conjunctive THEN COUNT(DISTINCT termid) = (SELECT COUNT(*) FROM tokens) ELSE 1 END
                    ),
                    subscores AS (
                        SELECT docs.docid,
                            len,
                            term_tf.termid,
                            tf,
                            df,
                            (log(((SELECT num_docs FROM %fts_schema%.stats) - df + 0.5) / (df + 0.5) + 1) * ((tf * (k + 1)/(tf + k * (1 - b + b * (len / (SELECT avgdl FROM %fts_schema%.stats))))))) AS subscore
                        FROM term_tf,
                            cdocs,
                            %fts_schema%.docs AS docs,
                            %fts_schema%.dict AS dict
                        WHERE term_tf.docid = cdocs.docid
                        AND term_tf.docid = docs.docid
                        AND term_tf.termid = dict.termid
                    ),
                    scores AS (
                        SELECT docid,
                            sum(subscore) AS score
                        FROM subscores
                        GROUP BY docid
                    )
                    SELECT score
                    FROM scores,
                        %fts_schema%.docs AS docs
                    WHERE scores.docid = docs.docid
                    AND docs.name = docname
                );
            """,
            )
            _sql(con, "CHECKPOINT;")
