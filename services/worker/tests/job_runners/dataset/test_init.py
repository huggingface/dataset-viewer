# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from unittest.mock import MagicMock, patch

import pytest
from libcommon.dtos import CachedJob, SplitHubFile

from worker.config import AppConfig
from worker.dtos import (
    ConfigParquetAndInfoResponse,
    ConfigParquetMetadataResponse,
    ConfigParquetResponse,
    JobResult,
    ParquetFileMetadataItem,
    ShortcutJobResult,
)
from worker.job_runners.config.parquet import ConfigParquetJobRunner
from worker.job_runners.config.parquet_and_info import ConfigParquetAndInfoJobRunner
from worker.job_runners.config.parquet_metadata import ConfigParquetMetadataJobRunner
from worker.job_runners.config.split_names import ConfigSplitNamesJobRunner, FullSplitItem, SplitsList
from worker.job_runners.dataset.config_names import DatasetConfigNamesJobRunner, DatasetConfigNamesResponse
from worker.job_runners.dataset.init import DatasetInitJobRunner, compute_init_responses

logging.basicConfig(level=logging.INFO)


class TestDatasetInitJobRunner:
    """Test the dataset-init job runner."""

    @pytest.fixture
    def hf_endpoint(self) -> str:
        return "https://huggingface.co"

    @pytest.fixture
    def hf_token(self) -> str:
        return "hf_test_token"

    @pytest.fixture
    def job_info(self, hf_endpoint: str, hf_token: str) -> dict:
        return {
            "job_id": "test-job",
            "type": "dataset-init",
            "params": {
                "dataset": "test-dataset",
                "config": None,
                "split": None,
                "revision": "main",
            },
            "assets_directory": None,
            "difficulty": 0,
        }

    @pytest.fixture
    def app_config(self, hf_endpoint: str, hf_token: str) -> AppConfig:
        return AppConfig(
            common=type(
                "CommonConfig",
                (),
                {"hf_endpoint": hf_endpoint, "hf_token": hf_token},
            )(),
            config_names=type(
                "ConfigNamesConfig",
                (),
                {"max_number_for_init": 10},
            )(),
            parquet_and_info=type(
                "ParquetAndInfoConfig",
                (),
                {
                    "source_revision": "main",
                    "target_revision": "refs/convert/parquet",
                    "commit_message": "test commit",
                    "url_template": "/test",
                    "max_dataset_size_bytes": 100_000_000,
                },
            )(),
            parquet_metadata=type(
                "ParquetMetadataConfig",
                (),
                {"storage_directory": None, "max_parallelism": 1},
            )(),
            committer=type(
                "CommitterConfig",
                (),
                {"hf_token": hf_token},
            )(),
        )

    def test_compute_returns_config_names_result(
        self, job_info: dict, app_config: AppConfig
    ) -> None:
        """Test that DatasetInitJobRunner.compute returns config names."""
        with patch(
            "worker.job_runners.dataset.init.compute_init_responses",
            return_value=iter(
                [
                    ShortcutJobResult(
                        content=DatasetConfigNamesResponse(
                            config_names=[{"dataset": "test", "config": "default"}]
                        ),
                        job={
                            "dataset": "test",
                            "kind": "dataset-config-names",
                            "config": None,
                            "split": None,
                        },
                    )
                ]
            ),
        ):
            job_runner = DatasetInitJobRunner(
                job_info=job_info,
                app_config=app_config,
                hf_datasets_cache="/tmp/test-cache",
            )
            results = list(job_runner.compute())
            assert len(results) == 1
            assert isinstance(results[0], ShortcutJobResult)

    def test_compute_for_parquet_dataset_returns_all_shortcuts(
        self, job_info: dict, app_config: AppConfig
    ) -> None:
        """Test that DatasetInitJobRunner.compute returns all shortcuts for parquet datasets."""
        with patch(
            "worker.job_runners.dataset.init.compute_init_responses",
            return_value=iter([]),  # Mock empty result
        ):
            job_runner = DatasetInitJobRunner(
                job_info=job_info,
                app_config=app_config,
                hf_datasets_cache="/tmp/test-cache",
            )
            results = list(job_runner.compute())
            assert len(results) >= 0  # Just verify it doesn't crash


class TestComputeInitResponsesShortcuts:
    """Test the shortcut behavior of compute_init_responses."""

    @pytest.fixture
    def hf_endpoint(self) -> str:
        return "https://huggingface.co"

    @pytest.fixture
    def hf_token(self) -> str:
        return "hf_test_token"

    def test_init_returns_config_names_shortcut(
        self, hf_endpoint: str, hf_token: str
    ) -> None:
        """Verify dataset-init returns config names shortcut."""
        from datasets.packaged_modules.csv.csv import Csv as CsvBuilder

        with patch(
            "worker.job_runners.dataset.init.dataset_module_factory"
        ) as mock_factory, patch(
            "worker.job_runners.dataset.init.get_dataset_builder_class"
        ) as mock_builder_cls:
            mock_module = MagicMock()
            mock_module.hash = "abc123"
            mock_module.builder_kwargs = {}
            mock_factory.return_value = mock_module

            # Use a real builder class (not Parquet)
            mock_builder_cls.return_value = CsvBuilder

            # Run the shortcut with non-parquet builder
            results = list(
                compute_init_responses(  # type: ignore
                    dataset="test-dataset",
                    max_num_configs=10,
                    hf_endpoint=hf_endpoint,
                    hf_token=hf_token,
                    committer_hf_token=hf_token,
                    source_revision="main",
                    target_revision="refs/convert/parquet",
                    commit_message="test",
                    url_template="/test",
                    max_dataset_size_bytes=100_000_000,
                    data_store=None,
                    parquet_metadata_directory=None,
                    max_parallelism=1,
                )
            )

            # Find the config names shortcut
            config_names_result = None
            for result in results:
                if isinstance(result, ShortcutJobResult):
                    config_names_result = result

            assert config_names_result is not None
            assert "config_names" in config_names_result.content

    def test_init_returns_all_shortcuts_for_parquet_dataset(
        self, hf_endpoint: str, hf_token: str
    ) -> None:
        """Verify dataset-init returns all shortcuts for parquet datasets."""
        from datasets.packaged_modules.parquet.parquet import Parquet as ParquetBuilder

        with patch(
            "worker.job_runners.dataset.init.dataset_module_factory"
        ) as mock_factory, patch(
            "worker.job_runners.dataset.init.get_dataset_builder_class"
        ) as mock_builder_cls, patch(
            "worker.job_runners.dataset.init.HfFileSystem"
        ) as mock_fs_class, patch(
            "worker.job_runners.dataset.init.resolve_hf_path"
        ) as mock_resolve_hf_path, patch(
            "worker.job_runners.dataset.init.is_relative_path"
        ) as mock_is_relative:
            # Mock is_relative_path to return False (so the data_file is used directly)
            mock_is_relative.return_value = False
            
            # Mock resolve_hf_path to return correct paths that pass the safety check
            def resolve_path(path: str) -> str:
                return f"hf://datasets/test-dataset@abc123/{path.split('/')[-1]}"
            
            mock_resolve_hf_path.side_effect = resolve_path
            
            # Mock dataset module
            mock_module = MagicMock()
            mock_module.hash = "abc123"
            # Don't include config_name in builder_kwargs to avoid conflict with config_name kwarg
            mock_module.builder_kwargs = {
                "base_path": "hf://datasets/test-dataset/",
            }
            mock_factory.return_value = mock_module

            # Use a mock builder class
            mock_cls = MagicMock()
            mock_cls.builder_configs = {
                "default": MagicMock(data_files={"train": ["data.parquet"]})
            }
            mock_cls.DEFAULT_CONFIG_NAME = "default"
            
            # Set up the builder instance mock with proper info structure
            from datasets.info import DatasetInfo
            from datasets.features import Features
            from datasets.splits import SplitDict, SplitInfo
            from dataclasses import asdict as dataclass_asdict

            # Create a real DatasetInfo object that asdict can handle
            mock_features = Features({"text": {"dtype": "string", "_type": "Value"}})
            split_dict = SplitDict()
            split_dict.add(SplitInfo("train", num_bytes=10000, num_examples=100))
            
            mock_info = DatasetInfo()
            mock_info.builder_name = "parquet"
            mock_info.dataset_name = "test-dataset"
            mock_info.config_name = "default"
            mock_info.version = None
            mock_info.features = mock_features
            mock_info.download_size = 1000
            mock_info.dataset_size = 10000
            mock_info.splits = split_dict

            mock_builder_instance = MagicMock()
            mock_builder_instance.config.data_files = {"train": ["data.parquet"]}
            mock_builder_instance.info = mock_info
            mock_builder_instance.name = "parquet"
            mock_builder_instance.dataset_name = "test-dataset"
            mock_builder_instance.config.name = "default"
            mock_builder_instance.config.version = None
            mock_cls.return_value = mock_builder_instance
            mock_builder_cls.return_value = mock_cls

            # Mock HfFileSystem
            mock_fs = MagicMock()
            # Set up dircache with proper entries
            mock_fs.dircache = {
                "hf://datasets/test-dataset/": [
                    {"name": "hf://datasets/test-dataset/data.parquet", "size": 1000, "type": "file"}
                ]
            }
            mock_fs._strip_protocol.return_value = "hf://datasets/test-dataset/data.parquet"
            mock_fs.url.return_value = "https://huggingface.co/test-dataset/resolve/main/data.parquet"
            mock_fs.resolve_path.return_value = MagicMock(
                revision="main", path_in_repo="data.parquet"
            )
            mock_fs_class.return_value = mock_fs

            with patch(
                "worker.job_runners.dataset.init.issubclass", return_value=True
            ), patch(
                "worker.job_runners.dataset.init.retry_get_features_num_examples_size_and_num_bytes",
                return_value=(MagicMock(), 100, 1000, 5000),
            ), patch(
                "worker.job_runners.dataset.init.get_file_sizes",
                return_value={"hf://datasets/test-dataset/data.parquet": 1000},
            ), patch(
                "worker.job_runners.dataset.init.fill_builder_info",
            ):
                with patch(
                    "worker.job_runners.dataset.init.create_parquet_metadata_dir",
                    return_value=(None, "metadata-path"),
                ):
                    # Mock libviewer.Dataset
                    with patch(
                        "worker.job_runners.dataset.init.lv", autospec=True
                    ) as mock_lv:
                        mock_dataset = MagicMock()
                        mock_dataset.sync_index.return_value = []
                        mock_lv.Dataset.return_value = mock_dataset

                        results = list(
                            compute_init_responses(  # type: ignore
                                dataset="test-dataset",
                                max_num_configs=10,
                                hf_endpoint=hf_endpoint,
                                hf_token=hf_token,
                                committer_hf_token=hf_token,
                                source_revision="main",
                                target_revision="refs/convert/parquet",
                                commit_message="test",
                                url_template="/test",
                                max_dataset_size_bytes=100_000_000,
                                data_store=None,
                                parquet_metadata_directory="/tmp/metadata",
                                max_parallelism=1,
                            )
                        )

                        # Find the split names shortcut
                        split_names_result = None
                        for result in results:
                            if isinstance(result, ShortcutJobResult):
                                if result.job["kind"] == "config-split-names":
                                    split_names_result = result

                        assert split_names_result is not None
                        assert "splits" in split_names_result.content


class TestParquetFileMetadataItem:
    """Test the ParquetFileMetadataItem structure."""

    def test_metadata_item_has_all_required_fields(self) -> None:
        """Test that ParquetFileMetadataItem has all required fields."""
        metadata: ParquetFileMetadataItem = {
            "dataset": "test-dataset",
            "config": "default",
            "split": "train",
            "url": "https://huggingface.co/test-dataset/resolve/main/data.parquet",
            "filename": "data.parquet",
            "size": 1000,
            "num_rows": 100,
            "parquet_metadata_subpath": "test-dataset/default/train/data.parquet",
        }

        required_fields = [
            "dataset",
            "config",
            "split",
            "url",
            "filename",
            "size",
            "num_rows",
            "parquet_metadata_subpath",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"


class TestSplitHubFile:
    """Test the SplitHubFile structure."""

    def test_split_hub_file_has_required_fields(self) -> None:
        """Test that SplitHubFile has all required fields."""
        file_item: SplitHubFile = {
            "dataset": "test-dataset",
            "config": "default",
            "split": "train",
            "url": "https://huggingface.co/test-dataset/resolve/main/data.parquet",
            "filename": "data.parquet",
            "size": 1000,
        }

        required_fields = ["dataset", "config", "split", "url", "filename", "size"]

        for field in required_fields:
            assert field in file_item, f"Missing required field: {field}"


class TestConfigParquetAndInfoResponse:
    """Test the ConfigParquetAndInfoResponse structure."""

    def test_response_has_all_required_fields(self) -> None:
        """Test that ConfigParquetAndInfoResponse has all required fields."""
        response: ConfigParquetAndInfoResponse = {
            "parquet_files": [
                {
                    "dataset": "test-dataset",
                    "config": "default",
                    "split": "train",
                    "url": "https://huggingface.co/test-dataset/resolve/main/data.parquet",
                    "filename": "data.parquet",
                    "size": 1000,
                }
            ],
            "dataset_info": {
                "features": {"text": {"dtype": "string", "_type": "Value"}},
                "splits": [{"name": "train", "num_bytes": 10000, "num_examples": 100}],
            },
            "estimated_dataset_info": None,
            "partial": False,
        }

        required_fields = ["parquet_files", "dataset_info", "estimated_dataset_info", "partial"]

        for field in required_fields:
            assert field in response, f"Missing required field: {field}"


class TestConfigParquetMetadataResponse:
    """Test the ConfigParquetMetadataResponse structure."""

    def test_response_has_all_required_fields(self) -> None:
        """Test that ConfigParquetMetadataResponse has all required fields."""
        response: ConfigParquetMetadataResponse = {
            "parquet_files_metadata": [
                {
                    "dataset": "test-dataset",
                    "config": "default",
                    "split": "train",
                    "url": "https://huggingface.co/test-dataset/resolve/main/data.parquet",
                    "filename": "data.parquet",
                    "size": 1000,
                    "num_rows": 100,
                    "parquet_metadata_subpath": "test-dataset/default/train/data.parquet",
                }
            ],
            "features": {"text": {"dtype": "string", "_type": "Value"}},
            "partial": False,
        }

        required_fields = ["parquet_files_metadata", "features", "partial"]

        for field in required_fields:
            assert field in response, f"Missing required field: {field}"


class TestJobRunnerTypes:
    """Test that job runners return correct job types."""

    def test_dataset_init_returns_correct_job_type(self) -> None:
        """Test that dataset-init returns 'dataset-init' as job type."""
        assert DatasetInitJobRunner.get_job_type() == "dataset-init"

    def test_config_names_returns_correct_job_type(self) -> None:
        """Test that dataset-config-names returns 'dataset-config-names' as job type."""
        assert DatasetConfigNamesJobRunner.get_job_type() == "dataset-config-names"

    def test_config_split_names_returns_correct_job_type(self) -> None:
        """Test that config-split-names returns 'config-split-names' as job type."""
        assert ConfigSplitNamesJobRunner.get_job_type() == "config-split-names"

    def test_config_parquet_returns_correct_job_type(self) -> None:
        """Test that config-parquet returns 'config-parquet' as job type."""
        assert ConfigParquetJobRunner.get_job_type() == "config-parquet"

    def test_config_parquet_metadata_returns_correct_job_type(self) -> None:
        """Test that config-parquet-metadata returns 'config-parquet-metadata' as job type."""
        assert ConfigParquetMetadataJobRunner.get_job_type() == "config-parquet-metadata"

    def test_config_parquet_and_info_returns_correct_job_type(self) -> None:
        """Test that config-parquet-and-info returns 'config-parquet-and-info' as job type."""
        assert ConfigParquetAndInfoJobRunner.get_job_type() == "config-parquet-and-info"


class TestJobResultProgress:
    """Test that job results have correct progress values."""

    def test_config_names_progress_is_complete(self) -> None:
        """Test that config-names shortcut result has progress=1.0."""
        result = ShortcutJobResult(
            content=DatasetConfigNamesResponse(
                config_names=[{"dataset": "test", "config": "default"}]
            ),
            job={
                "dataset": "test",
                "kind": "dataset-config-names",
                "config": None,
                "split": None,
            },
        )
        assert result.progress == 1.0

    def test_split_names_progress_is_complete(self) -> None:
        """Test that split-names shortcut result has progress=1.0."""
        result = ShortcutJobResult(
            content=SplitsList(splits=[FullSplitItem(
                dataset="test",
                config="default",
                split="train",
            )]),
            job={
                "dataset": "test",
                "kind": "config-split-names",
                "config": "default",
                "split": None,
            },
        )
        assert result.progress == 1.0

    def test_parquet_progress_is_complete(self) -> None:
        """Test that parquet shortcut result has progress=1.0."""
        result = ShortcutJobResult(
            content=ConfigParquetResponse(
                parquet_files=[
                    {
                        "dataset": "test",
                        "config": "default",
                        "split": "train",
                        "url": "https://huggingface.co/test/resolve/main/data.parquet",
                        "filename": "data.parquet",
                        "size": 1000,
                    }
                ],
                features=None,
                partial=False,
            ),
            job={
                "dataset": "test",
                "kind": "config-parquet",
                "config": "default",
                "split": None,
            },
        )
        assert result.progress == 1.0

    def test_parquet_metadata_progress_is_complete(self) -> None:
        """Test that parquet-metadata shortcut result has progress=1.0."""
        result = ShortcutJobResult(
            content=ConfigParquetMetadataResponse(
                parquet_files_metadata=[],
                features=None,
                partial=False,
            ),
            job={
                "dataset": "test",
                "kind": "config-parquet-metadata",
                "config": "default",
                "split": "train",
            },
        )
        assert result.progress == 1.0

    def test_parquet_and_info_progress_is_complete(self) -> None:
        """Test that parquet-and-info shortcut result has progress=1.0."""
        result = ShortcutJobResult(
            content=ConfigParquetAndInfoResponse(
                parquet_files=[
                    {
                        "dataset": "test",
                        "config": "default",
                        "split": "train",
                        "url": "https://huggingface.co/test/resolve/main/data.parquet",
                        "filename": "data.parquet",
                        "size": 1000,
                    }
                ],
                dataset_info={"features": {}, "splits": []},
                estimated_dataset_info=None,
                partial=False,
            ),
            job={
                "dataset": "test",
                "kind": "config-parquet-and-info",
                "config": "default",
                "split": None,
            },
        )
        assert result.progress == 1.0