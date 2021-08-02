from typing import List

from datasets import (
    load_dataset_builder,
)
from datasets.utils.streaming_download_manager import StreamingDownloadManager

from datasets_preview_backend.exceptions import (
    DatasetBuilderScriptError,
    DatasetBuilderNoSplitsError,
    ConfigNotFoundError,
)

# TODO: log the traces on every caught exception


def get_splits(dataset: str, config: str) -> List[str]:
    try:
        builder = load_dataset_builder(dataset, name=config)
    except ValueError as err:
        message = str(err)
        if message.startswith(f"BuilderConfig {config} not found"):
            raise ConfigNotFoundError(dataset=dataset, config=config)
        elif message.startswith(f"Config name is missing."):
            raise ConfigNotFoundError(dataset=dataset, config=config)
        else:
            raise
    except (ModuleNotFoundError, RuntimeError, TypeError):
        raise DatasetBuilderScriptError(dataset=dataset)

    if builder.info.splits is None:
        # try to get them from _split_generators
        try:
            splits = [
                split_generator.name
                for split_generator in builder._split_generators(
                    StreamingDownloadManager(base_path=builder.base_path)
                )
            ]
        except:
            raise DatasetBuilderNoSplitsError(dataset=dataset, config=config)
    else:
        splits = list(builder.info.splits.keys())
    return {"dataset": dataset, "config": config, "splits": splits}
