import logging
import json
from dataclasses import asdict

from typing import List

from datasets import (
    prepare_module,
    import_main_class,
)

from datasets_preview_backend.exceptions import (
    Status400Error,
    Status404Error,
)


def get_info(dataset: str) -> any:
    try:
        module_path, *_ = prepare_module(dataset, dataset=True)
        builder_cls = import_main_class(module_path, dataset=True)
        total_dataset_infos = builder_cls.get_all_exported_dataset_infos()
        info = {config_name: asdict(config_info) for config_name, config_info in total_dataset_infos.items()}
    except FileNotFoundError as err:
        raise Status404Error("The dataset info could not be found.") from err
    except Exception as err:
        raise Status400Error(
            "The dataset info could not be parsed from the dataset."
        ) from err

    return {"dataset": dataset, "info": info}
