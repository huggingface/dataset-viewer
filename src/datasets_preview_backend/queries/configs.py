import logging

from datasets import import_main_class, prepare_module

from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error


def get_configs(dataset: str):
    try:
        module_path, *_ = prepare_module(dataset, dataset=True)
        builder_cls = import_main_class(module_path, dataset=True)
    except FileNotFoundError as err:
        raise Status404Error("The dataset could not be found.") from err
    except Exception as err:
        raise Status400Error(
            "The config names could not be parsed from the dataset."
        ) from err

    configs = [c.name for c in builder_cls.BUILDER_CONFIGS] or [DEFAULT_CONFIG_NAME]
    logging.debug(f"The dataset builder has {len(configs)} configs: {configs}")
    return {"dataset": dataset, "configs": configs}
