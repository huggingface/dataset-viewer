import logging

from typing import List

from datasets import (
    prepare_module,
    import_main_class,
)

from datasets_preview_backend.exceptions import (
    Status400Error,
    Status404Error,
)


def get_configs(dataset: str) -> List[str]:
    try:
        module_path, *_ = prepare_module(dataset, dataset=True)
        builder_cls = import_main_class(module_path, dataset=True)
    except FileNotFoundError as err:
        raise Status404Error("The dataset could not be found.") from err
    except Exception as err:
        raise Status400Error(
            "The config names could not be parsed from the dataset."
        ) from err

    configs = [c.name for c in builder_cls.BUILDER_CONFIGS] or [None]
    logging.debug(f"The dataset builder has {len(configs)} configs: {configs}")
    return {"dataset": dataset, "configs": configs}
