import logging

from typing import List

from datasets import (
    prepare_module,
    import_main_class,
)

from datasets_preview_backend.exceptions import (
    DatasetBuilderScriptError,
    DatasetBuilderNotFoundError,
)

# TODO: log the traces on every caught exception


def get_configs(dataset: str) -> List[str]:
    try:
        module_path, *_ = prepare_module(dataset, dataset=True)
        builder_cls = import_main_class(module_path, dataset=True)
    except FileNotFoundError as err:
        raise DatasetBuilderNotFoundError(dataset=dataset)
    except (ModuleNotFoundError):
        raise DatasetBuilderScriptError(dataset=dataset)

    configs = [c.name for c in builder_cls.BUILDER_CONFIGS] or [None]
    logging.debug(f"The dataset builder has {len(configs)} configs: {configs}")
    return {"dataset": dataset, "configs": configs}
