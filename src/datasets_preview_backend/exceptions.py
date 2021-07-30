def print_config(config_name):
    if config_name is None:
        return "None"
    else:
        return f"'{config_name}'"


class Error(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)


class DatasetBuilderScriptError(Error):
    """Exception raised if the dataset script fails.

    Attributes:
        dataset_id -- the erroneous dataset id
    """

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        super().__init__(f"Dataset builder script error. Dataset: '{self.dataset_id}'")


class DatasetBuilderScriptConfigNoSplitsError(Error):
    """Exception raised if the builder script fails for this config.

    Attributes:
        dataset_id -- the erroneous dataset id
        config_name -- the erroneous dataset config_name
    """

    def __init__(self, dataset_id, config_name):
        self.dataset_id = dataset_id
        self.config_name = config_name
        super().__init__(
            f"Dataset builder script error: missing .info.splits. Dataset: '{self.dataset_id}', config: {print_config(self.config_name)}"
        )


class DatasetNotFoundError(Error):
    """Exception raised if a dataset has not been found.

    Attributes:
        dataset_id -- the erroneous dataset id
    """

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        super().__init__(f"Dataset not found. Dataset: '{self.dataset_id}'")


class ConfigNotFoundError(Error):
    """Exception raised for config builder not found.

    Attributes:
        dataset_id -- the erroneous dataset id
        config_name -- the erroneous dataset config_name
    """

    def __init__(self, dataset_id, config_name):
        self.dataset_id = dataset_id
        self.config_name = config_name
        super().__init__(
            f"Config not found. Dataset: '{self.dataset_id}', config: {print_config(self.config_name)}"
        )


class SplitError(Error):
    """Exception raised for errors in the split.

    Attributes:
        dataset_id -- the erroneous dataset id
        config_name -- the erroneous dataset config_name
        split -- the erroneous dataset split
    """

    def __init__(self, dataset_id, config_name, split):
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.split = split
        super().__init__(
            f"Split error. Dataset: '{self.dataset_id}', config: {print_config(self.config_name)}, split: '{self.split}'"
        )


class SplitNotImplementedError(Error):
    """Exception raised for NotImplementedError in the split.

    Attributes:
        dataset_id -- the erroneous dataset id
        config_name -- the erroneous dataset config_name
        split -- the erroneous dataset split
        extension -- the file extension not implemented yet
    """

    def __init__(self, dataset_id, config_name, split, extension):
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.split = split
        self.extension = extension
        extension_str = (
            "" if self.extension is None else f" for extension '{self.extension}'"
        )
        super().__init__(
            f"Extraction protocol not implemented{extension_str}. Dataset: '{self.dataset_id}', config: {print_config(self.config_name)}, split: '{self.split}'"
        )
