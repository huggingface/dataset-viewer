def print_config(config):
    if config is None:
        return "None"
    else:
        return f"'{config}'"


class Error(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)


class DatasetBuilderScriptError(Error):
    """Exception raised if the dataset script fails.

    Attributes:
        dataset -- the erroneous dataset id
    """

    def __init__(self, dataset):
        self.dataset = dataset
        super().__init__(f"Dataset builder script error. Dataset: '{self.dataset}'")


class DatasetBuilderNotFoundError(Error):
    """Exception raised if the dataset script could not be found.

    Attributes:
        dataset -- the erroneous dataset id
    """

    def __init__(self, dataset):
        self.dataset = dataset
        super().__init__(
            f"Dataset builder script could not be found. Dataset: '{self.dataset}'"
        )


class DatasetBuilderNoSplitsError(Error):
    """Exception raised if the builder script fails to provide the list of splits.

    Attributes:
        dataset -- the erroneous dataset id
        config -- the erroneous dataset config name
    """

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        super().__init__(
            f"Dataset builder script error: could not get the list of splits. Dataset: '{self.dataset}', config: {print_config(self.config)}"
        )


class DatasetNotFoundError(Error):
    """Exception raised if a dataset has not been found.

    Attributes:
        dataset -- the erroneous dataset id
    """

    def __init__(self, dataset):
        self.dataset = dataset
        super().__init__(f"Dataset not found. Dataset: '{self.dataset}'")


class ConfigNotFoundError(Error):
    """Exception raised for config builder not found.

    Attributes:
        dataset -- the erroneous dataset id
        config -- the erroneous dataset config name
    """

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        super().__init__(
            f"Config not found. Dataset: '{self.dataset}', config: {print_config(self.config)}"
        )


class SplitError(Error):
    """Exception raised for errors in the split.

    Attributes:
        dataset -- the erroneous dataset id
        config -- the erroneous dataset config name
        split -- the erroneous dataset split name
    """

    def __init__(self, dataset, config, split):
        self.dataset = dataset
        self.config = config
        self.split = split
        super().__init__(
            f"Split error. Dataset: '{self.dataset}', config: {print_config(self.config)}, split: '{self.split}'"
        )


class SplitNotImplementedError(Error):
    """Exception raised for NotImplementedError in the split.

    Attributes:
        dataset -- the erroneous dataset id
        config -- the erroneous dataset config name
        split -- the erroneous dataset split name
        extension -- the file extension not implemented yet
    """

    def __init__(self, dataset, config, split, extension):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.extension = extension
        extension_str = (
            "" if self.extension is None else f" for extension '{self.extension}'"
        )
        super().__init__(
            f"Extraction protocol not implemented{extension_str}. Dataset: '{self.dataset}', config: {print_config(self.config)}, split: '{self.split}'"
        )
