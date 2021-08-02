def print_config(config):
    if config is None:
        return "None"
    else:
        return f"'{config}'"


class StatusError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message, status_code):
        # TODO: log the traces on every caught exception

        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class Status400Error(StatusError):
    """Exception raised if the response must be a 400 status code.

    Attributes:
        message -- the content of the response
    """

    def __init__(self, message):
        super().__init__(message, 400)


class Status404Error(StatusError):
    """Exception raised if the response must be a 404 status code.

    Attributes:
        message -- the content of the response
    """

    def __init__(self, message):
        super().__init__(message, 404)
