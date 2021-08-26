class StatusError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message, status_code):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

    def as_dict(self):
        return {
            "status_code": self.status_code,
            "exception": type(self).__name__,
            "message": str(self),
            "cause": type(self.__cause__).__name__,
            "cause_message": str(self.__cause__),
        }


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
