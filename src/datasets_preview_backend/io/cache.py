# type: ignore

import functools as ft
import logging

from appdirs import user_cache_dir
from diskcache import Cache
from diskcache.core import ENOVAL, args_to_key, full_name

from datasets_preview_backend.config import (
    CACHE_DIRECTORY,
    CACHE_PERSIST,
    CACHE_SIZE_LIMIT,
)
from datasets_preview_backend.exceptions import StatusError

logger = logging.getLogger(__name__)

# singleton
cache_directory = None
if CACHE_PERSIST:
    if CACHE_DIRECTORY is None:
        # set it to the default cache location on the machine, in order to
        # persist the cache between runs
        cache_directory = user_cache_dir("datasets_preview_backend")
    else:
        cache_directory = CACHE_DIRECTORY

cache = Cache(directory=cache_directory, size_limit=CACHE_SIZE_LIMIT)


class CacheNotFoundError(Exception):
    pass


def show_cache_dir() -> None:
    logger.info(f"Cache directory: {cache.directory}")


# this function is complex. It's basically a copy of "diskcache" code:
# https://github.com/grantjenks/python-diskcache/blob/e1d7c4aaa6729178ca3216f4c8a75b835f963022/diskcache/core.py#L1812
# where we:
# - add the special `_refresh` argument, inspired by django-cache-memoize:
# https://github.com/peterbe/django-cache-memoize/blob/4da1ba4639774426fa928d4a461626e6f841b4f3/src/cache_memoize/__init__.py#L123 # noqa
# - add the special `_lookup` argument
# - removed what we don't use
#
# beware: the cache keys are different if the arguments are passed as args or kwargs!
# that's why we enforce to pass the arguments as kwargs


def memoize(
    cache: Cache,
):
    """Memoizing cache decorator.

    Decorator to wrap callable with memoizing function using cache.
    Repeated calls with the same arguments will lookup result in cache and
    avoid function evaluation.

    The original underlying function is accessible through the __wrapped__
    attribute. This is useful for introspection, for bypassing the cache,
    or for rewrapping the function with a different cache.

    >>> from diskcache import Cache
    >>> from datasets_preview_backend.cache import memoize
    >>> cache = Cache()
    >>> @memoize(cache)
    ... def fibonacci(number):
    ...     if number == 0:
    ...         return 0
    ...     elif number == 1:
    ...         return 1
    ...     else:
    ...         return fibonacci(number - 1) + fibonacci(number - 2)
    >>> print(fibonacci(100))
    354224848179261915075

    Calling the memoized function with the special boolean argument
    `_refresh` set to True (default is False) will bypass the cache and
    refresh the value

    Calling the memoized function with the special boolean argument
    `_lookup` set to True (default is False) will bypass the cache refresh
    and return diskcache.core.ENOVAL in case of cache miss.

    Calling the memoized function with the special boolean argument
    `_delete` set to True (default is False) will delete the cache entry after
    getting the value.

    :param cache: cache to store callable arguments and return values
    :return: callable decorator

    """
    # Caution: Nearly identical code exists in DjangoCache.memoize
    def decorator(func):
        "Decorator created by memoize() for callable `func`."
        base = (full_name(func),)

        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            "Wrapper for callable to cache arguments and return values."
            # The cache key string should never be dependent on special keyword
            # arguments like _refresh. So extract them into
            # variables as soon as possible.
            _refresh = bool(kwargs.pop("_refresh", False))
            _lookup = bool(kwargs.pop("_lookup", False))
            _delete = bool(kwargs.pop("_delete", False))
            key = args_to_key(base, args, kwargs, typed=False)

            if _delete:
                cache.delete(key, retry=True)
                return

            result = ENOVAL if _refresh else cache.get(key, default=ENOVAL, retry=True)

            if result is ENOVAL:
                if _lookup:
                    raise CacheNotFoundError()
                # If the function raises an exception we want to cache,
                # catch it, else let it propagate.
                try:
                    result = func(*args, **kwargs)
                except StatusError as exception:
                    result = exception
                cache.set(key, result, retry=True)

            # See https://github.com/peterbe/django-cache-memoize/blob/master/src/cache_memoize/__init__.py#L153-L156
            # If the result is an exception we've caught and cached, raise it
            # in the end as to not change the API of the function we're caching.
            if isinstance(result, StatusError):
                raise result
            return result

        return wrapper

    return decorator
