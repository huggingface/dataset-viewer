# type: ignore

import functools as ft
from time import time

from diskcache.core import ENOVAL, args_to_key, full_name

# this function is complex. It's basically a copy of "diskcache" code:
# https://github.com/grantjenks/python-diskcache/blob/e1d7c4aaa6729178ca3216f4c8a75b835f963022/diskcache/core.py#L1812
# where we:
# - add the special `_refresh` argument, inspired by django-cache-memoize:
# https://github.com/peterbe/django-cache-memoize/blob/4da1ba4639774426fa928d4a461626e6f841b4f3/src/cache_memoize/__init__.py#L123 # noqa
# - add the special `_return_expire` argument, inspired by the later
# - add the `__cache__` wrapper property

# beware: the cache keys are different if the arguments are passed as args or kwargs!
# that's why we enforce to pass the arguments as kwargs


def memoize(
    cache,
    name=None,
    typed=False,
    expire=None,
    tag=None,
):
    """Memoizing cache decorator.

    Decorator to wrap callable with memoizing function using cache.
    Repeated calls with the same arguments will lookup result in cache and
    avoid function evaluation.

    If name is set to None (default), the callable name will be determined
    automatically.

    When expire is set to zero, function results will not be set in the
    cache. Cache lookups still occur, however. Read
    :doc:`case-study-landing-page-caching` for example usage.

    If typed is set to True, function arguments of different types will be
    cached separately. For example, f(3) and f(3.0) will be treated as
    distinct calls with distinct results.

    The original underlying function is accessible through the __wrapped__
    attribute. This is useful for introspection, for bypassing the cache,
    or for rewrapping the function with a different cache.

    >>> from diskcache import Cache
    >>> from datasets_preview_backend.cache import memoize
    >>> cache = Cache()
    >>> @memoize(cache, expire=1, tag='fib')
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
    `_return_max_age` set to True (default is False) will return a tuple
    (value, max_age) where max_age is the number of seconds until
    expiration, or None if no expiry.

    Calling the memoized function with the special boolean argument
    `_refresh` set to True (default is False) will bypass the cache and
    refresh the value

    An additional `__cache__` attribute can be used to access the cache.

    An additional `__cache_key__` attribute can be used to generate the
    cache key used for the given arguments.

    >>> key = fibonacci.__cache_key__(100)
    >>> print(cache[key])
    354224848179261915075

    Remember to call memoize when decorating a callable. If you forget,
    then a TypeError will occur. Note the lack of parenthenses after
    memoize below:

    >>> @memoize
    ... def test():
    ...     pass
    Traceback (most recent call last):
        ...
    TypeError: name cannot be callable

    :param cache: cache to store callable arguments and return values
    :param str name: name given for callable (default None, automatic)
    :param bool typed: cache different types separately (default False)
    :param float expire: seconds until arguments expire
        (default None, no expiry)
    :param str tag: text to associate with arguments (default None)
    :return: callable decorator

    """
    # Caution: Nearly identical code exists in DjangoCache.memoize
    if callable(name):
        raise TypeError("name cannot be callable")

    if not (expire is None or expire > 0):
        raise TypeError("expire argument is not valid")

    def decorator(func):
        "Decorator created by memoize() for callable `func`."
        base = (full_name(func),) if name is None else (name,)

        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            "Wrapper for callable to cache arguments and return values."
            # The cache key string should never be dependent on special keyword
            # arguments like _refresh and _return_max_age. So extract them into
            # variables as soon as possible.
            _refresh = bool(kwargs.pop("_refresh", False))
            _return_max_age = bool(kwargs.pop("_return_max_age", False))
            # disable the token argument, and don't take it into account for the key
            kwargs.pop("token", False)
            key = wrapper.__cache_key__(*args, **kwargs)
            if _refresh:
                result = ENOVAL
            else:
                result, expire_time = cache.get(key, default=ENOVAL, retry=True, expire_time=True)
                max_age = None if expire_time is None else int(expire_time - time())

            if result is ENOVAL:
                result = func(*args, **kwargs)
                cache.set(key, result, expire, tag=tag, retry=True)
                max_age = expire

            return (result, max_age) if _return_max_age else result

        def __cache_key__(*args, **kwargs):
            "Make key for cache given function arguments."
            return args_to_key(base, args, kwargs, typed=False)

        wrapper.__cache_key__ = __cache_key__
        wrapper.__cache__ = cache
        return wrapper

    return decorator
