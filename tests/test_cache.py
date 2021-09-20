from time import sleep

from diskcache import Cache  # type: ignore

from datasets_preview_backend.cache import memoize  # type: ignore


def test_memoize() -> None:
    cache = Cache()

    @memoize(cache=cache, expire=100)  # type: ignore
    def fibonacci(number: int) -> int:
        if number == 0:
            return 0
        elif number == 1:
            return 1
        else:
            return fibonacci(number - 1) + fibonacci(number - 2)  # type: ignore

    result, max_age = fibonacci(20, _return_max_age=True)
    assert result == 6765
    assert max_age == 99

    result, max_age = fibonacci(20, _return_max_age=True)
    assert result == 6765
    assert max_age == 99

    result = fibonacci(20)
    assert result == 6765

    sleep(1)
    result, max_age = fibonacci(20, _return_max_age=True)
    assert result == 6765
    assert max_age == 98


def test_memoize_no_expire() -> None:

    cache = Cache()

    @memoize(cache=cache)  # type: ignore
    def fibonacci(number: int) -> int:
        if number == 0:
            return 0
        elif number == 1:
            return 1
        else:
            return fibonacci(number - 1) + fibonacci(number - 2)  # type: ignore

    result, max_age = fibonacci(20, _return_max_age=True)
    assert result == 6765
    assert max_age is None


def test_memoize_expired() -> None:

    cache = Cache()

    @memoize(cache=cache, expire=2)  # type: ignore
    def fibonacci(number: int) -> int:
        if number == 0:
            return 0
        elif number == 1:
            return 1
        else:
            return fibonacci(number - 1) + fibonacci(number - 2)  # type: ignore

    result, max_age = fibonacci(20, _return_max_age=True)
    assert max_age == 1
    sleep(1)
    result, max_age = fibonacci(20, _return_max_age=True)
    assert max_age == 0
    sleep(1)
    result, max_age = fibonacci(20, _return_max_age=True)
    assert result == 6765
    assert max_age == 1  # the cache had expired, the 2s TTL has restarted
