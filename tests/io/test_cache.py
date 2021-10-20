from diskcache import Cache  # type: ignore

from datasets_preview_backend.io.cache import memoize  # type: ignore


def test_memoize() -> None:
    cache = Cache()

    @memoize(cache=cache)  # type: ignore
    def fibonacci(number: int) -> int:
        if number == 0:
            return 0
        elif number == 1:
            return 1
        else:
            return fibonacci(number - 1) + fibonacci(number - 2)  # type: ignore

    result = fibonacci(20)
    assert result == 6765

    result = fibonacci(20)
    assert result == 6765


def test_memoize_refresh() -> None:

    cache = Cache()

    @memoize(cache=cache)  # type: ignore
    def fibonacci(number: int) -> int:
        if number == 0:
            return 0
        elif number == 1:
            return 1
        else:
            return fibonacci(number - 1) + fibonacci(number - 2)  # type: ignore

    result = fibonacci(20)
    assert result == 6765
    result = fibonacci(20, _refresh=True)
    assert result == 6765
