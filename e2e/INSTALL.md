# Install guide

Install Python 3.9 (consider [pyenv](https://github.com/pyenv/pyenv)) and [poetry]](https://python-poetry.org/docs/master/#installation) (don't forget to add `poetry` to the `PATH` environment variable).

If you use pyenv:

```bash
cd e2e/
pyenv install 3.9.6
pyenv local 3.9.6
poetry env use python3.9
```

then:

```
make install
```

It will create a virtual environment in a `./.venv/` subdirectory.
