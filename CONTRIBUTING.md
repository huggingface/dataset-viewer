# Contributing guide

The repository is structured as a monorepo, with Python applications in [services/](./services/) and Python libraries in [libs/](./libs/).

If you have access to the internal HF notion, see https://www.notion.so/huggingface2/Datasets-server-464848da2a984e999c540a4aa7f0ece5.

## Install

To start working on the project:

```bash
git clone git@github.com:huggingface/datasets-server.git
cd datasets-server
```

Install docker (see https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository and https://docs.docker.com/engine/install/linux-postinstall/)

```
make install
make start-from-local-code
```

To use the docker images already compiled using the CI:

```
make start-from-remote-images
```

Note that you must login to AWS to be able to download the docker images:

```
aws ecr get-login-password --region us-east-1 --profile=hub-prod \
    | docker login --username AWS --password-stdin 707930574880.dkr.ecr.us-east-1.amazonaws.com
```

To install a single library (in [libs](./libs)) or service (in [services](./services)), install Python 3.9 (consider [pyenv](https://github.com/pyenv/pyenv)) and [poetry]](https://python-poetry.org/docs/master/#installation) (don't forget to add `poetry` to the `PATH` environment variable).

If you use pyenv:

```bash
cd libs/libutils/
pyenv install 3.9.6
pyenv local 3.9.6
poetry env use python3.9
```

then:

```
make install
```

It will create a virtual environment in a `./.venv/` subdirectory.

If you use VSCode, it might be useful to use the ["monorepo" workspace](./.vscode/monorepo.code-workspace) (see a [blogpost](https://medium.com/rewrite-tech/visual-studio-code-tips-for-monorepo-development-with-multi-root-workspaces-and-extension-6b69420ecd12) for more explanations). It is a multi-root workspace, with one folder for each library and service (note that we hide them from the ROOT to avoid editing there). Each folder has its own Python interpreter, with access to the dependencies installed by Poetry. You might have to manually select the interpreter in every folder though on first access, then VSCode stores the information in its local storage.

## Quality

The CI checks the quality of the code through a [GitHub action](./.github/workflows/quality.yml). To manually format the code of a library or a service:

```bash
make style
```

To check the quality (which includes checking the style, but also security vulnerabilities):

```bash
make quality
```

## Tests

The CI checks the tests a [GitHub action](./.github/workflows/unit-tests.yml). To manually test a library or a service:

```bash
make test
```

Note that it requires the resources to be ready, ie. mongo and the storage for assets.

To launch the end to end tests:

```bash
make e2e
```

## Poetry

### Versions

We version the [libraries](./libs) as they are dependencies of the [services](./services). To update a library:

- change the version in its pyproject.yaml file
- build with `make build`
- version the new files in `dist/`

And then update the library version in the services that require the update, for example if the library is `libcache`:

```
poetry update libcache
```

If service is updated, we don't update its version in the `pyproject.yaml` file. But we have to update the [docker images file](./chart/docker-images.yaml) with the new image tag. Then the CI will test the new docker images, and we will be able to deploy them to the infrastructure.

## Pull requests

All the contributions should go through a pull request. The pull requests must be "squashed" (ie: one commit per pull request).

## GitHub Actions

You can use [act](https://github.com/nektos/act) to test the GitHub Actions (see [.github/workflows/](.github/workflows/)) locally. It reduces the retroaction loop when working on the GitHub Actions, avoid polluting the branches with empty pushes only meant to trigger the CI, and allows to only run specific actions.

For example, to launch the build and push of the docker images to ECR:

```
act -j build-and-push-image --secret-file my.secrets
```

with `my.secrets` a file with the secrets:

```
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
GITHUB_TOKEN=xxx
```

You might prefer to use [aws-vault](https://github.com/99designs/aws-vault) instead to set the environment variables, but you will still have to pass the GitHub token as a secret.
