# INSTALL

## Requirements

The requirements are:

- node (for pm2)
- Python 3.9.6+
- Poetry 1.1.7+
- make
- libicu-dev
- libsndfile 1.0.30+

We assume a machine running Ubuntu. Install packages:

```bash
sudo apt install python-is-python3 make libicu-dev ffmpeg libavcodec-extra
```

Also install `libsndfile` in version `v1.0.30`. As the version in ubuntu stable for the moment is `v1.0.28`, we can build from scratch (see details here: https://github.com/libsndfile/libsndfile)

```
sudo apt install -y autoconf autogen automake build-essential libasound2-dev libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev libmpg123-dev pkg-config;
cd /tmp;
git clone https://github.com/libsndfile/libsndfile.git;
cd libsndfile;
git checkout v1.0.30;
./autogen.sh;
./configure --enable-werror;
make;
sudo make install;
sudo ldconfig;
cd;
rm -rf /tmp/libsndfile
```

Also install node and npm (with [nvm](https://github.com/nvm-sh/nvm)), then:

```bash
npm i -g pm2@latest
```

Also [install poetry](https://python-poetry.org/docs/master/#installation). Don't forget to add `poetry` to the `PATH` environment variable.

## Install and configure

Install the job runner:

```bash

# See https://github.blog/2013-09-03-two-factor-authentication/#how-does-it-work-for-command-line-git for authentication
git clone https://github.com/huggingface/datasets-preview-backend.git
cd datasets-preview-backend/services/job_runner
make install
```

Copy and edit the environment variables file:

```bash
cd datasets-preview-backend/services/job_runner
cp .env.example .env
vi .env
```

In particular, set the following environment variables to get access to the common resources: `ASSETS_DIRECTORY`, `MONGO_CACHE_DATABASE`, `MONGO_QUEUE_DATABASE` and `MONGO_URL`.

## Deploy

Deploy the datasets job runners with:

```bash
pm2 start --name datasets-worker make -- -C /home/hf/datasets-preview-backend/services/job_runner/ datasets-worker
```

Deploy the splits job runners with:

```bash
pm2 start --name splits-worker make -- -C /home/hf/datasets-preview-backend/services/job_runner/ splits-worker
```

Launch the same command again to deploy one worker more.

Finally, ensure that pm2 will restart on reboot (see https://pm2.keymetrics.io/docs/usage/startup/):

- if it's the first time:
  ```bash
  pm2 startup
  # and follow the instructions
  ```
- else:
  ```bash
  pm2 save
  ```

Note that once a worker has processed a job, or has encountered an error, it quits. `pm2` will then restart the worker automatically, so that it can process the following jobs. Exiting after every job, instead of looping on the jobs, has two benefits: memory leaks are reduced, and we don't have to manage specifically a runtime error.

## Manage

Use [pm2](https://pm2.keymetrics.io/docs/usage/quick-start/#cheatsheet) to manage the workers.

```bash
pm2 list
pm2 logs
```

## Upgrade

To deploy a new version of the job-runner, first update the code

```
cd /home/hf/datasets-preview-backend/
git fetch --tags
git checkout XXXX # <- the latest release tag (https://github.com/huggingface/datasets-preview-backend/releases/latest)
```

If the Python version has been increased to 3.9.6, for example, [run](https://stackoverflow.com/a/65589331/7351594):

```
cd services/job_runner
pyenv install 3.9.6
pyenv local 3.9.6
poetry env use python3.9
```

Install the dependencies

```
make install
```

Check is new environment variables are available and edit the environment variables in `.env`:

```
cd services/job_runner
diff .env.example .env
vi .env
```

Apply the database migrations (see [libs/libcache/src/libcache/migrations/README.md](./../../libs/libcache/migrations/README.md)) if any (in this case: ensure to upgrade the other services too).

```
# see https://github.com/huggingface/datasets-preview-backend/blob/main/libs/libcache/migrations/README.md
```

If you want to be extra-sure, check that all the tests are passing

```
make test
```

Restart

```
pm2 restart datasets-worker
pm2 restart splits-worker
```
