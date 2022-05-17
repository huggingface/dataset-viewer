# Install guide

Follow the [general INSTALL](../INSTALL.md) to be sure to setup the assets directory and the databases.

## Requirements

The requirements are:

- Python 3.9.6+ (consider [pyenv](https://github.com/pyenv/pyenv))
- Poetry 1.1.7+
- make

We assume a machine running Ubuntu. Install packages:

```bash
sudo apt install python-is-python3 make
```

Also install node and npm (with [nvm](https://github.com/nvm-sh/nvm)), then:

```bash
npm i -g pm2@latest
```

Also [install poetry](https://python-poetry.org/docs/master/#installation). Don't forget to add `poetry` to the `PATH` environment variable.

## Install and configure

Install the API service:

```bash
cd
# See https://github.blog/2013-09-03-two-factor-authentication/#how-does-it-work-for-command-line-git for authentication
git clone https://github.com/huggingface/datasets-server.git
cd datasets-server/services/admin
make install
```

Copy and edit the environment variables file:

```bash
cd datasets-server/services/admin
cp .env.example .env
vi .env
```

Note that we assume `ASSETS_DIRECTORY=/data` in the nginx configuration. If you set the assets directory to another place, or let the default, ensure the nginx configuration is setup accordingly. Beware: the default directory inside `/home/hf/.cache` is surely not readable by the nginx user.
