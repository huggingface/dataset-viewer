# libcache

A Python library to manage the storage of precomputed API responses in a mongo database (the "cache").

## Configuration

Set environment variables to configure the following aspects:

- `CACHE_ASSETS_DIRECTORY`: directory where the asset files are stored. Defaults to empty, in which case the assets are located in the `datasets_server_assets` subdirectory inside the OS default cache directory.
- `CACHE_MONGO_DATABASE`: the name of the database used for storing the cache. Defaults to `"datasets_server_cache"`.
- `CACHE_MONGO_URL`: the URL used to connect to the mongo db server. Defaults to `"mongodb://localhost:27017"`.
