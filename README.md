# Datasets preview backend

> API to extract rows of 🤗 datasets

## Services

- [api_service](./services/api_service): the API
- [job_runner](./services/job_runner): the workers that preprocess API responses for the `/rows` and `/splits` endpoints.

## Python libraries

- [libcache](./libs/libcache): the database that stores the preprocessed API responses for the `/rows` and `/splits` endpoints.
- [libqueue](./libs/libqueue): the database that stores the list of jobs.
- [libutils](./libs/libutils): common code
