# e2e

End to end tests, written in Python

Install (see [INSTALL.md](./INSTALL.md))

You must also login to AWS to be able to download the docker images:

```
aws ecr get-login-password --region us-east-1 --profile=hub-prod \
    | docker login --username AWS --password-stdin 707930574880.dkr.ecr.us-east-1.amazonaws.com
```

Then:

```
make e2e
```
