from fsspec.registry import known_implementations

for name in list(known_implementations):
    if name not in ["hf", "s3", "zip", "file"]:
        del known_implementations[name]
