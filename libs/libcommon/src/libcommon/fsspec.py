from fsspec.registry import known_implementations

for name in list(known_implementations):
    if name != "hf":
        del known_implementations[name]
