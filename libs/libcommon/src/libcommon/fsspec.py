from fsspec.registry import _registry, known_implementations

ALLOWED_PROTOCOLS = ["hf", "s3", "zip", "file", "local"]

for name in list(known_implementations):
    if name not in ALLOWED_PROTOCOLS:
        del known_implementations[name]

# `get_filesystem_class` looks the protocol up in the registry of the already imported filesystems
# before falling back to `known_implementations`, so removing it from the latter is not enough: a
# protocol that was imported before this module runs would stay usable. Note that it only stays
# usable under the name it was imported with, e.g. "https" keeps working while "http" is refused.
for name in list(_registry):
    if name not in ALLOWED_PROTOCOLS:
        del _registry[name]
