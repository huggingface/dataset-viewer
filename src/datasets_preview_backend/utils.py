def get_int_value(d, key, default):
    try:
        value = int(d.get(key))
    except TypeError:
        value = default
    return value


def get_str_value(d, key, default):
    if key not in d:
        return default
    value = str(d.get(key)).strip()
    return default if value == "" else value
