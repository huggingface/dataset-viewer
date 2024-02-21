def get_record_set(dataset: str, config_name: str) -> str:
    # Identical keys are not supported in Croissant
    # The current workaround that is used in /croissant endpoint
    # is to prefix the config name with `record_set_` if necessary.
    if dataset != config_name:
        return config_name
    else:
        return f"record_set_{config_name}"
