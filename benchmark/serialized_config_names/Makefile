.PHONY: clean all

tmpDir = ../tmp/
scriptsDir = ../scripts/

serialized_config_names_filename = $(addprefix $(tmpDir), serialized_config_names.txt)
get_configs_reports_filename = $(addprefix $(tmpDir), get_configs_reports.json)

GET_SERIALIZED_CONFIG_NAMES = $(addprefix $(scriptsDir), get_serialized_config_names.py)

all: $(serialized_config_names_filename)

# get the list of serialized config names
$(serialized_config_names_filename): $(get_configs_reports_filename)
	@mkdir -p $(tmpDir)
	poetry run python $(GET_SERIALIZED_CONFIG_NAMES) $(get_configs_reports_filename) $(serialized_config_names_filename)

clean:
	rm -rf $(serialized_config_names_filename)
