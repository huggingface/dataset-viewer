.PHONY: clean all

tmpDir = ../tmp/
scriptsDir = ../scripts/

sorted_serialized_config_names_filename = $(addprefix $(tmpDir), sorted_serialized_config_names.txt)
serialized_config_names_filename = $(addprefix $(tmpDir), serialized_config_names.txt)
get_configs_reports_filename = $(addprefix $(tmpDir), get_configs_reports.json)

GET_SERIALIZED_CONFIG_NAMES = $(addprefix $(scriptsDir), get_serialized_config_names.py)

all: $(serialized_config_names_filename)

# get the list of serialized config names
# the names are shuffled to avoid processing configs from the same dataset together wince the data server might ban us
$(serialized_config_names_filename): $(get_configs_reports_filename)
	@mkdir -p $(tmpDir)
	poetry run python $(GET_SERIALIZED_CONFIG_NAMES) $(get_configs_reports_filename) $(sorted_serialized_config_names_filename)
	shuf $(sorted_serialized_config_names_filename) -o $(serialized_config_names_filename)

clean:
	rm -rf $(serialized_config_names_filename) $(sorted_serialized_config_names_filename)
