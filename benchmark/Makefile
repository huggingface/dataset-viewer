.PHONY: clean all

tmpDir = ./tmp/

serialized_dataset_names_filename = $(addprefix $(tmpDir), serialized_dataset_names.txt)
serialized_config_names_filename = $(addprefix $(tmpDir), serialized_config_names.txt)
get_info_reports_filename = $(addprefix $(tmpDir), get_info_reports.json)
get_configs_reports_filename = $(addprefix $(tmpDir), get_configs_reports.json)
report_filename = $(addprefix $(tmpDir), report.json)

all: $(report_filename) $(serialized_config_names_filename)

$(report_filename): $(get_configs_report_filename) $(get_info_report_filename)
#	TODO -> generate the report from the files
#	{ info: <$info_filename, configs: <$configs_filename, ... }
#	jq -s '.' $^ > $@

# TODO: add splits and rows
# we need to generate the list of pairs (dataset, config), and of tuples (dataset, config, split), one per line, in order to allow parallelization

$(serialized_config_names_filename): $(get_configs_reports_filename)
	$(MAKE) -C serialized_config_names

$(get_configs_reports_filename): $(serialized_dataset_names_filename)
	$(MAKE) -C get_configs_reports -j 5

$(get_info_reports_filename): $(serialized_dataset_names_filename)
	$(MAKE) -C get_info_reports -j 5

$(serialized_dataset_names_filename):
	$(MAKE) -C serialized_dataset_names

clean:
	$(MAKE) -C serialized_config_names/ clean
	$(MAKE) -C get_configs_reports/ clean
	$(MAKE) -C get_info_reports/ clean
	$(MAKE) -C serialized_dataset_names/ clean
