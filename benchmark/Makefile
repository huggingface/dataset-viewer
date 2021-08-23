.PHONY: clean all

tmpDir = ./tmp/

serialized_dataset_names_filename = $(addprefix $(tmpDir), serialized_dataset_names.txt)
serialized_config_names_filename = $(addprefix $(tmpDir), serialized_config_names.txt)
serialized_split_names_filename = $(addprefix $(tmpDir), serialized_split_names.txt)
get_info_reports_filename = $(addprefix $(tmpDir), get_info_reports.json)
get_configs_reports_filename = $(addprefix $(tmpDir), get_configs_reports.json)
get_splits_reports_filename = $(addprefix $(tmpDir), get_splits_reports.json)
report_filename = $(addprefix $(tmpDir), report.json)


all: $(get_configs_reports_filename) $(get_info_reports_filename) $(get_splits_reports_filename) $(serialized_split_names_filename)
#	TODO -> generate the report from the files
#	{ info: <$info_filename, configs: <$configs_filename, ... }
#	jq -s '.' $^ > $@

# TODO: add splits and rows
# we need to generate the list of pairs (dataset, config), and of tuples (dataset, config, split), one per line, in order to allow parallelization

$(serialized_split_names_filename): $(get_splits_reports_filename)
	$(MAKE) -C serialized_split_names

$(get_splits_reports_filename): $(serialized_config_names_filename)
	$(MAKE) -C get_splits_reports -j 5

$(serialized_config_names_filename): $(get_configs_reports_filename)
	$(MAKE) -C serialized_config_names

$(get_configs_reports_filename): $(serialized_dataset_names_filename)
	$(MAKE) -C get_configs_reports -j 5

$(get_info_reports_filename): $(serialized_dataset_names_filename)
	$(MAKE) -C get_info_reports -j 5

$(serialized_dataset_names_filename):
	$(MAKE) -C serialized_dataset_names

clean:
	$(MAKE) -C serialized_split_names/ clean
	$(MAKE) -C get_splits_reports/ clean
	$(MAKE) -C serialized_config_names/ clean
	$(MAKE) -C get_configs_reports/ clean
	$(MAKE) -C get_info_reports/ clean
	$(MAKE) -C serialized_dataset_names/ clean
