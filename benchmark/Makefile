# Launch with "make -j -l 7" to enable parallel execution
.PHONY: clean all

tmpDir = ./tmp/

serialized_dataset_names_filename = $(addprefix $(tmpDir), serialized_dataset_names.txt)
serialized_config_names_filename = $(addprefix $(tmpDir), serialized_config_names.txt)
serialized_split_names_filename = $(addprefix $(tmpDir), serialized_split_names.txt)
get_info_reports_filename = $(addprefix $(tmpDir), get_info_reports.json)
get_configs_reports_filename = $(addprefix $(tmpDir), get_configs_reports.json)
get_splits_reports_filename = $(addprefix $(tmpDir), get_splits_reports.json)
get_rows_reports_filename = $(addprefix $(tmpDir), get_rows_reports.json)
report_filename = $(addprefix $(tmpDir), report.json)

COMPILE_REPORT = $(addprefix $(scriptsDir), compile_report.py)

all: $(report_filename)

$(report_filename): $(get_info_reports_filename) $(get_configs_reports_filename) $(get_splits_reports_filename) $(get_rows_reports_filename)
	poetry run python $(MERGE_REPORTS) $(get_info_reports_filename) $(get_configs_reports_filename) $(get_splits_reports_filename) $(get_rows_reports_filename) $@

$(get_rows_reports_filename): $(serialized_split_names_filename)
	$(MAKE) -C get_rows_reports

$(serialized_split_names_filename): $(get_splits_reports_filename)
	$(MAKE) -C serialized_split_names

$(get_splits_reports_filename): $(serialized_config_names_filename)
	$(MAKE) -C get_splits_reports

$(serialized_config_names_filename): $(get_configs_reports_filename)
	$(MAKE) -C serialized_config_names

$(get_configs_reports_filename): $(serialized_dataset_names_filename)
	$(MAKE) -C get_configs_reports

$(get_info_reports_filename): $(serialized_dataset_names_filename)
	$(MAKE) -C get_info_reports

$(serialized_dataset_names_filename):
	$(MAKE) -C serialized_dataset_names

clean:
	$(MAKE) -C get_rows_reports/ clean
	$(MAKE) -C serialized_split_names/ clean
	$(MAKE) -C get_splits_reports/ clean
	$(MAKE) -C serialized_config_names/ clean
	$(MAKE) -C get_configs_reports/ clean
	$(MAKE) -C get_info_reports/ clean
	$(MAKE) -C serialized_dataset_names/ clean
