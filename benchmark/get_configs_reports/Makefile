.PHONY: clean all

tmpDir = ../tmp/
reportsDir = $(addprefix $(tmpDir), get_configs_reports/)
scriptsDir = ../scripts/

serialized_dataset_names_filename = $(addprefix $(tmpDir), serialized_dataset_names.txt)
serialized_dataset_names = $(file < $(serialized_dataset_names_filename))
individual_report_filenames = $(addprefix $(reportsDir), $(addsuffix .json, $(serialized_dataset_names)))
merged_report_filename = $(addprefix $(tmpDir), get_configs_reports.json)

GET_CONFIGS_REPORT = $(addprefix $(scriptsDir), get_configs_report.py)
MERGE_REPORTS = $(addprefix $(scriptsDir), merge_reports.py)

all: $(merged_report_filename)

# merge all the reports in one JSON

$(merged_report_filename): $(individual_report_filenames)
	poetry run python $(MERGE_REPORTS) $(serialized_dataset_names_filename) $(reportsDir) $@

# generate a report for every dataset (get_configs.py) -> list of configs, and potential exceptions
# this is $(individual_report_filenames)
../tmp/get_configs_reports/%.json:
	@mkdir -p $(reportsDir)
	poetry run python $(GET_CONFIGS_REPORT) $* $@

clean:
	rm -rf $(merged_report_filename)
	rm -rf $(reportsDir)
