.PHONY: clean all

tmpDir = ../tmp/
reportsDir = $(addprefix $(tmpDir), get_splits_reports/)
scriptsDir = ../scripts/

serialized_config_names_filename = $(addprefix $(tmpDir), serialized_config_names.txt)
serialized_config_names = $(file < $(serialized_config_names_filename))
individual_report_filenames = $(addprefix $(reportsDir), $(addsuffix .json, $(serialized_config_names)))
merged_report_filename = $(addprefix $(tmpDir), get_splits_reports.json)

GET_SPLITS_REPORT = $(addprefix $(scriptsDir), get_splits_report.py)

all: $(merged_report_filename)

# merge all the reports in one JSON
$(merged_report_filename): $(individual_report_filenames)
	jq -s '.' $^ > $@

# generate a report for every config (get_splits_report.py) -> list of splits, and potential exceptions
# this is $(individual_report_filenames)
../tmp/get_splits_reports/%.json:
	@mkdir -p $(reportsDir)
	poetry run python $(GET_SPLITS_REPORT) $* $@

clean:
	rm -rf $(merged_report_filename)
	rm -rf $(reportsDir)
