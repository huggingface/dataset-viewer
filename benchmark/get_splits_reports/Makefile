.PHONY: clean all

tmpDir = ../tmp/
splitsDir = $(addprefix $(tmpDir), get_splits_reports/)
scriptsDir = ../scripts/

serialized_config_names_filename = $(addprefix $(tmpDir), serialized_config_names.txt)
serialized_config_names = $(file < $(serialized_config_names_filename))
individual_report_filenames = $(addprefix $(splitsDir), $(addsuffix .json, $(serialized_config_names)))
merged_report_filename = $(addprefix $(tmpDir), get_splits_reports.json)

GET_SPLITS_REPORT = $(addprefix $(scriptsDir), get_splits_report.py)

all: $(merged_report_filename)

# merge all the reports in one JSON
$(merged_report_filename): $(individual_report_filenames)
	jq -s '.' $^ > $@

# generate a report for every config (get_splits_report.py) -> list of splits, and potential exceptions
# this is $(individual_report_filenames)
../tmp/get_splits_reports/%.json:
	@mkdir -p $(splitsDir)
	poetry run python $(GET_SPLITS_REPORT) $* $@

clean:
	rm -rf $(merged_report_filename)
	rm -rf $(splitsDir)
