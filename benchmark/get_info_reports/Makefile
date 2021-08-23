.PHONY: clean all

tmpDir = ../tmp/
reportsDir = $(addprefix $(tmpDir), get_info_reports/)
scriptsDir = ../scripts/

serialized_dataset_names_filename = $(addprefix $(tmpDir), serialized_dataset_names.txt)
serialized_dataset_names = $(file < $(serialized_dataset_names_filename))
individual_report_filenames = $(addprefix $(reportsDir), $(addsuffix .json, $(serialized_dataset_names)))
merged_report_filename = $(addprefix $(tmpDir), get_info_reports.json)

GET_INFO_REPORT = $(addprefix $(scriptsDir), get_info_report.py)

all: $(merged_report_filename)

# merge all the reports in one JSON
$(merged_report_filename): $(individual_report_filenames)
	jq -s '.' $^ > $@

# generate a report for every dataset (get_info.py) -> list of info, and potential exceptions
# this is $(individual_report_filenames)
../tmp/get_info_reports/%.json:
	@mkdir -p $(reportsDir)
	poetry run python $(GET_INFO_REPORT) $* $@

clean:
	rm -rf $(merged_report_filename)
	rm -rf $(reportsDir)
