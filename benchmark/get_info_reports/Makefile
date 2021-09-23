.PHONY: clean all

tmpDir = ../tmp/
reportsDir = $(addprefix $(tmpDir), get_info_reports/)
scriptsDir = ../scripts/

serialized_dataset_names_filename = $(addprefix $(tmpDir), serialized_dataset_names.txt)
serialized_dataset_names = $(file < $(serialized_dataset_names_filename))
individual_report_filenames = $(addprefix $(reportsDir), $(addsuffix .json, $(serialized_dataset_names)))
merged_report_filename = $(addprefix $(tmpDir), get_info_reports.json)

GET_REPORTS = $(addprefix $(scriptsDir), get_reports.py)
MERGE_REPORTS = $(addprefix $(scriptsDir), merge_reports.py)

# TODO: use env vars + add an env var for the scheme (http/https)
URL = "http://localhost:8000"
ENDPOINT = "info"

all: $(merged_report_filename)

# merge all the reports in one JSON
$(merged_report_filename): $(individual_report_filenames)
	poetry run python $(MERGE_REPORTS) $(serialized_dataset_names_filename) $(reportsDir) $@

# generate a report for every dataset (/info) -> list of info, and potential exceptions
# this is $(individual_report_filenames)
../tmp/get_info_reports/%.json:
	@mkdir -p $(reportsDir)
	poetry run python $(GET_REPORTS) $(URL) $(ENDPOINT) $* $@

clean:
	rm -rf $(merged_report_filename)
	rm -rf $(reportsDir)
