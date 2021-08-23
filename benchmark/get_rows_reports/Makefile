.PHONY: clean all

tmpDir = ../tmp/
reportsDir = $(addprefix $(tmpDir), get_rows_reports/)
scriptsDir = ../scripts/

serialized_split_names_filename = $(addprefix $(tmpDir), serialized_split_names.txt)
serialized_split_names = $(file < $(serialized_split_names_filename))
individual_report_filenames = $(addprefix $(reportsDir), $(addsuffix .json, $(serialized_split_names)))
merged_report_filename = $(addprefix $(tmpDir), get_rows_reports.json)

GET_ROWS_REPORT = $(addprefix $(scriptsDir), get_rows_report.py)

all: $(merged_report_filename)

# merge all the reports in one JSON
$(merged_report_filename): $(individual_report_filenames)
	jq -s '.' $^ > $@

# generate a report for every split (get_rows_report.py) -> list of rows, and potential exceptions
# this is $(individual_report_filenames)
../tmp/get_rows_reports/%.json:
	@mkdir -p $(reportsDir)
	poetry run python $(GET_ROWS_REPORT) $* $@

clean:
	rm -rf $(merged_report_filename)
	rm -rf $(reportsDir)
