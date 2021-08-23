.PHONY: clean all

tmpDir = ../tmp/
scriptsDir = ../scripts/

serialized_split_names_filename = $(addprefix $(tmpDir), serialized_split_names.txt)
get_splits_reports_filename = $(addprefix $(tmpDir), get_splits_reports.json)

GET_SERIALIZED_SPLIT_NAMES = $(addprefix $(scriptsDir), get_serialized_split_names.py)

all: $(serialized_split_names_filename)

# get the list of serialized split names
$(serialized_split_names_filename): $(get_splits_reports_filename)
	@mkdir -p $(tmpDir)
	poetry run python $(GET_SERIALIZED_SPLIT_NAMES) $(get_splits_reports_filename) $(serialized_split_names_filename)

clean:
	rm -rf $(serialized_split_names_filename)
