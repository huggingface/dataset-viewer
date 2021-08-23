.PHONY: clean all

tmpDir = ../tmp/
scriptsDir = ../scripts/

sorted_serialized_split_names_filename = $(addprefix $(tmpDir), sorted_serialized_split_names.txt)
serialized_split_names_filename = $(addprefix $(tmpDir), serialized_split_names.txt)
get_splits_reports_filename = $(addprefix $(tmpDir), get_splits_reports.json)

GET_SERIALIZED_SPLIT_NAMES = $(addprefix $(scriptsDir), get_serialized_split_names.py)

all: $(serialized_split_names_filename)

# get the list of serialized split names
# the names are shuffled to avoid processing splits from the same dataset together wince the data server might ban us
$(serialized_split_names_filename): $(get_splits_reports_filename)
	@mkdir -p $(tmpDir)
	poetry run python $(GET_SERIALIZED_SPLIT_NAMES) $(get_splits_reports_filename) $(sorted_serialized_split_names_filename)
	shuf $(sorted_serialized_split_names_filename) -o $(serialized_split_names_filename)

clean:
	rm -rf $(serialized_split_names_filename) $(sorted_serialized_split_names_filename)
