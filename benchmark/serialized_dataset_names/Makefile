.PHONY: clean all

tmpDir = ../tmp/
scriptsDir = ../scripts/

sorted_serialized_dataset_names_filename = $(addprefix $(tmpDir), sorted_serialized_dataset_names.txt)
serialized_dataset_names_filename = $(addprefix $(tmpDir), serialized_dataset_names.txt)

GET_SERIALIZED_DATASET_NAMES = $(addprefix $(scriptsDir), get_serialized_dataset_names.py)

all: $(serialized_dataset_names_filename)

# get the list of serialized dataset names (with '/' replaced with "___SLASH___")
$(serialized_dataset_names_filename):
	@mkdir -p $(tmpDir)
	poetry run python $(GET_SERIALIZED_DATASET_NAMES) $(sorted_serialized_dataset_names_filename)
	shuf $(sorted_serialized_dataset_names_filename) -o $(serialized_dataset_names_filename)

clean:
	rm -rf $(serialized_dataset_names_filename) $(sorted_serialized_dataset_names_filename)
