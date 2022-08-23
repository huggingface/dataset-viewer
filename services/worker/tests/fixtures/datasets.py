import pytest
from datasets import Audio, Dataset, Features


@pytest.fixture(scope="session")
def audio_dataset() -> Dataset:
    sampling_rate = 16_000
    return Dataset.from_dict(
        {"audio_column": [{"array": [0.1, 0.2, 0.3], "sampling_rate": sampling_rate}]},
        Features({"audio_column": Audio(sampling_rate=sampling_rate)}),
    )
