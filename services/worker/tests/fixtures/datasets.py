import pytest
from datasets import Audio, Dataset, Features, Image
from pathlib import Path


@pytest.fixture(scope="session")
def audio_dataset() -> Dataset:
    sampling_rate = 16_000
    return Dataset.from_dict(
        {"audio_column": [{"array": [0.1, 0.2, 0.3], "sampling_rate": sampling_rate}]},
        Features({"audio_column": Audio(sampling_rate=sampling_rate)}),
    )


@pytest.fixture(scope="session")
def image_dataset() -> Dataset:
    return Dataset.from_dict(
        {"image_column": [str(Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg")]}
    ).cast_column("image_column", Image())
