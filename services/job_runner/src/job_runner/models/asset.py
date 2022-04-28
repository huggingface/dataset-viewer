import logging
import os
from typing import List, Tuple, TypedDict

import soundfile  # type:ignore
from libcache.asset import get_assets_dir
from numpy import ndarray  # type:ignore
from PIL import Image  # type: ignore
from pydub import AudioSegment  # type:ignore

logger = logging.getLogger(__name__)

DATASET_SEPARATOR = "--"
ASSET_DIR_MODE = 0o755


def create_asset_dir(dataset: str, config: str, split: str, row_idx: int, column: str) -> Tuple[str, str]:
    assets_dir = get_assets_dir()
    # TODO: add assets_dir as an argument?
    dir_path = os.path.join(assets_dir, dataset, DATASET_SEPARATOR, config, split, str(row_idx), column)
    url_dir_path = f"{dataset}/{DATASET_SEPARATOR}/{config}/{split}/{row_idx}/{column}"
    os.makedirs(dir_path, ASSET_DIR_MODE, exist_ok=True)
    return dir_path, url_dir_path


def create_asset_file(
    dataset: str, config: str, split: str, row_idx: int, column: str, filename: str, data: bytes
) -> str:
    dir_path, url_dir_path = create_asset_dir(dataset, config, split, row_idx, column)
    file_path = os.path.join(dir_path, filename)
    with open(file_path, "wb") as f:
        f.write(data)
    return f"assets/{url_dir_path}/{filename}"


def create_image_file(
    dataset: str, config: str, split: str, row_idx: int, column: str, filename: str, image: Image.Image
) -> str:
    dir_path, url_dir_path = create_asset_dir(dataset, config, split, row_idx, column)
    file_path = os.path.join(dir_path, filename)
    image.save(file_path)
    return f"assets/{url_dir_path}/{filename}"


class AudioSource(TypedDict):
    src: str
    type: str


def create_audio_files(
    dataset: str, config: str, split: str, row_idx: int, column: str, array: ndarray, sampling_rate: int
) -> List[AudioSource]:
    wav_filename = "audio.wav"
    mp3_filename = "audio.mp3"
    dir_path, url_dir_path = create_asset_dir(dataset, config, split, row_idx, column)
    wav_file_path = os.path.join(dir_path, wav_filename)
    mp3_file_path = os.path.join(dir_path, mp3_filename)
    soundfile.write(wav_file_path, array, sampling_rate)
    segment = AudioSegment.from_wav(wav_file_path)
    segment.export(mp3_file_path, format="mp3")
    return [
        {"src": f"assets/{url_dir_path}/{mp3_filename}", "type": "audio/mpeg"},
        {"src": f"assets/{url_dir_path}/{wav_filename}", "type": "audio/wav"},
    ]


# TODO: add a function to flush all the assets of a dataset
