from typing import List, Optional

DEFAULT_ASSETS_DIRECTORY: None = None
DEFAULT_DATASETS_REVISION: str = "master"
DEFAULT_HF_TOKEN: Optional[str] = None
DEFAULT_MAX_SIZE_FALLBACK: int = 100_000_000
DEFAULT_ROWS_MAX_NUMBER: int = 100

# below 100 bytes, the cell content will not be truncated
DEFAULT_MIN_CELL_BYTES: int = 100

# these datasets take too much time, we block them beforehand
DATASETS_BLOCKLIST: List[str] = [
    "Alvenir/nst-da-16khz",
    "bigscience/P3",
    "clips/mqa",
    "echarlaix/gqa-lxmert",
    "echarlaix/vqa-lxmert",
    "fractalego/QA_to_statements",
    "hyperpartisan_news_detection",
    "imthanhlv/binhvq_news21_raw",
    "Graphcore/gqa-lxmert",
    "Graphcore/vqa-lxmert",
    "kiyoung2/aistage-mrc",
    "lewtun/gem-multi-dataset-predictions",
    "lukesjordan/worldbank-project-documents",
    "math_dataset",
    "midas/ldke3k_medium",
    "midas/ldke3k_small",
    "midas/ldkp3k_small",
    "qr/cefr_book_sentences",
    "SaulLu/Natural_Questions_HTML_reduced_all",
    "SaulLu/Natural_Questions_HTML_Toy",
    "unicamp-dl/mmarco",
    "z-uo/squad-it",
]
