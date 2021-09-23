from datasets_preview_backend.serialize import (
    deserialize,
    deserialize_params,
    serialize,
    serialize_params,
)


def test_serialize() -> None:
    assert serialize("") == ""
    assert serialize("glue") == "glue"
    assert serialize("user/dataset") == "user___SLASHdataset"
    # https://huggingface.co/datasets/tommy19970714/common_voice
    assert serialize("tommy19970714/common_voice") == "tommy19970714___SLASHcommon_voice"
    assert serialize("Chinese (Hong Kong)") == "Chinese___SPACE___PARHong___SPACEKong___END_PAR"
    assert serialize("user___SLASHdataset") == "user___SLASHdataset"
    assert (
        serialize("Chinese___SPACE___PARHong___SPACEKong___END_PAR")
        == "Chinese___SPACE___PARHong___SPACEKong___END_PAR"
    )


def test_deserialize() -> None:
    assert deserialize("") == ""
    assert deserialize("glue") == "glue"
    assert deserialize("user___SLASHdataset") == "user/dataset"
    # https://huggingface.co/datasets/tommy19970714/common_voice
    assert deserialize("tommy19970714___SLASHcommon_voice") == "tommy19970714/common_voice"
    assert deserialize("Chinese___SPACE___PARHong___SPACEKong___END_PAR") == "Chinese (Hong Kong)"
    assert deserialize("Chinese (Hong Kong)") == "Chinese (Hong Kong)"


def test_serialize_params() -> None:
    assert serialize_params({}) == ""
    assert serialize_params({"unknown_key": "value"}) == ""
    assert serialize_params({"dataset": "d"}) == "___DATASETd"
    assert serialize_params({"dataset": "d", "split": "s"}) == "___DATASETd"
    assert serialize_params({"dataset": "d", "config": "c"}) == "___DATASETd___CONFIGc"
    assert serialize_params({"dataset": "d", "config": "c", "split": "s"}) == "___DATASETd___CONFIGc___SPLITs"
    assert serialize_params({"config": "c", "split": "s"}) == ""
    assert (
        serialize_params({"dataset": "d", "config": "c", "split": "s", "something_else": "a"})
        == "___DATASETd___CONFIGc___SPLITs"
    )
    assert (
        serialize_params({"dataset": "tommy19970714/common_voice", "config": "Chinese (Hong Kong)", "split": "train"})
        == "___DATASETtommy19970714___SLASHcommon_voice___CONFIGChinese___SPACE___PARHong___SPACEKong___END_PAR___SPLITtrain"
    )


def test_deserialize_params() -> None:
    assert deserialize_params("") == {}
    assert deserialize_params("___DATASETd") == {"dataset": "d"}
    assert deserialize_params("___DATASETd___CONFIGc") == {"dataset": "d", "config": "c"}
    assert deserialize_params("___DATASETd___CONFIGc___SPLITs") == {"dataset": "d", "config": "c", "split": "s"}
    assert deserialize_params(
        "___DATASETtommy19970714___SLASHcommon_voice___CONFIGChinese___SPACE___PARHong___SPACEKong___END_PAR___SPLITtrain"
    ) == {"dataset": "tommy19970714/common_voice", "config": "Chinese (Hong Kong)", "split": "train"}
