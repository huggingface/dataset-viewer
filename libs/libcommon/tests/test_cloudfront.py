# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from datetime import datetime, timezone

from libcommon.cloudfront import CloudFront


def test_cloudfront() -> None:
    # see https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-canned-policy.html
    EXAMPLE_DATETIME = datetime(2013, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    KEY_PAIR_ID = "K2JCJMDEHXQW5F"
    URL = "https://d111111abcdef8.cloudfront.net/image.jpg"

    # this private key is only for the tests, and is the same as used in libapi
    PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIBOQIBAAJAZTmplhS/Jd73ycVut7TglMObheQqXM7RZYlwazLU4wpfIVIwOh9I
sCZGSgLyFq42KWIikKLEs/yqx3pRGfq+rwIDAQABAkAMyF9WCICq86Eu5bO5lynV
H26AVfPTjHp87AI6R00C7p9n8hO/DhHaHpc3InOSsXsw9d2hmz37jwwBFiwMHMMh
AiEAtbttHlIO+yO29oXw4P6+yO11lMy1UpT1sPVTnR9TXbUCIQCOl7Zuyy2ZY9ZW
pDhW91x/14uXjnLXPypgY9bcfggJUwIhAJQG1LzrzjQWRUPMmgZKuhBkC3BmxhM8
LlwzmCXVjEw5AiA7JnAFEb9+q82T71d3q/DxD0bWvb6hz5ASoBfXK2jGBQIgbaQp
h4Tk6UJuj1xgKNs75Pk3pG2tj8AQiuBk3l62vRU=
-----END RSA PRIVATE KEY-----"""

    # can parse the private key and create the signer
    cloudfront = CloudFront(key_pair_id=KEY_PAIR_ID, private_key=PRIVATE_KEY)
    # can sign an URL
    signed_url = cloudfront.sign_url(url=URL, date_less_than=EXAMPLE_DATETIME)

    EXPECTED_BASE_URL = URL
    EXPECTED_SECONDS = "1357034400"
    EXPECTED_SIGNATURE = "JvFbQI~xSRStWOmvHk88XeE4tJ5zGFJN62eY6FK0JH7aLCmvvWmsbLXuhMsT8QZ1frCFsZfv4dJLPlqZXj7Mjw__"
    EXPECTED_KEY_PAIR_ID = KEY_PAIR_ID
    assert (
        signed_url
        == EXPECTED_BASE_URL
        + "?Expires="
        + EXPECTED_SECONDS
        + "&Signature="
        + EXPECTED_SIGNATURE
        + "&Key-Pair-Id="
        + EXPECTED_KEY_PAIR_ID
    )
