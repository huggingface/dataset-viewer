# List splits and subsets

Datasets typically have splits and may also have subsets. A _split_ is a subset of the dataset, like `train` and `test`, that are used during different stages of training and evaluating a model. A _subset_ (also called _configuration_) is a sub-dataset contained within a larger dataset. Subsets are especially common in multilingual speech datasets where there may be a different subset for each language. If you're interested in learning more about splits and subsets, check out the [conceptual guide on "Splits and subsets"](./configs_and_splits)!

![split-configs-server](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/split-configs-server.gif)

This guide shows you how to use the dataset viewer's `/splits` endpoint to retrieve a dataset's splits and subsets programmatically. Feel free to also try it out with [Postman](https://www.postman.com/huggingface/workspace/hugging-face-apis/request/23242779-f0cde3b9-c2ee-4062-aaca-65c4cfdd96f8), [RapidAPI](https://rapidapi.com/hugging-face-hugging-face-default/api/hugging-face-datasets-api), or [ReDoc](https://redocly.github.io/redoc/?url=https://datasets-server.huggingface.co/openapi.json#operation/listSplits)

The `/splits` endpoint accepts the dataset name as its query parameter:

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/splits?dataset=ibm/duorc"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
```
</python>
<js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://datasets-server.huggingface.co/splits?dataset=ibm/duorc",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const result = await response.json();
    return result;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js>
<curl>
```curl
curl https://datasets-server.huggingface.co/splits?dataset=ibm/duorc \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

The endpoint response is a JSON containing a list of the dataset's splits and subsets. For example, the [ibm/duorc](https://huggingface.co/datasets/ibm/duorc) dataset has six splits and two subsets:

```json
{
  "splits": [
    { "dataset": "ibm/duorc", "config": "ParaphraseRC", "split": "train" },
    { "dataset": "ibm/duorc", "config": "ParaphraseRC", "split": "validation" },
    { "dataset": "ibm/duorc", "config": "ParaphraseRC", "split": "test" },
    { "dataset": "ibm/duorc", "config": "SelfRC", "split": "train" },
    { "dataset": "ibm/duorc", "config": "SelfRC", "split": "validation" },
    { "dataset": "ibm/duorc", "config": "SelfRC", "split": "test" }
  ],
  "pending": [],
  "failed": []
}
```
