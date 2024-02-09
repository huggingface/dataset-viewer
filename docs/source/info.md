# Get dataset information

Datasets Server provides an `/info` endpoint for exploring the general information about dataset, including such fields as description, citation, homepage, license and features.

The `/info` endpoint accepts two query parameters:

- `dataset`: the dataset name
- `config`: the configuration name

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/info?dataset=ibm/duorc&config=SelfRC"
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
        "https://datasets-server.huggingface.co/info?dataset=ibm/duorc&config=SelfRC",
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
curl https://datasets-server.huggingface.co/info?dataset=ibm/duorc&config=SelfRC \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

The endpoint response is a JSON with the `dataset_info` key. Its structure and content correspond to [DatasetInfo](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.DatasetInfo) object of the `datasets` library.

```json
{
   "dataset_info":{
      "description":"",
      "citation":"",
      "homepage":"",
      "license":"",
      "features":{
         "plot_id":{
            "dtype":"string",
            "_type":"Value"
         },
         "plot":{
            "dtype":"string",
            "_type":"Value"
         },
         "title":{
            "dtype":"string",
            "_type":"Value"
         },
         "question_id":{
            "dtype":"string",
            "_type":"Value"
         },
         "question":{
            "dtype":"string",
            "_type":"Value"
         },
         "answers":{
            "feature":{
               "dtype":"string",
               "_type":"Value"
            },
            "_type":"Sequence"
         },
         "no_answer":{
            "dtype":"bool",
            "_type":"Value"
         }
      },
      "builder_name":"parquet",
      "dataset_name":"duorc",
      "config_name":"SelfRC",
      "version":{
         "version_str":"0.0.0",
         "major":0,
         "minor":0,
         "patch":0
      },
      "splits":{
         "train":{
            "name":"train",
            "num_bytes":248966361,
            "num_examples":60721,
            "dataset_name":null
         },
         "validation":{
            "name":"validation",
            "num_bytes":56359392,
            "num_examples":12961,
            "dataset_name":null
         },
         "test":{
            "name":"test",
            "num_bytes":51022318,
            "num_examples":12559,
            "dataset_name":null
         }
      },
      "download_size":21001846,
      "dataset_size":356348071
   },
   "partial":false
}
```