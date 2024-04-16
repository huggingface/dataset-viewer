# Explore statistics over split data

The dataset viewer provides a `/statistics` endpoint for fetching some basic statistics precomputed for a requested dataset. This will get you a quick insight on how the data is distributed.

<Tip warning={true}>
  Currently, statistics are computed only for <a href="./parquet">datasets with Parquet exports</a>.
</Tip>

The `/statistics` endpoint requires three query parameters:

- `dataset`: the dataset name, for example `nyu-mll/glue`
- `config`: the configuration name, for example `cola`
- `split`: the split name, for example `train`

Let's get some stats for `nyu-mll/glue` dataset, `cola` config, `train` split:

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/statistics?dataset=nyu-mll/glue&config=cola&split=train"
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
        "https://datasets-server.huggingface.co/statistics?dataset=nyu-mll/glue&config=cola&split=train",
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
curl https://datasets-server.huggingface.co/statistics?dataset=nyu-mll/glue&config=cola&split=train \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

The response JSON contains three keys:
* `num_examples` - number of samples in a split or number of samples in the first chunk of data if dataset is larger than 5GB (see `partial` field below).
* `statistics` - list of dictionaries of statistics per each column, each dictionary has three keys: `column_name`, `column_type`, and `column_statistics`. Content of `column_statistics` depends on a column type, see [Response structure by data types](./statistics#response-structure-by-data-type) for more details
* `partial` - `true` if statistics are computed on the first 5 GB of data, not on the full split, `false` otherwise.

```json
{
  "num_examples": 8551,
  "statistics": [
    {
      "column_name": "idx",
      "column_type": "int",
      "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0,
        "min": 0,
        "max": 8550,
        "mean": 4275,
        "median": 4275,
        "std": 2468.60541,
        "histogram": {
          "hist": [
            856,
            856,
            856,
            856,
            856,
            856,
            856,
            856,
            856,
            847
          ],
          "bin_edges": [
            0,
            856,
            1712,
            2568,
            3424,
            4280,
            5136,
            5992,
            6848,
            7704,
            8550
          ]
        }
      }
    },
    {
      "column_name": "label",
      "column_type": "class_label",
      "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0,
        "no_label_count": 0,
        "no_label_proportion": 0,
        "n_unique": 2,
        "frequencies": {
          "unacceptable": 2528,
          "acceptable": 6023
        }
      }
    },
    {
      "column_name": "sentence",
      "column_type": "string_text",
      "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0,
        "min": 6,
        "max": 231,
        "mean": 40.70074,
        "median": 37,
        "std": 19.14431,
        "histogram": {
          "hist": [
            2260,
            4512,
            1262,
            380,
            102,
            26,
            6,
            1,
            1,
            1
          ],
          "bin_edges": [
            6,
            29,
            52,
            75,
            98,
            121,
            144,
            167,
            190,
            213,
            231
          ]
        }
      }
    }
  ],
  "partial": false
}
```

## Response structure by data type

Currently, statistics are supported for strings, float and integer numbers, lists, audio and image data and the special [`datasets.ClassLabel`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.ClassLabel) feature type of the [`datasets`](https://huggingface.co/docs/datasets/) library.

`column_type` in response can be one of the following values:

* `class_label` - for [`datasets.ClassLabel`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.ClassLabel) feature which represents categorical data
* `float` - for float data types
* `int` - for integer data types
* `bool` - for boolean data type
* `string_label` - for string data types being treated as categories (see below)
* `string_text` - for string data types if they do not represent categories (see below)
* `list` - for lists of any other data types (including lists)
* `audio` - for audio data
* `image` - for image data

### `class_label`

This type represents categorical data encoded as [`ClassLabel`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.ClassLabel) feature. The following measures are computed:

* number and proportion of `null` values
* number and proportion of values with no label
* number of unique values (excluding `null` and `no label`)
* value counts for each label (excluding `null` and `no label`)

<details><summary>Example </summary>
<p>

```json
{
  "column_name": "label",
  "column_type": "class_label",
  "column_statistics": {
    "nan_count": 0,
    "nan_proportion": 0,
    "no_label_count": 0,
    "no_label_proportion": 0,
    "n_unique": 2,
    "frequencies": {
      "unacceptable": 2528,
      "acceptable": 6023
    }
  }
}
```

</p>
</details>

### float

The following measures are returned for float data types:

* minimum, maximum, mean, and standard deviation values
* number and proportion of `null` values
* histogram with 10 bins

<details><summary>Example </summary>
<p>

```json
{
  "column_name": "clarity",
  "column_type": "float",
  "column_statistics": {
    "nan_count": 0,
    "nan_proportion": 0,
    "min": 0,
    "max": 2,
    "mean": 1.67206,
    "median": 1.8,
    "std": 0.38714,
    "histogram": {
      "hist": [
        17,
        12,
        48,
        52,
        135,
        188,
        814,
        15,
        1628,
        2048
      ],
      "bin_edges": [
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1,
        1.2,
        1.4,
        1.6,
        1.8,
        2
      ]
    }
  }
}
```

</p>
</details>

### int

The following measures are returned for integer data types:

* minimum, maximum, mean, and standard deviation values
* number and proportion of `null` values
* histogram with less than or equal to 10 bins

<details><summary>Example </summary>
<p>

```json
{
    "column_name": "direction",
    "column_type": "int",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 0,
        "max": 1,
        "mean": 0.49925,
        "median": 0.0,
        "std": 0.5,
        "histogram": {
            "hist": [
                50075,
                49925
            ],
            "bin_edges": [
                0,
                1,
                1
            ]
        }
    }
}
```

</p>
</details>

### bool

The following measures are returned for bool data type:

* number and proportion of `null` values
* value counts for `'True'` and `'False'` values

<details><summary>Example </summary>
<p>

```json
{
  "column_name": "penalty",
  "column_type": "bool",
  "column_statistics":
    {
        "nan_count": 3,
        "nan_proportion": 0.15,
        "frequencies": {
            "False": 7,
            "True": 10
        }
    }
}
```

</p>
</details>


### string_label

If the proportion of unique values in a string column within requested split is lower than or equal to 0.2 and the number of unique values is lower than 1000, or if the number of unique values is lower or equal to 10 (independently of the proportion), it is considered to be a category. The following measures are returned:

* number and proportion of `null` values
* number of unique values (excluding `null`)
* value counts for each label (excluding `null`)

<details><summary>Example </summary>
<p>

```json
{
  "column_name": "answerKey",
  "column_type": "string_label",
  "column_statistics": {
    "nan_count": 0,
    "nan_proportion": 0,
    "n_unique": 4,
    "frequencies": {
      "D": 1221,
      "C": 1146,
      "A": 1378,
      "B": 1212
    }
  }
}

```

</p>
</details>

### string_text

If string column does not satisfy the conditions to be treated as a `string_label`, it is considered to be a column containing texts and response contains statistics over text lengths. The following measures are computed:

* minimum, maximum, mean, and standard deviation of text lengths
* number and proportion of `null` values
* histogram of text lengths with 10 bins

<details><summary>Example </summary>
<p>

```json
{
  "column_name": "sentence",
  "column_type": "string_text",
  "column_statistics": {
    "nan_count": 0,
    "nan_proportion": 0,
    "min": 6,
    "max": 231,
    "mean": 40.70074,
    "median": 37,
    "std": 19.14431,
    "histogram": {
      "hist": [
        2260,
        4512,
        1262,
        380,
        102,
        26,
        6,
        1,
        1,
        1
      ],
      "bin_edges": [
        6,
        29,
        52,
        75,
        98,
        121,
        144,
        167,
        190,
        213,
        231
      ]
    }
  }
}
```

</p>
</details>

### list

For lists, the distribution of their lengths is computed. The following measures are returned:

* minimum, maximum, mean, and standard deviation of lists lengths
* number and proportion of `null` values
* histogram of lists lengths with up to 10 bins

<details><summary>Example </summary>
<p>

```json
{
    "column_name": "chat_history",
    "column_type": "list",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 1,
        "max": 3,
        "mean": 1.01741,
        "median": 1.0,
        "std": 0.13146,
        "histogram": {
            "hist": [
                11177,
                196,
                1
            ],
            "bin_edges": [
                1,
                2,
                3,
                3
            ]
        }
    }
}
```

</p>
</details>

Note that dictionaries of lists are not supported.


### audio

For audio data, the distribution of audio files durations is computed. The following measures are returned:

* minimum, maximum, mean, and standard deviation of audio files durations
* number and proportion of `null` values
* histogram of audio files durations with 10 bins


<details><summary>Example </summary>
<p>

```json
{
    "column_name": "audio",
    "column_type": "audio",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0,
        "min": 1.02,
        "max": 15,
        "mean": 13.93042,
        "median": 14.77,
        "std": 2.63734,
        "histogram": {
            "hist": [
                32,
                25,
                18,
                24,
                22,
                17,
                18,
                19,
                55,
                1770
            ],
            "bin_edges": [
                1.02,
                2.418,
                3.816,
                5.214,
                6.612,
                8.01,
                9.408,
                10.806,
                12.204,
                13.602,
                15
            ]
        }
    }
}
```

</p>
</details>


### audio

For image data, the distribution of images widths is computed. The following measures are returned:

* minimum, maximum, mean, and standard deviation of widths of image files
* number and proportion of `null` values
* histogram of images widths with 10 bins

<details><summary>Example </summary>
<p>

```json
{
    "column_name": "image",
    "column_type": "image",
    "column_statistics": {
        "nan_count": 0,
        "nan_proportion": 0.0,
        "min": 256,
        "max": 873,
        "mean": 327.99339,
        "median": 341.0,
        "std": 60.07286,
        "histogram": {
            "hist": [
                1734,
                1637,
                1326,
                121,
                10,
                3,
                1,
                3,
                1,
                2
            ],
            "bin_edges": [
                256,
                318,
                380,
                442,
                504,
                566,
                628,
                690,
                752,
                814,
                873
            ]
        }
    }
}
```

</p>
</details>
