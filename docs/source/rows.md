# Download slices of rows

The dataset viewer provides a `/rows` endpoint for visualizing any slice of rows of a dataset. This will let you walk-through and inspect the data contained in a dataset.

<div class="flex justify-center">
    <img style="margin-bottom: 0;" class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets-server/oasst1_light.png"/>
    <img style="margin-bottom: 0;" class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets-server/oasst1_dark.png"/>
</div>

<Tip warning={true}>
  Currently, only {" "}
  <a href="./parquet">datasets with parquet exports</a>
  are supported so the dataset viewer can extract any slice of rows without downloading the
  whole dataset.
</Tip>

This guide shows you how to use the dataset viewer's `/rows` endpoint to download slices of a dataset.
Feel free to also try it out with [Postman](https://www.postman.com/huggingface/workspace/hugging-face-apis/request/23242779-32d6a8be-b800-446a-8cee-f6b5ca1710df),
[RapidAPI](https://rapidapi.com/hugging-face-hugging-face-default/api/hugging-face-datasets-api),
or [ReDoc](https://redocly.github.io/redoc/?url=https://datasets-server.huggingface.co/openapi.json#operation/listFirstRows).

The `/rows` endpoint accepts five query parameters:

- `dataset`: the dataset name, for example `nyu-mll/glue` or `mozilla-foundation/common_voice_10_0`
- `config`: the subset name, for example `cola`
- `split`: the split name, for example `train`
- `offset`: the offset of the slice, for example `150`
- `length`: the length of the slice, for example `10` (maximum: `100`)

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/rows?dataset=ibm/duorc&config=SelfRC&split=train&offset=150&length=10"
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
        "https://datasets-server.huggingface.co/rows?dataset=ibm/duorc&config=SelfRC&split=train&offset=150&length=10",
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
curl https://datasets-server.huggingface.co/rows?dataset=ibm/duorc&config=SelfRC&split=train&offset=150&length=10 \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

The endpoint response is a JSON containing two keys:

- The [`features`](https://huggingface.co/docs/datasets/about_dataset_features) of a dataset, including the column's name and data type.
- The slice of `rows` of a dataset and the content contained in each column of a specific row.

For example, here are the `features` and the slice of `rows` of the `ibm/duorc`/`SelfRC` train split from 150 to 151:

```json
// https://datasets-server.huggingface.co/rows?dataset=ibm/duorc&config=SelfRC&split=train&offset=150&length=2
{
  "features": [
    {
      "feature_idx": 0,
      "name": "plot_id",
      "type": { "dtype": "string", "_type": "Value" }
    },
    {
      "feature_idx": 1,
      "name": "plot",
      "type": { "dtype": "string", "_type": "Value" }
    },
    {
      "feature_idx": 2,
      "name": "title",
      "type": { "dtype": "string", "_type": "Value" }
    },
    {
      "feature_idx": 3,
      "name": "question_id",
      "type": { "dtype": "string", "_type": "Value" }
    },
    {
      "feature_idx": 4,
      "name": "question",
      "type": { "dtype": "string", "_type": "Value" }
    },
    {
      "feature_idx": 5,
      "name": "answers",
      "type": {
        "feature": { "dtype": "string", "_type": "Value" },
        "_type": "List"
      }
    },
    {
      "feature_idx": 6,
      "name": "no_answer",
      "type": { "dtype": "bool", "_type": "Value" }
    }
  ],
  "rows": [
    {
      "row_idx": 150,
      "row": {
        "plot_id": "/m/03wj_q",
        "plot": "The film is centered on Mortal Kombat, a fighting tournament between the representatives of the realms of Earth and Outworld conceived by the Elder Gods amid looming invasion of the Earth by Outworld. If the realm of Outworld wins Mortal Kombat ten consecutive times, its Emperor Shao Kahn will be able to invade and conquer the Earth realm.\nShaolin monk Liu Kang and his comrades, movie star Johnny Cage and military officer Sonya Blade were handpicked by Raiden, the god of thunder and defender of the Earth realm, to overcome their powerful adversaries in order to prevent Outworld from winning their tenth straight Mortal Kombat tournament. Each of the three has his or her own reason for competing: Liu seeks revenge against the tournament host Shang Tsung for killing his brother Chan; Sonya seeks revenge on an Australian crime lord Kano; and Cage, having been branded as a fake by the media, seeks to prove otherwise.\nAt Shang Tsung's island, Liu is attracted to Princess Kitana, Shao Kahn's adopted daughter. Aware that Kitana is a dangerous adversary because she is the rightful heir to Outworld and that she will attempt to ally herself with the Earth warriors, Tsung orders the creature Reptile to spy on her. Liu defeats his first opponent and Sonya gets her revenge on Kano by snapping his neck. Cage encounters and barely beats Scorpion. Liu engages in a brief duel with Kitana, who secretly offers him cryptic advice for his next battle. Liu's next opponent is Sub-Zero, whose defense seems untouched because of his freezing abilities, until Liu recalls Kitana's advice and uses it to kill Sub-Zero.\nPrince Goro enters the tournament and mercilessly crushes every opponent he faces. One of Cage's peers, Art Lean, is defeated by Goro as well and has his soul taken by Shang Tsung. Sonya worries that they may not win against Goro, but Raiden disagrees. He reveals their own fears and egos are preventing them from winning the tournament.\nDespite Sonya's warning, Cage comes to Tsung to request a fight with Goro. The sorcerer accepts on the condition that he be allowed to challenge any opponent of his choosing, anytime and anywhere he chooses. Raiden tries to intervene, but the conditions are agreed upon before he can do so. After Shang Tsung leaves, Raiden confronts Cage for what he has done in challenging Goro, but is impressed when Cage shows his awareness of the gravity of the tournament. Cage faces Goro and uses guile and the element of surprise to defeat the defending champion. Now desperate, Tsung takes Sonya hostage and takes her to Outworld, intending to fight her as his opponent. Knowing that his powers are ineffective there and that Sonya cannot defeat Tsung by herself, Raiden sends Liu and Cage into Outworld in order to rescue Sonya and challenge Tsung. In Outworld, Liu is attacked by Reptile, but eventually gains the upper hand and defeats him. Afterward, Kitana meets up with Cage and Liu, revealing to the pair the origins of both herself and Outworld. Kitana allies with them and helps them to infiltrate Tsung's castle.\nInside the castle tower, Shang Tsung challenges Sonya to fight him, claiming that her refusal to accept will result in the Earth realm forfeiting Mortal Kombat (this is, in fact, a lie on Shang's part). All seems lost for Earth realm until Kitana, Liu, and Cage appear. Kitana berates Tsung for his treachery to the Emperor as Sonya is set free. Tsung challenges Cage, but is counter-challenged by Liu. During the lengthy battle, Liu faces not only Tsung, but the souls that Tsung had forcibly taken in past tournaments. In a last-ditch attempt to take advantage, Tsung morphs into Chan. Seeing through the charade, Liu renews his determination and ultimately fires an energy bolt at the sorcerer, knocking him down and impaling him on a row of spikes. Tsung's death releases all of the captive souls, including Chan's. Before ascending to the afterlife, Chan tells Liu that he will remain with him in spirit until they are once again reunited, after Liu dies.\nThe warriors return to Earth realm, where a victory celebration is taking place at the Shaolin temple. The jubilation abruptly stops, however, when Shao Kahn's giant figure suddenly appears in the skies. When the Emperor declares that he has come for everyone's souls, the warriors take up fighting stances.",
        "title": "Mortal Kombat",
        "question_id": "40c1866a-b214-11ba-be57-8979d2cefa90",
        "question": "Where is Sonya taken to?",
        "answers": ["Outworld"],
        "no_answer": false
      },
      "truncated_cells": []
    },
    {
      "row_idx": 151,
      "row": {
        "plot_id": "/m/03wj_q",
        "plot": "The film is centered on Mortal Kombat, a fighting tournament between the representatives of the realms of Earth and Outworld conceived by the Elder Gods amid looming invasion of the Earth by Outworld. If the realm of Outworld wins Mortal Kombat ten consecutive times, its Emperor Shao Kahn will be able to invade and conquer the Earth realm.\nShaolin monk Liu Kang and his comrades, movie star Johnny Cage and military officer Sonya Blade were handpicked by Raiden, the god of thunder and defender of the Earth realm, to overcome their powerful adversaries in order to prevent Outworld from winning their tenth straight Mortal Kombat tournament. Each of the three has his or her own reason for competing: Liu seeks revenge against the tournament host Shang Tsung for killing his brother Chan; Sonya seeks revenge on an Australian crime lord Kano; and Cage, having been branded as a fake by the media, seeks to prove otherwise.\nAt Shang Tsung's island, Liu is attracted to Princess Kitana, Shao Kahn's adopted daughter. Aware that Kitana is a dangerous adversary because she is the rightful heir to Outworld and that she will attempt to ally herself with the Earth warriors, Tsung orders the creature Reptile to spy on her. Liu defeats his first opponent and Sonya gets her revenge on Kano by snapping his neck. Cage encounters and barely beats Scorpion. Liu engages in a brief duel with Kitana, who secretly offers him cryptic advice for his next battle. Liu's next opponent is Sub-Zero, whose defense seems untouched because of his freezing abilities, until Liu recalls Kitana's advice and uses it to kill Sub-Zero.\nPrince Goro enters the tournament and mercilessly crushes every opponent he faces. One of Cage's peers, Art Lean, is defeated by Goro as well and has his soul taken by Shang Tsung. Sonya worries that they may not win against Goro, but Raiden disagrees. He reveals their own fears and egos are preventing them from winning the tournament.\nDespite Sonya's warning, Cage comes to Tsung to request a fight with Goro. The sorcerer accepts on the condition that he be allowed to challenge any opponent of his choosing, anytime and anywhere he chooses. Raiden tries to intervene, but the conditions are agreed upon before he can do so. After Shang Tsung leaves, Raiden confronts Cage for what he has done in challenging Goro, but is impressed when Cage shows his awareness of the gravity of the tournament. Cage faces Goro and uses guile and the element of surprise to defeat the defending champion. Now desperate, Tsung takes Sonya hostage and takes her to Outworld, intending to fight her as his opponent. Knowing that his powers are ineffective there and that Sonya cannot defeat Tsung by herself, Raiden sends Liu and Cage into Outworld in order to rescue Sonya and challenge Tsung. In Outworld, Liu is attacked by Reptile, but eventually gains the upper hand and defeats him. Afterward, Kitana meets up with Cage and Liu, revealing to the pair the origins of both herself and Outworld. Kitana allies with them and helps them to infiltrate Tsung's castle.\nInside the castle tower, Shang Tsung challenges Sonya to fight him, claiming that her refusal to accept will result in the Earth realm forfeiting Mortal Kombat (this is, in fact, a lie on Shang's part). All seems lost for Earth realm until Kitana, Liu, and Cage appear. Kitana berates Tsung for his treachery to the Emperor as Sonya is set free. Tsung challenges Cage, but is counter-challenged by Liu. During the lengthy battle, Liu faces not only Tsung, but the souls that Tsung had forcibly taken in past tournaments. In a last-ditch attempt to take advantage, Tsung morphs into Chan. Seeing through the charade, Liu renews his determination and ultimately fires an energy bolt at the sorcerer, knocking him down and impaling him on a row of spikes. Tsung's death releases all of the captive souls, including Chan's. Before ascending to the afterlife, Chan tells Liu that he will remain with him in spirit until they are once again reunited, after Liu dies.\nThe warriors return to Earth realm, where a victory celebration is taking place at the Shaolin temple. The jubilation abruptly stops, however, when Shao Kahn's giant figure suddenly appears in the skies. When the Emperor declares that he has come for everyone's souls, the warriors take up fighting stances.",
        "title": "Mortal Kombat",
        "question_id": "f1fdefcf-1191-b5f9-4cae-4ce4d0a59da7",
        "question": "Who took Goro's soul?",
        "answers": ["Shang Tsung."],
        "no_answer": false
      },
      "truncated_cells": []
    }
  ],
  "num_rows_total":60721,
  "num_rows_per_page":100,
  "partial":false
}
```

## Image and audio samples

Image and audio are represented by a URL that points to the file.

### Images

Images are represented as a JSON object with three fields:

- `src`: URL to the image file. It's a [signed URL](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-signed-urls.html) that expires after a certain time.
- `height`: height (in pixels) of the image
- `width`: width (in pixels) of the image

Here is an example of image, from the first row of the uoft-cs/cifar100 dataset:

```json
// https://datasets-server.huggingface.co/rows?dataset=uoft-cs/cifar100&config=cifar100&split=train&offset=0&length=1
{
  "features": [
    { "feature_idx": 0, "name": "img", "type": { "_type": "Image" } },
    ...
  ],
  "rows": [
    {
      "row_idx": 0,
      "row": {
        "img": {
          "src": "https://datasets-server.huggingface.co/cached-assets/uoft-cs/cifar100/--/aadb3af77e9048adbea6b47c21a81e47dd092ae5/--/cifar100/train/0/img/image.jpg?Expires=1710283469&Signature=A1v0cG07nuaBxYbuPR5EUZpJ9Se072SBDr4935gEsOESHGVyeqvd3qmvdsy1fuqbHk0dnx~p6MLtQ-hg3aCBOJ8eIJ5ItIoyYT4riJRuPQC0VFUb~b1maEwU8LRoXXuvrSysSz2QhBbC~ofv6cQudm~~bgGxXWAslDs180KnmPDsMU55ySsKyKQYNEkQKyuYvrGIJbFeg4lEps0f5CEwUstAwRAwlk~mzRpzUDBq7nJ~DcujTlllLv36nJX~too8mMnFn6dCn2nfGOFYwUiyYM73Czv-laLhVaIVUzcuJum90No~KNGzfYeFZpPqktA7MjCzRLf1gz5kA7wBqnY-8Q__&Key-Pair-Id=K3EI6M078Z3AC3",
          "height": 32,
          "width": 32
        },
        "fine_label": 19,
        "coarse_label": 11
      },
      "truncated_cells": []
    }
  ],
  "num_rows_total":50000,
  "num_rows_per_page":100,
  "partial":false
}
```

If the result has `partial: true` it means that the slices couldn't be run on the full dataset because it's too big.


### Caching

The images and audio samples are cached by the dataset viewer temporarily.
Internally we empty the cached assets of certain datasets from time to time based on usage.

If a certain asset is not available, you may have to call `/rows` again.


## Truncated responses

Unlike `/first-rows`, there is currently no truncation in `/rows`.
The `truncated_cells` field is still there but is always empty.
