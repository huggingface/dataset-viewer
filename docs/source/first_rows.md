# Preview a dataset

The dataset viewer provides a `/first-rows` endpoint for visualizing the first 100 rows of a dataset. This'll give you a good idea of the data types and example data contained in a dataset.

This guide shows you how to use the dataset viewer's `/first-rows` endpoint to preview a dataset. Feel free to also try it out with [Postman](https://www.postman.com/huggingface/workspace/hugging-face-apis/request/23242779-32d6a8be-b800-446a-8cee-f6b5ca1710df), [RapidAPI](https://rapidapi.com/hugging-face-hugging-face-default/api/hugging-face-datasets-api), or [ReDoc](https://redocly.github.io/redoc/?url=https://datasets-server.huggingface.co/openapi.json#operation/listFirstRows).

The `/first-rows` endpoint accepts three query parameters:

- `dataset`: the dataset name, for example `nyu-mll/glue` or `mozilla-foundation/common_voice_10_0`
- `config`: the subset name, for example `cola`
- `split`: the split name, for example `train`

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/first-rows?dataset=ibm/duorc&config=SelfRC&split=train"
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
        "https://datasets-server.huggingface.co/first-rows?dataset=ibm/duorc&config=SelfRC&split=train",
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
curl https://datasets-server.huggingface.co/first-rows?dataset=ibm/duorc&config=SelfRC&split=train \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

The endpoint response is a JSON containing two keys:

- The [`features`](https://huggingface.co/docs/datasets/about_dataset_features) of a dataset, including the column's name and data type.
- The first 100 `rows` of a dataset and the content contained in each column of a specific row.

For example, here are the `features` and the first 100 `rows` of the `ibm/duorc`/`SelfRC` train split:

```json
{
  "dataset": "ibm/duorc",
  "config": "SelfRC",
  "split": "train",
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
      "row_idx": 0,
      "row": {
        "plot_id": "/m/03vyhn",
        "plot": "200 years in the future, Mars has been colonized by a high-tech company.\nMelanie Ballard (Natasha Henstridge) arrives by train to a Mars mining camp which has cut all communication links with the company headquarters. She's not alone, as she is with a group of fellow police officers. They find the mining camp deserted except for a person in the prison, Desolation Williams (Ice Cube), who seems to laugh about them because they are all going to die. They were supposed to take Desolation to headquarters, but decide to explore first to find out what happened.They find a man inside an encapsulated mining car, who tells them not to open it. However, they do and he tries to kill them. One of the cops witnesses strange men with deep scarred and heavily tattooed faces killing the remaining survivors. The cops realise they need to leave the place fast.Desolation explains that the miners opened a kind of Martian construction in the soil which unleashed red dust. Those who breathed that dust became violent psychopaths who started to build weapons and kill the uninfected. They changed genetically, becoming distorted but much stronger.The cops and Desolation leave the prison with difficulty, and devise a plan to kill all the genetically modified ex-miners on the way out. However, the plan goes awry, and only Melanie and Desolation reach headquarters alive. Melanie realises that her bosses won't ever believe her. However, the red dust eventually arrives to headquarters, and Melanie and Desolation need to fight once again.",
        "title": "Ghosts of Mars",
        "question_id": "b440de7d-9c3f-841c-eaec-a14bdff950d1",
        "question": "How did the police arrive at the Mars mining camp?",
        "answers": ["They arrived by train."],
        "no_answer": false
      },
      "truncated_cells": []
    },
    {
      "row_idx": 1,
      "row": {
        "plot_id": "/m/03vyhn",
        "plot": "200 years in the future, Mars has been colonized by a high-tech company.\nMelanie Ballard (Natasha Henstridge) arrives by train to a Mars mining camp which has cut all communication links with the company headquarters. She's not alone, as she is with a group of fellow police officers. They find the mining camp deserted except for a person in the prison, Desolation Williams (Ice Cube), who seems to laugh about them because they are all going to die. They were supposed to take Desolation to headquarters, but decide to explore first to find out what happened.They find a man inside an encapsulated mining car, who tells them not to open it. However, they do and he tries to kill them. One of the cops witnesses strange men with deep scarred and heavily tattooed faces killing the remaining survivors. The cops realise they need to leave the place fast.Desolation explains that the miners opened a kind of Martian construction in the soil which unleashed red dust. Those who breathed that dust became violent psychopaths who started to build weapons and kill the uninfected. They changed genetically, becoming distorted but much stronger.The cops and Desolation leave the prison with difficulty, and devise a plan to kill all the genetically modified ex-miners on the way out. However, the plan goes awry, and only Melanie and Desolation reach headquarters alive. Melanie realises that her bosses won't ever believe her. However, the red dust eventually arrives to headquarters, and Melanie and Desolation need to fight once again.",
        "title": "Ghosts of Mars",
        "question_id": "a9f95c0d-121f-3ca9-1595-d497dc8bc56c",
        "question": "Who has colonized Mars 200 years in the future?",
        "answers": [
          "A high-tech company has colonized Mars 200 years in the future."
        ],
        "no_answer": false
      },
      "truncated_cells": []
    }
    ...
  ],
  "truncated": false
}
```

## Truncated responses

For some datasets, the response size from `/first-rows` may exceed 1MB, in which case the response is truncated until the size is under 1MB. This means you may not get 100 rows in your response because the rows are truncated, in which case the `truncated` field would be `true`.

In some cases, if even the first few rows generate a response that exceeds 1MB, some of the columns are truncated and converted to a string. You'll see these listed in the `truncated_cells` field.

For example, the [`GEM/SciDuet`](https://datasets-server.huggingface.co/first-rows?dataset=GEM/SciDuet&config=default&split=train) dataset only returns 10 rows, and the `paper_abstract`, `paper_content`, `paper_headers`, `slide_content_text` and `target` columns are truncated:

```json
  ...
  "rows": [
    {
      {
         "row_idx":8,
         "row":{
            "gem_id":"GEM-SciDuet-train-1#paper-954#slide-8",
            "paper_id":"954",
            "paper_title":"Incremental Syntactic Language Models for Phrase-based Translation",
            "paper_abstract":"\"This paper describes a novel technique for incorporating syntactic knowledge into phrasebased machi",
            "paper_content":"{\"paper_content_id\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29",
            "paper_headers":"{\"paper_header_number\":[\"1\",\"2\",\"3\",\"3.1\",\"3.3\",\"4\",\"4.1\",\"6\",\"7\"],\"paper_header_content\":[\"Introduc",
            "slide_id":"GEM-SciDuet-train-1#paper-954#slide-8",
            "slide_title":"Does an Incremental Syntactic LM Help Translation",
            "slide_content_text":"\"but will it make my BLEU score go up?\\nMotivation Syntactic LM Decoder Integration Questions?\\nMose",
            "target":"\"but will it make my BLEU score go up?\\nMotivation Syntactic LM Decoder Integration Questions?\\nMose",
            "references":[]
         },
         "truncated_cells":[
            "paper_abstract",
            "paper_content",
            "paper_headers",
            "slide_content_text",
            "target"
         ]
      },
      {
         "row_idx":9,
         "row":{
            "gem_id":"GEM-SciDuet-train-1#paper-954#slide-9",
            "paper_id":"954",
            "paper_title":"Incremental Syntactic Language Models for Phrase-based Translation",
            "paper_abstract":"\"This paper describes a novel technique for incorporating syntactic knowledge into phrasebased machi",
            "paper_content":"{\"paper_content_id\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29",
            "paper_headers":"{\"paper_header_number\":[\"1\",\"2\",\"3\",\"3.1\",\"3.3\",\"4\",\"4.1\",\"6\",\"7\"],\"paper_header_content\":[\"Introduc",
            "slide_id":"GEM-SciDuet-train-1#paper-954#slide-9",
            "slide_title":"Perplexity Results",
            "slide_content_text":"\"Language models trained on WSJ Treebank corpus\\nMotivation Syntactic LM Decoder Integration Questio",
            "target":"\"Language models trained on WSJ Treebank corpus\\nMotivation Syntactic LM Decoder Integration Questio",
            "references":[
               
            ]
         },
         "truncated_cells":[
            "paper_abstract",
            "paper_content",
            "paper_headers",
            "slide_content_text",
            "target"
         ]
      }
      "truncated_cells": ["target", "feat_dynamic_real"]
    },
  ...
  ],
  truncated: true
```
