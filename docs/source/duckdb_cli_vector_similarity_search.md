# Perform vector similarity search

The Fixed-Length Arrays feature was added in DuckDB version 0.10.0. This lets you use vector embeddings in DuckDB tables, making your data analysis even more powerful.

Additionally, the array_cosine_similarity function was introduced. This function measures the cosine of the angle between two vectors, indicating their similarity. A value of 1 means they’re perfectly aligned, 0 means they’re perpendicular, and -1 means they’re completely opposite.

Let's explore how to use this function for similarity searches. In this section, we’ll show you how to perform similarity searches using DuckDB.

We will use the [asoria/awesome-chatgpt-prompts-embeddings](https://huggingface.co/datasets/asoria/awesome-chatgpt-prompts-embeddings) dataset.

First, let's preview a few records from the dataset:

```bash
FROM 'hf://datasets/asoria/awesome-chatgpt-prompts-embeddings/data/*.parquet' SELECT act, prompt, len(embedding) as embed_len LIMIT 3;

┌──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────┐
│         act          │                                                                                    prompt                                                                                    │ embed_len │
│       varchar        │                                                                                   varchar                                                                                    │   int64   │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
│ Linux Terminal       │ I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output insid…  │       384 │
│ English Translator…  │ I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer…  │       384 │
│ `position` Intervi…  │ I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the `position` position. I want you to only reply as the inte…  │       384 │
└──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────┘

```

Next, let's choose an embedding to use for the similarity search:

```bash
FROM 'hf://datasets/asoria/awesome-chatgpt-prompts-embeddings/data/*.parquet' SELECT  embedding  WHERE act = 'Linux Terminal';

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                    embedding                                                                                                    │
│                                                                                                     float[]                                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [-0.020781303, -0.029143505, -0.0660217, -0.00932716, -0.02601602, -0.011426172, 0.06627567, 0.11941507, 0.0013917526, 0.012889079, 0.053234346, -0.07380514, 0.04871567, -0.043601237, -0.0025319182, 0.0448…  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

Now, let's use the selected embedding to find similar records:


```bash
SELECT act,
       prompt,
       array_cosine_similarity(embedding::float[384], (SELECT embedding FROM 'hf://datasets/asoria/awesome-chatgpt-prompts-embeddings/data/*.parquet' WHERE  act = 'Linux Terminal')::float[384]) AS similarity 
FROM 'hf://datasets/asoria/awesome-chatgpt-prompts-embeddings/data/*.parquet'
ORDER BY similarity DESC
LIMIT 3;

┌──────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────┐
│         act          │                                                                                   prompt                                                                                    │ similarity │
│       varchar        │                                                                                   varchar                                                                                   │   float    │
├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────┤
│ Linux Terminal       │ I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output insi…  │        1.0 │
│ JavaScript Console   │ I want you to act as a javascript console. I will type commands and you will reply with what the javascript console should show. I want you to only reply with the termin…  │  0.7599728 │
│ R programming Inte…  │ I want you to act as a R interpreter. I'll type commands and you'll reply with what the terminal should show. I want you to only reply with the terminal output inside on…  │  0.7303775 │
└──────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────┘

```

That's it! You have successfully performed a vector similarity search using DuckDB.
