# Perform SQL operations

Performing SQL operations with DuckDB opens up a world of possibilities for querying datasets efficiently. Let's dive into some examples showcasing the power of DuckDB functions.

For our demonstration, we'll explore a fascinating dataset. The [MMLU](https://huggingface.co/datasets/cais/mmlu) dataset is a multitask test containing multiple-choice questions spanning various knowledge domains.

To preview the dataset, let's select a sample of 3 rows:

```bash
FROM 'hf://datasets/cais/mmlu/all/test-*.parquet' USING SAMPLE 3;

┌──────────────────────┬──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────┐
│       question       │       subject        │                                                                         choices                                                                          │ answer │
│       varchar        │       varchar        │                                                                        varchar[]                                                                         │ int64  │
├──────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ Dr. Harry Holliday…  │ professional_psych…  │ [discuss his vacation plans with his current clients ahead of time so that they know he’ll be unavailable during that time., give his clients a phone …  │      2 │
│ A resident of a st…  │ professional_law     │ [The resident would succeed, because the logging company's selling of the timber would entitle the resident to re-enter and terminate the grant to the…  │      2 │
│ Moderate and frequ…  │ miscellaneous        │ [dispersed alluvial fan soil, heavy-textured soil, such as silty clay, light-textured soil, such as loamy sand, region of low humidity]                  │      2 │
└──────────────────────┴──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────┘

```

This command retrieves a random sample of 3 rows from the dataset for us to examine.

Let's start by examining the schema of our dataset. The following table outlines the structure of our dataset:

```bash
DESCRIBE FROM 'hf://datasets/cais/mmlu/all/test-*.parquet' USING SAMPLE 3;
┌─────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│ column_name │ column_type │  null   │   key   │ default │  extra  │
│   varchar   │   varchar   │ varchar │ varchar │ varchar │ varchar │
├─────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ question    │ VARCHAR     │ YES     │         │         │         │
│ subject     │ VARCHAR     │ YES     │         │         │         │
│ choices     │ VARCHAR[]   │ YES     │         │         │         │
│ answer      │ BIGINT      │ YES     │         │         │         │
└─────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┘

```
Next, let's analyze if there are any duplicated records in our dataset:

```bash
SELECT   *,
         COUNT(*) AS counts
FROM     'hf://datasets/cais/mmlu/all/test-*.parquet'
GROUP BY ALL
HAVING   counts > 2; 

┌──────────┬─────────┬───────────┬────────┬────────┐
│ question │ subject │  choices  │ answer │ counts │
│ varchar  │ varchar │ varchar[] │ int64  │ int64  │
├──────────┴─────────┴───────────┴────────┴────────┤
│                      0 rows                      │
└──────────────────────────────────────────────────┘

```

Fortunately, our dataset doesn't contain any duplicate records.

Let's see the proportion of questions based on the subject in a bar representation:

```bash
SELECT 
    subject, 
    COUNT(*) AS counts, 
    BAR(COUNT(*), 0, (SELECT COUNT(*) FROM 'hf://datasets/cais/mmlu/all/test-*.parquet')) AS percentage 
FROM 
    'hf://datasets/cais/mmlu/all/test-*.parquet' 
GROUP BY 
    subject 
ORDER BY 
    counts DESC;

┌──────────────────────────────┬────────┬────────────────────────────────────────────────────────────────────────────────┐
│           subject            │ counts │                                   percentage                                   │
│           varchar            │ int64  │                                    varchar                                     │
├──────────────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────┤
│ professional_law             │   1534 │ ████████▋                                                                      │
│ moral_scenarios              │    895 │ █████                                                                          │
│ miscellaneous                │    783 │ ████▍                                                                          │
│ professional_psychology      │    612 │ ███▍                                                                           │
│ high_school_psychology       │    545 │ ███                                                                            │
│ high_school_macroeconomics   │    390 │ ██▏                                                                            │
│ elementary_mathematics       │    378 │ ██▏                                                                            │
│ moral_disputes               │    346 │ █▉                                                                             │
├──────────────────────────────┴────────┴────────────────────────────────────────────────────────────────────────────────┤
│ 57 rows (8 shown)                                                                                           3 columns  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

Now, let's prepare a subset of the dataset containing questions related to **nutrition** and create a mapping of questions to correct answers.
Notice that we have the column **choices** from which we can get the correct answer using the **answer** column as an index.

```bash
SELECT *
FROM   'hf://datasets/cais/mmlu/all/test-*.parquet'
WHERE  subject = 'nutrition' LIMIT 3;

┌──────────────────────┬───────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────┐
│       question       │  subject  │                                                                               choices                                                                               │ answer │
│       varchar        │  varchar  │                                                                              varchar[]                                                                              │ int64  │
├──────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ Which foods tend t…  │ nutrition │ [Meat, Confectionary, Fruits and vegetables, Potatoes]                                                                                                              │      2 │
│ In which one of th…  │ nutrition │ [If the incidence rate of the disease falls., If survival time with the disease increases., If recovery of the disease is faster., If the population in which the…  │      1 │
│ Which of the follo…  │ nutrition │ [The flavonoid class comprises flavonoids and isoflavonoids., The digestibility and bioavailability of isoflavones in soya food products are not changed by proce…  │      0 │
└──────────────────────┴───────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────┘

```

```bash
SELECT question,
       choices[answer] AS correct_answer
FROM   'hf://datasets/cais/mmlu/all/test-*.parquet'
WHERE  subject = 'nutrition' LIMIT 3;

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────┐
│                                                              question                                                               │               correct_answer                │
│                                                               varchar                                                               │                   varchar                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Which foods tend to be consumed in lower quantities in Wales and Scotland (as of 2020)?\n                                           │ Confectionary                               │
│ In which one of the following circumstances will the prevalence of a disease in the population increase, all else being constant?\n │ If the incidence rate of the disease falls. │
│ Which of the following statements is correct?\n                                                                                     │                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────┘

```

To ensure data cleanliness, let's remove any newline characters at the end of the questions and filter out any empty answers:

```bash
SELECT regexp_replace(question, '\n', '') AS question,
       choices[answer] AS correct_answer
FROM   'hf://datasets/cais/mmlu/all/test-*.parquet'
WHERE  subject = 'nutrition' AND LENGTH(correct_answer) > 0 LIMIT 3;

┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────┐
│                                                             question                                                              │               correct_answer                │
│                                                              varchar                                                              │                   varchar                   │
├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Which foods tend to be consumed in lower quantities in Wales and Scotland (as of 2020)?                                           │ Confectionary                               │
│ In which one of the following circumstances will the prevalence of a disease in the population increase, all else being constant? │ If the incidence rate of the disease falls. │
│ Which vitamin is a major lipid-soluble antioxidant in cell membranes?                                                             │ Vitamin D                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────┘

```

Finally, lets hightlight some of the DuckDB functions used in this section:
- `DESCRIBE`, returns the table schema.
- `USING SAMPLE`, samples are used to randomly select a subset of a dataset.
- `BAR`, draws a band whose width is proportional to (x - min) and equal to width characters when x = max. Width defaults to 80.
- `string[begin:end]`, extracts a string using slice conventions. Missing begin or end arguments are interpreted as the beginning or end of the list respectively. Negative values are accepted.
- `regexp_replace`, if the string contains the regexp pattern, replaces the matching part with replacement.
- `LENGTH`, gets the number of characters in the string.

<Tip>

There are plenty of useful functions available in DuckDB's [SQL functions overview](https://duckdb.org/docs/sql/functions/overview). The best part is that you can use them directly on Hugging Face datasets.

</Tip>
