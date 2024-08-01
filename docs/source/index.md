# 🤗 Dataset viewer

The dataset page includes a table with the dataset's contents, arranged by pages of 100 rows. You can navigate between pages using the buttons at the bottom of the table, filter, search, look at basic statistics, and more.

<div class="flex justify-center">
  <img
    style="margin-bottom: 0;"
    class="block dark:hidden"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets-server/openbookqa_light.png"
  />
  <img
    style="margin-bottom: 0;"
    class="hidden dark:block"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets-server/openbookqa_dark.png"
  />
</div>

<p style="text-align: center; font-style: italic; margin-top: 0;">
  Dataset viewer of the
  <a href="https://huggingface.co/datasets/openbookqa" rel="nofollow">
    OpenBookQA dataset
  </a>
</p>

## Contents of the documentation

These documentation pages are focused on the **dataset viewer's backend** (code in https://github.com/huggingface/dataset-viewer), which provides the table with pre-computed data through an API for all the datasets on the Hub. You can explore the sections if you want to consume the API for your application or to understand how we preprocess the datasets.

Otherwise, if you want to learn about creating datasets from the Hub's web-based interface, [**configuring the dataset viewer**](https://huggingface.co/docs/hub/datasets-data-files-configuration) for data, [images](https://huggingface.co/docs/hub/datasets-image), or audio, or fixing errors, you might prefer reading the [Datasets Hub documentation pages](https://huggingface.co/docs/hub/datasets). Take also a look to the [example datasets](https://huggingface.co/datasets-examples) collections: [splits configuration](https://huggingface.co/collections/datasets-examples/file-names-and-splits-655e28af4471bd95709eb135), [subsets configuration](https://huggingface.co/collections/datasets-examples/manual-configuration-655e293cea26da0acab95b87), [CSV data files](https://huggingface.co/collections/datasets-examples/format-csv-and-tsv-655f681cb9673a4249cccb3d) and [image datasets](https://huggingface.co/collections/datasets-examples/image-dataset-6568e7cf28639db76eb92d65).

## Dataset viewer's backend

The dataset viewer's backend provides an API for visualizing and exploring all types of datasets - computer vision, speech, text, and tabular - stored on the Hugging Face [Hub](https://huggingface.co/datasets).

The main feature of the dataset viewer's backend is to auto-convert all the [Hub datasets](https://huggingface.co/datasets) to [Parquet](https://parquet.apache.org/). Read more in the [Parquet section](./parquet).

As datasets increase in size and data type richness, the cost of preprocessing (storage and compute) these datasets can be challenging and time-consuming.
To help users access these modern datasets, The dataset viewer runs a server behind the scenes to generate the API responses ahead of time and stores them in a database so they are instantly returned when you make a query through the API.

Let the dataset viewer take care of the heavy lifting so you can use a simple **REST API** on any of the **100,000+ datasets on Hugging Face** to:

- List the **dataset splits, column names and data types**
- Get the **dataset size** (in number of rows or bytes)
- Download and view **rows at any index** in the dataset
- **Search** a word in the dataset
- **Filter** rows based on a query string
- Get insightful **statistics** about the data
- Access the dataset as **parquet files** to use in your favorite **processing or analytics framework**

Join the growing community on the [forum](https://discuss.huggingface.co/) or [Discord](https://discord.com/invite/JfAtkvEtRb) today, and give the [dataset viewer repository](https://github.com/huggingface/dataset-viewer) a ⭐️ if you're interested in the latest updates!
