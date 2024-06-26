# 🤗 Dataset viewer

The dataset viewer is a lightweight web API for visualizing and exploring all types of datasets - computer vision, speech, text, and tabular - stored on the Hugging Face [Hub](https://huggingface.co/datasets).

The main feature of the dataset viewer is to auto-convert all the [Hub datasets](https://huggingface.co/datasets) to [Parquet](https://parquet.apache.org/). Read more in the [Parquet section](./parquet).

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

Join the growing community on the [forum](https://discuss.huggingface.co/) or [Discord](https://discord.com/invite/JfAtkvEtRb) today, and give the [dataset viewer repository](https://github.com/huggingface/dataset-viewer) a ⭐️ if you're interested in the latest updates!
