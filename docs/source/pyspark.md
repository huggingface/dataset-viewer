# PySpark

[pyspark](https://spark.apache.org/docs/latest/api/python) is the Python interface for Apache Spark, enabling large-scale data processing and real-time analytics in a distributed environment using Python.

<Tip>

For a detailed guide on how to analyze datasets on the Hub with PySpark, check out this [blog](https://huggingface.co/blog/asoria/pyspark-hugging-face-datasets).

</Tip>

To start working with Parquet files in PySpark, you'll first need to add the file(s) to a Spark context. Below is an example of how to read a single Parquet file:

```py
from pyspark import SparkFiles, SparkContext, SparkFiles
from pyspark.sql import SparkSession

# Initialize a Spark session
spark = SparkSession.builder.appName("WineReviews").getOrCreate()

# Add the Parquet file to the Spark context
spark.sparkContext.addFile("https://huggingface.co/api/datasets/james-burton/wine_reviews/parquet/default/train/0.parquet")

# Read the Parquet file into a DataFrame
df = spark.read.parquet(SparkFiles.get("0.parquet"))

```
If your dataset is sharded into multiple Parquet files, you'll need to add each file to the Spark context individually. Here's how to do it:

```py
import requests

# Fetch the URLs of the Parquet files for the train split
r = requests.get('https://huggingface.co/api/datasets/james-burton/wine_reviews/parquet')
train_parquet_files = r.json()['default']['train']

# Add each Parquet file to the Spark context
for url in train_parquet_files:
  spark.sparkContext.addFile(url)

# Read all Parquet files into a single DataFrame
df = spark.read.parquet(SparkFiles.getRootDirectory() + "/*.parquet")

```

Once you've loaded the data into a PySpark DataFrame, you can perform various operations to explore and analyze it:

```py
print(f"Shape of the dataset: {df.count()}, {len(df.columns)}")

# Display first 10 rows
df.show(n=10)

# Get a statistical summary of the data
df.describe().show()

# Print the schema of the DataFrame
df.printSchema()

```