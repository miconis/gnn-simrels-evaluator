import json
import torch
from pyspark.sql import SparkSession
import os
import xxhash  # pip install xxhash
from src.utils.config import ORCIDS_PATH, RAW_DIR, SIMRELS_FOR_TESTING_PATH, SIMRELS_SCORES_PATH

os.environ.pop("SPARK_HOME", None) # to prevent conflicts

spark = SparkSession.builder \
    .appName("Dataset Processor") \
    .master("local[*]") \
    .config("spark.driver.memory", "15g") \
    .config("spark.driver.maxResultSize", "0") \
    .getOrCreate()

sc = spark.sparkContext
scores = torch.load(SIMRELS_SCORES_PATH)
simrels = torch.load(SIMRELS_FOR_TESTING_PATH)

orcids_map = sc.textFile(RAW_DIR + "/authors") \
            .map(eval) \
            .map(lambda x: json.loads(x[0])['orcid']) \
            .distinct() \
            .collect()

authors = sc.textFile(RAW_DIR + "/authors") \
            .map(eval) \
            .map(lambda x: (x[1], json.loads(x[0])['orcid'])) \
            .map(lambda x: (x[0], orcids_map.index(x[1]))) \
            .sortBy(lambda x: x[0], ascending=True) \
            .map(lambda x: x[1])

orcids = torch.LongTensor(authors.collect())

os.makedirs(os.path.dirname(ORCIDS_PATH), exist_ok=True)
torch.save(orcids, ORCIDS_PATH)

print(authors.take(5))
print(orcids)
