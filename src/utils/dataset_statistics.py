from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from src.utils.utility import *


def reduction_ratio(authors):
    n = authors.count()  # number of entities

    p = n*(n-1)/2   # number of pairs without blocking

    # number of pairs after blocking
    pb = authors\
            .filter(lambda x: x['key'] != "")\
            .filter(lambda x: x['wellformed'] is True)\
            .map(lambda x: (x['key'], 1))\
            .reduceByKey(lambda a, b: a+b)\
            .map(lambda x: (x[1]*(x[1]-1))/2)\
            .reduce(lambda a, b: a+b)

    return 1 - (pb/p)


def correct_pairs(l):
    c = 0
    for i in range(0, len(l) - 1):
        for j in range(i+1, len(l)):
            if l[i] == l[j]:
                c += 1

    return c


def pairs_completeness(authors):

    # number of correct pairs
    pt = authors\
            .filter(lambda x: x['orcid'] != "")\
            .map(lambda x: (x['orcid'], 1))\
            .reduceByKey(lambda a, b: a+b)\
            .map(lambda x: (x[1]*(x[1]-1))/2)\
            .reduce(lambda a, b: a+b)

    # number of correct pairs after blocking
    pb = authors \
        .filter(lambda x: x['key'] != "") \
        .filter(lambda x: x['wellformed'] is True)\
        .map(lambda x: (x['key'], x['orcid']))\
        .groupByKey()\
        .map(lambda x: correct_pairs(list(x[1])))\
        .reduce(lambda a, b: a+b)

    return 1 - (pb/pt)


conf = SparkConf()\
    .setAppName("Dataset Processor") \
    .set("spark.driver.memory", "15g") \
    .setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

publications = sc.textFile("../../dataset/processed_pubmed_subgraph/publications").map(json.loads)
relations = sc.textFile("../../dataset/processed_pubmed_subgraph/relations").map(json.loads)
authors = sc.textFile("../../dataset/processed_pubmed_subgraph/authors").map(json.loads)

print("Number of publications: ", publications.count())
print("Number of relations: ", relations.count())
for item in relations.map(
        lambda x: (str(x['sourceType']) + "<--(" + str(x['relClass']) + ")-->" + str(x['targetType']), 1)).reduceByKey(
        lambda a, b: a + b).collect():
    print(" - " + str(item[1]), item[0])
print("Number of raw authors: ", authors.count())
print(" - " + str(authors.filter(lambda x: x['orcid'] != "").count()), "with orcid")
print(" - " + str(authors.filter(lambda x: x['wellformed'] is True).count()), "well formed")
print(" - " + str(authors.filter(lambda x: x['key'] != "").count()), "with key")
print(" - " + str(authors.map(lambda x: (x['orcid'], 1)).reduceByKey(lambda a, b: a + b).count()), "unique")
print("Number of blocks: ", authors.map(lambda x: (x['key'], 1)).reduceByKey(lambda a, b: a + b).count())
print("Reduction ratio: ", reduction_ratio(authors))
print("Pair completeness: ", pairs_completeness(authors))

