#  process raw pubmed subgraph to extract authors and create relations
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import json
from src.utils.utility import *
import copy
import random
import string
from transliterate import translit

conf = SparkConf()\
    .setAppName("Dataset Processor") \
    .set("spark.driver.memory", "15g") \
    .setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

languages = ['el', 'hy', 'ka', 'ru']


def insert_error(author):
    n = random.random()

    author['wellformed'] = True
    if not is_author_wellformed(author):
        author['wellformed'] = False
        return author

    if 0.55 < n <= 0.7:  # no name and surname
        author['name'] = ""
        author['surname'] = ""
        author['wellformed'] = False

    if 0.7 < n <= 0.85:  # typo
        char1 = random.choice(string.ascii_lowercase)  # random character1
        char2 = random.choice(string.ascii_lowercase)  # random character2
        ran_pos1 = random.randint(0, len(author['name']) - 1)     # random index1
        ran_pos2 = random.randint(0, len(author['surname']) - 1)  # random index2
        author['name'] = replace_char(author['name'], ran_pos1, char1)
        author['surname'] = replace_char(author['surname'], ran_pos2, char2)
        author['wellformed'] = False

    if 0.85 < n <= 1.0:  # transliterated
        i = random.randint(0, len(languages) - 1)
        author['name'] = translit(author['name'], languages[i])
        author['surname'] = translit(author['surname'], languages[i])
        author['fullname'] = translit(author['fullname'], languages[i])

        author['wellformed'] = False
    return author


def mapAuthors(publication):
    pub_id = publication['id']
    coauthors_full = []
    authors = []
    relations = []

    for author in publication['author']:
        try:
            orcid = author['pid'][0]['value']
        except:
            orcid = ""

        try:
            name = author['name']
        except:
            name = ""

        try:
            surname = author['surname']
        except:
            surname = ""

        a = dict(name=name,
                 surname=surname,
                 fullname=author['fullname'],
                 orcid=orcid,
                 id=create_author_id(author['fullname'], publication['id']))
        coauthors_full.append(a)

    # add coAuthors relations
    for i in range(0, len(coauthors_full)-1):
        for j in range(i+1, len(coauthors_full)):
            if coauthors_full[i]['orcid'] != "" and coauthors_full[j]['orcid'] != "":
                relations.append(dict(source=coauthors_full[i]['id'],target=coauthors_full[j]['id'],relClass="coAuthor",sourceType="author",targetType="author"))

    for author in coauthors_full:
        if author['orcid'] != '':
            coauthors = copy.deepcopy(coauthors_full)
            coauthors.remove(author)

            a = dict(name=author['name'],
                     surname=author['surname'],
                     fullname=author['fullname'],
                     orcid=author['orcid'],
                     id=author['id'],
                     pub_id=pub_id,
                     coauthors=coauthors,
                     key=lnfi(author))

            # add publication-author relation
            relations.append(dict(source=pub_id, target=author['id'], relClass="writtenBy", sourceType="publication", targetType="author"))
            authors.append(a)

    return authors, relations


if __name__ == "__main__":

    # read raw data as extracted from the OpenAIRE dump
    publications = sc.textFile("../../dataset/raw_pubmed_subgraph/publications").map(json.loads)
    relations = sc.textFile("../../dataset/raw_pubmed_subgraph/relations").map(json.loads)

    # create authors and relations
    authors_relations = publications.map(mapAuthors)
    authors = authors_relations.flatMap(lambda x: x[0])
    relations = relations.union(authors_relations.flatMap(lambda x: x[1]))

    # insert errors in the dataset
    authors = authors.map(insert_error)

    # print dataset statistics
    print("Number of publications: ", publications.count())
    print("Number of relations: ", relations.count())
    for item in relations.map(lambda x: (str(x['sourceType']) + "<--(" + str(x['relClass']) + ")-->" + str(x['targetType']), 1)).reduceByKey(lambda a, b: a+b).collect():
        print(" - " + str(item[1]), item[0])
    print("Number of raw authors: ", authors.count())
    print(" - " + str(authors.filter(lambda x: x['orcid'] != "").count()), "with orcid")
    print(" - " + str(authors.filter(lambda x: x['wellformed'] is True).count()), "well formed")
    print(" - " + str(authors.filter(lambda x: x['key'] != "").count()), "with key")
    print(" - " + str(authors.map(lambda x: (x['orcid'], 1)).reduceByKey(lambda a, b: a+b).count()), "unique")
    print("Number of blocks: ", authors.map(lambda x: (x['key'], 1)).reduceByKey(lambda a, b: a+b).count())

    publications.map(lambda x: json.dumps(x, ensure_ascii=False)).saveAsTextFile(path="../../dataset/processed_pubmed_subgraph/publications", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
    relations.map(lambda x: json.dumps(x, ensure_ascii=False)).saveAsTextFile(path="../../dataset/processed_pubmed_subgraph/relations", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
    authors.map(lambda x: json.dumps(x, ensure_ascii=False)).saveAsTextFile(path="../../dataset/processed_pubmed_subgraph/authors", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

