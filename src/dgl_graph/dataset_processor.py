#  process raw pubmed subgraph to extract authors and create relations
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
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


def inject_embedding(x):
    pub = x[1][0]
    pub['bert_embedding'] = x[1][1]
    return pub


def mapAuthors(publication):
    pub_id = publication['id']
    pub_bert_embedding = publication['bert_embedding']
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
                     pub_embedding=pub_bert_embedding,
                     coauthors=coauthors,
                     key=lnfi(author))

            # add publication-author relation
            relations.append(dict(source=pub_id, target=author['id'], relClass="writtenBy", sourceType="publication", targetType="author"))
            authors.append(a)

    return authors, relations


if __name__ == "__main__":

    # read raw data as extracted from the OpenAIRE dump
    publications = sc.textFile("../../dataset/raw_pubmed_subgraph/publications").map(json.loads).map(lambda x: (x['id'], x))
    publications = publications.reduceByKey(lambda a, b: a)  # remove repeating ids
    publications_embeddings = spark.read.load("../../dataset/raw_pubmed_subgraph/publications_pubmed_bert_embeddings").rdd.map(lambda x: (x['id'], x['bert_embedding']))
    publications_embeddings = publications_embeddings.reduceByKey(lambda a, b: a)  # remove repeating ids

    # inject bert embeddings into the OpenAIRE data
    publications = publications.leftOuterJoin(publications_embeddings).map(inject_embedding)

    relations = sc.textFile("../../dataset/raw_pubmed_subgraph/relations").map(json.loads)

    # create authors and relations
    authors_relations = publications.map(mapAuthors)
    authors = authors_relations.flatMap(lambda x: x[0]).map(lambda x: (x['id'], x)).reduceByKey(lambda a, b: a).map(lambda x: x[1])
    relations = relations.union(authors_relations.flatMap(lambda x: x[1]))

    # insert errors in the dataset
    authors = authors.map(insert_error)

    save_rdd(publications, "../../dataset/processed_pubmed_subgraph/publications")
    save_rdd(relations, "../../dataset/processed_pubmed_subgraph/relations")
    save_rdd(authors, "../../dataset/processed_pubmed_subgraph/authors")

