from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import json
from dgl.data import DGLDataset
from src.utils.utility import *
import torch
from dgl.data.utils import save_graphs, load_graphs
import dgl
import os
import random
import numpy as np

random.seed(1234)
np.random.seed(1234)

class PubmedSubgraph(DGLDataset):
    """
    Dataset for Graph Blocking.

    Parameters
    ----------
    url : URL to download the raw dataset
    raw_dir : directory that will store (or already stores) the downloaded data
    save_dir : directory to save preprocessed data
    force_reload : whether to reload dataset
    verbose : whether to print out progress information
    """
    def __init__(self,
                 dataset_name,
                 subgraph_base_path,
                 num_links,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):

        self.conf = SparkConf() \
                    .setAppName("Dataset Processor") \
                    .set("spark.driver.memory", "15g") \
                    .set("spark.driver.maxResultSize", "0")\
                    .setMaster("local[*]")

        self.sc = SparkContext(conf=self.conf)

        self.spark = SparkSession.builder.config(conf=self.conf).getOrCreate()

        self.subgraph_base_path = subgraph_base_path

        self.graph = None

        self.num_links = num_links

        self.pos_links = None
        self.neg_links = None

        super().__init__(name=dataset_name,
                         url=url,
                         raw_dir=raw_dir,
                         save_dir=save_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def download(self):
        """
        Download the raw data to local disk (create single files to be processed)
        """
        print("Downloading data!")

        if len(os.listdir(self.raw_dir)) <= 1:

            # PROCESS AUTHORS
            authors = self.sc.textFile(self.subgraph_base_path + "/authors").map(json.loads)
            authors.map(lambda x: dict(pub_embedding=x['pub_embedding'], id=x['id'])).map(json.dumps).zipWithIndex().saveAsTextFile(self.raw_dir + "/authors", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # PROCESS PUBLICATIONS
            publications = self.sc.textFile(self.subgraph_base_path + "/publications").map(json.loads)
            publications.map(lambda x: dict(bert_embedding=x['bert_embedding'], id=x['id'])).map(json.dumps).zipWithIndex().saveAsTextFile(self.raw_dir + "/publications", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # PROCESS RELATIONS
            all_relations = self.sc.textFile(self.subgraph_base_path + "/relations").map(json.loads)

            # Citations
            cites_rels = all_relations.filter(lambda x: x['relClass'] == 'Cites').map(lambda x: (x['source'], x['target']))
            cites_rels.saveAsTextFile(self.raw_dir + "/cites_rels", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # Co-Project
            coproject_rels = all_relations.filter(lambda x: x['relClass'] == 'isProducedBy').map(lambda x: (x['target'], x['source']))  # format: (project_id, publication_id)
            coproject_rels = coproject_rels.join(coproject_rels).map(lambda x: (x[1][0], x[1][1]))
            coproject_rels.saveAsTextFile(self.raw_dir + "/coprojected_rels", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # Collaborates
            collaborates_rels = all_relations.filter(lambda x: x['relClass'] == 'coAuthor').map(lambda x: (x['source'], x['target']))
            collaborates_rels.saveAsTextFile(self.raw_dir + "/collaborates_rels", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # Writes
            writes_rels = all_relations.filter(lambda x: x['relClass'] == 'writtenBy').map(lambda x: (x['target'], x['source']))
            writes_rels.saveAsTextFile(self.raw_dir + "/writes_rels", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # PotentiallyEquivalent
            keys_rels = authors.filter(lambda x: x['key'] != "").filter(lambda x: x['wellformed'] != False).map(lambda x: (x['key'], x['id']))
            keys_rels = keys_rels.join(keys_rels).map(lambda x: (x[1][0], x[1][1]))
            keys_rels.saveAsTextFile(self.raw_dir + "/potentiallyequivalent_rels", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # Equivalent
            orcid_rels = authors.filter(lambda x: x['orcid'] != "").map(lambda x: (x['orcid'], x['id']))
            orcid_rels = orcid_rels.join(orcid_rels).map(lambda x: (x[1][0], x[1][1]))
            orcid_rels.saveAsTextFile(self.raw_dir + "/equivalent_rels", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

        print("Data downloaded!")

    def process(self):
        """
        Process raw data to create the graph
        """
        # CREATE AUTHORS TENSOR
        print("Processing AUTHOR Nodes")
        authors = self.sc.textFile(self.raw_dir + "/authors").map(eval).map(lambda x: (x[1], x[0]))  # format: (index, pub_embedding)
        authors_tensor = torch.FloatTensor(authors.sortByKey(ascending=True).map(lambda x: json.loads(x[1])['pub_embedding']).collect())
        authors = authors.map(lambda x: (json.loads(x[1])['id'], x[0]))  # format: (id, index)

        # CREATE PUBLICATIONS TENSOR
        print("Processing PUBLICATION Nodes")
        publications = self.sc.textFile(self.raw_dir + "/publications").map(eval).map(lambda x: (x[1], x[0]))  # format: (index, bert_embedding)
        publications_tensor = torch.FloatTensor(publications.sortByKey(ascending=True).map(lambda x: json.loads(x[1])['bert_embedding']).collect())
        publications = publications.map(lambda x: (json.loads(x[1])['id'], x[0]))  # format: (id, index)

        # CITES RELS TENSOR
        print("Processing CITES Relations")
        cites_rels = self.sc.textFile(self.raw_dir + "/cites_rels").map(eval)  # format: (id_source, id_target)
        cites_rels = cites_rels.join(publications)  # format: (id_source, (id_target, index_source))
        cites_rels = cites_rels.map(lambda x: (x[1][0], x[1][1]))  # format: (id_target, index_source)
        cites_rels = cites_rels.join(publications)  # format: (id_target, (index_source, index_target))
        cites_rels = cites_rels.map(lambda x: [x[1][0], x[1][1]])  # format: (index_source, index_target)
        cites_rels_tensor = torch.LongTensor(cites_rels.collect())

        # COLLABORATES RELS TENSOR
        print("Processing COLLABORATES Relations")
        collaborates_rels = self.sc.textFile(self.raw_dir + "/collaborates_rels").map(eval)  # format: (id_source, id_target)
        collaborates_rels = collaborates_rels.join(authors)  # format: (id_source, (id_target, index_source))
        collaborates_rels = collaborates_rels.map(lambda x: (x[1][0], x[1][1]))  # format: (id_target, index_source)
        collaborates_rels = collaborates_rels.join(authors)  # format: (id_target, (index_source, index_target))
        collaborates_rels = collaborates_rels.map(lambda x: [x[1][0], x[1][1]])  # format: (index_source, index_target)
        collaborates_rels_tensor = torch.LongTensor(collaborates_rels.collect())

        # COPROJECTED RELS TENSOR
        print("Processing COPROJECTED Relations")
        coprojected_rels = self.sc.textFile(self.raw_dir + "/coprojected_rels").map(eval)
        coprojected_rels = coprojected_rels.join(publications)
        coprojected_rels = coprojected_rels.map(lambda x: (x[1][0], x[1][1]))
        coprojected_rels = coprojected_rels.join(publications)
        coprojected_rels = coprojected_rels.map(lambda x: [x[1][0], x[1][1]])
        coprojected_rels_tensor = torch.LongTensor(coprojected_rels.collect())

        # WRITES RELS TENSOR (INCLUDED IN COLLABORATES RELS)
        print("Processing WRITES Relations")
        writes_rels = self.sc.textFile(self.raw_dir + "/writes_rels").map(eval)  # format: (id_author, id_publication)
        writes_rels = writes_rels.join(authors)  # format: (id_author, (id_publication, index_author))
        writes_rels = writes_rels.map(lambda x: (x[1][0], x[1][1]))  # format: (id_publication, index_author)
        writes_rels = writes_rels.join(publications)  # format: (id_publication, (index_author, index_publication))
        writes_rels = writes_rels.map(lambda x: [x[1][0], x[1][1]])  # format: (index_author, index_publication)
        writes_rels_tensor = torch.LongTensor(writes_rels.collect())

        # POTENTIALLY EQUIVALENT RELS
        print("Processing POTENTIALLY EQUIVALENT Relations")
        potentiallyequivalent_rels = self.sc.textFile(self.raw_dir + "/potentiallyequivalent_rels").map(eval)
        potentiallyequivalent_rels = potentiallyequivalent_rels.join(authors)
        potentiallyequivalent_rels = potentiallyequivalent_rels.map(lambda x: (x[1][0], x[1][1]))
        potentiallyequivalent_rels = potentiallyequivalent_rels.join(authors)
        potentiallyequivalent_rels = potentiallyequivalent_rels.map(lambda x: [x[1][0], x[1][1]])
        potentiallyequivalent_rels_tensor = torch.LongTensor(potentiallyequivalent_rels.collect())

        # EQUIVALENT RELS
        print("Processing EQUIVALENT Relations")
        equivalent_rels = self.sc.textFile(self.raw_dir + "/equivalent_rels").map(eval)
        equivalent_rels = equivalent_rels.join(authors)
        equivalent_rels = equivalent_rels.map(lambda x: (x[1][0], x[1][1]))
        equivalent_rels = equivalent_rels.join(authors)
        equivalent_rels = equivalent_rels.map(lambda x: [x[1][0], x[1][1]])
        equivalent_rels_tensor = torch.LongTensor(equivalent_rels.collect())

        self.graph = dgl.heterograph(
            data_dict={
                ("publication", "cites", "publication"): (cites_rels_tensor[:, 0], cites_rels_tensor[:, 1]),
                ("author", "collaborates", "author"): (collaborates_rels_tensor[:, 0], collaborates_rels_tensor[:, 1]),
                ("publication", "coprojected", "publication"): (coprojected_rels_tensor[:, 0], coprojected_rels_tensor[:, 1]),
                ("author", "writes", "publication"): (writes_rels_tensor[:, 0], writes_rels_tensor[:, 1]),
                ("author", "potentially_equates", "author"): (potentiallyequivalent_rels_tensor[:, 0], potentiallyequivalent_rels_tensor[:, 1]),
                ("author", "equates", "author"): (equivalent_rels_tensor[:, 0], equivalent_rels_tensor[:, 1])
            },
            num_nodes_dict={
                "author": authors_tensor.shape[0],
                "publication": publications_tensor.shape[0]
            }
        )

        self.graph.ndata["feat"] = {
            "publication": publications_tensor,
            "author": authors_tensor
        }

        self.pos_links = self.generate_positive_links()
        self.neg_links = self.generate_negative_links()

    def generate_positive_links(self):
        """
        Generates positive links for link prediction training (potentially equivalent links)
        """
        print("Generating positive links")
        g_pe = dgl.edge_type_subgraph(self.graph, etypes=['potentially_equates'])

        u, v = g_pe.edges()

        eids = np.random.permutation(np.arange(g_pe.num_edges()))  # random permutation of numbers between 0 and num_edges
        u = u[eids[:self.num_links]]
        v = v[eids[:self.num_links]]

        return torch.stack((u, v), 0)

    def generate_negative_links(self):
        """
        Generates negative links for link prediction training (not equates links)
        """
        print("Generating negative links")
        g_e = dgl.edge_type_subgraph(self.graph, etypes=['equates'])
        neg_u_list = []
        neg_v_list = []
        for i in range(0, self.num_links):  # generate random negative edges
            u = random.randint(0, g_e.num_nodes() - 1)
            v = random.randint(0, g_e.num_nodes() - 1)
            if not g_e.has_edges_between(u, v):
                neg_u_list.append(u)
                neg_v_list.append(v)

        neg_u = torch.IntTensor(neg_u_list)
        neg_v = torch.IntTensor(neg_v_list)

        return torch.stack((neg_u, neg_v), 0)

    def get_graph(self):
        """
        Returns the Graph
        """
        return self.graph

    def get_pos_links(self):
        """
        Returns the positive edges
        """
        return self.pos_links

    def get_neg_links(self):
        """
        Returns the negative edges
        """
        return self.neg_links

    def save(self):
        """
        Save processed data to directory (self.save_path)
        """
        print("Saving graph on disk")
        save_graphs(self.save_dir + "/pubmed_graph.dgl", [self.graph])
        torch.save(self.pos_links, self.save_dir + "/positive_links.pt")
        torch.save(self.neg_links, self.save_dir + "/negative_links.pt")

    def load(self):
        """
        Load processed data from directory (self.save_path)
        """
        print("Load graph from disk")
        self.graph = load_graphs(self.save_dir + "/pubmed_graph.dgl")[0]
        self.pos_links = torch.load(self.save_dir + "/positive_links.pt")
        self.neg_links = torch.load(self.save_dir + "/negative_links.pt")

    def has_cache(self):
        """
        Check whether there are processed data
        """
        return os.path.exists(self.save_dir + "/pubmed_graph.dgl")

