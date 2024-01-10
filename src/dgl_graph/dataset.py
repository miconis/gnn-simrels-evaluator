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
    dataset_name : name of the dataset
    subgraph_base_path : path to download the base subgraph
    pos_etype : edge type of positive edges for link prediction
    neg_etype : edge type of edges to be used to compute negative graph
    url : URL to download the raw dataset
    raw_dir : directory that will store (or already stores) the downloaded data
    save_dir : directory to save preprocessed data
    force_reload : whether to reload dataset
    verbose : whether to print out progress information
    """
    def __init__(self,
                 dataset_name,
                 subgraph_base_path,
                 neg_etype,
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

        self.neg_etype = neg_etype

        self.graph = None
        self.negative_edges = None
        self.wellformed_mask = None
        self.key_block_size_mask = None
        self.label_block_size_mask = None
        self.labels = None
        self.keys = None

        super().__init__(name=dataset_name,
                         url=url,
                         raw_dir=raw_dir,
                         save_dir=save_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def get_sc(self):
        return self.sc

    def get_spark(self):
        return self.spark

    def download(self):
        """
        Download the raw data to local disk (create single files to be processed)
        """
        print("Downloading data!")

        if len(os.listdir(self.raw_dir)) <= 1:

            # PROCESS AUTHORS
            authors = self.sc.textFile(self.subgraph_base_path + "/authors").map(json.loads)\
                .map(lambda x: dict(pub_embedding=x['pub_embedding'], id=x['id'], key=x['key'], wellformed=x['wellformed'], orcid=x['orcid']))\
                .zipWithIndex()  # (json, index)
            authors.map(lambda x: (json.dumps(x[0]), x[1])).saveAsTextFile(self.raw_dir + "/authors", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # PROCESS PUBLICATIONS
            publications = self.sc.textFile(self.subgraph_base_path + "/publications").map(json.loads)\
                .map(lambda x: dict(bert_embedding=x['bert_embedding'], id=x['id']))\
                .zipWithIndex()  # (json, index)
            publications.map(lambda x: (json.dumps(x[0]), x[1])).saveAsTextFile(self.raw_dir + "/publications", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

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
            keys_rels = authors.map(lambda x: x[0]).filter(lambda x: x['key'] != "").filter(lambda x: x['wellformed'] != False).map(lambda x: (x['key'], x['id']))
            keys_rels = keys_rels.join(keys_rels).map(lambda x: (x[1][0], x[1][1]))
            keys_rels.saveAsTextFile(self.raw_dir + "/potentiallyequivalent_rels", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # Equivalent
            orcid_rels = authors.map(lambda x: x[0]).filter(lambda x: x['orcid'] != "").map(lambda x: (x['orcid'], x['id']))
            orcid_rels = orcid_rels.join(orcid_rels).map(lambda x: (x[1][0], x[1][1]))
            orcid_rels.saveAsTextFile(self.raw_dir + "/equivalent_rels", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

            # PROCESS SIMRELS
            simrels = self.sc.textFile(self.subgraph_base_path + "/simrels").map(json.loads).map(lambda x: (x['source'], x['target']))
            simrels.saveAsTextFile(self.raw_dir + "/simrels", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

        print("Data downloaded!")

    def process(self):
        """
        Process raw data to create the graph
        """
        # CREATE AUTHORS TENSOR
        print("Processing AUTHOR Nodes")
        authors = self.sc.textFile(self.raw_dir + "/authors").map(eval).map(lambda x: (json.loads(x[0]), x[1])).sortBy(lambda x: x[1], ascending=True)  # format: (json, index)
        authors_tensor = torch.FloatTensor(authors.map(lambda x: x[0]['pub_embedding']).collect())
        authors_for_join = authors.map(lambda x: (x[0]['id'], x[1]))  # (id, index)

        # CREATE PUBLICATIONS TENSOR
        print("Processing PUBLICATION Nodes")
        publications = self.sc.textFile(self.raw_dir + "/publications").map(eval).map(lambda x: (json.loads(x[0]), x[1])).sortBy(lambda x: x[1], ascending=True)  # format: (json, index)
        publications_tensor = torch.FloatTensor(publications.map(lambda x: x[0]['bert_embedding']).collect())
        publications_for_join = publications.map(lambda x: (x[0]['id'], x[1]))  # (id, index)

        # CITES RELS TENSOR
        print("Processing CITES Relations")
        cites_rels = self.sc.textFile(self.raw_dir + "/cites_rels").map(eval)  # format: (id_source, id_target)
        cites_rels = cites_rels.join(publications_for_join)  # format: (id_source, (id_target, index_source))
        cites_rels = cites_rels.map(lambda x: (x[1][0], x[1][1]))  # format: (id_target, index_source)
        cites_rels = cites_rels.join(publications_for_join)  # format: (id_target, (index_source, index_target))
        cites_rels = cites_rels.map(lambda x: [x[1][0], x[1][1]])  # format: (index_source, index_target)
        cites_rels_tensor = torch.LongTensor(cites_rels.collect())

        # COLLABORATES RELS TENSOR
        print("Processing COLLABORATES Relations")
        collaborates_rels = self.sc.textFile(self.raw_dir + "/collaborates_rels").map(eval)  # format: (id_source, id_target)
        collaborates_rels = collaborates_rels.join(authors_for_join)  # format: (id_source, (id_target, index_source))
        collaborates_rels = collaborates_rels.map(lambda x: (x[1][0], x[1][1]))  # format: (id_target, index_source)
        collaborates_rels = collaborates_rels.join(authors_for_join)  # format: (id_target, (index_source, index_target))
        collaborates_rels = collaborates_rels.map(lambda x: [x[1][0], x[1][1]])  # format: (index_source, index_target)
        collaborates_rels_tensor = torch.LongTensor(collaborates_rels.collect())

        # COPROJECTED RELS TENSOR
        print("Processing COPROJECTED Relations")
        coprojected_rels = self.sc.textFile(self.raw_dir + "/coprojected_rels").map(eval)
        coprojected_rels = coprojected_rels.join(publications_for_join)
        coprojected_rels = coprojected_rels.map(lambda x: (x[1][0], x[1][1]))
        coprojected_rels = coprojected_rels.join(publications_for_join)
        coprojected_rels = coprojected_rels.map(lambda x: [x[1][0], x[1][1]])
        coprojected_rels_tensor = torch.LongTensor(coprojected_rels.collect())

        # WRITES RELS TENSOR (INCLUDED IN COLLABORATES RELS)
        print("Processing WRITES Relations")
        writes_rels = self.sc.textFile(self.raw_dir + "/writes_rels").map(eval)  # format: (id_author, id_publication)
        writes_rels = writes_rels.join(authors_for_join)  # format: (id_author, (id_publication, index_author))
        writes_rels = writes_rels.map(lambda x: (x[1][0], x[1][1]))  # format: (id_publication, index_author)
        writes_rels = writes_rels.join(publications_for_join)  # format: (id_publication, (index_author, index_publication))
        writes_rels = writes_rels.map(lambda x: [x[1][0], x[1][1]])  # format: (index_author, index_publication)
        writes_rels_tensor = torch.LongTensor(writes_rels.collect())

        # POTENTIALLY EQUIVALENT RELS
        print("Processing POTENTIALLY EQUIVALENT Relations")
        potentiallyequivalent_rels = self.sc.textFile(self.raw_dir + "/potentiallyequivalent_rels").map(eval)
        potentiallyequivalent_rels = potentiallyequivalent_rels.join(authors_for_join)
        potentiallyequivalent_rels = potentiallyequivalent_rels.map(lambda x: (x[1][0], x[1][1]))
        potentiallyequivalent_rels = potentiallyequivalent_rels.join(authors_for_join)
        potentiallyequivalent_rels = potentiallyequivalent_rels.map(lambda x: [x[1][0], x[1][1]])
        potentiallyequivalent_rels_tensor = torch.LongTensor(potentiallyequivalent_rels.collect())

        # EQUIVALENT RELS
        print("Processing EQUIVALENT Relations")
        equivalent_rels = self.sc.textFile(self.raw_dir + "/equivalent_rels").map(eval)
        equivalent_rels = equivalent_rels.join(authors_for_join)
        equivalent_rels = equivalent_rels.map(lambda x: (x[1][0], x[1][1]))
        equivalent_rels = equivalent_rels.join(authors_for_join)
        equivalent_rels = equivalent_rels.map(lambda x: [x[1][0], x[1][1]])
        equivalent_rels_tensor = torch.LongTensor(equivalent_rels.collect())

        # SIMRELS
        print("Processing SIMILARITY Relations")
        similarity_rels = self.sc.textFile(self.raw_dir + "/simrels").map(eval)
        similarity_rels = similarity_rels.join(authors_for_join)
        similarity_rels = similarity_rels.map(lambda x: (x[1][0], x[1][1]))
        similarity_rels = similarity_rels.join(authors_for_join)
        similarity_rels = similarity_rels.map(lambda x: [x[1][0], x[1][1]])
        similarity_rels_tensor = torch.LongTensor(similarity_rels.collect())


        self.graph = dgl.heterograph(
            data_dict={
                ("publication", "cites", "publication"): (cites_rels_tensor[:, 0], cites_rels_tensor[:, 1]),
                ("author", "collaborates", "author"): (collaborates_rels_tensor[:, 0], collaborates_rels_tensor[:, 1]),
                ("publication", "coprojected", "publication"): (coprojected_rels_tensor[:, 0], coprojected_rels_tensor[:, 1]),
                ("author", "writes", "publication"): (writes_rels_tensor[:, 0], writes_rels_tensor[:, 1]),
                ("publication", "is_written_by", "author"): (writes_rels_tensor[:, 1], writes_rels_tensor[:, 0]),
                ("author", "potentially_equates", "author"): (potentiallyequivalent_rels_tensor[:, 0], potentiallyequivalent_rels_tensor[:, 1]),
                ("author", "equates", "author"): (equivalent_rels_tensor[:, 0], equivalent_rels_tensor[:, 1]),
                ("author", "similar", "author"): (similarity_rels_tensor[:, 0], similarity_rels_tensor[:, 1])
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

        self.negative_edges = self.generate_negative_edges(etype=self.neg_etype)
        self.keys = self.generate_keys(authors=authors)
        self.wellformed_mask = self.generate_wellformed_mask(authors=authors)
        self.singleton_label_mask = self.generate_label_block_size_mask(authors=authors)
        self.singleton_key_mask = self.generate_key_block_size_mask(authors=authors)
        self.labels = self.generate_labels(authors=authors)

    def get_graph(self):
        """
        Returns the Graph
        """
        return self.graph

    def get_wellformed_mask(self):
        """
        Returns the wellformed mask
        """
        return self.wellformed_mask

    def get_key_block_size_mask(self):
        """
        Returns the singleton mask
        """
        return self.key_block_size_mask

    def get_labels_block_size_mask(self):
        """
        Returns the singleton mask
        """
        return self.label_block_size_mask

    def get_labels(self):
        """
        Return labels for authors
        """
        return self.labels

    def get_keys(self):
        """
        Return keys for authors
        """
        return self.keys

    def save(self):
        """
        Save processed data to directory (self.save_path)
        """
        print("Saving graph on disk")
        save_graphs(self.save_dir + "/pubmed_graph.dgl", [self.graph])
        torch.save(self.negative_edges, self.save_dir + "/negative_edges.pt")
        torch.save(self.keys, self.save_dir + "/author_clean_keys.pt")
        torch.save(self.wellformed_mask, self.save_dir + "/wellformed_mask.pt")
        torch.save(self.labels, self.save_dir + "/author_labels.pt")
        torch.save(self.label_block_size_mask, self.save_dir + "/label_block_size_mask.pt")
        torch.save(self.key_block_size_mask, self.save_dir + "/key_block_size_mask.pt")

    def load(self):
        """
        Load processed data from directory (self.save_path)
        """
        print("Load graph from disk")
        self.graph = load_graphs(self.save_dir + "/pubmed_graph.dgl")[0]
        self.negative_edges = torch.load(self.save_dir + "/negative_edges.pt")
        self.keys = torch.load(self.save_dir + "/author_clean_keys.pt")
        self.wellformed_mask = torch.load(self.save_dir + "/wellformed_mask.pt")
        self.labels = torch.load(self.save_dir + "/author_labels.pt")
        self.label_block_size_mask = torch.load(self.save_dir + "/label_block_size_mask.pt")
        self.key_block_size_mask = torch.load(self.save_dir + "/key_block_size_mask.pt")


    def get_clean_keys(self):
        return self.keys.long()

    def has_cache(self):
        """
        Check whether there are processed data
        """
        return os.path.exists(self.save_dir + "/pubmed_graph.dgl")

    def generate_negative_edges(self, etype):
        """
        Generates negative edges of the given type
        Parameters:
            - pos_etype: positive edges
            - neg_etype: edges to be used to compute negative graph
        """
        print("Generating negative edges")
        n_edges = self.graph.number_of_edges(etype=etype)  # to generate the same number of negative edges

        # create negative edges
        base_graph = dgl.edge_type_subgraph(self.graph, etypes=[etype])
        neg_src_list = []
        neg_dst_list = []
        count = 0
        while count < n_edges:
            src_id, dst_id = random.randint(0, base_graph.num_nodes() - 1), random.randint(0, base_graph.num_nodes() - 1)
            if not base_graph.has_edges_between(src_id, dst_id):
                neg_src_list.append(src_id)
                neg_dst_list.append(dst_id)
                count += 1

        neg_src = torch.LongTensor(neg_src_list)
        neg_dst = torch.LongTensor(neg_dst_list)

        return torch.stack((neg_src, neg_dst), 0)

    def generate_keys(self, authors):
        """
        Generate keys for groups
        Parameters:
            - the author RDD in the form (JSON, index)
        """
        keys_list = authors.map(lambda x: x[0]['key']).distinct().collect()  # used to convert the label name to numbers
        keys = authors.map(lambda x: x[0]['key']).map(lambda x: keys_list.index(x))
        keys_tensor = torch.IntTensor(keys.collect())

        return keys_tensor

    def generate_labels(self, authors):
        """
        Generate labels for groups
        Parameters:
            - the author RDD in the form (JSON, index)
        """
        print("Generating keys")
        labels_list = authors.map(lambda x: x[0]['orcid']).distinct().collect()  # used to convert the orcids to numbers
        labels = authors.map(lambda x: x[0]['orcid']).map(lambda x: labels_list.index(x))
        labels_tensor = torch.IntTensor(labels.collect())

        return labels_tensor

    def generate_wellformed_mask(self, authors):
        """
        Generate wellformed mask
        Parameters:
            - the author RDD in the form (JSON, index)
        """
        print("Generating wellformed mask")
        wellformed_mask_tensor = torch.IntTensor(authors.map(lambda x: x[0]['wellformed']).map(lambda x: 1 if x==True else 0).collect())

        return wellformed_mask_tensor

    def generate_label_block_size_mask(self, authors):
        """
        Generate mask to count number of same labels
        Parameters:
            - the author RDD in the form (JSON, index)
        """
        print("Generating singleton mask")
        block_size = authors.map(lambda x: (x[0]['orcid'], 1)).reduceByKey(lambda a,b: a+b)

        size_mask_tensor = torch.IntTensor(authors.map(lambda x: (x[0]['orcid'], x[1])).leftOuterJoin(block_size).map(lambda x: (x[1][0], x[1][1])).sortByKey(ascending=True).map(lambda x: 0 if x[1]==None else x[1]).collect())

        return size_mask_tensor

    def generate_key_block_size_mask(self, authors):
        """
        Generate mask to count number of same keys
        Parameters:
            - the author RDD in the form (JSON, index)
        """
        print("Generating singleton mask")
        block_size = authors.map(lambda x: (x[0]['key'], 1)).reduceByKey(lambda a,b: a+b)

        size_mask_tensor = torch.IntTensor(authors.map(lambda x: (x[0]['key'], x[1])).leftOuterJoin(block_size).map(lambda x: (x[1][0], x[1][1])).sortByKey(ascending=True).map(lambda x: 0 if x[1]==None else x[1]).collect())

        return size_mask_tensor

    def get_node_embeddings_graphs(self):
        """
        Return graphs to be used for the node embedding part of the architecture
        """
        potentially_equates_graph = dgl.edge_type_subgraph(self.graph[0], ['potentially_equates'])
        colleague_graph = dgl.metapath_reachable_graph(self.graph[0], ['writes', 'coprojected', 'is_written_by'])
        citation_graph = dgl.metapath_reachable_graph(self.graph[0], ['writes', 'cites', 'is_written_by'])
        collaboration_graph = dgl.edge_type_subgraph(self.graph[0], etypes=['collaborates'])

        return dgl.add_self_loop(potentially_equates_graph), dgl.add_self_loop(colleague_graph), dgl.add_self_loop(citation_graph), dgl.add_self_loop(collaboration_graph)

    def get_link_prediction_graphs(self, pos_etype, train_ratio):
        """
        Returns splittings for training and testing
        """
        n_edges = self.graph[0].number_of_edges(etype=pos_etype)
        n_nodes = self.graph[0].edge_type_subgraph(etypes=[pos_etype]).num_nodes()
        train_size = int(train_ratio * n_edges)
        # random permutation to split nodes randomly
        eids = np.arange(n_edges)
        eids = np.random.permutation(eids)

        # positive edges
        pos_src, pos_dst = self.graph[0].edges(etype=pos_etype)
        train_src, train_dst = pos_src[eids[:train_size]], pos_dst[eids[:train_size]]
        test_src, test_dst = pos_src[eids[train_size:]], pos_dst[eids[train_size:]]

        # negative edges
        neg_src, neg_dst = self.negative_edges[0], self.negative_edges[1]
        train_neg_src, train_neg_dst = neg_src[eids[:train_size]], neg_dst[eids[:train_size]]
        test_neg_src, test_neg_dst = neg_src[eids[train_size:]], neg_dst[eids[train_size:]]

        # create graphs
        train_pos_graph = dgl.graph((train_src, train_dst), num_nodes=n_nodes)
        test_pos_graph = dgl.graph((test_src, test_dst), num_nodes=n_nodes)
        train_neg_graph = dgl.graph((train_neg_src, train_neg_dst), num_nodes=n_nodes)
        test_neg_graph = dgl.graph((test_neg_src, test_neg_dst), num_nodes=n_nodes)

        return train_pos_graph, train_neg_graph, test_pos_graph, test_neg_graph

    def get_random_regularization_graph(self, n_reg_edges, etype):
        """
        Returns the random regularization graph based on the given etype
        Parameters:
            - n_reg_edges: number of edges to be created
            - etype: type of edge to be used
        """
        reg_src, reg_dst = self.graph[0].edges(etype=etype)
        n_edges = self.graph[0].number_of_edges(etype=etype)
        n_nodes = self.graph[0].edge_type_subgraph(etypes=[etype]).num_nodes()

        # random permutation
        eids = np.arange(n_edges)
        eids = np.random.permutation(eids)

        regularization_graph = dgl.graph((reg_src[eids][0:n_reg_edges], reg_dst[eids][0:n_reg_edges]), num_nodes=n_nodes)

        return regularization_graph

    def get_simrels_graph(self):
        return dgl.edge_type_subgraph(self.graph[0], ['similar'])

    def get_simrel_splittings(self, train_ratio):
        """
        Returns splittings for training and testing
        """
        n_edges = self.graph[0].number_of_edges(etype="similar")
        n_nodes = self.graph[0].edge_type_subgraph(etypes=["similar"]).num_nodes()

        orcids_graph = self.graph[0].edge_type_subgraph(etypes=["equates"])
        sim_src, sim_dst = self.graph[0].edges(etype="similar")
        correct_simrel_mask = orcids_graph.has_edges_between(sim_src, sim_dst).long()
        positive_index = (correct_simrel_mask == 1).nonzero(as_tuple=True)
        negative_index = (correct_simrel_mask == 0).nonzero(as_tuple=True)

        n_pos_edges = correct_simrel_mask.sum()
        n_neg_edges = n_edges - n_pos_edges
        n_train_pos = int(train_ratio * n_pos_edges)
        n_train_neg = int(train_ratio * n_neg_edges)

        # positive edges
        train_src, train_dst = sim_src[positive_index][:n_train_pos], sim_dst[positive_index][:n_train_pos]
        test_src, test_dst = sim_src[positive_index][n_train_pos:], sim_dst[positive_index][n_train_pos:]

        # negative edges
        train_neg_src, train_neg_dst = sim_src[negative_index][:n_train_neg], sim_dst[negative_index][:n_train_neg]
        test_neg_src, test_neg_dst = sim_src[negative_index][n_train_neg:], sim_dst[negative_index][n_train_neg:]

        # create graphs
        train_pos_graph = dgl.graph((train_src, train_dst), num_nodes=n_nodes)
        test_pos_graph = dgl.graph((test_src, test_dst), num_nodes=n_nodes)
        train_neg_graph = dgl.graph((train_neg_src, train_neg_dst), num_nodes=n_nodes)
        test_neg_graph = dgl.graph((test_neg_src, test_neg_dst), num_nodes=n_nodes)

        return train_pos_graph, train_neg_graph, test_pos_graph, test_neg_graph

    def get_orcids_graph(self):
        return dgl.edge_type_subgraph(self.graph[0], ['equates'])

