from pyspark.sql import SparkSession
from dgl.data import DGLDataset
from src.utils.functions import *
import torch
from dgl.data.utils import save_graphs, load_graphs
import dgl
import os
import random
import numpy as np
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from tqdm import tqdm

random.seed(1234)
np.random.seed(1234)
os.environ.pop("SPARK_HOME", None) # to prevent conflicts


class PubmedSubgraph(DGLDataset):
    """
    Dataset for Similarity Relationships Evaluation.

    Parameters
    ----------
    dataset_name : name of the dataset
    url : URL of a ZIP or TAR archive containing the raw dataset
    raw_dir : directory that will store (or already stores) the downloaded data
    save_dir : directory to save preprocessed data
    force_reload : whether to reload dataset
    verbose : whether to print out progress information
    """
    RAW_DATA_DIRECTORIES = (
        "authors",
        "publications",
        "cites_rels",
        "coprojected_rels",
        "collaborates_rels",
        "writes_rels",
        "potentiallyequivalent_rels",
        "equivalent_rels",
        "simrels",
        "simrels_dedup",
    )

    def __init__(self,
                 dataset_name,
                 url,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):

        self.spark = SparkSession.builder \
            .appName("Dataset Processor") \
            .master("local[*]") \
            .config("spark.driver.memory", "15g") \
            .config("spark.driver.maxResultSize", "0") \
            .getOrCreate()

        self.sc = self.spark.sparkContext

        self.dataset_url = url
        self.graph = None

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
        Download and extract the raw dataset, unless it is already available.

        The archive may contain the expected directories at its root or inside
        one or more wrapper directories. Its contents are validated in a
        temporary directory before being copied to ``raw_dir``.
        """
        if self._raw_data_available(self.raw_dir):
            print(f"Raw data already available in {self.raw_dir}; skipping download")
            return

        if not self.dataset_url:
            raise ValueError("A dataset archive URL is required")

        raw_dir = os.path.abspath(self.raw_dir)
        os.makedirs(raw_dir, exist_ok=True)
        print(f"Downloading raw data from {self.dataset_url}")

        with tempfile.TemporaryDirectory(
                prefix="pubmed-subgraph-", dir=os.path.dirname(raw_dir)) as work_dir:
            archive_path = os.path.join(work_dir, "dataset.archive")
            extracted_dir = os.path.join(work_dir, "extracted")
            os.makedirs(extracted_dir)

            self._download_archive(self.dataset_url, archive_path)
            self._extract_archive(archive_path, extracted_dir)
            dataset_root = self._find_dataset_root(extracted_dir)
            self._copy_raw_data(dataset_root, raw_dir)

        if not self._raw_data_available(raw_dir):
            raise RuntimeError(
                f"The downloaded archive did not produce a complete dataset in {raw_dir}"
            )
        print(f"Raw data ready in {raw_dir}")

    @classmethod
    def _raw_data_available(cls, root):
        if not root or not os.path.isdir(root):
            return False
        return all(
            cls._directory_has_data(os.path.join(root, directory_name))
            for directory_name in cls.RAW_DATA_DIRECTORIES
        )

    @staticmethod
    def _directory_has_data(path):
        if not os.path.isdir(path):
            return False
        for _, _, filenames in os.walk(path):
            if any(not name.startswith((".", "_")) for name in filenames):
                return True
        return False

    @staticmethod
    def _download_archive(url, destination):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "gnn-simrels-evaluator/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response, \
                    open(destination, "wb") as archive_file:
                content_length = response.headers.get("Content-Length")
                total_size = int(content_length) if content_length else None
                with tqdm(
                        total=total_size,
                        desc="Downloading dataset",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                ) as progress:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        archive_file.write(chunk)
                        progress.update(len(chunk))
        except Exception as error:
            raise RuntimeError(f"Unable to download dataset from {url}") from error

    @classmethod
    def _extract_archive(cls, archive_path, destination):
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                cls._validate_zip_members(archive, destination)
                members = archive.infolist()
                total_size = sum(member.file_size for member in members)
                with tqdm(
                        total=total_size,
                        desc="Extracting dataset",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                ) as progress:
                    for member in members:
                        archive.extract(member, destination)
                        progress.update(member.file_size)
            return

        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                cls._validate_tar_members(archive, destination)
                members = archive.getmembers()
                total_size = sum(member.size for member in members if member.isfile())
                with tqdm(
                        total=total_size,
                        desc="Extracting dataset",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                ) as progress:
                    for member in members:
                        archive.extract(member, destination)
                        if member.isfile():
                            progress.update(member.size)
            return

        raise ValueError("The dataset URL must point to a ZIP or TAR archive")

    @classmethod
    def _copy_raw_data(cls, dataset_root, raw_dir):
        sources = [
            os.path.join(dataset_root, directory_name)
            for directory_name in cls.RAW_DATA_DIRECTORIES
        ]
        total_size = sum(
            os.path.getsize(os.path.join(root, filename))
            for source in sources
            for root, _, filenames in os.walk(source)
            for filename in filenames
        )

        with tqdm(
                total=total_size,
                desc="Copying raw data",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
        ) as progress:
            for source in sources:
                relative_source = os.path.relpath(source, dataset_root)
                destination_root = os.path.join(raw_dir, relative_source)

                for root, directories, filenames in os.walk(source):
                    relative_root = os.path.relpath(root, source)
                    destination = os.path.join(destination_root, relative_root)
                    os.makedirs(destination, exist_ok=True)
                    for directory in directories:
                        os.makedirs(os.path.join(destination, directory), exist_ok=True)

                    for filename in filenames:
                        source_file = os.path.join(root, filename)
                        destination_file = os.path.join(destination, filename)
                        with open(source_file, "rb") as input_file, \
                                open(destination_file, "wb") as output_file:
                            while True:
                                chunk = input_file.read(1024 * 1024)
                                if not chunk:
                                    break
                                output_file.write(chunk)
                                progress.update(len(chunk))
                        shutil.copystat(source_file, destination_file)

    @staticmethod
    def _safe_archive_path(destination, member_name):
        destination = os.path.realpath(destination)
        member_path = os.path.realpath(os.path.join(destination, member_name))
        try:
            return os.path.commonpath((destination, member_path)) == destination
        except ValueError:
            return False

    @classmethod
    def _validate_zip_members(cls, archive, destination):
        for member in archive.infolist():
            mode = member.external_attr >> 16
            if not cls._safe_archive_path(destination, member.filename):
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not allowed in ZIP archives: {member.filename}")

    @classmethod
    def _validate_tar_members(cls, archive, destination):
        for member in archive.getmembers():
            if not cls._safe_archive_path(destination, member.name):
                raise ValueError(f"Unsafe path in TAR archive: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Links and device files are not allowed in TAR archives: {member.name}")

    @classmethod
    def _find_dataset_root(cls, extracted_dir):
        candidates = []
        for root, directories, _ in os.walk(extracted_dir):
            if set(cls.RAW_DATA_DIRECTORIES).issubset(directories) \
                    and cls._raw_data_available(root):
                candidates.append(root)

        if not candidates:
            expected = ", ".join(cls.RAW_DATA_DIRECTORIES)
            raise ValueError(
                "The downloaded archive has an unexpected layout. "
                f"Expected directories: {expected}"
            )

        return min(candidates, key=lambda path: path.count(os.sep))

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
        print("Processing SIMILARITY Relations for training")
        similarity_rels = self.sc.textFile(self.raw_dir + "/simrels").map(eval)
        similarity_rels = similarity_rels.join(authors_for_join)
        similarity_rels = similarity_rels.map(lambda x: (x[1][0], x[1][1]))
        similarity_rels = similarity_rels.join(authors_for_join)
        similarity_rels = similarity_rels.map(lambda x: [x[1][0], x[1][1]])
        similarity_rels_tensor = torch.LongTensor(similarity_rels.collect())

        print("Processing SIMILARITY Relations from dedup")
        similarity_rels_dedup = self.sc.textFile(self.raw_dir + "/simrels_dedup").map(eval)
        similarity_rels_dedup = similarity_rels_dedup.join(authors_for_join)
        similarity_rels_dedup = similarity_rels_dedup.map(lambda x: (x[1][0], x[1][1]))
        similarity_rels_dedup = similarity_rels_dedup.join(authors_for_join)
        similarity_rels_dedup = similarity_rels_dedup.map(lambda x: [x[1][0], x[1][1]])
        similarity_rels_dedup_tensor = torch.LongTensor(similarity_rels_dedup.collect())

        self.graph = dgl.heterograph(
            data_dict={
                ("publication", "cites", "publication"): (cites_rels_tensor[:, 0], cites_rels_tensor[:, 1]),
                ("author", "collaborates", "author"): (collaborates_rels_tensor[:, 0], collaborates_rels_tensor[:, 1]),
                ("publication", "coprojected", "publication"): (coprojected_rels_tensor[:, 0], coprojected_rels_tensor[:, 1]),
                ("author", "writes", "publication"): (writes_rels_tensor[:, 0], writes_rels_tensor[:, 1]),
                ("publication", "is_written_by", "author"): (writes_rels_tensor[:, 1], writes_rels_tensor[:, 0]),
                ("author", "potentially_equates", "author"): (potentiallyequivalent_rels_tensor[:, 0], potentiallyequivalent_rels_tensor[:, 1]),
                ("author", "equates", "author"): (equivalent_rels_tensor[:, 0], equivalent_rels_tensor[:, 1]),
                ("author", "similar", "author"): (similarity_rels_tensor[:, 0], similarity_rels_tensor[:, 1]),
                ("author", "similar_for_dedup", "author"): (similarity_rels_dedup_tensor[:, 0], similarity_rels_dedup_tensor[:, 1])
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

    def get_graph(self):
        """
        Returns the Graph
        """
        return self.graph

    def save(self):
        """
        Save processed data to directory (self.save_path)
        """
        print("Saving graph on disk")
        save_graphs(self.save_dir + "/pubmed_graph.dgl", [self.graph])

    def load(self):
        """
        Load processed data from directory (self.save_path)
        """
        print("Load graph from disk")
        graphs, _ = load_graphs(self.save_dir + "/pubmed_graph.dgl")
        self.graph = graphs[0]

    def has_cache(self):
        """
        Check whether there are processed data
        """
        return os.path.exists(self.save_dir + "/pubmed_graph.dgl")

    def get_node_embeddings_graphs(self):
        """
        Return graphs to be used for the node embedding part of the architecture
        """
        potentially_equates_graph = dgl.edge_type_subgraph(self.graph, ['potentially_equates'])
        colleague_graph = dgl.metapath_reachable_graph(self.graph, ['writes', 'coprojected', 'is_written_by'])
        citation_graph = dgl.metapath_reachable_graph(self.graph, ['writes', 'cites', 'is_written_by'])
        collaboration_graph = dgl.edge_type_subgraph(self.graph, etypes=['collaborates'])

        return dgl.add_self_loop(potentially_equates_graph), dgl.add_self_loop(colleague_graph), dgl.add_self_loop(citation_graph), dgl.add_self_loop(collaboration_graph)

    def get_simrels_graph(self):
        return dgl.edge_type_subgraph(self.graph, ['similar'])

    def get_simrel_splittings(self, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
        """
        Returns splittings for training, validation and testing
        """
        n_edges = self.graph.number_of_edges(etype="similar")
        n_nodes = self.graph.edge_type_subgraph(etypes=["similar"]).num_nodes()

        orcids_graph = self.graph.edge_type_subgraph(etypes=["equates"])
        sim_src, sim_dst = self.graph.edges(etype="similar")
        correct_simrel_mask = orcids_graph.has_edges_between(sim_src, sim_dst).long()
        positive_index = (correct_simrel_mask == 1).nonzero(as_tuple=True)
        negative_index = (correct_simrel_mask == 0).nonzero(as_tuple=True)

        n_pos_edges = correct_simrel_mask.sum()
        n_neg_edges = n_edges - n_pos_edges
        min_edges = min(n_pos_edges, n_neg_edges)

        pos_src = sim_src[positive_index][:min_edges]
        pos_dst = sim_dst[positive_index][:min_edges]
        neg_src = sim_src[negative_index][:min_edges]
        neg_dst = sim_dst[negative_index][:min_edges]

        # POSITIVE EDGES
        train_pos_src, train_pos_dst = pos_src[:int(train_ratio*min_edges)], pos_dst[:int(train_ratio*min_edges)]
        valid_pos_src, valid_pos_dst = pos_src[int(train_ratio*min_edges):int((train_ratio+valid_ratio)*min_edges)], pos_dst[int(train_ratio*min_edges):int((train_ratio+valid_ratio)*min_edges)]
        test_pos_src, test_pos_dst = pos_src[int((train_ratio+valid_ratio)*min_edges):], pos_dst[int((train_ratio+valid_ratio)*min_edges):]

        # NEGATIVE EDGES
        train_neg_src, train_neg_dst = neg_src[:int(train_ratio*min_edges)], neg_dst[:int(train_ratio*min_edges)]
        valid_neg_src, valid_neg_dst = neg_src[int(train_ratio*min_edges):int((train_ratio+valid_ratio)*min_edges)], neg_dst[int(train_ratio*min_edges):int((train_ratio+valid_ratio)*min_edges)]
        test_neg_src, test_neg_dst = neg_src[int((train_ratio+valid_ratio)*min_edges):], neg_dst[int((train_ratio+valid_ratio)*min_edges):]

        # create graphs
        train_pos_graph = dgl.graph((train_pos_src, train_pos_dst), num_nodes=n_nodes)
        valid_pos_graph = dgl.graph((valid_pos_src, valid_pos_dst), num_nodes=n_nodes)
        test_pos_graph = dgl.graph((test_pos_src, test_pos_dst), num_nodes=n_nodes)

        train_neg_graph = dgl.graph((train_neg_src, train_neg_dst), num_nodes=n_nodes)
        valid_neg_graph = dgl.graph((valid_neg_src, valid_neg_dst), num_nodes=n_nodes)
        test_neg_graph = dgl.graph((test_neg_src, test_neg_dst), num_nodes=n_nodes)

        return train_pos_graph, train_neg_graph, valid_pos_graph, valid_neg_graph, test_pos_graph, test_neg_graph

    def get_orcids_graph(self):
        return dgl.edge_type_subgraph(self.graph, ['equates'])

    def get_dedup_graph(self):
        return dgl.edge_type_subgraph(self.graph, ['similar_for_dedup'])
