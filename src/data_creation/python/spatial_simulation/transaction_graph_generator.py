"""
Generate a base transaction graph used in the simulator
"""

import networkx as nx
import numpy as np
import itertools
import random
import csv
import yaml
import os
import sys
import logging
from scipy import stats

from collections import Counter, defaultdict

from .utils.nominator import Nominator
from .utils.normal_model import NormalModel
from .utils.random_amount import RandomAmount
from .utils.rounded_amount import RoundedAmount


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Attribute keys
MAIN_ACCT_KEY = "main_acct"  # Main account ID (SAR typology subgraph attribute)
IS_SAR_KEY = "is_sar"  # SAR flag (account vertex attribute)

DEFAULT_MARGIN_RATIO = 0.1  # Each member will keep this ratio of the received amount


# Utility functions parsing values
def parse_int(value):
    """ Convert string to int
    :param value: string value
    :return: int value if the parameter can be converted to str, otherwise None
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def parse_float(value):
    """ Convert string to amount (float)
    :param value: string value
    :return: float value if the parameter can be converted to float, otherwise None
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_flag(value):
    """ Convert string to boolean (True or false)
    :param value: string value
    :return: True if the value is equal to "true" (case-insensitive), otherwise False
    """
    return type(value) == str and value.lower() == "true"


def get_positive_or_none(value):
    """ Get positive value or None (used to parse simulation step parameters)
    :param value: Numerical value or None
    :return: If the value is positive, return this value. Otherwise, return None.
    """
    if value is None:
        return None
    else:
        return value if value > 0 else None


def directed_configuration_model(_in_deg, _out_deg, seed=0):
    """Generate a directed random graph with the given degree sequences without self loop.
    Based on nx.generators.degree_seq.directed_configuration_model
    :param _in_deg: Each list entry corresponds to the in-degree of a node.
    :param _out_deg: Each list entry corresponds to the out-degree of a node.
    :param seed: Seed for random number generator
    :return: MultiDiGraph without self loop
    """
    if not sum(_in_deg) == sum(_out_deg):
        raise nx.NetworkXError('Invalid degree sequences. Sequences must have equal sums.')

    random.seed(seed)
    n_in = len(_in_deg)
    n_out = len(_out_deg)
    if n_in < n_out:
        _in_deg.extend((n_out - n_in) * [0])
    else:
        _out_deg.extend((n_in - n_out) * [0])

    num_nodes = len(_in_deg)
    _g = nx.empty_graph(num_nodes, nx.MultiDiGraph())
    if num_nodes == 0 or max(_in_deg) == 0:
        return _g  # No edges exist

    in_tmp_list = list()
    out_tmp_list = list()
    for n in _g.nodes():
        in_tmp_list.extend(_in_deg[n] * [n])
        out_tmp_list.extend(_out_deg[n] * [n])
    random.shuffle(in_tmp_list)
    random.shuffle(out_tmp_list)

    added_edges = set()  # To prevent duplicate edges
    for i in range(len(in_tmp_list)):
        _src = out_tmp_list[i]
        _dst = in_tmp_list[i]

        # Ensure no self-loops or duplicates
        if _src != _dst and (_src, _dst) not in added_edges:
            _g.add_edge(_src, _dst)
            added_edges.add((_src, _dst))
        else:
            # Find a new pair to avoid self-loop or duplicate
            for j in range(i + 1, len(in_tmp_list)):
                potential_dst = in_tmp_list[j]
                if _src != potential_dst and (_src, potential_dst) not in added_edges:
                    # Swap to avoid the self-loop or duplicate and break the inner loop
                    in_tmp_list[i], in_tmp_list[j] = in_tmp_list[j], in_tmp_list[i]
                    _g.add_edge(_src, in_tmp_list[i])
                    added_edges.add((_src, in_tmp_list[i]))
                    break
    return _g


def get_degrees(deg_csv, num_v):
    """
    :param deg_csv: Degree distribution parameter CSV file
    :param num_v: Number of total account vertices
    :return: In-degree and out-degree sequence list
    """
    with open(deg_csv, "r") as rf:  # Load in/out-degree sequences from parameter CSV file for each account
        reader = csv.reader(rf)
        next(reader)
        return get_in_and_out_degrees(reader, num_v)


def get_in_and_out_degrees(iterable, num_v):
    _in_deg = list()  # In-degree sequence
    _out_deg = list()  # Out-degree sequence
    
    for row in iterable:
        if row[0].startswith("#"):
            continue
        count = int(row[0])
        _in_deg.extend([int(row[1])] * count)
        _out_deg.extend([int(row[2])] * count)

    in_len, out_len = len(_in_deg), len(_out_deg)
    if in_len != out_len:
        raise ValueError("The length of in-degree (%d) and out-degree (%d) sequences must be same."
                         % (in_len, out_len))

    total_in_deg, total_out_deg = sum(_in_deg), sum(_out_deg)
    if total_in_deg != total_out_deg:
        raise ValueError("The sum of in-degree (%d) and out-degree (%d) must be same."
                         % (total_in_deg, total_out_deg))

    if num_v % in_len != 0:
        raise ValueError("The number of total accounts (%d) "
                         "must be a multiple of the degree sequence length (%d)."
                         % (num_v, in_len))

    repeats = num_v // in_len
    _in_deg = _in_deg * repeats
    _out_deg = _out_deg * repeats

    # shuffle in and out degrees
    _in_out_deg = list(zip(_in_deg, _out_deg))
    random.shuffle(_in_out_deg)
    _in_deg, _out_deg = zip(*_in_out_deg)
    _in_deg, _out_deg = list(_in_deg), list(_out_deg)
    
    return _in_deg, _out_deg


class TransactionGenerator:

    def __init__(self, conf, sim_name=None):
        """Initialize transaction network from parameter files.
        :param conf_file: JSON file as configurations
        :param sim_name: Simulation name (overrides the content in the `conf_json`)
        """
        self.g = nx.DiGraph()  # Transaction graph object
        self.num_accounts = 0  # Number of total accounts
        self.hubs = set()  # Hub account vertices (main account candidates of AML typology subgraphs)
        self.attr_names = list()  # Additional account attribute names
        self.bank_to_accts = defaultdict(set)  # Bank ID -> account set
        self.acct_to_bank = dict()  # Account ID -> bank ID
        self.normal_model_counts = dict()
        self.normal_models = list()
        self.normal_model_id = 1

        self.conf = conf

        general_conf = self.conf["general"]

        # Set random seed
        seed = general_conf.get("random_seed")
        env_seed = os.getenv("RANDOM_SEED")
        if env_seed is not None:
            seed = env_seed  # Overwrite random seed if specified as an environment variable
        self.seed = seed if seed is None else int(seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        logger.info("Random seed: " + str(self.seed))

        # Get simulation name
        if sim_name is None:
            sim_name = general_conf["simulation_name"]
        logger.info("Simulation name: " + sim_name)

        self.total_steps = parse_int(general_conf["total_steps"])

        # Set default amounts, steps and model ID
        default_conf = self.conf["default"]
        self.default_min_amount = parse_float(default_conf.get("min_amount"))
        self.default_max_amount = parse_float(default_conf.get("max_amount"))
        self.default_min_balance = parse_float(default_conf.get("min_balance"))
        self.default_max_balance = parse_float(default_conf.get("max_balance"))
        self.default_start_step = parse_int(default_conf.get("start_step"))
        self.default_end_step = parse_int(default_conf.get("end_step"))
        self.default_start_range = parse_int(default_conf.get("start_range"))
        self.default_end_range = parse_int(default_conf.get("end_range"))
        self.default_model = parse_int(default_conf.get("transaction_model"))
        self.default_prob_sar_participate = parse_float(default_conf.get("prob_participate_in_multiple_sars"))
        self.sar_participation = {0: []}
        
        # The ratio of amount intermediate accounts receive
        self.margin_ratio = parse_float(default_conf.get("margin_ratio", DEFAULT_MARGIN_RATIO))
        if not 0.0 <= self.margin_ratio <= 1.0:
            raise ValueError("Margin ratio in AML typologies (%f) must be within [0.0, 1.0]" % self.margin_ratio)

        # Probability of adding random edges in bipartite pattern
        self.bipartite_edge_prob = parse_float(default_conf.get("bipartite_edge_prob", 0.8))
        if not 0.0 <= self.bipartite_edge_prob <= 1.0:
            raise ValueError("Bipartite edge probability (%f) must be within [0.0, 1.0]" % self.bipartite_edge_prob)

        self.default_bank_id = default_conf.get("bank_id")  # Default bank ID if not specified at parameter files

        # Get input file names and properties
        input_conf = self.conf["input"]
        self.input_dir = input_conf["directory"]  # The directory name of input files
        self.account_file = input_conf["accounts"]  # Account list file
        self.alert_file = input_conf["alert_patterns"]  # AML typology definition file
        self.normal_models_file = input_conf["normal_models"] # Normal models definition file
        self.degree_file = input_conf["degree"]  # Degree distribution file
        self.type_file = input_conf["transaction_type"]  # Transaction type
        self.is_aggregated = input_conf["is_aggregated_accounts"]  # Flag whether the account list is aggregated

        # Get output file names
        output_conf = self.conf["temporal"]  # The output directory of the graph generator is temporal one
        self.output_dir = os.path.join(output_conf["directory"])  # The directory name of temporal files
        self.out_tx_file = output_conf["transactions"]  # All transaction list CSV file
        self.out_account_file = output_conf["accounts"]  # All account list CSV file
        self.out_alert_member_file = output_conf["alert_members"]  # Account list of AML typology members CSV file
        self.out_normal_models_file = output_conf["normal_models"] # List of normal models CSV file
 
        # High risk settings (optional)
        other_conf = self.conf.get("graph_generator", {})
        high_risk_countries_str = other_conf.get("high_risk_countries", "")
        high_risk_business_str = other_conf.get("high_risk_business", "")
        self.high_risk_countries = set(high_risk_countries_str.split(",")) if high_risk_countries_str else set()
        self.high_risk_business = set(high_risk_business_str.split(",")) if high_risk_business_str else set()

        self.edge_id = 0  # Edge ID. Formerly Transaction ID
        self.alert_id = 0  # Alert ID from the alert parameter file
        self.alert_groups = dict()  # Alert ID and alert transaction subgraph
        # TODO: Move the mapping of AML pattern to configuration JSON file
        self.alert_types = {"fan_out": 1, "fan_in": 2, "cycle": 3, "bipartite": 4, "stack": 5,
                            "random": 6, "scatter_gather": 7, "gather_scatter": 8}  # Pattern name and model ID

        self.acct_file = os.path.join(self.input_dir, self.account_file)

        def get_types(type_csv):
            """Read out the transaction types

            Args:
                type_csv (string): Transaction types csv file

            Returns:
                list: transaction types
            """            
            tx_types = list()
            with open(type_csv, "r") as _rf:
                reader = csv.reader(_rf)
                next(reader)
                for row in reader:
                    if row[0].startswith("#"):
                        continue
                    ttype = row[0]
                    tx_types.extend([ttype] * int(row[1]))
            return tx_types

        self.tx_types = get_types(os.path.join(self.input_dir, self.type_file))


    def check_hub_exists(self):
        """Validate whether one or more hub accounts exist as main accounts of AML typologies
        """
        if not self.hubs:
            raise ValueError("No main account candidates found in the transaction graph.")


    def set_main_acct_candidates(self):
        """ Set self.hubs to be a set of all nodes as main account candidates
            Throw an error if not done successfully.
        """
        self.hubs = set(self.g.nodes())
        self.check_hub_exists()


    def check_account_exist(self, aid):
        """Validate an existence of a specified account. If absent, it raises KeyError.
        :param aid: Account ID
        """
        if not self.g.has_node(aid):
            raise KeyError("Account %s does not exist" % str(aid))


    def check_account_absent(self, aid):
        """Validate an absence of a specified account
        :param aid: Account ID
        :return: True if an account of the specified ID is not yet added
        """
        if self.g.has_node(aid):
            logger.warning("Account %s already exists" % str(aid))
            return False
        else:
            return True


    def get_all_bank_ids(self):
        """Get a list of all bank IDs
        :return: Bank ID list
        """
        return list(self.bank_to_accts.keys())


    def get_typology_members(self, num, bank_id=""):
        """Choose accounts randomly as members of AML typologies from one or multiple banks.
        :param num: Number of total account vertices (including the main account)
        :param bank_id: If specified, it chooses members from a single bank with the ID.
        If empty (default), it chooses members from all banks randomly.
        :return: Main account and list of member account IDs
        """
        if num <= 1:
            raise ValueError("The number of members must be more than 1")

        if bank_id in self.bank_to_accts:  # Choose members from the same bank as the main account
            bank_accts = self.bank_to_accts[bank_id] # Get account set of the specified bank
            members = []
            for m in range(num):
                bin = 0
                while random.random() > stats.logser.cdf(bin+1, self.default_prob_sar_participate):
                    if bin+1 not in self.sar_participation: # if there are no accounts in the next bin, remain in the current
                        break
                    elif all([candidate not in bank_accts for candidate in self.sar_participation[bin+1]]): # if all accounts in the next bin are not in the bank, remain in the current
                        break
                    elif all([candidate in members for candidate in self.sar_participation[bin+1]]): # if all accounts in the next bin are already participating, remain in the current
                        break
                    else:
                        bin += 1
                
                candidate = None 
                while candidate is None:
                    if bin == 0:
                        available_set = set(bank_accts) - set(members) # get the set of accounts that are in the bank and not already participating
                    else:
                        available_set = set(self.sar_participation[bin]) - set(members) # get the set of accounts that are in the current bin and not already participating
                    
                    if available_set == set() and (bin+1) in self.sar_participation: # if there are no accounts in the current bin and there are accounts in the next bin
                        bin += 1
                    elif available_set == set() and (bin+1) not in self.sar_participation:
                        bin = max(bin-1, 0) # if there are no accounts in the current bin and no accounts in next bin, move to the previous bin
                    else:
                        candidate = random.choice(list(available_set)) # choose a random account from the available set
                members.append(candidate)
                
            for member in members:
                bins = list(self.sar_participation.keys())
                for bin in bins:
                    if member in self.sar_participation[bin]:
                        if bin+1 not in self.sar_participation:
                            self.sar_participation[bin+1] = [member]
                        else:
                            self.sar_participation[bin+1].append(member)
                        self.sar_participation[bin].remove(member)
                        break
            main_acct = random.choice(members)
            return main_acct, members

        elif bank_id == "":  # Choose members from all accounts
            members = [] # initiate list of participating accounts in alert
            for m in range(num): # for each account in the alert
                bin = 0
                while random.random() > stats.logser.cdf(bin+1, self.default_prob_sar_participate):
                    if bin+1 not in self.sar_participation: # if there are no accounts in the next bin, remain in the current
                        break
                    elif all([candidate in members for candidate in self.sar_participation[bin+1]]): # if all accounts in the next bin are already participating, remain in the current
                        break
                    else:
                        bin += 1 # move to the next bin
                
                candidate = None 
                while candidate is None:
                    available_set = set(self.sar_participation[bin]) - set(members) # get the set of accounts that are in the current bin and not already participating
                    if available_set == set() and (bin+1) in self.sar_participation: # if there are no accounts in the current bin and there are accounts in the next bin
                        bin += 1
                    elif available_set == set() and (bin+1) not in self.sar_participation:
                        bin = max(bin-1, 0) # if there are no accounts in the current bin and no accounts in next bin, move to the previous bin
                    else:
                        candidate = random.choice(list(available_set)) # choose a random account from the available set
                members.append(candidate)
                
            for member in members:
                bins = list(self.sar_participation.keys())
                for bin in bins:
                    if member in self.sar_participation[bin]:
                        if bin+1 not in self.sar_participation:
                            self.sar_participation[bin+1] = [member]
                        else:
                            self.sar_participation[bin+1].append(member)
                        self.sar_participation[bin].remove(member)
                        break
            main_acct = random.choice(members)
            
            return main_acct, members

        else:
            raise KeyError("No such bank ID: %s" % bank_id)

        
    def load_account_list(self):
        """Load and add account vertices from a CSV file
        """
        if self.is_aggregated:
            self.load_account_list_param()
        else:
            self.load_account_list_raw()


    def load_account_list_raw(self):
        """Load and add account vertices from a CSV file with raw account info
        header: uuid,seq,first_name,last_name,street_addr,city,state,zip,gender,phone_number,birth_date,ssn
        :param acct_file: Raw account list file path
        """
        if self.default_min_balance is None:
            raise KeyError("Option 'default_min_balance' is required to load raw account list")
        min_balance = self.default_min_balance

        if self.default_max_balance is None:
            raise KeyError("Option 'default_max_balance' is required to load raw account list")
        max_balance = self.default_max_balance

        start_day = get_positive_or_none(self.default_start_step)
        end_day = get_positive_or_none(self.default_end_step)
        start_range = get_positive_or_none(self.default_start_range)
        end_range = get_positive_or_none(self.default_end_range)
        default_model = self.default_model if self.default_model is not None else 1

        self.attr_names.extend(["first_name", "last_name", "street_addr", "city", "state", "zip",
                                "gender", "phone_number", "birth_date", "ssn", "lon", "lat"])

        with open(self.acct_file, "r") as rf:
            reader = csv.reader(rf)
            header = next(reader)
            name2idx = {n: i for i, n in enumerate(header)}
            idx_aid = name2idx["uuid"]
            idx_first_name = name2idx["first_name"]
            idx_last_name = name2idx["last_name"]
            idx_street_addr = name2idx["street_addr"]
            idx_city = name2idx["city"]
            idx_state = name2idx["state"]
            idx_zip = name2idx["zip"]
            idx_gender = name2idx["gender"]
            idx_phone_number = name2idx["phone_number"]
            idx_birth_date = name2idx["birth_date"]
            idx_ssn = name2idx["ssn"]
            idx_lon = name2idx["lon"]
            idx_lat = name2idx["lat"]

            count = 0
            for row in reader:
                if row[0].startswith("#"):  # Comment line
                    continue
                aid = row[idx_aid]
                first_name = row[idx_first_name]
                last_name = row[idx_last_name]
                street_addr = row[idx_street_addr]
                city = row[idx_city]
                state = row[idx_state]
                zip_code = row[idx_zip]
                gender = row[idx_gender]
                phone_number = row[idx_phone_number]
                birth_date = row[idx_birth_date]
                ssn = row[idx_ssn]
                lon = row[idx_lon]
                lat = row[idx_lat]
                model = default_model

                if start_day is not None and start_range is not None:
                    start = start_day + random.randrange(start_range)
                else:
                    start = -1

                if end_day is not None and end_range is not None:
                    end = end_day - random.randrange(end_range)
                else:
                    end = -1

                attr = {"first_name": first_name, "last_name": last_name, "street_addr": street_addr,
                        "city": city, "state": state, "zip": zip_code, "gender": gender,
                        "phone_number": phone_number, "birth_date": birth_date, "ssn": ssn, "lon": lon, "lat": lat}

                init_balance = random.uniform(min_balance, max_balance)  # Generate the initial balance
                self.add_account(aid, init_balance=init_balance, is_sar=False, **attr)
                count += 1


    def set_num_accounts(self):
        """Read the number of accounts from the account list file
        """        
        with open(self.acct_file, "r") as rf:
            reader = csv.reader(rf)
            # Parse header
            header = next(reader)

            count = 0
            for row in reader:
                if row[0].startswith("#"):
                    continue
                num = int(row[header.index('count')])
                count += num

        self.num_accounts = count


    def load_account_list_param(self):

        """Load and add account vertices from a CSV file with aggregated parameters
        Each row may represent two or more accounts
        :param acct_file: Account parameter file path
        """

        with open(self.acct_file, "r") as rf:
            reader = csv.reader(rf)
            # Parse header
            header = next(reader)

            acct_id = 0
            for row in reader:
                if row[0].startswith("#"):
                    continue
                num = int(row[header.index('count')])
                min_balance = parse_float(row[header.index('min_balance')])
                max_balance = parse_float(row[header.index('max_balance')])
                bank_id = row[header.index('bank_id')]
                if bank_id is None:
                    bank_id = self.default_bank_id

                # Generate accounts with random initial balance
                for i in range(num):
                    init_balance = random.uniform(min_balance, max_balance)  # Generate amount TODO: use distribution instead
                    self.add_account(acct_id, init_balance=init_balance, bank_id=bank_id, is_sar=False, normal_models=list())
                    acct_id += 1

        logger.info("Generated %d accounts." % self.num_accounts)


    def generate_normal_transactions(self):
        """Generate a base directed graph from degree sequences
        TODO: Add options to call scale-free generator functions directly instead of loading degree CSV files
        :return: Directed graph as the base transaction graph (not complete transaction graph)
        """
        deg_file = os.path.join(self.input_dir, self.degree_file) # read in degree.csv
        in_deg, out_deg = get_degrees(deg_file, self.num_accounts) # read out the in and out degree distributions
        G = directed_configuration_model(in_deg, out_deg, self.seed)
        G = nx.DiGraph(G)
        self.g = G

        logger.info("Add %d base transactions" % self.g.number_of_edges())
        nodes = list(self.g.nodes())
        for src_i, dst_i in self.g.edges():
            src = nodes[src_i]
            dst = nodes[dst_i]
            self.add_edge_info(src, dst)  # Add edge info.


    def add_account(self, acct_id, **attr):
        """Add an account vertex
        :param acct_id: Account ID
        :param init_balance: Initial amount
        :param start: The day when the account opened
        :param end: The day when the account closed
        :param bank_id: Bank ID
        :param attr: Optional attributes
        :return:
        """
        
        if attr['bank_id'] is None:
            attr['bank_id'] = self.default_bank_id

        self.g.nodes[acct_id].update(attr)

        self.bank_to_accts[attr['bank_id']].add(acct_id)
        self.acct_to_bank[acct_id] = attr['bank_id']
        self.sar_participation[0].append(acct_id) 


    def remove_typology_candidate(self, acct):
        """Remove an account vertex from AML typology member candidates
        :param acct: Account ID
        """
        self.hubs.discard(acct) # remove from hubs
        bank_id = self.acct_to_bank[acct] 
        del self.acct_to_bank[acct] # remove from bank mapping
        self.bank_to_accts[bank_id].discard(acct)


    def add_edge_info(self, orig, bene):
        """Adds info to edge. Based on add_transaction.
        Add transaction will go away eventually.
        :param orig: Originator account ID
        :param bene: Beneficiary account ID
        :return:
        """
        self.check_account_exist(orig)  # Ensure the originator and beneficiary accounts exist
        self.check_account_exist(bene)
        if orig == bene:
            raise ValueError("Self loop from/to %s is not allowed for transaction networks" % str(orig))
        self.g.edges[orig, bene]['edge_id'] = self.edge_id
        self.edge_id += 1


    # Load Custom Topology Files
    def add_subgraph(self, members, topology):
        """Add subgraph from existing account vertices and given graph topology
        :param members: Account vertex list
        :param topology: Topology graph
        :return:
        """
        if len(members) != topology.number_of_nodes():
            raise nx.NetworkXError("The number of account vertices does not match")

        node_map = dict(zip(members, topology.nodes()))
        for e in topology.edges():
            src = node_map[e[0]]
            dst = node_map[e[1]]
            self.g.add_edge(src, dst)
            self.add_edge_info(src, dst)


    def load_edgelist(self, members, csv_name):
        """Load edgelist and add edges with existing account vertices
        :param members: Account vertex list
        :param csv_name: Edgelist file name
        :return:
        """
        topology = nx.DiGraph()
        topology = nx.read_edgelist(csv_name, delimiter=",", create_using=topology)
        self.add_subgraph(members, topology)


    def mark_active_edges(self):
        nx.set_edge_attributes(self.g, False, 'active')
        for normal_model in self.normal_models:
            subgraph = self.g.subgraph(normal_model.node_ids)
            nx.set_edge_attributes(subgraph, True, 'active')


    def load_normal_models(self):
        """Load a Normal Model parameter file
        """
        normal_models_file = os.path.join(self.input_dir, self.normal_models_file)
        with open(normal_models_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            self.read_normal_models(reader)


    def read_normal_models(self, reader):
        """Parse the Normal Model parameter file
        """
        header = next(reader)

        self.nominator = Nominator(self.g)

        for row in reader:
            count = int(row[header.index('count')])
            type = row[header.index('type')]
            # schedule_id, min_period, max_period removed - timing now generated in temporal simulation
            schedule_id = 2  # Default: unordered (not used in temporal sim anymore)
            min_accounts = int(row[header.index('min_accounts')])
            max_accounts = int(row[header.index('max_accounts')])
            min_period = 1  # Default: not used in temporal sim anymore
            max_period = 100  # Default: not used in temporal sim anymore
            bank_id = row[header.index('bank_id')]
            if bank_id == "":
                bank_id = None

            self.nominator.initialize_count(type, count, schedule_id, min_accounts, max_accounts, min_period, max_period, bank_id)
        self.nominator.initialize_candidates() # create candidate lists for the types considered


    def build_normal_models(self):
        """ Go through the accounts and attach normal models to them
        """        
        while(self.nominator.has_more()):
            for type in self.nominator.types():
                count = self.nominator.count(type)
                if count > 0:
                    success = self.choose_normal_model(type)
                    self.normal_model_id += success
        logger.info(f"Generated {len(self.normal_models)} normal models.")
        logger.info("Normal model counts %s", self.nominator.used_count_dict)
        return self.normal_models # just to get access in unit test, probably not a good solution
        

    def choose_normal_model(self, type):
        """Choose a normal model based on the type

        Args:
            type (string): Type of normal model
        """        
        if type == 'fan_in':
            success = self.fan_in_model(type)
        elif type == 'fan_out':
            success = self.fan_out_model(type)
        elif type == 'forward':
            success = self.forward_model(type)
        elif type == 'single':
            success = self.single_model(type)
        elif type == 'mutual':
            success = self.mutual_model(type)
        elif type == 'periodical':
            success = self.periodical_model(type)

        return success
        
    def fan_in_model(self, type):     
        node_id = self.nominator.next(type) # get the next node_id for this type

        if node_id is None:
            return False

        candidates = self.nominator.find_available_candidate_neighbors(type, node_id)

        if not candidates:
            raise ValueError('should always be candidates')

        # Create the normal pattern
        schedule_id, min_accounts, max_accounts, start_step, end_step, _ = self.nominator.model_params_dict[type][self.nominator.current_candidate_index[type]]
        result_ids = candidates | { node_id }
        normal_model = NormalModel(self.normal_model_id, type, result_ids, node_id)
        normal_model.set_params(schedule_id, start_step, end_step)

        for result_id in result_ids:
            self.g.nodes[result_id]['normal_models'].append(normal_model)

        self.normal_models.append(normal_model)
        
        self.nominator.post_update(node_id, type)
        
        return True


    def fan_out_model(self, type):
        node_id = self.nominator.next(type) # get the next node_id for this type

        if node_id is None:
            return False

        candidates = self.nominator.find_available_candidate_neighbors(type, node_id) # get the neighboring nodes that are in a potential fan-out relationship

        if not candidates:
            raise ValueError('should always be candidates')

        schedule_id, _, _, start_step, end_step, _ = self.nominator.model_params_dict[type][self.nominator.current_candidate_index[type]]
        result_ids = candidates | { node_id }
        normal_model = NormalModel(self.normal_model_id, type, result_ids, node_id)
        normal_model.set_params(schedule_id, start_step, end_step)
        for id in result_ids:
            self.g.nodes[id]['normal_models'].append(normal_model)

        self.normal_models.append(normal_model)
        
        self.nominator.post_update(node_id, type)

        return True

    def forward_model(self, type):
        node_id = self.nominator.next(type) # get the next node_id for this type
        # min and max accounts are not used in forward
        schedule_id, _, _, start_step, end_step, bank_id = self.nominator.model_params_dict[type][self.nominator.current_candidate_index[type]]
        
        if node_id is None:
            return False

        succ_ids = list(self.g.successors(node_id))
        pred_ids = list(self.g.predecessors(node_id))
        # if bank_id is not None:
        #     succ_ids = [succ_id for succ_id in succ_ids if self.acct_to_bank[succ_id] == bank_id]
        #     pred_ids = [pred_id for pred_id in pred_ids if self.acct_to_bank[pred_id] == bank_id]

        # find all input-node_id-output sets avialable where input and output are different
        sets = [[node_id, pred_id, succ_id] for pred_id in pred_ids for succ_id in succ_ids if pred_id != succ_id]
        random.shuffle(sets)
        # find the first set of nodes that is not in a forward relationship with this node_id
        chosen_nodes = next((nodes for nodes in sets if not self.nominator.is_in_type_relationship_ordered(type, nodes[1], nodes)), None)
        if chosen_nodes is None:
            raise ValueError('should always be candidates')
        main_node =  chosen_nodes[1]
        
        # Create normal models
        normal_model = NormalModel(self.normal_model_id, type, chosen_nodes, main_node)
        normal_model.set_params(schedule_id, start_step, end_step)
        
        for id in chosen_nodes:
            self.g.nodes[id]['normal_models'].append(normal_model)

        self.normal_models.append(normal_model)
        self.nominator.post_update(node_id, type)
        return True

    def single_model(self, type):
        node_id = self.nominator.next(type) # get the next node_id for this type

        if node_id is None:
            return False
        
        succ_ids = list(self.g.successors(node_id)) # find the accounts connected to this node_id
        # find the first account that is not in a single relationship with this node_id
        succ_id = next(succ_id for succ_id in succ_ids if not self.nominator.is_in_type_relationship(type, node_id, {node_id, succ_id})) # TODO: this takes a lot of time... 
        
        result_ids = { node_id, succ_id }
        normal_model = NormalModel(self.normal_model_id, type, result_ids, node_id) # create a normal model with the node_id and the connected account
        schedule_id, min_accounts, max_accounts, start_step, end_step, bank_id = self.nominator.model_params_dict[type][self.nominator.current_candidate_index[type]]
        normal_model.set_params(schedule_id, start_step, end_step)
        for id in result_ids:
            self.g.nodes[id]['normal_models'].append(normal_model) # add the normal model to the nodes

        self.normal_models.append(normal_model)

        self.nominator.post_update(node_id, type)
        return True
    
    def periodical_model(self, type):
        node_id = self.nominator.next(type)

        if node_id is None:
            return False
        
        succ_ids = list(self.g.successors(node_id))
        succ_id = next(succ_id for succ_id in succ_ids if not self.nominator.is_in_type_relationship(type, node_id, {node_id, succ_id}))

        result_ids = { node_id, succ_id }
        normal_model = NormalModel(self.normal_model_id, type, result_ids, node_id)
        schedule_id, min_accounts, max_accounts, start_step, end_step, _ = self.nominator.model_params_dict[type][self.nominator.current_candidate_index[type]]
        normal_model.set_params(schedule_id, start_step, end_step)
        for id in result_ids:
            self.g.nodes[id]['normal_models'].append(normal_model)

        self.normal_models.append(normal_model)

        self.nominator.post_update(node_id, type)
        return True
    
    def mutual_model(self, type):
        node_id = self.nominator.next(type)

        if node_id is None:
            return False
        
        succ_ids = list(self.g.successors(node_id))
        succ_id = next(succ_id for succ_id in succ_ids if not self.nominator.is_in_type_relationship(type, node_id, {node_id, succ_id}))

        result_ids = { node_id, succ_id }
        normal_model = NormalModel(self.normal_model_id, type, result_ids, node_id)
        schedule_id, min_accounts, max_accounts, start_step, end_step, _ = self.nominator.model_params_dict[type][self.nominator.current_candidate_index[type]]
        normal_model.set_params(schedule_id, start_step, end_step)
        for id in result_ids:
            self.g.nodes[id]['normal_models'].append(normal_model)

        self.normal_models.append(normal_model)

        self.nominator.post_update(node_id, type)
        return True

    def load_alert_patterns(self):
        """Load an AML typology parameter file
        :return:
        """
        alert_file = os.path.join(self.input_dir, self.alert_file)

        idx_num = None
        idx_type = None
        idx_schedule = None
        idx_min_accts = None
        idx_max_accts = None
        idx_min_period = None
        idx_max_period = None
        idx_bank = None
        idx_sar = None

        with open(alert_file, "r") as rf:
            reader = csv.reader(rf)
            # Parse header
            header = next(reader)
            for i, k in enumerate(header):
                if k == "count":  # Number of pattern subgraphs
                    idx_num = i
                elif k == "type":  # AML typology type (e.g. fan-out and cycle)
                    idx_type = i
                elif k == "schedule_id":  # Transaction scheduling type
                    idx_schedule = i
                elif k == "min_accounts":  # Minimum number of involved accounts
                    idx_min_accts = i
                elif k == "max_accounts":  # Maximum number of involved accounts
                    idx_max_accts = i
                elif k == "min_period":  # Minimum overall transaction period (number of simulation steps)
                    idx_min_period = i
                elif k == "max_period":  # Maximum overall transaction period (number of simulation steps)
                    idx_max_period = i
                elif k == "bank_id":  # Bank ID for internal-bank transactions
                    idx_bank = i
                elif k == "is_sar":  # SAR flag
                    idx_sar = i
                elif k == "source_type": # Source type 
                    idx_source_type = i
                else:
                    logger.warning("Unknown column name in %s: %s" % (alert_file, k))

            # Generate transaction set
            count = 0
            for row in reader:
                if len(row) == 0 or row[0].startswith("#"):
                    continue
                num_patterns = int(row[idx_num])  # Number of alert patterns
                typology_name = row[idx_type]
                # schedule_id, min_period, max_period now optional (timing generated in temporal simulation)
                schedule = int(row[idx_schedule]) if idx_schedule is not None else 2  # Default: unordered
                min_accts = int(row[idx_min_accts])
                max_accts = int(row[idx_max_accts])
                min_period = parse_int(row[idx_min_period]) if idx_min_period is not None else 0
                max_period = parse_int(row[idx_max_period]) if idx_max_period is not None else 100
                bank_id = row[idx_bank] if idx_bank is not None else ""  # If empty, it has inter-bank transactions
                is_sar = parse_flag(row[idx_sar])
                source_type = row[idx_source_type]

                if typology_name not in self.alert_types:
                    logger.warning("Pattern type name (%s) must be one of %s"
                                   % (typology_name, str(self.alert_types.keys())))
                    continue
                # Generate alert patterns
                for i in range(num_patterns):
                    num_accts = random.randrange(min_accts, max_accts + 1) # Number of accounts
                    self.add_aml_typology(is_sar, typology_name, num_accts, min_period, max_period, bank_id, schedule, source_type)
                    count += 1
                    if count % 1000 == 0:
                        logger.info("Created %d alerts" % count)

    def add_aml_typology(self, is_sar, typology_name, num_accounts, min_period, max_period, bank_id="", schedule=1, source_type='TRANSFER'):
        """Add an AML typology transaction set (graph structure only, amounts sampled in temporal simulation)
        :param is_sar: Whether the alerted transaction set is SAR (True) or false-alert (False)
        :param typology_name: Name of pattern type
            ("fan_in", "fan_out", "cycle", "random", "stack", "scatter_gather" or "gather_scatter")
        :param num_accounts: Number of transaction members (accounts)
        :param min_period: earliest date for all transactions
        :param max_period: latest date for all transactions
        :param bank_id: Bank ID which it chooses members from. If empty, it chooses members from all banks.
        :param schedule: AML pattern transaction schedule model ID
        """

        def add_node(_acct, _bank_id, phase=0):
            """Set an attribute of bank ID to a member account
            :param _acct: Account ID
            :param _bank_id: Bank ID
            :param phase: Phase number for multi-phase patterns (scatter_gather, gather_scatter, stack)
            """
            attr_dict = self.g.nodes[_acct].copy() # Get attributes of the account from main transaction graph
            attr_dict[IS_SAR_KEY] = True # Set SAR flag
            attr_dict["phase"] = phase # Set phase for multi-phase patterns

            sub_g.add_node(_acct, **attr_dict) # Add the account to the AML typology subgraph

        def add_main_acct():
            """Create a main account ID and a bank ID from hub accounts
            :return: main account ID and bank ID
            """
            self.check_hub_exists() # Check if there is a hub account (an account with large number of transactions)
            _main_acct = random.sample(self.hubs, 1)[0] # Choose a hub account randomly
            _main_bank_id = self.acct_to_bank[_main_acct] # Get bank ID of the hub account
            self.remove_typology_candidate(_main_acct)
            add_node(_main_acct, _main_bank_id)
            return _main_acct, _main_bank_id

        def add_edge(_orig, _bene, _amount, _date):
            """Add transaction edge to the AML typology subgraph as well as the whole transaction graph
            :param _orig: Originator account ID
            :param _bene: Beneficiary account ID
            :param _amount: Transaction amount
            :param _date: Transaction timestamp
            """
            sub_g.add_edge(_orig, _bene, amount=_amount, date=_date)
            self.g.add_edge(_orig, _bene) # add edge in original graph
            self.add_edge_info(_orig, _bene)

        # Decide if it is an inter-bank transaction TODO: make random assignment
        if bank_id == "" and len(self.bank_to_accts) >= 2:
            is_external = True
        elif bank_id != "" and bank_id not in self.bank_to_accts:  # Invalid bank ID
            raise KeyError("No such bank ID: %s" % bank_id)
        else:
            is_external = False

        # Decide transaction start and end dates
        start_date = min_period
        end_date = max_period # end_date is inclusive

        # Validate period is valid for generating random dates
        if start_date >= end_date:
            logger.warning(f"Invalid period for {typology_name} pattern: start_date ({start_date}) >= end_date ({end_date}). Skipping pattern.")
            return

        # Create subgraph structure with transaction attributes
        model_id = self.alert_types[typology_name]  # alert model ID
        sub_g = nx.DiGraph(model_id=model_id, reason=typology_name, scheduleID=schedule,
                           start=start_date, end=end_date, source_type=source_type)  # Create a subgraph for the AML typology with given attributes

        if typology_name == "fan_in":  # fan_in pattern (multiple accounts --> single (main) account)            
            amount = None  # Amount will be sampled in temporal simulation
            if is_external:
                main_acct, members = self.get_typology_members(num_accounts)
            else:
                main_acct, members = self.get_typology_members(num_accounts, bank_id)
            add_node(main_acct, self.acct_to_bank[main_acct])
            #self.remove_typology_candidate(main_acct)
            for member in members:
                if member == main_acct:
                    continue
                add_node(member, self.acct_to_bank[member])
                date = random.randrange(start_date, end_date)
                add_edge(member, main_acct, amount, date)

        elif typology_name == "fan_out":  # fan_out pattern (single (main) account --> multiple accounts)
            amount = None  # Amount will be sampled in temporal simulation
            if is_external:
                main_acct, members = self.get_typology_members(num_accounts)
            else:
                main_acct, members = self.get_typology_members(num_accounts, bank_id)
            add_node(main_acct, self.acct_to_bank[main_acct])
            for member in members:
                if member == main_acct:
                    continue
                add_node(member, self.acct_to_bank[member])
                date = random.randrange(start_date, end_date)
                add_edge(main_acct, member, amount, date)

        elif typology_name == "bipartite":  # bipartite (originators -> many-to-many -> beneficiaries)
            if num_accounts < 4:
                logger.warning(f"Bipartite pattern requires at least 4 accounts, got {num_accounts}. Skipping pattern.")
                return
            if is_external:
                main_acct, members = self.get_typology_members(num_accounts)
            else:
                main_acct, members = self.get_typology_members(num_accounts, bank_id)
            for member in members:
                add_node(member, self.acct_to_bank[member])
            num_orig_accts = random.randint(2, num_accounts - 2) # At least 2 originators and 2 beneficiaries, otherwise it is a fan-in or fan-out
            benes = members[num_orig_accts:]
            origs = members[:num_orig_accts]
            for orig, bene in itertools.product(origs, benes):  # All-to-all transaction edges
                amount = None  # Amount will be sampled in temporal simulation
                date = random.randrange(start_date, end_date)
                add_edge(orig, bene, amount, date)

        elif typology_name == "stack":  # stacked bipartite layers
            if is_external:
                main_acct, members = self.get_typology_members(num_accounts)
            else:
                main_acct, members = self.get_typology_members(num_accounts, bank_id)

            # Determine number of layers based on available accounts
            # Each layer should have at least 1 account
            total_accounts = len(members)
            min_layers = 2  # At least 2 layers needed for transactions
            max_layers = min(total_accounts, 5)  # Cap at 5 layers for reasonable structure

            if total_accounts < min_layers:
                # Not enough accounts for stacked bipartite, fallback to simple chain
                num_layers = total_accounts
            else:
                num_layers = random.randint(min_layers, max_layers)

            random.shuffle(members)

            # Assign members to layers with better distribution
            # Try to ensure each layer has at least 1 account
            layers = []
            remaining_nodes = len(members)

            for i in range(num_layers):
                if i == num_layers - 1:
                    # Last layer gets all remaining nodes
                    layer_size = remaining_nodes
                else:
                    # Calculate minimum size needed for remaining layers (1 per layer)
                    min_needed_for_remaining = (num_layers - i - 1)
                    # Calculate maximum we can allocate to this layer
                    max_for_this_layer = remaining_nodes - min_needed_for_remaining
                    # Allocate at least 1, at most max_for_this_layer
                    layer_size = random.randint(1, max(1, max_for_this_layer))

                layer = members[:layer_size]
                layers.append(layer)
                members = members[layer_size:]
                remaining_nodes -= layer_size

            # Add nodes with phase information (each layer gets its own phase number)
            for phase_num, layer in enumerate(layers):
                for member in layer:
                    add_node(member, self.acct_to_bank[member], phase=phase_num)
            
            # Ensure the layers perform their transaction before next layer initiates
            n_periods = num_layers - 1

            # Check if period is long enough for the number of layers (need at least 1 step per period)
            period_length = end_date - start_date
            if period_length < n_periods:
                logger.warning(f"Stack pattern period too short ({period_length} steps) for {num_layers} layers. Reducing layers.")
                num_layers = min(num_layers, period_length + 1)
                n_periods = num_layers - 1
                # Re-distribute members to fewer layers
                layers = []
                remaining_nodes = len(members)
                for i in range(num_layers):
                    if i == num_layers - 1:
                        layer_size = remaining_nodes
                    else:
                        min_needed_for_remaining = (num_layers - i - 1)
                        max_for_this_layer = remaining_nodes - min_needed_for_remaining
                        layer_size = random.randint(1, max(1, max_for_this_layer))
                    layer = members[:layer_size]
                    layers.append(layer)
                    members = members[layer_size:]
                    remaining_nodes -= layer_size

            proportions = np.random.dirichlet(alpha=np.ones(n_periods), size=1)[0]
            scaled_proportions = proportions * period_length
            intervals = (np.floor(scaled_proportions).astype(int) + 1)
            while np.sum(intervals) != period_length:
                difference = np.sum(intervals) - period_length
                if difference > 0:
                    # Only decrement intervals that are > 1 to avoid invalid ranges
                    candidates = np.where(intervals > 1)[0]
                    if len(candidates) > 0:
                        candidate_values = intervals[candidates]
                        max_idx = candidates[np.argmax(candidate_values)]
                        intervals[max_idx] -= 1
                    else:
                        # All intervals are 1, can't fit - this shouldn't happen with check above
                        logger.warning(f"Cannot distribute {period_length} steps across {n_periods} periods")
                        break
                else:
                    intervals[np.argmin(intervals)] += 1
    
            periods = []
            current_start = start_date
            for interval in intervals:
                current_end = current_start + interval
                periods.append((current_start, current_end))
                current_start = current_end
 
            for i in range(len(layers) - 1):
                added_edges = set()
                origs = layers[i]
                benes = layers[i+1]
                current_start, current_end = periods[i] 
                # Ensure each orig connects with at least one bene
                for orig in origs:
                    bene = random.choice(benes)
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(current_start, current_end)
                    add_edge(orig, bene, amount, date)
                    added_edges.add((orig, bene))
                
                # Ensure all benes are connected to at least one orig
                added_benes = [bene for (_,bene) in added_edges]
                for bene in benes:
                    if bene not in added_benes:
                        orig = random.choice(origs)
                        amount = None  # Amount will be sampled in temporal simulation
                        date = random.randrange(current_start, current_end)
                        add_edge(orig, bene, amount, date)
                        added_edges.add((orig, bene))

                # Add additional random connections
                for orig, bene in itertools.product(origs, benes):
                    # Randomly decide whether to add a connection
                    if random.random() < self.bipartite_edge_prob and (orig, bene) not in added_edges:
                        amount = None  # Amount will be sampled in temporal simulation
                        date = random.randrange(current_start, current_end)
                        add_edge(orig, bene, amount, date)

        elif typology_name == "random":  # Random transactions among members
            if is_external:
                main_acct, members = self.get_typology_members(num_accounts)
            else:
                main_acct, members = self.get_typology_members(num_accounts, bank_id)
            for member in members:
                add_node(member, self.acct_to_bank[member])
            members.remove(main_acct)
            for member in members:
                num_txs = random.randrange(1, num_accounts)
                benes = random.sample(members, num_txs)
                for bene in benes:
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(start_date, end_date + 1)
                    add_edge(member, bene, amount, date)
                
        elif typology_name == "cycle":  # Cycle transactions
            if is_external:
                main_acct, members = self.get_typology_members(num_accounts)
            else:
                main_acct, members = self.get_typology_members(num_accounts, bank_id)
            amount = None  # Amount will be sampled in temporal simulation
            dates = sorted([random.randrange(start_date, end_date) for _ in range(num_accounts)])
            orig = main_acct
            members.remove(main_acct)
            add_node(orig, self.acct_to_bank[orig])
            #self.remove_typology_candidate(orig)
            for bene, date in zip(members, dates):
                add_node(bene, self.acct_to_bank[bene])
                #self.remove_typology_candidate(bene)
                add_edge(orig, bene, amount, date)
                orig = bene
            add_edge(orig, main_acct, amount, dates[-1])

        elif typology_name == "scatter_gather":  # Scatter-Gather (originators -> intermediaries -> beneficiaries)
            if is_external:
                main_acct, members = self.get_typology_members(num_accounts)
            else:
                main_acct, members = self.get_typology_members(num_accounts, bank_id)

            # Distribute accounts into 3 phases with random allocation
            members.remove(main_acct)
            random.shuffle(members)

            # Randomly decide how many originators and beneficiaries
            # At least 1 originator (main_acct), 1 intermediary, 1 beneficiary
            if len(members) < 2:
                # Not enough accounts for 3 phases, use simple pattern
                orig_accts = [main_acct]
                mid_accts = members if len(members) > 0 else []
                bene_accts = [main_acct] if len(mid_accts) == 0 else []
            else:
                # Randomly split members into intermediaries and beneficiaries
                # Optionally add more originators from members
                n_extra_origs = random.randint(0, len(members) // 3) if len(members) >= 3 else 0
                n_benes = random.randint(1, max(1, (len(members) - n_extra_origs) // 2))

                orig_accts = [main_acct] + members[:n_extra_origs]
                bene_accts = members[n_extra_origs:n_extra_origs + n_benes]
                mid_accts = members[n_extra_origs + n_benes:]

            # Add nodes with phase information
            for orig in orig_accts:
                add_node(orig, self.acct_to_bank[orig], phase=0)
            for mid in mid_accts:
                add_node(mid, self.acct_to_bank[mid], phase=1)
            for bene in bene_accts:
                add_node(bene, self.acct_to_bank[bene], phase=2)

            # Phase 1: Random connections from originators to intermediaries
            # Ensure each intermediary receives from at least one originator
            added_edges = set()
            for mid in mid_accts:
                orig = random.choice(orig_accts)
                amount = None  # Amount will be sampled in temporal simulation
                date = random.randrange(start_date, end_date)
                add_edge(orig, mid, amount, date)
                added_edges.add((orig, mid))

            # Ensure each originator sends to at least one intermediary
            for orig in orig_accts:
                if not any(o == orig for o, _ in added_edges):
                    mid = random.choice(mid_accts)
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(start_date, end_date)
                    add_edge(orig, mid, amount, date)
                    added_edges.add((orig, mid))

            # Add additional random connections
            for orig, mid in itertools.product(orig_accts, mid_accts):
                if (orig, mid) not in added_edges and random.random() < self.bipartite_edge_prob:
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(start_date, end_date)
                    add_edge(orig, mid, amount, date)

            # Phase 2: Random connections from intermediaries to beneficiaries
            # Ensure each beneficiary receives from at least one intermediary
            added_edges = set()
            for bene in bene_accts:
                mid = random.choice(mid_accts)
                amount = None  # Amount will be sampled in temporal simulation
                date = random.randrange(start_date, end_date)
                add_edge(mid, bene, amount, date)
                added_edges.add((mid, bene))

            # Ensure each intermediary sends to at least one beneficiary
            for mid in mid_accts:
                if not any(m == mid for m, _ in added_edges):
                    bene = random.choice(bene_accts)
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(start_date, end_date)
                    add_edge(mid, bene, amount, date)
                    added_edges.add((mid, bene))

            # Add additional random connections
            for mid, bene in itertools.product(mid_accts, bene_accts):
                if (mid, bene) not in added_edges and random.random() < self.bipartite_edge_prob:
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(start_date, end_date)
                    add_edge(mid, bene, amount, date)

        elif typology_name == "gather_scatter":  # Gather-Scatter (sources -> hubs -> targets)
            if is_external:
                main_acct, members = self.get_typology_members(num_accounts)
            else:
                main_acct, members = self.get_typology_members(num_accounts, bank_id)

            # Distribute accounts into 3 phases with random allocation
            members.remove(main_acct)
            random.shuffle(members)

            # Randomly decide how many hubs, sources, and targets
            # At least 1 hub (main_acct), optionally more hubs from members
            if len(members) < 2:
                # Not enough accounts, use simple pattern
                hub_accts = [main_acct]
                source_accts = members if len(members) > 0 else []
                target_accts = [main_acct] if len(source_accts) == 0 else []
            else:
                # Optionally add more hubs from members
                n_extra_hubs = random.randint(0, len(members) // 3) if len(members) >= 3 else 0
                n_sources = random.randint(1, max(1, (len(members) - n_extra_hubs) // 2))

                hub_accts = [main_acct] + members[:n_extra_hubs]
                source_accts = members[n_extra_hubs:n_extra_hubs + n_sources]
                target_accts = members[n_extra_hubs + n_sources:]

            # Add nodes with phase information
            for hub in hub_accts:
                add_node(hub, self.acct_to_bank[hub], phase=0)
            for source in source_accts:
                add_node(source, self.acct_to_bank[source], phase=1)
            for target in target_accts:
                add_node(target, self.acct_to_bank[target], phase=2)

            # Phase 1: Random connections from sources to hubs (gather)
            # Ensure each hub receives from at least one source
            added_edges = set()
            for hub in hub_accts:
                source = random.choice(source_accts)
                amount = None  # Amount will be sampled in temporal simulation
                date = random.randrange(start_date, end_date)
                add_edge(source, hub, amount, date)
                added_edges.add((source, hub))

            # Ensure each source sends to at least one hub
            for source in source_accts:
                if not any(s == source for s, _ in added_edges):
                    hub = random.choice(hub_accts)
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(start_date, end_date)
                    add_edge(source, hub, amount, date)
                    added_edges.add((source, hub))

            # Add additional random connections
            for source, hub in itertools.product(source_accts, hub_accts):
                if (source, hub) not in added_edges and random.random() < self.bipartite_edge_prob:
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(start_date, end_date)
                    add_edge(source, hub, amount, date)

            # Phase 2: Random connections from hubs to targets (scatter)
            # Ensure each target receives from at least one hub
            added_edges = set()
            for target in target_accts:
                hub = random.choice(hub_accts)
                amount = None  # Amount will be sampled in temporal simulation
                date = random.randrange(start_date, end_date)
                add_edge(hub, target, amount, date)
                added_edges.add((hub, target))

            # Ensure each hub sends to at least one target
            for hub in hub_accts:
                if not any(h == hub for h, _ in added_edges):
                    target = random.choice(target_accts)
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(start_date, end_date)
                    add_edge(hub, target, amount, date)
                    added_edges.add((hub, target))

            # Add additional random connections
            for hub, target in itertools.product(hub_accts, target_accts):
                if (hub, target) not in added_edges and random.random() < self.bipartite_edge_prob:
                    amount = None  # Amount will be sampled in temporal simulation
                    date = random.randrange(start_date, end_date)
                    add_edge(hub, target, amount, date)

        # TODO: User-defined typology implementations goes here

        else:
            logger.warning("Unknown AML typology name: %s" % typology_name)
            return

        # Add the generated transaction edges to whole transaction graph
        sub_g.graph[MAIN_ACCT_KEY] = main_acct  # Main account ID
        sub_g.graph[IS_SAR_KEY] = is_sar  # SAR flag
        self.alert_groups[self.alert_id] = sub_g
        self.alert_id += 1


    def write_account_list(self):
        """Write account list to a CSV file.
        """        
        os.makedirs(self.output_dir, exist_ok=True)
        acct_file = os.path.join(self.output_dir, self.out_account_file)
        with open(acct_file, "w") as wf:
            writer = csv.writer(wf)
            base_attrs = ["ACCOUNT_ID", "CUSTOMER_ID", "INIT_BALANCE", "IS_SAR", "BANK_ID"] # column names
            writer.writerow(base_attrs + self.attr_names) # add user-defined attributes
            for n in self.g.nodes(data=True): # loop over all nodes with access to their attributes
                aid = n[0]  # Account ID
                cid = "C_" + str(aid)  # Customer ID bounded to this account
                prop = n[1]  # Account attributes
                balance = "{0:.2f}".format(prop["init_balance"])  # Initial balance
                is_sar = "true" if prop[IS_SAR_KEY] else "false"  # Whether this account is involved in SAR
                bank_id = prop["bank_id"]  # Bank ID
                values = [aid, cid, balance, is_sar, bank_id]
                for attr_name in self.attr_names:
                    values.append(prop[attr_name])
                writer.writerow(values)
        logger.info("Exported %d accounts to %s" % (self.g.number_of_nodes(), acct_file))

    def write_transaction_list(self):
        """Write transaction list to a CSV file.
        """        
        tx_file = os.path.join(self.output_dir, self.out_tx_file)
        with open(tx_file, "w") as wf:
            writer = csv.writer(wf)
            writer.writerow(["id", "src", "dst", "ttype"])
            for e in self.g.edges(data=True): # go through all transactions in graph
                src = e[0]
                dst = e[1]
                attr = e[2]
                tid = attr.get('edge_id', None)
                is_active = attr.get('active', False)
    
                tid = attr['edge_id']
                tx_type = random.choice(self.tx_types)
                if is_active:
                    writer.writerow([tid, src, dst, tx_type])
        logger.info("Exported %d transactions to %s" % (self.g.number_of_edges(), tx_file))

    def write_alert_account_list(self):
        """Write alert account list to a CSV file (renamed to alert_models.csv for consistency).
        """
        def get_out_edge_attrs(g, vid, name):
            return [v for k, v in nx.get_edge_attributes(g, name).items() if (k[0] == vid or k[1] == vid)]

        acct_count = 0
        # Renamed alert_members.csv -> alert_models.csv for consistency with normal_models.csv
        alert_model_file = os.path.join(self.output_dir, self.out_alert_member_file)
        logger.info("Output alert model list to: " + alert_model_file)
        with open(alert_model_file, "w") as wf:
            writer = csv.writer(wf)
            # Unified column names: modelID (was alertID), type (was reason)
            # Removed: isSAR (redundant - determined by file), bankID (can be looked up from account)
            base_attrs = ["modelID", "type", "accountID", "isMain", "sourceType", "phase"]
            writer.writerow(base_attrs + self.attr_names)
            for gid, sub_g in self.alert_groups.items(): # go over all subgraphs of alert groups
                main_id = sub_g.graph[MAIN_ACCT_KEY] # get main account ID
                pattern_type = sub_g.graph["reason"] # get the type of money laundering
                source_type = sub_g.graph["source_type"] # source type
                for n in sub_g.nodes(): # go over all nodes in the subgraph
                    is_main = "true" if n == main_id else "false"
                    # Get phase from node attributes (default to 0 if not set)
                    phase = sub_g.nodes[n].get("phase", 0)
                    values = [gid, pattern_type, n, is_main, source_type, phase]
                    prop = self.g.nodes[n] # get the current node from the main graph
                    for attr_name in self.attr_names: # read out all the user-defined attributes
                        values.append(prop[attr_name]) # append the values to the list
                    writer.writerow(values) # write the values to the CSV file
                    acct_count += 1

        logger.info("Exported %d members for %d AML typologies to %s" %
                    (acct_count, len(self.alert_groups), alert_model_file))

    def write_normal_models(self):
        """Write normal models to a CSV file (unified structure with alert_models.csv).
        """
        output_file = os.path.join(self.output_dir, self.out_normal_models_file)
        with open(output_file, "w") as wf:
            writer = csv.writer(wf)
            # Unified column names with alert_models.csv
            # Removed: isSAR (redundant - determined by file type)
            # Note: normal models don't have sourceType or phase, so we omit them
            column_headers = ["modelID", "type", "accountID", "isMain"]
            writer.writerow(column_headers)

            for normal_model in self.normal_models: # go over the normal models
                for account_id in normal_model.node_ids: # go over the accounts in the normal model
                    is_main = normal_model.is_main(account_id)
                    values = [normal_model.id, normal_model.type, account_id, is_main]
                    writer.writerow(values)

    def count__patterns(self, threshold=2):
        """Count the number of fan-in and fan-out patterns in the generated transaction graph
        """
        in_deg = Counter(self.g.in_degree().values())  # in-degree, count
        out_deg = Counter(self.g.out_degree().values())  # out-degree, count
        for th in range(2, threshold + 1):
            num_fan_in = sum([c for d, c in in_deg.items() if d >= th])
            num_fan_out = sum([c for d, c in out_deg.items() if d >= th])
            logger.info("\tNumber of fan-in / fan-out patterns with %d neighbors: %d / %d"
                        % (th, num_fan_in, num_fan_out))

        main_in_deg = Counter()
        main_out_deg = Counter()
        for sub_g in self.alert_groups.values():
            main_acct = sub_g.graph[MAIN_ACCT_KEY]
            main_in_deg[self.g.in_degree(main_acct)] += 1
            main_out_deg[self.g.out_degree(main_acct)] += 1
        for th in range(2, threshold + 1):
            num_fan_in = sum([c for d, c in main_in_deg.items() if d >= threshold])
            num_fan_out = sum([c for d, c in main_out_deg.items() if d >= threshold])
            logger.info("\tNumber of alerted fan-in / fan-out patterns with %d neighbors: %d / %d"
                        % (th, num_fan_in, num_fan_out))


def generate_transaction_graph_from_config(conf, sim_name=None):
    """
    Generate transaction graph from config dictionary (with absolute paths).

    Args:
        conf: Configuration dictionary with absolute paths
        sim_name: Optional simulation name override
    """
    if sim_name is None:
        sim_name = conf['general']['simulation_name']

    txg = TransactionGenerator(conf, sim_name)
    txg.set_num_accounts()  # Read out the number of accounts to be created
    txg.generate_normal_transactions()  # Load a parameter CSV file for the base transaction types
    txg.load_account_list()  # Load account list CSV file and write accounts to nodes in network
    txg.load_normal_models()  # Load a parameter CSV file for Normal Models
    txg.build_normal_models()  # Build normal models from the base transaction types
    txg.set_main_acct_candidates()  # Identify accounts with large amounts of in and out edges
    txg.load_alert_patterns()  # Load alert patterns CSV file and create AML typology subgraphs
    txg.mark_active_edges()  # mark all edges in the normal models as active
    txg.write_account_list()  # Export accounts to a CSV file
    txg.write_transaction_list()  # Export transactions to a CSV file
    txg.write_alert_account_list()  # Export alert accounts to a CSV file
    txg.write_normal_models()

def generate_transaction_graph(conf_file):
    """Generate transaction graph from config file path (for CLI usage)."""
    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)
        sim_name = conf['general']['simulation_name']
    txg = TransactionGenerator(conf, sim_name)
    txg.set_num_accounts()  # Read out the number of accounts to be created
    txg.generate_normal_transactions()  # Load a parameter CSV file for the base transaction types
    txg.load_account_list()  # Load account list CSV file and write accounts to nodes in network
    txg.load_normal_models()  # Load a parameter CSV file for Normal Models
    txg.build_normal_models()  # Build normal models from the base transaction types
    txg.set_main_acct_candidates()  # Identify accounts with large amounts of in and out edges
    txg.load_alert_patterns()  # Load alert patterns CSV file and create AML typology subgraphs
    txg.mark_active_edges()  # mark all edges in the normal models as active
    txg.write_account_list()  # Export accounts to a CSV file
    txg.write_transaction_list()  # Export transactions to a CSV file
    txg.write_alert_account_list()  # Export alert accounts to a CSV file
    txg.write_normal_models()


def main():
    """Main entry point for generating transaction graph"""
    argv = sys.argv
    if len(argv) < 2:
        print("Error: Configuration file is required")
        print("Usage: python transaction_graph_generator.py <config.yaml> [simulation_name]")
        print("Example: python transaction_graph_generator.py experiments/template/config/data.yaml")
        sys.exit(1)

    _conf_file = argv[1]
    _sim_name = argv[2] if len(argv) >= 3 else None

    # Validation option for graph contractions
    deg_param = os.getenv("DEGREE")
    degree_threshold = 0 if deg_param is None else int(deg_param)

    with open(_conf_file, "r") as rf:
        conf = yaml.safe_load(rf)

    txg = TransactionGenerator(conf, _sim_name)
    txg.set_num_accounts() # Read out the number of accounts to be created
    txg.generate_normal_transactions()  # Load a parameter CSV file for the base transaction types
    txg.load_account_list()  # Load account list CSV file and write accounts to nodes in network
    if degree_threshold > 0:
        logger.info("Generated normal transaction network")
        txg.count_fan_in_out_patterns(degree_threshold)
    txg.load_normal_models() # Load a parameter CSV file for Normal Models
    #cProfile.run('txg.build_normal_models()')
    txg.build_normal_models() # Build normal models from the base transaction types
    txg.set_main_acct_candidates() # Identify accounts with large amounts of in and out edges
    txg.load_alert_patterns()  # Load alert patterns CSV file and create AML typology subgraphs
    txg.mark_active_edges() # mark all edges in the normal models as active

    if degree_threshold > 0:
        logger.info("Added alert transaction patterns")
        txg.count_fan_in_out_patterns(degree_threshold)
    txg.write_account_list()  # Export accounts to a CSV file
    txg.write_transaction_list()  # Export transactions to a CSV file
    txg.write_alert_account_list()  # Export alert accounts to a CSV file
    txg.write_normal_models()


if __name__ == "__main__":
    main()
