import numpy as np
import random
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch_geometric
import torch_geometric.transforms
import inspect

def set_random_seed(seed:int=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tensordatasets(train_df:pd.DataFrame, val_df:pd.DataFrame=None, test_df:pd.DataFrame=None, normalize=True, device='cpu'):
    train_np = train_df.to_numpy()
    x_train = train_np[:, :-1]
    y_train = train_np[:, -1]
    if normalize:
        scaler = MinMaxScaler().fit(x_train)
        x_train = scaler.transform(x_train)
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.int64).to(device)
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    
    if val_df is not None:
        val_np = val_df.to_numpy()
        x_val = val_np[:, :-1]
        y_val = val_np[:, -1]
        if normalize:
            x_val = scaler.transform(x_val)
        x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.int64).to(device)
        valset = torch.utils.data.TensorDataset(x_val, y_val)
    elif val_df is None:
        valset = None
    
    if test_df is not None:
        test_np = test_df.to_numpy()
        x_test = test_np[:, :-1]
        y_test = test_np[:, -1]
        if normalize:
            x_test = scaler.transform(x_test)
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.int64).to(device)
        testset = torch.utils.data.TensorDataset(x_test, y_test)
    elif test_df is None:
        testset = None

    return trainset, valset, testset

def dataloaders(trainset, valset, testset, batch_size=64, sampler=None, generator=None):
    if trainset is not None:
        shuffle = False if sampler is not None else True
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, generator=generator)
    else:
        trainloader = None
    
    if valset is not None:
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    else:
        valloader = None
    
    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    else:
        testloader = None
    
    return trainloader, valloader, testloader

def decrease_lr(optimizer, factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor

def graphdataset(train_nodes_df:pd.DataFrame, train_edges_df:pd.DataFrame, val_nodes_df:pd.DataFrame, val_edges_df:pd.DataFrame, test_nodes_df:pd.DataFrame, test_edges_df:pd.DataFrame, device='cpu', directed=False):
    node_to_index = {id: idx for idx, id in enumerate(train_nodes_df['node'])}
    train_nodes_np = train_nodes_df.drop(columns='node').to_numpy()
    x_train_nodes = train_nodes_np[:, :-1]
    y_train_nodes = train_nodes_np[:, -1]
    scaler = MinMaxScaler().fit(x_train_nodes)
    x_train_nodes = scaler.transform(x_train_nodes)
    x_train_nodes = torch.tensor(x_train_nodes, dtype=torch.float32).to(device)
    y_train_nodes = torch.tensor(y_train_nodes, dtype=torch.int64).to(device)
    train_edges_df['src'] = train_edges_df['src'].map(node_to_index)
    train_edges_df['dst'] = train_edges_df['dst'].map(node_to_index)
    train_edges = train_edges_df[['src', 'dst']].to_numpy()
    train_edges_index = torch.tensor(train_edges.T).to(device)
    trainset = torch_geometric.data.Data(x=x_train_nodes, edge_index=train_edges_index, y=y_train_nodes)
    if not directed:
        trainset = torch_geometric.transforms.ToUndirected()(trainset)
    
    if val_nodes_df is None:
        valset = None
    else:
        node_to_index = {id: idx for idx, id in enumerate(val_nodes_df['node'])}
        val_nodes_np = val_nodes_df.drop(columns='node').to_numpy()
        x_val_nodes = val_nodes_np[:, :-1]
        y_val_nodes = val_nodes_np[:, -1]
        x_val_nodes = scaler.transform(x_val_nodes)
        x_val_nodes = torch.tensor(x_val_nodes, dtype=torch.float32).to(device)
        y_val_nodes = torch.tensor(y_val_nodes, dtype=torch.int64).to(device)
        val_edges_df['src'] = val_edges_df['src'].map(node_to_index)
        val_edges_df['dst'] = val_edges_df['dst'].map(node_to_index)
        val_edges = val_edges_df[['src', 'dst']].to_numpy()
        val_edge_index = torch.tensor(val_edges.T).to(device)
        valset = torch_geometric.data.Data(x=x_val_nodes, edge_index=val_edge_index, y=y_val_nodes)
        if not directed:
            valset = torch_geometric.transforms.ToUndirected()(valset)
    
    if test_nodes_df is None:
        testset = None
    else:
        node_to_index = {id: idx for idx, id in enumerate(test_nodes_df['node'])}
        test_nodes_np = test_nodes_df.drop(columns='node').to_numpy()
        x_test_nodes = test_nodes_np[:, :-1]
        y_test_nodes = test_nodes_np[:, -1]
        x_test_nodes = scaler.transform(x_test_nodes)
        x_test_nodes = torch.tensor(x_test_nodes, dtype=torch.float32).to(device)
        y_test_nodes = torch.tensor(y_test_nodes, dtype=torch.int64).to(device)
        test_edges_df['src'] = test_edges_df['src'].map(node_to_index)
        test_edges_df['dst'] = test_edges_df['dst'].map(node_to_index)
        test_edges = test_edges_df[['src', 'dst']].to_numpy()
        test_edge_index = torch.tensor(test_edges.T).to(device)
        testset = torch_geometric.data.Data(x=x_test_nodes, edge_index=test_edge_index, y=y_test_nodes)
        if not directed:
            testset = torch_geometric.transforms.ToUndirected()(testset)
    
    return trainset, valset, testset

def filter_args(class_type, args):
    """
    Filters kwargs to keep only valid arguments for a given class type.

    Args:
        class_type: The class (e.g., torch.optim.Adam, torch.nn.CrossEntropyLoss).
        kwargs (dict): Dictionary of all potential arguments.

    Returns:
        dict: Filtered dictionary containing only relevant arguments.
    """
    valid_params = inspect.signature(class_type).parameters.keys()
    return {k: v for k, v in args.items() if k in valid_params}

def get_optimal_params(config:dict, params_path:str) -> dict:
    """
    Reads the optimal hyperparameters from a file.

    Args:
        config (dict): Dictionary containing the default hyperparameters.
        params_path (str): Path to the file containing the optimal hyperparameters.

    Returns:
        dict: Dictionary containing the optimal hyperparameters.
    """
    with open(f"{params_path}/best_trials.txt", 'r') as f:
        lines = f.readlines()
    optimal_params = {}
    for line in lines:
        if 'values' in line:
            values = line.split(': ')[1].strip()[1:-1].split(', ')
            values = [float(val) if '.' in val else int(val) for val in values]
            optimal_params['values'] = values
        elif ':' in line:
            key, val = line.split(': ')
            key = key.strip()
            val = val.strip()
            if val.isdigit():
                val = int(val)
            elif val.replace('.', '', 1).isdigit():
                val = float(val)
            optimal_params[key] = val
    
    # add only the parameters that are in the config
    optimal_params = {k: v for k, v in optimal_params.items() if k in config}
            
    return config | optimal_params
    