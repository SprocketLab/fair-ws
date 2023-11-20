import numpy as np
import os
import torch

DEFAULT_DATA_PATH = '../data'
FAIRNESS_DATASET_LIST = ["adult", "bank", "bank_marketing", "utkface", "hatexplain", "celeba", "civilcomments"]

def load_dataset(dataset_name, data_base_path=DEFAULT_DATA_PATH, use_torch=False):

    data_folder_name = get_data_folder_name(dataset_name)
    
    # load dataset
    
    train = np.load(os.path.join(data_base_path, data_folder_name, 
                                    f'{data_folder_name}_train.npz'),
                    allow_pickle=True)
    test = np.load(os.path.join(data_base_path, data_folder_name,
                                f'{data_folder_name}_test.npz'),
                    allow_pickle=True)

    # unpack dataset
    x_train, y_train, a_train = train['x'], train['y'], train['a']
    x_test, y_test, a_test = test['x'], test['y'], test['a']
    
    y_train, a_train = y_train.squeeze(), a_train.squeeze()
    y_test, a_test = y_test.squeeze(), a_test.squeeze()
    
    # np to torch
    if use_torch:
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()
        a_train = torch.tensor(a_train, dtype=torch.float32).squeeze()
        
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).squeeze()
        a_test = torch.tensor(a_test, dtype=torch.float32).squeeze()
    return x_train, y_train, a_train, x_test, y_test, a_test

def load_wrench_dataset(dataset_name, data_base_path=DEFAULT_DATA_PATH, use_torch=False):
    """
    load wrench dataset
    The only difference from load_dataset is that a_train, a_test don't exist
    """

    data_folder_name = get_data_folder_name(dataset_name)
    
    # load dataset
    train = np.load(os.path.join(data_base_path, data_folder_name, 
                                    f'{data_folder_name}_train.npz'),
                    allow_pickle=True)
    test = np.load(os.path.join(data_base_path, data_folder_name,
                                f'{data_folder_name}_test.npz'),
                    allow_pickle=True)

    # unpack dataset
    x_train, y_train = train['x'], train['y']
    x_test, y_test = test['x'], test['y']
    
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    
    # np to torch
    if use_torch:
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()
        
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).squeeze()
    return x_train, y_train, x_test, y_test


def load_LIFT_embedding(dataset_name, data_base_path=DEFAULT_DATA_PATH, use_torch=False):

    data_folder_name = get_data_folder_name(dataset_name)
    
    # load dataset
    train = np.load(os.path.join(data_base_path, data_folder_name, 
                                     f'{data_folder_name}_embedding_train.npz'),
                        allow_pickle=True)
    test = np.load(os.path.join(data_base_path, data_folder_name,
                                f'{data_folder_name}_embedding_test.npz'),
                allow_pickle=True)

    # unpack dataset
    x_embedding_train, x_embedding_test = train['x'], test['x']
    
    # np to torch
    if use_torch:
        x_embedding_train = torch.tensor(x_embedding_train, dtype=torch.float32)
        x_embedding_test = torch.tensor(x_embedding_test, dtype=torch.float32)
        
    return x_embedding_train, x_embedding_test

def load_LF(dataset_name, data_base_path=DEFAULT_DATA_PATH):
    data_folder_name = get_data_folder_name(dataset_name)
    L = np.load(os.path.join(data_base_path, data_folder_name, f'{data_folder_name}_LF.npy'))
    
    # convert negative class -1 --> 0
    if dataset_name.lower() in FAIRNESS_DATASET_LIST:
        L = preprocess_binary_array(L, min_val=0)
    return L

def check_sbm_mapping_path(dataset_name, ot_type=None, use_LIFT_embedding=False, data_base_path=DEFAULT_DATA_PATH):
    
    data_folder_name = get_data_folder_name(dataset_name)
    mapping_base_path = os.path.join(data_base_path, data_folder_name, "SBM_mapping")
    if use_LIFT_embedding:
        sbm_mapping_01_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_embedding_SBM_mapping_{ot_type}_0->1.pt')
        sbm_mapping_10_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_embedding_SBM_mapping_{ot_type}_1->0.pt')
    else:
        sbm_mapping_01_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_SBM_mapping_{ot_type}_0->1.pt')
        sbm_mapping_10_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_SBM_mapping_{ot_type}_1->0.pt')
    
    return os.path.exists(sbm_mapping_01_path) & os.path.exists(sbm_mapping_10_path)

def load_sbm_mapping(dataset_name, ot_type=None, use_LIFT_embedding=False, data_base_path=DEFAULT_DATA_PATH):
    
    data_folder_name = get_data_folder_name(dataset_name)
    mapping_base_path = os.path.join(data_base_path, data_folder_name, "SBM_mapping")
    if use_LIFT_embedding:
        sbm_mapping_01 = torch.load(os.path.join(mapping_base_path,
            f'{data_folder_name}_embedding_SBM_mapping_{ot_type}_0->1.pt'))
        sbm_mapping_10 = torch.load(os.path.join(mapping_base_path,
            f'{data_folder_name}_embedding_SBM_mapping_{ot_type}_1->0.pt'))
    else:
        sbm_mapping_01 = torch.load(os.path.join(mapping_base_path,
            f'{data_folder_name}_SBM_mapping_{ot_type}_0->1.pt'))
        sbm_mapping_10 = torch.load(os.path.join(mapping_base_path,
            f'{data_folder_name}_SBM_mapping_{ot_type}_1->0.pt'))
    
    sbm_mapping_01 = sbm_mapping_01.type(torch.long)
    sbm_mapping_10 = sbm_mapping_10.type(torch.long)
    '''
    # Usage exmaple (group 0 --> group 1 for LF i)
    
    L[a_train==0, i] = L[sbm_mapping_01, i]
    
    * Note that domain of counterfactual is the source group indices, not full indices
    '''
    return sbm_mapping_01, sbm_mapping_10

def save_sbm_mapping(sbm_mapping_01, sbm_mapping_10, dataset_name,
                     ot_type=None, use_LIFT_embedding=False, data_base_path=DEFAULT_DATA_PATH):
    
    data_folder_name = get_data_folder_name(dataset_name)
    mapping_base_path = os.path.join(data_base_path, data_folder_name, "SBM_mapping")
    if use_LIFT_embedding:
        sbm_mapping_01_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_embedding_SBM_mapping_{ot_type}_0->1.pt')
        sbm_mapping_10_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_embedding_SBM_mapping_{ot_type}_1->0.pt')
    else:
        sbm_mapping_01_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_SBM_mapping_{ot_type}_0->1.pt')
        sbm_mapping_10_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_SBM_mapping_{ot_type}_1->0.pt')
    
    if not os.path.exists(mapping_base_path):
        os.makedirs(mapping_base_path)
        
    torch.save(sbm_mapping_01, sbm_mapping_01_path)
    torch.save(sbm_mapping_10, sbm_mapping_10_path)
    
    print(f"SBM ({ot_type}) saved in {sbm_mapping_01_path}, {sbm_mapping_01_path}!")
    

def get_data_folder_name(dataset_name):
    if dataset_name.lower() == 'adult':
        return "adult"
    elif (dataset_name.lower() == 'bank') or (dataset_name.lower() == 'bank_marketing'):
        return "bank_marketing"
    elif dataset_name.lower() == 'civilcomments':
        return "CivilComments"
    elif (dataset_name.lower() == 'hate') or dataset_name.lower() == 'hatexplain':
        return "hateXplain"
    elif (dataset_name.lower() == 'celeba'):
        return "CelebA"
    elif (dataset_name.lower() == 'utkface'):
        return "UTKFace"
    else:
        return dataset_name.lower()
 
def find_knn(data, queries, k, batch_size=16):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)
    if isinstance(queries, np.ndarray):
        queries = torch.tensor(queries)
        
    num_batches = int(np.ceil(data.shape[0] / batch_size))
    
    knn_indices, knn_dists = torch.zeros((queries.shape[0], k)), torch.zeros((queries.shape[0], k))
    
    if len(queries.shape)==1:
        knn_indices, knn_dists = find_knn_batch(data, queries.unsqueeze(0), k)
    else:
        for i in range(num_batches):
            batch_start_idx = i*batch_size
            batch_end_idx = (i+1)*batch_size
            
            knn_indices_batch, knn_dists_batch = find_knn_batch(data, queries[batch_start_idx:batch_end_idx], k)
            knn_indices[batch_start_idx:batch_end_idx] = knn_indices_batch
            knn_dists[batch_start_idx:batch_end_idx] = knn_dists_batch
    
    return knn_indices, knn_dists

def find_knn_batch(data, queries, k):
    # calculate the L2 distance between the queries and all data points
    dist = torch.sum((data.unsqueeze(0) - queries.unsqueeze(1))**2, dim=-1)
    
    # find the indices of the k nearest neighbors for each query
    knn_dists, knn_indices = torch.topk(dist, k, dim=-1, largest=False)
    return knn_indices, knn_dists
    

def preprocess_binary_array(arr, min_val=-1):
    
    # type conversion
    if not isinstance(arr, torch.Tensor):
        arr = torch.tensor(arr)
    
    # {0, 1} array -> {-1, 1} array or vice versa.
    if arr.min()!=min_val:
        if min_val==-1:
            arr = arr*2 - 1
        elif min_val==0:
            arr = (arr+1)/2
        else:
            raise ValueError("Currently, min_val only considers -1 or 0")
        
    # squeeze if needed
    if len(arr.shape)>1:
        if arr.shape[1]==1:
            arr = arr.squeeze()
            
    return arr
