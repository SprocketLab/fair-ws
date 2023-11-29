import numpy as np
import os
import torch
import joblib


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
        sbm_mapping_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_embedding_SBM_mapping_{ot_type}.joblib')
    else:
        sbm_mapping_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_SBM_mapping_{ot_type}.joblib')
    
    return os.path.exists(sbm_mapping_path)

def load_sbm_mapping(dataset_name, ot_type=None, use_LIFT_embedding=False, data_base_path=DEFAULT_DATA_PATH):
    
    data_folder_name = get_data_folder_name(dataset_name)
    mapping_base_path = os.path.join(data_base_path, data_folder_name, "SBM_mapping")
    if use_LIFT_embedding:
        sbm_mapping = joblib.load(os.path.join(mapping_base_path,
            f'{data_folder_name}_embedding_SBM_mapping_{ot_type}.joblib'))
    else:
        sbm_mapping = joblip.load(os.path.join(mapping_base_path,
            f'{data_folder_name}_SBM_mapping_{ot_type}.joblib'))
    
    for key in sbm_mapping:
        sbm_mapping[key] = sbm_mapping[key].type(torch.long)
        
    return sbm_mapping

def save_sbm_mapping(sbm_mapping, dataset_name,
                     ot_type=None, use_LIFT_embedding=False, data_base_path=DEFAULT_DATA_PATH):
    
    data_folder_name = get_data_folder_name(dataset_name)
    mapping_base_path = os.path.join(data_base_path, data_folder_name, "SBM_mapping")
    
    if use_LIFT_embedding:
        sbm_mapping_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_embedding_SBM_mapping_{ot_type}.joblib')
    else:
        sbm_mapping_path = os.path.join(mapping_base_path,
            f'{data_folder_name}_SBM_mapping_{ot_type}.joblib')
        
    if not os.path.exists(mapping_base_path):
        os.makedirs(mapping_base_path)
        
    joblib.dump(sbm_mapping, sbm_mapping_path)
    
    print(f"SBM ({ot_type}) saved in {sbm_mapping_path}!")
    

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
