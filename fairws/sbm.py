import torch
import numpy as np
import sys
import tqdm
import ot
import faiss
sys.path.append('../fairws')
sys.path.append('../../fairws')
from snorkel.labeling.model import LabelModel
from data_util import preprocess_binary_array, save_sbm_mapping, load_sbm_mapping, check_sbm_mapping_path

np.random.seed(2023)
torch.manual_seed(2023)

DEFAULT_DATA_PATH = '../data'


class FaissKNN:
    # Efficient KNN using faiss (https://github.com/facebookresearch/faiss)
    # Source: https://gist.github.com/j-adamczyk/74ee808ffd53cd8545a49f185a908584#file-knn_with_faiss-py
    def __init__(self, k=1):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        
    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        return indices

def correct_bias(L, a_train, sbm_mapping, diff_threshold=0.05):
    """
    Correct bias related to a group variable a (a_train) using sbm_mapping
    """
    if isinstance(L, np.ndarray):
        L = torch.tensor(L)
        
    m = L.shape[1]
    
    # Estimate accuracies
    groupby_E_LY = estimate_accuracies(L, a_train)
    L_raw = L.clone()
    
    a_train = torch.tensor(a_train)
    
    
    unique_a = np.unique(a_train)
    n_groups = len(unique_a)
    
    for i in range(m):
        L_i = L_raw[:, i]
        
        # Find max accuracy group
        max_E_LY = -1
        max_acc_group = -1
    
        for group in unique_a:
            if groupby_E_LY[group][i] > max_E_LY:
                max_E_LY = groupby_E_LY[group][i]
                max_acc_group = group
        
        # Transport
        for src_group in unique_a:
            src_group_indices = torch.where(a_train==src_group)[0]
            
            # if the accuracy gap between max accuracy group and src group is enough, then transport
            if groupby_E_LY[max_acc_group][i] >= groupby_E_LY[src_group][i] + diff_threshold: 
                for j, idx in enumerate(src_group_indices):
                    src_dst_mapping = sbm_mapping[(src_group, max_acc_group)]
                    L[idx, i] = L_i[src_dst_mapping[j]]
    return L

def _triplet_median_i(L_cov, i):
    """
    get one accuracy using triplet method
    """
    m = L_cov.shape[1]
    E_LiY_estimates = []
    
    for j in range(m):
        if j==i:
            continue
        else:
            for k in range(j+1, m):
                if k==i:
                    continue
                estimate = torch.sqrt(L_cov[i,j]*L_cov[i,k]/L_cov[j,k])
                if not np.isnan(estimate):
                    E_LiY_estimates.append(estimate)
    return np.median(E_LiY_estimates)

def triplet_median(L):
    """
    Estimate accuracy using triplet (median)
    """
    n = L.shape[0]
    m = L.shape[1]
    L = preprocess_binary_array(L)
    L_cov = L.T@L/n
    E_LY = []
    for i in range(m):
        E_LY.append(_triplet_median_i(L_cov, i))
    return torch.tensor(E_LY)

def estimate_accuracies(L, a):
    """
    Estimate accuraices groupby
    """
    unique_a = np.unique(a)
    groupby_E_LY = {}
    for group in unique_a:
        L_a = L[a==group]
        E_LY_A = triplet_median(L_a)
        groupby_E_LY[group] = E_LY_A
    return groupby_E_LY

def find_sbm_mapping(X, A, ot_type):
    """
    Compute pairwise SBM mapping
    """
    sbm_mapping = {}
    
    unique_a = np.unique(A)
    
    # Compute pairwise SBM mapping
    for src_group in unique_a:
        for dst_group in unique_a:
            if src_group==dst_group:
                continue
            else:
                sbm_mapping[(src_group, dst_group)] = _find_sbm_mapping_one_direction(X, A,
                                                                                      src_group=src_group,
                                                                                      dst_group=dst_group,
                                                                                      ot_type=ot_type)
    return sbm_mapping
    

def _find_sbm_mapping_one_direction(X, A, src_group, dst_group, ot_type=None, k=1, batch_size=4):
    """
    Learn mapping a source distribution (group) to a target distribution (group)
    """
    if len(A.shape)>1:
        if A.shape[1]==1:
            A = A.squeeze()
    
    X = X.astype(np.float64)
    src_indices = np.where(A==src_group)[0]
    dst_indices = np.where(A==dst_group)[0]
    
    X_src, X_dst_ = X[src_indices], X[dst_indices]
    
    if ot_type is not None:
        if ot_type=="linear":
            ot_map = ot.da.LinearTransport()
        elif ot_type=="sinkhorn":
            ot_map = ot.da.SinkhornTransport()
        else:
            raise
            
        ot_num_sample_threshold = 10000
        if max(X_src.shape[0], X_dst_.shape[0]) > ot_num_sample_threshold:
            ot_num_samples = min(X_src.shape[0], X_dst_.shape[0], ot_num_sample_threshold)
            ot_sample_indices_src = np.random.choice(range(X_src.shape[0]), size=ot_num_samples, replace=False)
            ot_sample_indices_dst_ = np.random.choice(range(X_dst_.shape[0]), size=ot_num_samples, replace=False)
            
            ot_map.fit(Xs=X_src[ot_sample_indices_src], Xt=X_dst_[ot_sample_indices_dst_])
        else:
            ot_map.fit(Xs=X_src, Xt=X_dst_)
        X_src = ot_map.transform(X_src)
            
    
    # To get indices from the original X...
    X_dst = X.copy()
    X_dst[dst_indices] = X_dst_
    X_dst[src_indices] += 1e18 # almost infinitely far for the positive class so that never matched
    
    knn = FaissKNN()
    knn.fit(X_dst)
    mapping = knn.predict(X_src)
    
    return torch.tensor(mapping.squeeze(), dtype=torch.long)

def get_baseline_pseudolabel(L, y_train=None):
    """
    Wrapper function for WS baselines
    """
    
    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L_train=L,
                    n_epochs=1000, log_freq=100, seed=123)
    y_train_pseudo = label_model.predict(L, tie_break_policy="random")  
    
    return y_train_pseudo


def get_sbm_pseudolabel(L, x_train, a_train, dataset_name,
                        ot_type=None, diff_threshold=0.05, use_LIFT_embedding=False,
                        mapping_cache=True, data_base_path=DEFAULT_DATA_PATH):
    """
    Wrapper function for SBM
    """
    
    if check_sbm_mapping_path(dataset_name, ot_type, use_LIFT_embedding, data_base_path):
        sbm_mapping = load_sbm_mapping(dataset_name, ot_type,
                                       use_LIFT_embedding, data_base_path)
    else:
        sbm_mapping = find_sbm_mapping(x_train, a_train, ot_type)
        if mapping_cache:
            save_sbm_mapping(sbm_mapping, dataset_name, ot_type, use_LIFT_embedding)
            
    L = correct_bias(L, a_train, sbm_mapping, diff_threshold)
    label_model = LabelModel(cardinality=2, verbose=False) # assume binary classification
    label_model.fit(L_train=L, n_epochs=1000, log_freq=100, seed=123)
    y_train_pseudo = label_model.predict(L, tie_break_policy="random")  
    
    return y_train_pseudo