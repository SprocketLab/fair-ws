import torch
import numpy as np
import sys
import tqdm
import ot
sys.path.append('../fairws')
sys.path.append('../../fairws')
from snorkel.labeling.model import LabelModel
from data_util import preprocess_binary_array, save_sbm_mapping, load_sbm_mapping, check_sbm_mapping_path

np.random.seed(2023)
torch.manual_seed(2023)

DEFAULT_DATA_PATH = '../data'

def get_baseline_pseudolabel(L, y_train=None):
    
    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L_train=L,
                    n_epochs=1000, log_freq=100, seed=123)
    y_train_pseudo = label_model.predict(L, tie_break_policy="random")  
    
    return y_train_pseudo


def get_sbm_pseudolabel(L, x_train, a_train, dataset_name,
                        ot_type=None, diff_threshold=0.05, use_LIFT_embedding=False,
                        mapping_cache=True, data_base_path=DEFAULT_DATA_PATH):
    
    if check_sbm_mapping_path(dataset_name, ot_type, use_LIFT_embedding, data_base_path):
        sbm_mapping_01, sbm_mapping_10 = load_sbm_mapping(dataset_name, ot_type,
                                                        use_LIFT_embedding, data_base_path)
    else:
        sbm_mapping_01, sbm_mapping_10 = find_sbm_mapping(x_train, a_train, ot_type)
        if mapping_cache:
            save_sbm_mapping(sbm_mapping_01, sbm_mapping_10, dataset_name, ot_type, use_LIFT_embedding)
            
    
    
    L = correct_bias(L, a_train, sbm_mapping_01, sbm_mapping_10, diff_threshold)
    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L_train=L, n_epochs=1000, log_freq=100, seed=123)
    y_train_pseudo = label_model.predict(L, tie_break_policy="random")  
    
    return y_train_pseudo

def correct_bias(L, a_train, sbm_mapping_01, sbm_mapping_10, diff_threshold):
    if isinstance(L, np.ndarray):
        L = torch.tensor(L)
        
    m = L.shape[1]
    
    # Estimate accuracies
    E_LY_Ap, E_LY_An = estimate_accuracies(L, a_train)
    L_raw = L.clone()
    
    a_train = torch.tensor(a_train)
    Ap_indices = torch.where(a_train==1)[0]
    An_indices = torch.where(a_train!=1)[0]
    
    for i in range(m):
        L_i = L_raw[:, i]

        if E_LY_Ap[i] >= E_LY_An[i] + diff_threshold: # LF|A=1 is more accurate than LF|A=0
            for j, idx in enumerate(An_indices):
                L[idx, i] = L_i[sbm_mapping_01[j]]

        elif E_LY_An[i] >= E_LY_Ap[i] + diff_threshold: # LF|A=0 is more accurate than LF|A=1
            for j, idx in enumerate(Ap_indices):
                L[idx, i] = L_i[sbm_mapping_10[j]]
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
    L_Ap = L[a==1]
    L_An = L[a!=1]
    
    E_LY_Ap = triplet_median(L_Ap)
    E_LY_An = triplet_median(L_An)
    return E_LY_Ap, E_LY_An



def find_knn(data, queries, k):
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)
    if isinstance(queries, np.ndarray):
        queries = torch.tensor(queries)
        
    # calculate the L2 distance between the queries and all data points
    dist = torch.sum((data.unsqueeze(0) - queries.unsqueeze(1))**2, dim=-1)
    # find the indices of the k nearest neighbors for each query
    knn_dists, knn_indices = torch.topk(dist, k, dim=-1, largest=False)
    return knn_indices, knn_dists

def find_sbm_mapping(X, A, ot_type):
    sbm_mapping_01 = _find_sbm_mapping_one_direction(X, A, ot_type, src_group=0)
    sbm_mapping_10 = _find_sbm_mapping_one_direction(X, A, ot_type, src_group=1)
    
    return sbm_mapping_01, sbm_mapping_10
    

def _find_sbm_mapping_one_direction(X, A, ot_type=None, src_group=1, k=1, batch_size=4):
    X = torch.tensor(X, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)
    
    
    if len(A.shape)>1:
        if A.shape[1]==1:
            A = A.squeeze()
    
    if src_group==1:
        dst_group=0
    else:
        dst_group=1
    Ap_indices = torch.where(A==src_group)[0]
    An_indices = torch.where(A==dst_group)[0]
    
    X_Ap, X_An_ = X[Ap_indices], X[An_indices]
    
    if ot_type is not None:
        if ot_type=="linear":
            ot_map = ot.da.LinearTransport()
        elif ot_type=="sinkhorn":
            ot_map = ot.da.SinkhornTransport()
        else:
            raise
            
        X_Ap, X_An_ = torch.tensor(X_Ap).double(), torch.tensor(X_An_).double()
        
        ot_num_sample_threshold = 10000
        if max(X_Ap.shape[0], X_An_.shape[0]) > ot_num_sample_threshold:
            ot_num_samples = min(X_Ap.shape[0], X_An_.shape[0], ot_num_sample_threshold)
            # print("X_Ap.shape[0], X_An.shape[0]", X_Ap.shape[0], X_An_.shape[0], "do sampling...", ot_num_samples)
            ot_sample_indices_Ap = np.random.choice(range(X_Ap.shape[0]), size=ot_num_samples, replace=False)
            ot_sample_indices_An_ = np.random.choice(range(X_An_.shape[0]), size=ot_num_samples, replace=False)
            
            ot_map.fit(Xs=X_Ap[ot_sample_indices_Ap], Xt=X_An_[ot_sample_indices_An_])
        else:
            ot_map.fit(Xs=X_Ap, Xt=X_An_)
        X_Ap = ot_map.transform(X_Ap)
            
    
    # To get indices from the original X...
    X_An = X.clone()
    X_An[An_indices] = torch.tensor(X_An_).float()
    X_An[Ap_indices] += 1e18 # almost infinitely far for the positive class so that never matched
    
    
    mapping = torch.zeros((X_Ap.shape[0], k))
    mapping_dists = torch.zeros((X_Ap.shape[0], k))

    num_batches = X_Ap.shape[0] // batch_size + int((X.shape[0]%batch_size) > 0)
    
    for batch_idx in tqdm.trange(num_batches, desc="computing sbm mapping..."):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        knn_indices, knn_dists =  find_knn(X_An, X_Ap[batch_start:batch_end], k=k)
        mapping[batch_start:batch_end] = knn_indices
        mapping_dists[batch_start:batch_end] = knn_dists
    
    return mapping.squeeze().type(torch.long)