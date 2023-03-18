import pandas as pd
import numpy as np
import torch
import sys
import math
sys.path.append('../')
sys.path.append('../fairws')
sys.path.append('../../')
sys.path.append('../../fairws')
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_util import preprocess_binary_array


def exp_eval(y_true, y_pred, a=None, fairness=True, cond=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    result = {}
    
    if cond is not None:
        result['condition'] = cond
        
    result["accuracy"] = accuracy
    result["fscore"] = fscore
    result["precision"] = precision
    result["recall"] = recall
    
    
    if fairness & (a is not None):
        dp = demographic_diff(y_pred, a)
        eo = equal_opportunity(y_pred, a, y_true)
        
        result["demographic_parity_gap"] = dp
        result["equal_opportunity_gap"] = eo
        
    return result

def merge_two_evals(eval_test, lm_test):
    for key in lm_test:
        eval_test['lm_'+key] = lm_test[key]
    return eval_test

def demographic_diff(LY, A):
    LY = preprocess_binary_array(LY)
    A = preprocess_binary_array(A)
    
    if len(LY.shape) > 1:
        m = LY.shape[1]
    else:
        m = 1
        
    mean_LYp_Ap = []
    mean_LYp_An = []
    
    if m > 1:
        for i in range(m):
            LYi_p = torch.tensor(LY[:, i]==1).type(torch.float32).squeeze()
            Ap = torch.tensor(A==1).type(torch.float32)
            An = torch.tensor(A!=1).type(torch.float32)
            
            mean_LYp_Ap.append((LYi_p*Ap).sum()/Ap.sum())
            mean_LYp_An.append((LYi_p*An).sum()/An.sum())
    else:
        LYi_p = (LY==1).type(torch.float32).squeeze()
        Ap = (A==1).type(torch.float32)
        An = (A!=1).type(torch.float32)
        
        mean_LYp_Ap.append((LYi_p*Ap).sum()/Ap.sum())
        mean_LYp_An.append((LYi_p*An).sum()/An.sum())
            
    
    
    mean_LYp_Ap = torch.tensor(mean_LYp_Ap)
    mean_LYp_An = torch.tensor(mean_LYp_An)
    
    if len(mean_LYp_Ap) > 1:
        return torch.abs(mean_LYp_Ap - mean_LYp_An)
    else:
        return torch.abs(mean_LYp_Ap - mean_LYp_An).item()


def equal_opportunity(LY, A, Y):
    
    LY = preprocess_binary_array(LY)
    A = preprocess_binary_array(A)
    Y = preprocess_binary_array(Y)
    
    if len(LY.shape) > 1:
        m = LY.shape[1]
    else:
        m = 1
    
    mean_LYp_Ap_Yp = []
    mean_LYp_An_Yp = []
    
    if m > 1:
        for i in range(m):
            LYi_p = (LY[:, i]==1).type(torch.float32).squeeze()
            
            Ap = (A==1).type(torch.float32)
            An = (A!=1).type(torch.float32)
            Yp = (Y==1).type(torch.float32)
            
            mean_LYp_Ap_Yp.append(((LYi_p*Ap*Yp).sum()) / ((Ap*Yp).sum()))
            mean_LYp_An_Yp.append(((LYi_p*An*Yp).sum()) / ((An*Yp).sum()))
    else:
        LYi_p = (LY==1).type(torch.float32).squeeze()
        
        Ap = (A==1).type(torch.float32)
        An = (A!=1).type(torch.float32)
        Yp = (Y==1).type(torch.float32)
        
        # print('((LYi_p*Ap*Yp).sum())', ((LYi_p*Ap*Yp).sum()) )
        # print('((Ap*Yp).sum())', ((Ap*Yp).sum()))
        # print('((An*Yp).sum())', ((An*Yp).sum()))
        mean_LYp_Ap_Yp.append(((LYi_p*Ap*Yp).sum()) / ((Ap*Yp).sum()))
        mean_LYp_An_Yp.append(((LYi_p*An*Yp).sum()) / ((An*Yp).sum()))
        
    mean_LYp_Ap_Yp = torch.tensor(mean_LYp_Ap_Yp)
    mean_LYp_An_Yp = torch.tensor(mean_LYp_An_Yp)
    
    if math.isnan(mean_LYp_Ap_Yp.item()):
        mean_LYp_Ap_Yp = 0
    if math.isnan(mean_LYp_An_Yp.item()):
        mean_LYp_An_Yp = 0
    # print('mean_LYp_Ap_Yp', mean_LYp_Ap_Yp)
    # print('mean_LYp_An_Yp', mean_LYp_An_Yp)
    
    # print(np.abs(mean_LYp_Ap_Yp - mean_LYp_An_Yp))
    if len(mean_LYp_Ap_Yp) > 1:
        return torch.abs(mean_LYp_Ap_Yp - mean_LYp_An_Yp)
    else:
        return torch.abs(mean_LYp_Ap_Yp - mean_LYp_An_Yp).item()
