import matplotlib
matplotlib.use("Agg")

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import torch
import pandas as pd
from time import time
from dhg import Hypergraph
from transformers import T5EncoderModel, AutoTokenizer

from HyperAMP import HGNN
from utils import *
torch.manual_seed(1234)

# Hyperparameters
batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(model, data_loader):
    preds = torch.Tensor()
    true_labels = torch.Tensor()
    model.eval()
    with torch.no_grad():
        for i, (emb, params, label) in enumerate(data_loader):
            emb = emb.to(device)
            params = [para.to(device) for para in params]
            label = label.to(device)
            
            output = model(emb.to(torch.float32), params)
            preds = torch.cat((preds, output.cpu()), 0)
            true_labels = torch.cat((true_labels, label.cpu()), 0)
    return preds.numpy().flatten(), true_labels.numpy().flatten()

def test(data_test, model_path):
    # add 'premodel, tokenizer, device' parameters for Ankh RLloop 23-08-08
    premodel, tokenizer = ankh.load_base_model()
    premodel.eval()
    premodel.to(device = device)

    # print("--------start testing-------")
    test_dataset = load_data(data_test.index.values, data_test.label.values, data_test, premodel, tokenizer, device)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_padding)
    
    model = HGNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    preds, labels = predict(model, test_loader)
    return preds

def test_local(data_test, model_path):
    # replace ankh.load_base_model() from locally saved file, for FLEXS and tf38 env
    local_path = '/home/dong_rh/ankh/models--ElnaggarLab--ankh-base/snapshots/c8957643a8a415e218a14de8a97fea06ed27c0f0'
    premodel = T5EncoderModel.from_pretrained(local_path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    premodel.eval()
    premodel.to(device = device)

    # print("--------start testing-------")
    test_dataset = load_data(data_test.index.values, data_test.label.values, data_test, premodel, tokenizer, device)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_padding)
    
    model = HGNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    preds, labels = predict(model, test_loader)
    return preds

def read_seqs(seqs):
    target_keys = []
    y = []
    for i, __ in enumerate(seqs):
        target_keys.append('seq' + str(i))
        y.append(0)
    df_data = pd.DataFrame(zip(target_keys, seqs, y))
    df_data.rename(columns={0:'name', 1: 'seq', 2:'label'}, inplace=True)
    print('The number of peptides: ', len(df_data))
    return df_data

if __name__ == "__main__":
    # load test data files
    test_file = '../data/predict.txt'
    data_test = read_dataset(test_file)
    # df_seqs = pd.read_csv('./predict_input_ankh.txt', header=None)
    # test_seqs = df_seqs.iloc[:10, 0].values.tolist()
    # data_test = read_seqs(test_seqs)
    
    # run
    model_name= 'hyp_best_model_V1'
    test_preds = test(data_test, model_name) # label is reward score
    df_output = pd.DataFrame(test_preds)
    df_output.to_csv('predict_output_ankh.txt', header=False, index=False)
