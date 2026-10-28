import matplotlib
matplotlib.use("Agg")

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import pandas as pd
from time import time
from copy import deepcopy
from dhg import Hypergraph

from transformers import logging
logging.set_verbosity_error()

from HyperAMP import HGNN
from utils import *
torch.manual_seed(1234)

# Hyperparameters
learning_rate = 1e-4
weight_decay = 5e-4
batch_size = 128
epoches = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model training
def train(model, train_loader, optimizer, epoch):
    loss_record = []
    loss_fn = torch.nn.MSELoss()
    
    model.train()
    t0 = time()
    print("--------start training-------")
    for i, (emb, params, label) in enumerate(train_loader):
        emb = emb.to(device)
        params = [para.to(device) for para in params]
        label = label.to(device)

        output = model(emb.to(torch.float32), params)
        loss = loss_fn(output.to(torch.float32), label.to(torch.float32))

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        loss_record.append(loss.item())

        t1 = time()
        epoch_time = t1 - t0
        if i % 10 == 0:
            print('Epoch {}, Loss {:.4f}, time {:.2f}'.format(epoch, loss.item(), epoch_time))
    # with open('loss_record.txt','a') as f:
        # f.writelines(str(loss_record) + '\n')


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


def main(premodel, tokenizer, device, data_train, data_test, data_val = None):
    metrics_train = []
    metrics_val = []
    metrics_test = []

    # defined val data
    df_train = data_train
    df_train.index = range(len(df_train))
    df_val = data_val
    df_val.index = range(len(df_val))
    
    train_dataset = load_data(df_train.index.values, df_train.label.values, df_train, premodel, tokenizer, device)
    val_dataset = load_data(df_val.index.values, df_val.label.values, df_val, premodel, tokenizer, device)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padding)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_padding)
    
    model = HGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    best_rmse = 100
    best_epoch = 0        
    for epoch in range(epoches):
        train(model, train_loader, optimizer, epoch)
        # train metrics
        preds_train, real_labels_train = predict(model, train_loader)
        tr_spearman, tr_pearson, tr_r2, tr_rmse = evalute(preds_train, real_labels_train)
        metrics_train.append([tr_spearman, tr_pearson, tr_r2, tr_rmse])

        # validation
        preds, real_labels = predict(model, val_loader)
        val_spearman, val_pearson, val_r2, val_rmse = evalute(preds, real_labels)
        print("Validation at epoch {}:, Spearman {:.4f}, \
                Pearson {:.4f}, R2 {:.4f}, RMSE {:.4f}".format(epoch, val_spearman, val_pearson, val_r2, val_rmse))
        if val_rmse < best_rmse:
            print("RMSE decreases from {:.4f} to {:.4f}, save the model".format(best_rmse, val_rmse))
            best_rmse = val_rmse
            best_epoch = epoch
            best_metrics = [val_spearman, val_pearson, val_r2, val_rmse]
            # model_name = 'Fold_' + str(fold+1)
            model_name = deepcopy(model.state_dict())
            # torch.save(model.state_dict(), 'hyp_model_logMIC_V1')
        else:
            print("No improvement in RMSE since epoch {}".format(best_epoch))
    metrics_val.append(best_metrics)
    # remember to ensemble k-folds results
    metrics, test_preds = test(data_test, model_name, premodel, tokenizer, device)
    metrics_test.append(metrics)

    metric_header = ['Spearman', 'Pearson', 'R2', 'RMSE']
    results_train = pd.DataFrame(metrics_train, columns=metric_header)
    results_train.to_csv('train_results.csv')
    print("training results: \n", results_train.mean())

    results_val = pd.DataFrame(metrics_val, columns=metric_header)
    results_val.to_csv('val_results.csv')
    print("validation results: \n", results_val.mean())

    results_test = pd.DataFrame(metrics_test, columns=metric_header)
    results_test.to_csv('test_results.csv')
    print("test results: \n", results_test.mean())


def test(data_test, model_path, premodel, tokenizer, device):
    print("--------start testing-------")
    test_dataset = load_data(data_test.index.values, data_test.label.values, data_test, premodel, tokenizer, device)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_padding)
    
    model = HGNN().to(device)
    model.load_state_dict(model_path)
    # model.load_state_dict(torch.load(model_path, map_location=device))

    preds, labels = predict(model, test_loader)
    Spearman, Pearson, R2, RMSE = evalute(preds, labels)
    print("Test results: Spearman {:.4f}, \
        Pearson {:.4f}, R2 {:.4f}, RMSE {:.4f}".format(Spearman, Pearson, R2, RMSE))
    metrics = [Spearman, Pearson, R2, RMSE]
    return metrics, preds


if __name__ == "__main__":
    # data files
    train_file = '../data/train.txt'
    valid_file = '../data/valid.txt'
    test_file = '../data/test.txt'
    # load model
    premodel, tokenizer = ankh.load_base_model()
    premodel.eval()
    premodel.to(device = device)
    # load data
    data_train = read_dataset(train_file)
    data_val = read_dataset(valid_file)
    data_test = read_dataset(test_file)
    # run
    main(premodel, tokenizer, device, data_train, data_test, data_val)
