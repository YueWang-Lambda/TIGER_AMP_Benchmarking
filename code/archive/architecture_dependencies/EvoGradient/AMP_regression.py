import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import pandas as pd
import math
import argparse
from Bio import SeqIO

# Parse command-line arguments
parser = argparse.ArgumentParser(description="AMP Classification")
parser.add_argument("--testPath", type=str, required=True, help="Path to the test dataset")
parser.add_argument("--savePath", type=str, required=True, help="Path to save the results")
args = parser.parse_args()

testPath = args.testPath
savePath = args.savePath

# Data paths
trainPath = "./data/regression/train.csv"
validatePath = "./data/regression/test.csv"


# Configuration parameters

batch_size = 256
MAX_MIC = math.log10(8192)
My_MAX_MIC = math.log10(600)


# Model paths
model_list = {
    #
    "CNN": "./model/regression/CNN.pth",
    "Transformer": "./model/regression/Transformer.pth",
    "Attention": "./model/regression/Attention.pth",
    "LSTM": "./model/regression/LSTM.pth",
}

nameList = model_list.keys()
weight = {"CNN": 0.25000594, "Transformer": 0.2500046, "Attention": 0.25000825, "LSTM": 0.24998219}

mydict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}
myInvDict = dict([val, key] for key, val in mydict.items())


def fasta_to_csv(fasta_path, csv_path):

    sequences = []
    lengths = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        lengths.append(len(record.seq))

    df = pd.DataFrame({"Sequence": sequences, "Length": lengths})

    print(csv_path)
    df.to_csv(csv_path, index=False)
    return csv_path


# transform fasta to csv
testPath = fasta_to_csv(testPath, testPath[:-5] + ".csv")


def dataProcessPipeline(seq):

    # this function first transform peptide sequences into numerical sequence,
    # transformer it into onehot vector and padding them into a fix length
    # returning the padding vector and mask

    testest = seq
    num_seq = [mydict[character.upper()] for character in seq]

    seq = np.array(num_seq, dtype=int)
    len = seq.shape[0]
    torch_seq = torch.tensor(seq)
    if torch.sum(torch_seq[torch_seq < 0]) != 0:
        print(torch_seq[torch_seq < 0])
        print("wrong seq:", seq)
        print(testest)
    onehotSeq = torch.nn.functional.one_hot(torch_seq, num_classes=20)
    # onehotSeq = torch.nn.functional.one_hot(c
    pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 100 - len))
    mask = np.zeros(100, dtype=int)
    mask[len:] = 1
    mask = torch.tensor(mask)

    pad_seq = pad(onehotSeq)

    return pad_seq, mask


class TrainDataset(Dataset):
    def __init__(self, data_path, transform=dataProcessPipeline):
        df = pd.read_csv(data_path, header=0)

        self.df = df

        self.seqs = list(self.df["Sequence"])
        self.values = self.df["value"]

        self.values[self.values > MAX_MIC] = MAX_MIC
        self.values = list(self.values)
        self.transform = transform

    def __getitem__(self, idex):
        seq = self.seqs[idex]
        num_seq, mask = self.transform(seq)
        label = self.values[idex]

        return num_seq, mask, label

    def __len__(self):
        return len(self.seqs)


class TestDataset(Dataset):
    def __init__(self, data_path, transform=dataProcessPipeline):
        df = pd.read_csv(data_path, header=0)

        self.df = df

        self.seqs = self.df["Sequence"]
        self.transform = transform

    def __getitem__(self, idex):
        seq = self.seqs[idex]
        num_seq, mask = self.transform(seq)

        return num_seq, mask, seq

    def __len__(self):
        return len(self.seqs)


class FastaDataset(Dataset):
    def __init__(self, fasta_path, transform=dataProcessPipeline):
        self.seqs = [record.seq for record in SeqIO.parse(fasta_path, "fasta")]
        self.transform = transform

    def __getitem__(self, index):
        seq = str(self.seqs[index])
        num_seq, mask = self.transform(seq)
        return num_seq, mask, seq

    def __len__(self):
        return len(self.seqs)


class PositionalEncoding(nn.Module):
    def __init__(self, len, d_model=20, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(len, d_model)
        position = torch.arange(0, len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        # x = x + self.pe[:,:x.size(0), :]
        return x


class AttentionNetwork(nn.Module):

    def __init__(self, batch_size=128, embedding_size=20, num_tokens=100, num_classes=1, num_heads=4):
        super(AttentionNetwork, self).__init__()
        self.pe = PositionalEncoding(len=num_tokens, d_model=embedding_size)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout = 0.2
        self.relu = nn.ReLU()

        self.LN = nn.LayerNorm(normalized_shape=self.hidden1)
        self.fc1 = nn.Linear(self.embedding_size, self.hidden1)

        self.multihead_att = nn.MultiheadAttention(embed_dim=self.hidden1, num_heads=self.num_heads, batch_first=1, dropout=self.dropout)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(self.hidden1 * self.num_tokens, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.new_fc4 = nn.Linear(self.hidden3, self.num_classes)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask):
        x = self.pe(x)
        x = self.fc1(x)

        mask = mask.to(torch.bool)
        x, w1 = self.multihead_att.forward(x, x, x, key_padding_mask=mask)

        x = self.flatten(x)
        x = self.fc2(x)
        x = self.dropout(x)

        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.new_fc4(x)

        return x


trainData = TrainDataset(data_path=trainPath)
validateData = TrainDataset(data_path=validatePath)
testData = TestDataset(data_path=testPath)

train_loader = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=testData, batch_size=batch_size, shuffle=False)
validate_loader = DataLoader(dataset=validateData, batch_size=batch_size, shuffle=False)

testData1 = TestDataset(data_path=testPath)
test_loader1 = DataLoader(dataset=testData1, batch_size=batch_size, shuffle=0, num_workers=4)


frames = []

result_df = pd.read_csv(testPath, header=0)

model_out = {}
for modelName in nameList:

    modelPath = model_list[modelName]
    modelPreName = modelPath.split("/")[-1][:-4]
    id = modelPath.split("/")[-2]
    model_out[modelName] = []

    t_model = torch.load(modelPath)

    t_model.cuda()
    t_model.zero_grad()

    def test_eval(test_loader):
        t_model.eval()
        total_loss = []
        loss_function = nn.MSELoss()
        for i, data in enumerate(test_loader):
            inputs, masks, labels = data
            inputs = inputs.float()
            masks = masks.float()
            labels = labels.float()

            inputs = inputs.cuda()
            masks = masks.cuda()
            if modelName != "Attention":
                out = t_model(inputs)
            else:
                out = t_model(inputs, masks)
            out = torch.squeeze(out)

            out = out.cpu()
            loss = loss_function(out, labels)
            total_loss.append(loss.detach().numpy())

        ave = np.mean(total_loss)
        return ave

    loss0 = test_eval(validate_loader)
    print(modelName, " MSE loss in validation set:", str(loss0))

    for i, data in enumerate(test_loader1):
        inputs, masks, seqs = data
        inputs = inputs.float()
        masks = masks.float()

        t_model.eval()
        inputs = inputs.cuda()
        masks = masks.cuda()
        if modelName != "Attention":
            out = t_model(inputs)
        else:
            out = t_model(inputs, masks)
        out = out.cpu()
        out_ori = torch.squeeze(out)

        out_numpy = list(out_ori.detach().numpy())
        out_numpy = [round(v, 3) for v in out_numpy]
        model_out[modelName] = list(model_out[modelName]) + out_numpy

# summarize the results
for k, v in model_out.items():
    result_df[k] = v

result_df["Length"] = [len(v) for v in result_df["Sequence"]]
result_df = result_df[["Sequence", "Length", "CNN", "Transformer", "Attention", "LSTM"]]
df = result_df
y = result_df["CNN"] * weight["CNN"] + result_df["Transformer"] * weight["Transformer"] + result_df["Attention"] * weight["Attention"] + result_df["LSTM"] * weight["LSTM"]

df["Ensemble"] = [round(v, 3) for v in y]

df = df[["Sequence", "Ensemble", "CNN", "Transformer", "Attention", "LSTM"]]

print(df)

df.to_csv(savePath, index=0)
print(f"Regression test result is saved to ./{savePath} ")
