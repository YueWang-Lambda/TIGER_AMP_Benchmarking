import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
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
trainPath = "./data/classification/train.csv"
validatePath = "./data/classification/test.csv"

# Configuration parameters
batch_size = 256
embedding_size = 20
num_tokens = 100
num_classes = 2
num_heads = 4

# Model paths
model_list = {
    "CNN": "./model/classification/CNN.pth",
    "Transformer": "./model/classification/Transformer.pth",
    "Attention": "./model/classification/Attention.pth",
    "LSTM": "./model/classification/LSTM.pth",
}
nameList = model_list.keys()

# Sequence to numerical mapping
mydict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}

softmax = nn.functional.softmax


def fasta_to_csv(fasta_path, csv_path):
    """
    Convert a FASTA file to a CSV file.

    Parameters:
    fasta_path (str): Path to the input FASTA file.
    csv_path (str): Path to the output CSV file.

    Returns:
    str: Path to the output CSV file.
    """
    sequences = []
    lengths = []

    # Parse the FASTA file and extract sequences and their lengths
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        lengths.append(len(record.seq))

    # Create a DataFrame with sequences and their lengths
    df = pd.DataFrame({"Sequence": sequences, "Length": lengths})

    # Save the DataFrame to a CSV file
    print(csv_path)
    df.to_csv(csv_path, index=False)
    return csv_path


# Transform the test FASTA file to CSV
testPath = fasta_to_csv(testPath, testPath[:-5] + ".csv")


def dataProcessPipeline(seq):
    """
    Process a sequence into a padded one-hot encoded tensor and a mask.

    Parameters:
    seq (str): The input sequence to process.

    Returns:
    tuple: A tuple containing the padded one-hot encoded tensor and the mask tensor.
    """
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
    # Pad the sequence to a length of 100
    pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 100 - len))
    mask = np.zeros(100, dtype=int)
    mask[len:] = 1
    mask = torch.tensor(mask)
    pad_seq = pad(onehotSeq)

    return pad_seq, mask


# train dataset
class TrainDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path, header=0)
        df = df[df["Length"] <= 100]
        self.seqs = list(df["Sequence"])
        self.labels = list(df["label"])

    def __getitem__(self, index):
        seq = self.seqs[index]
        num_seq, mask = dataProcessPipeline(seq)
        label = self.labels[index]
        return num_seq, mask, label

    def __len__(self):
        return len(self.seqs)


# test dataset
class TestDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path, header=0).reset_index()
        self.seqs = df["Sequence"]

    def __getitem__(self, index):
        seq = self.seqs[index]
        num_seq, mask = dataProcessPipeline(seq)
        return num_seq, mask, seq

    def __len__(self):
        return len(self.seqs)


class FastaDataset(Dataset):
    def __init__(self, data_path, transform=dataProcessPipeline):
        """
        Initialize the dataset from a FASTA file.

        Parameters:
        data_path (str): Path to the FASTA file.
        transform (function): Function to process the sequences.
        """
        self.seqs = [record.seq for record in SeqIO.parse(data_path, "fasta")]
        self.transform = transform

    def __getitem__(self, index):
        seq = str(self.seqs[index])
        num_seq, mask = self.transform(seq)
        return num_seq, mask, seq

    def __len__(self):
        return len(self.seqs)


class PositionalEncoding(nn.Module):
    def __init__(self, length, d_model=20):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe
        return x


"""attention model"""


class AttentionNetwork(nn.Module):

    def __init__(self, batch_size=128, embedding_size=20, num_tokens=100, num_classes=2, num_heads=4):

        super(AttentionNetwork, self).__init__()
        self.pe = PositionalEncoding(len=num_tokens, d_model=embedding_size)

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout = 0.5

        self.relu = nn.ReLU()
        self.LN = nn.LayerNorm(normalized_shape=self.hidden1)
        self.fc1 = nn.Linear(self.embedding_size, self.hidden1)
        self.multihead_att = nn.MultiheadAttention(embed_dim=self.hidden1, num_heads=self.num_heads, batch_first=1, dropout=self.dropout)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(self.hidden1 * self.num_tokens, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.fc4 = nn.Linear(self.hidden3, self.num_classes)
        self.dropout = nn.Dropout(self.dropout)
        self.softmax = nn.functional.softmax

    def forward(self, x, mask):
        x = self.pe(x)
        x = self.fc1(x)

        mask = mask.to(torch.bool)
        x, _ = self.multihead_att.forward(x, x, x, key_padding_mask=mask)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.dropout(x)

        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)

        return x


trainData = TrainDataset(data_path=trainPath)
validateData = TrainDataset(data_path=validatePath)
testData = TestDataset(data_path=testPath)

train_loader = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=testData, batch_size=batch_size, shuffle=False)
validate_loader = DataLoader(dataset=validateData, batch_size=batch_size, shuffle=False)

loss_function = nn.MSELoss()

result_df = pd.read_csv(testPath, header=0)

model_out = {}
# process with all the models
for modelName in nameList:
    modelPath = model_list[modelName]
    id = modelPath.split("/")[-2]
    model_out[modelName] = []

    t_model = torch.load(modelPath)
    t_model.cuda()

    # evaluate models
    def score(test_loader):
        t_model.eval()
        epi = 0.000001
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        total = 0
        count = 0

        for data in test_loader:
            inputs, masks, labels = data
            inputs = inputs.float()
            masks = masks.float()
            inputs, masks, labels = Variable(inputs), Variable(masks), Variable(labels)

            inputs = inputs.cuda()
            masks = masks.cuda()

            if modelName != "Attention" and modelName != "Transformer2":
                out = t_model(inputs)
            else:
                out = t_model(inputs, masks)
            out = torch.squeeze(out)

            out = torch.argmax(out, -1)
            out = out.cpu()
            for i, pre in enumerate(out):
                total += 1
                if pre == labels[i]:
                    count += 1
                    if pre == 0:
                        tn += 1
                    else:
                        tp += 1
                if pre != labels[i]:
                    if pre == 0:
                        fn += 1
                    else:
                        fp += 1

        print("AMP classification result:")
        print("Precision:", np.round(tp / (tp + fp + epi), 3))
        print("Recall:", np.round(tp / (tp + fn + epi), 3))
        print("Specificity:", np.round(tn / (tn + fp + epi), 3))
        print("F1:", np.round(2 * tp / (2 * tp + fp + fn + epi), 3))
        print("Accuracy：", np.round(count / total, 3))
        print()

    print()
    print("Model:", modelName)
    score(validate_loader)

    # use model to predict test data
    for i, data in enumerate(test_loader):
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
        if "LSTM" in modelName:
            out = out.unsqueeze(0)
        out_ori = torch.squeeze(out)

        out_ori = torch.squeeze(out)
        out_soft = softmax(out_ori, -1)
        out_soft_AMP = out_soft[:, 1]

        out_soft_numpy = list(out_soft_AMP.detach().numpy())
        out_soft_numpy = [round(v, 3) for v in out_soft_numpy]
        model_out[modelName] = list(model_out[modelName]) + out_soft_numpy

# summarize the results
for k, v in model_out.items():
    result_df[k] = v

result_df = result_df[["Sequence", "CNN", "Transformer", "Attention", "LSTM"]]

y = (result_df["CNN"] > 0.5) * (result_df["Transformer"] > 0.5) * (result_df["LSTM"] > 0.5) * (result_df["Attention"] > 0.5)
result_df["Ensemble"] = y

result_df.to_csv(savePath, index=0)
print(result_df)
print(f"Test result is saved to ./{savePath} ")
