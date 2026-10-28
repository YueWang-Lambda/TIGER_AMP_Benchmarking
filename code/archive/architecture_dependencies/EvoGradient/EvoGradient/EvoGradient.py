import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
import warnings
import argparse

warnings.filterwarnings("ignore")

# Set up argument parser
parser = argparse.ArgumentParser(description="EvoGradient")
parser.add_argument("--peptide", type=str, required=True, help="Peptide to optimize")
args = parser.parse_args()
to_opt = args.peptide


# Dictionary to map amino acids to numerical values
mydict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}
# Inverse dictionary to map numerical values back to amino acids
myInvDict = dict([val, key] for key, val in mydict.items())
MAX_MIC = math.log10(8192)
max_mic_buffer = 0.1
My_MAX_MIC = math.log10(600)


def num2seq(narr, len):
    """
    Convert a numerical array to a sequence of amino acids.
    
    Parameters:
    narr (numpy array): Array of numerical values representing amino acids.
    len (int): Length of the sequence to return.
    
    Returns:
    list: Sequence of amino acids.
    """
    numlist = np.argmax(narr, axis=1)
    seq = [myInvDict[value] for value in numlist]
    seq = seq[:len]
    return seq


def colorstr(*input):
    """
    Colors a string using ANSI escape codes.
    
    Parameters:
    *input (str): Colors and the string to color.
    
    Returns:
    str: Colored string.
    """
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def standout(seq1, seq2):
    """
    Compare two sequences and highlight differences.
    
    Parameters:
    seq1 (str): First sequence.
    seq2 (str): Second sequence.
    
    Returns:
    list: Second sequence with differences highlighted.
    """
    # compare bewteen two seqs
    index = [1 if seq1[j] == seq2[j] else 0 for j in range(len(seq1))]
    newSeq2 = list(seq2)
    for i in range(len(seq1)):
        if index[i] == 0:
            newSeq2[i] = colorstr("blue", seq2[i])

    newSeq2 = "".join(newSeq2)

    return seq1, newSeq2


def colorPrint(ls):
    """
    Print the list with colored strings.
    
    Parameters:
    ls (list): List of sequences to print.
    """
    newls = [ls[0]]
    for i in range(len(ls) - 1):
        s1, s2 = standout(ls[i], ls[i + 1])
        newls.append(s2)

    for i in newls:
        print(i)


def colorShow(ls):
    """
    Highlight the changes with colored strings.
    
    Parameters:
    ls (list): List of sequences to compare and highlight.
    """
    length = len(ls[0])
    colorls = [ls[0]]
    flag = [0 for i in range(length)]  # record if the position was changed
    for i in range(len(ls) - 1):
        index = [1 if ls[i][j] == ls[i + 1][j] else 0 for j in range(length)]
        for k in range(length):
            if index[k] == 0:
                flag[k] = 1

        colorSeq = list(ls[i + 1])
        for k in range(length):
            if flag[k] == 1:
                colorSeq[k] = colorstr("blue", colorSeq[k])

        colorSeq = "".join(colorSeq)
        colorls.append(colorSeq)

    for seq in colorls:
        print(seq)


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
    pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 100 - len))
    mask = np.zeros(100, dtype=int)
    mask[len:] = 1
    mask = torch.tensor(mask)

    pad_seq = pad(onehotSeq)

    return pad_seq, mask


def num2onehot(array2d):
    """
    Convert a numerical array to a one-hot encoded tensor.
    
    Parameters:
    array2d (torch.Tensor): The input numerical array.
    
    Returns:
    torch.Tensor: The one-hot encoded tensor.
    """
    result = torch.zeros_like(array2d)
    index = torch.argmax(array2d, dim=-1)
    for i in range(index.shape[0]):
        result[i, index[i]] = 1

    return result


# Define the train dataset class
class TrainDataset(Dataset):
    def __init__(self, data_path, transform=dataProcessPipeline):
        """
        Initialize the dataset.
        
        Parameters:
        data_path (str): Path to the CSV file containing the data.
        transform (function): Function to process the sequences.
        """
        df = pd.read_csv(data_path, header=0)
        df = df[df["Length"] <= 100]
        self.df = df
        self.seqs = list(self.df["Sequence"])
        self.values = self.df["value"]
        self.values[self.values > MAX_MIC] = MAX_MIC
        self.values = list(self.values)

        self.transform = transform

    def __getitem__(self, idex):
        """
        Get an item from the dataset.
        
        Parameters:
        idex (int): Index of the item to retrieve.
        
        Returns:
        tuple: A tuple containing the processed sequence, mask, label, and original sequence.
        """
        seq = self.seqs[idex]
        num_seq, mask = self.transform(seq)
        label = self.values[idex]

        return num_seq, mask, label, seq

    def __len__(self):
        return len(self.seqs)


# Define the test dataset class
class TestDataset(Dataset):
    def __init__(self, data_path, transform=dataProcessPipeline):
        self.df = pd.read_csv(data_path, header=0)
        self.seqs = self.df["Sequence"]

        self.transform = transform

    def __getitem__(self, idex):
        seq = self.seqs[idex]
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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        return x


pe = PositionalEncoding(len=100, d_model=20)


class AttentionNetwork(nn.Module):

    def __init__(self, batch_size=128, embedding_size=20, num_tokens=100, num_classes=1, num_heads=4):
        super(AttentionNetwork, self).__init__()
        self.pe = PositionalEncoding(len=num_tokens, d_model=embedding_size)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        # self.hidden1 = 20
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
        self.softmax = nn.functional.softmax

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

# use attention model to optimize the sequence for demostration
model_list = {
    "Attention": "../model/regression/Attention.pth",
}

# models used to evaluate the optimized sequences
test_model_list = {"CNN": "../model/regression/CNN.pth", "Transformer": "../model/regression/Transformer.pth", "Attention": "../model/regression/Attention.pth", "LSTM": "../model/regression/LSTM.pth"}

opt_seqls = [to_opt]

# Set the number of iterations and learning rate for each model
iter_dict = {"CNN": 500, "Transformer": 500, "Attention": 500, "RCNN": 500}
lr_dict = {"CNN": 0.01, "Transformer": 0.0005, "Attention": 0.005, "RCNN": 0.001}

for seq in opt_seqls:
    tseq = seq
    ModelNameList = model_list.keys()

    oriseq = tseq

    # Create a DataFrame to store the sequences, their lengths, and labels
    df = pd.DataFrame(columns=["Sequence", "Length", "label"])
    items = [{"Sequence": oriseq, "Length": len(oriseq)}]
    
    # use _append or append for different pandas version
    # df = df.append(items, ignore_index=1) 
    df = df._append(items, ignore_index=1) 
    
    df.to_csv("./EvoResult/" + oriseq + ".csv", index=False)
    SeqPath = "./EvoResult/" + oriseq + ".csv"

    testData1 = TrainDataset(data_path=r"../data/regression/test.csv")
    test_loader1 = DataLoader(dataset=testData1, batch_size=4, drop_last=True)

    # Iterate over the models
    for modelName in ModelNameList:
        alpha = lr_dict[modelName]
        iters = iter_dict[modelName]

        iternum = 0

        testData = TestDataset(data_path=SeqPath)
        test_loader = DataLoader(dataset=testData, batch_size=1)
        t_model = torch.load(model_list[modelName])

        t_model.cuda()
        t_model.zero_grad()

        print("Using", modelName, " to optimize this sequence:")
        flag = 1

        for data in test_loader:
            resultList = []
            # ensamble_values = []
            resultSeq = [oriseq]
            outMIC = []
            # t_model.zero_grad()
            inputs, masks, seqs = data

            inputs = inputs.float()
            masks = masks.float()

            inputs = inputs.cuda()
            inputs.requires_grad = True
            masks = masks.cuda()
            print(seqs[0])

            t_model.eval()
            for iter in range(iters):
                t_model.zero_grad()
                inputs.retain_grad = True

                if modelName != "Attention":
                    out = t_model(inputs)
                else:
                    out = t_model(inputs, masks)

                # out = torch.squeeze(out)
                out = out.cpu()
                conloss = out
                conloss.backward()
                grad = inputs.grad

                colindex = masks[0] == 1
                grad[0][masks[0] == 1] = 0
                mylen = 100 - colindex.sum()

                ori_onehot = num2onehot(inputs[0].cpu())
                result = inputs[0] - alpha * grad[0]
                result[mylen:, :] = 0
                tempt_onehot = num2onehot(result.cpu())

                if (tempt_onehot == ori_onehot).all():  # if no AAs (after projection) was changed
                    flag = 0
                else:
                    result = tempt_onehot
                    flag = 1
                with torch.no_grad():
                    inputs[0] = result

                result = result.cpu().detach().numpy()
                seq = num2seq(result, len=mylen)
                seq = "".join(seq)
                if flag == 1:
                    resultSeq.append(seq)

        print()
        colorShow(resultSeq)
        print()

        optSeqDir = "./EvoResult/" + oriseq
        if not os.path.exists(optSeqDir):
            os.makedirs(optSeqDir)
        optSeqSavePath = optSeqDir + "/" + modelName + ".csv"

        result_df = pd.DataFrame(columns=["Sequence", "label", "Length"])
        items = []
        for seq in resultSeq:
            item = {"Sequence": seq, "Length": len(seq)}
            items.append(item)

        # use _append or append for different pandas version
        # result_df = result_df.append(items)
        result_df = result_df._append(items)
    
        result_df.to_csv(optSeqSavePath)

        result_soli = []

        testModelNameList = test_model_list.keys()

        preList = {}

        mylen = result_df.shape[0]
        numslist = []

        model_out = {}
        for testModelName in testModelNameList:
            ls = []
            model_out[testModelName] = []
            testData = TestDataset(data_path=optSeqSavePath)
            test_loader = DataLoader(dataset=testData, batch_size=64)

            testData1 = TrainDataset(data_path=r"../data/regression/test.csv")
            test_loader1 = DataLoader(dataset=testData1, batch_size=4, drop_last=True)

            t_model = torch.load(test_model_list[testModelName])

            t_model.cuda()
            t_model.zero_grad()
            t_model.eval()

            for data in test_loader:
                resultList = []
                t_model.zero_grad()
                inputs, masks, seqs = data
                inputs = inputs.float()
                masks = masks.float()

                inputs = inputs.cuda()
                masks = masks.cuda()
                t_model.zero_grad()

                if testModelName != "Attention":
                    out = t_model(inputs)
                else:
                    out = t_model(inputs, masks)

                out = out.cpu()
                if len(out.shape) > 0:
                    out_ori = torch.squeeze(out)
                else:
                    out_ori = out.unsqueeze(0)

                out_numpy = list(out_ori.detach().numpy())
                out_numpy = [round(v, 3) for v in out_numpy]
                model_out[testModelName] = list(model_out[testModelName]) + out_numpy

        # summarize the results
        for k, v in model_out.items():
            result_df[k] = v
        resultPath = optSeqSavePath[:-4] + "_result.csv"

        ensamble_values = [(result_df["Attention"][k] + result_df["Transformer"][k] + result_df["CNN"][k] + result_df["LSTM"][k]) / 4 for k in range(result_df.shape[0])]
        ensamble_values = [round(v, 3) for v in ensamble_values]
        result_df["Ensemble"] = ensamble_values

        result_df = result_df[["Sequence", "Ensemble", "Length", "CNN", "Transformer", "Attention", "LSTM"]]

        result_df.to_csv(resultPath, index=0)
        print(f"Optimization result is saved to ./{resultPath} ")
