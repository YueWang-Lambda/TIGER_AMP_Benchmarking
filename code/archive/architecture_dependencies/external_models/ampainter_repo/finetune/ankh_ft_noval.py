import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import pathlib
import random
import ankh
import pandas as pd
from tqdm.auto import tqdm
from modlamp.core import read_fasta
from torch.utils.data import Dataset, DataLoader

seed = 1234
torch.manual_seed(seed)
random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Available device:', device)

# hyperparameters
learning_rate = 3e-4
weight_decay = 5e-4
batch_size = 16
epoches = 1

# To load model:
model, tokenizer = ankh.load_base_model(generation=True)
model.to(device=device)

# Load fine-tune datasets
masking_ratio = 0.2  
max_length = 101
num_beams = 10
temperature = 2.0
df_all = pd.read_csv('./data/maskseq1by1_' + str(masking_ratio) + '.csv')
df = df_all[['source', 'target']]

class load_data(Dataset):

    def __init__(self, dataframe, tokenizer, max_length, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus([source_text], add_special_tokens = True, max_length = self.max_length, padding = 'max_length', return_tensors = 'pt')
        target = self.tokenizer.batch_encode_plus([target_text], add_special_tokens = True, max_length = self.max_length, padding = 'max_length', return_tensors = 'pt')
      
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

train_set = load_data(df, tokenizer, max_length, 'source', 'target')
train_dataloader = DataLoader(train_set, shuffle = True, batch_size = batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
# training
model.train()
for epoch in range(epoches):
    loss_record = []
    for idx, data in enumerate(tqdm(train_dataloader)):
        y_ids = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y_ids.contiguous()
        y_ids = y_ids.clone().detach()
        y_ids[y_ids == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        # outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids = y_ids, labels = lm_labels)
        outputs = model(input_ids = ids, attention_mask = mask, labels = y_ids)
        loss = outputs.loss
        loss_record.append(loss.item())

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(f'\nEpoch {epoch} loss in iteration {idx}: {loss}')
    with open('loss_record.txt','a') as f:
        f.writelines(str(loss_record) + '\n')

# save model after fine-tuning
output_dir = './model_1by1/'
name = 'lr' + str(learning_rate) + '_ba' + str(batch_size) + '_epo' + str(epoches) + '_temp' + str(temperature) + '_beam' + str(num_beams)
print('save model: ', name)
path = os.path.join(output_dir, name)
model.save_pretrained(path)
tokenizer.save_pretrained(path)
