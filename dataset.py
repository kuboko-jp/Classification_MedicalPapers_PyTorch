import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import random


class TextClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, use_col='title_abstract', token_max_length=512, argument=False, upsample_pos_n=1):

        if upsample_pos_n > 1:
            df_pos = df.loc[df.judgement==1]
            df_pos = pd.concat([df_pos for i in range(int(upsample_pos_n))], axis=0).reset_index(drop=True)
            df_neg = df.loc[df.judgement==0]
            self.df = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)
        else:
            self.df = df
        
        self.tokenizer = tokenizer
        self.argument = argument
        self.use_col = use_col

    def text_argument(self, text, drop_min_seq=3, seq_sort=True):
        seq_list = text.split('. ')
        seq_len = len(seq_list)
        if seq_len >= drop_min_seq:
            orig_idx_list = list(range(0, seq_len))
            idx_list = random.sample(orig_idx_list, random.randint(round(seq_len * 0.7), seq_len))
            if seq_sort:
                idx_list = sorted(idx_list)
            insert_idx_list = random.sample(orig_idx_list, random.randint(0, seq_len//3))
            for x in insert_idx_list:
                idx = random.randint(0, len(idx_list))
                idx_list.insert(idx, x)
            seq_list = [seq_list[i] for i in idx_list]
        text = '. '.join(seq_list)
        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        text = self.df.loc[idx, self.use_col]

        if self.argument:
            text = self.text_argument(text, drop_min_seq=3, seq_sort=True)

        token = self.tokenizer.encode_plus(
            text,
            padding = 'max_length', max_length = hps.token_max_length, truncation = True,
            return_attention_mask=True, return_tensors='pt'
        )

        sample = dict(
            input_ids=token['input_ids'][0],
            attention_mask=token['attention_mask'][0]
        )
        
        label = torch.tensor(self.df.loc[idx, 'judgement'], dtype=torch.float32)
        return sample, label
        