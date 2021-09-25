# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim
from torch.optim import lr_scheduler

import transformers
import pandas as pd
import numpy as np
import os
import random
import time
from tqdm import tqdm
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import datetime as dt
import copy


class Hparams:
    def __init__(self):
        self.random_seed = 2021
        self.data_dir = './data'
        self.output_dir = './outputs'
        self.batch_size = 256
        self.token_max_length = 256
        self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.num_epochs = 1
        self.class_1_weight = 150
        self.initial_lr = 2e-5  # 2e-5
        self.model_type = 'lstm'  # cnn, lstm
        self.upsample_pos_n = 1
        self.use_col = 'title_abstract'  # title, abstract, title_abstract
        self.train_argument = True
hps = Hparams()

def seed_torch(seed:int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
        

class BertCnnModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.conv1d_1 = nn.Conv1d(hidden_size, 256, kernel_size=2, padding=1)
        self.conv1d_2 = nn.Conv1d(256, 1, kernel_size=2, padding=1)
        self.linear = nn.Linear(258, 1)
    
    def forward(self, input_ids, attention_mask):
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out['last_hidden_state'].permute(0, 2, 1)
        conv_embed = torch.relu(self.conv1d_1(last_hidden_state))
        conv_embed = self.conv1d_2(conv_embed).squeeze()
        #out = self.linear(conv_embed).squeeze()
        logits = torch.sigmoid(self.linear(conv_embed)).squeeze()
        return logits

class BertLstmModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.regressor = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(outputs['last_hidden_state'], None)
        sequence_output = out[:, -1, :]
        logits = torch.sigmoid(torch.flatten(self.regressor(sequence_output)))
        return logits

class ModelCheckpoint:
    def __init__(self, save_dir:str, model_name:str):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.model_name = model_name
        jst = dt.timezone(dt.timedelta(hours=+9), 'JST')
        dt_now = dt.datetime.now(jst)
        self.dt_now_str = dt_now.strftime('%Y%m%d_%H%M')
        self.best_loss = self.best_acc = self.best_fbeta_score = 0.0
        self.best_epoch = 0

    def get_checkpoint_name(self, epoch):
        checkpoint_name = f"{self.model_name.replace('/', '_')}__epoch{epoch:03}__{self.dt_now_str}.pth"
        checkpoint_name = os.path.join(self.save_dir, checkpoint_name)
        return checkpoint_name

    def save_checkpoint(self, model, epoch):
        torch.save(model.state_dict(), self.get_checkpoint_name(epoch))

    def load_checkpoint(self, model=None, epoch=1, manual_name=None):
        if manual_name is None:
            checkpoint_name = self.get_checkpoint_name(epoch)
        else:
            checkpoint_name = manual_name
        print(checkpoint_name)
        model.load_state_dict(torch.load(checkpoint_name))
        return model

def fit(dataloaders, model, optimizer, num_epochs, device, batch_size, lr_scheduler):

    print(f"class 1 weight : {hps.class_1_weight}")

    checkpoint = ModelCheckpoint(save_dir='model_weights', model_name=hps.model_name)
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Using device : {device}")
    for epoch in range(num_epochs):
        print(f"【Epoch {epoch+1: 3}/{num_epochs: 3}】   LR -> ", end='')
        for i, params in enumerate(optimizer.param_groups):
            print(f"Group{i}: {params['lr']:.7f}", end=' / ')
        print('')

        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            running_fbeta_score = 0.0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.where(outputs >= 0.5, 1, 0)
                    pos_weight = torch.tensor([hps.class_1_weight for i in range(input_ids.size(0))]).to(device)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(preds == labels)
                running_fbeta_score += fbeta_score(labels.to('cpu').detach().numpy(), preds.to('cpu').detach().numpy(), beta=7.0, zero_division=0)                    

                if phase == 'train':
                    if i % 20 == 19:
                        total_num = float((i * batch_size) + input_ids.size(0))
                        print(f"{i+1: 4}/{len(dataloaders[phase]): 4}  <{phase}> Loss:{(running_loss/total_num):.4f}  Acc:{(running_corrects/total_num):.4f}  fbScore:{(running_fbeta_score/(i+1)):.4f}")

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_fbscore = running_fbeta_score / len(dataloaders[phase])
            
            print(f"<{phase}> Loss:{epoch_loss:.4f}  Acc:{epoch_acc:.4f}  fbScore:{epoch_fbscore:.4f}")

            if phase == 'val' and epoch_fbscore > checkpoint.best_fbeta_score:
                print(f"Checkpoints have been updated to the epoch {epoch+1} weights.")
                checkpoint.best_loss = epoch_loss
                checkpoint.best_acc = epoch_acc
                checkpoint.best_fbeta_score = epoch_fbscore
                checkpoint.best_epoch = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())

        lr_scheduler.step()
        print('-' * 150)

    model.load_state_dict(best_model_wts)
    checkpoint.save_checkpoint(model, epoch)
    torch.cuda.empty_cache()

    return model

def inference(model, dataloader, device):

    print(f"class 1 weight : {hps.class_1_weight}")
    
    running_loss = 0.0
    running_corrects = 0
    running_fbeta_score = 0.0

    preds_labels_dict = dict(preds = np.empty(0), labels = np.empty(0))

    for i, (inputs, labels) in enumerate(tqdm(dataloader)):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.where(outputs >= 0.5, 1, 0)
            pos_weight = torch.tensor([hps.class_1_weight for i in range(input_ids.size(0))]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(outputs, labels)

            running_loss += loss.item() + input_ids.size(0)
            running_corrects += torch.sum(preds == labels)
            running_fbeta_score += fbeta_score(labels.to('cpu').detach().numpy(), preds.to('cpu').detach().numpy(), beta=7.0, zero_division=0)   
            preds_labels_dict['preds']  = np.hstack([preds_labels_dict['preds'], preds.to('cpu').detach().numpy().copy()])
            preds_labels_dict['labels']  = np.hstack([preds_labels_dict['labels'], labels.to('cpu').detach().numpy().copy()])

    loss = running_loss / len(dataloader)
    acc = running_corrects / len(dataloader.dataset)
    fbscore = running_fbeta_score / len(dataloader)
    print(f"Loss:{loss:.4f}  Acc:{acc:.4f}  fbScore:{fbscore:.4f}")
    return preds_labels_dict



def main():

    for w in range(100, 201, 5):

        os.environ["CUDA_VISIBLE_DEVICES"]='0,1,3,5'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")


        seed_torch(hps.random_seed)

        orig_df = pd.read_csv(os.path.join(hps.data_dir, 'train.csv'), index_col=0)
        submit_df = pd.read_csv(os.path.join(hps.data_dir, 'test.csv'), index_col=0)
        sample_submit_df = pd.read_csv(os.path.join(hps.data_dir, 'sample_submit.csv'), index_col=0)


        orig_df.loc[2488, 'judgement'] = 0
        orig_df.loc[7708, 'judgement'] = 0


        orig_df['abstract'].fillna('', inplace=True)
        orig_df['title_abstract'] = orig_df.title + orig_df.abstract



        train_df, test_df = train_test_split(orig_df, test_size=0.2, random_state=hps.random_seed, shuffle=True, stratify=orig_df.judgement)
        train_df, valid_df = train_test_split(train_df, test_size=0.25, random_state=hps.random_seed, shuffle=True, stratify=train_df.judgement)
        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        print(f"Train  ->  label_1:{train_df.judgement.sum()} / all:{train_df.judgement.count()}   ({train_df.judgement.sum() / train_df.judgement.count() * 100:.3f}%)")
        print(f"Valid  ->  label_1:{valid_df.judgement.sum()} / all:{valid_df.judgement.count()}   ({valid_df.judgement.sum() / valid_df.judgement.count() * 100:.3f}%)")
        print(f"Test   ->  label_1:{test_df.judgement.sum()} / all:{test_df.judgement.count()}   ({test_df.judgement.sum() / test_df.judgement.count() * 100:.3f}%)")


        base_tokenizer = transformers.AutoTokenizer.from_pretrained(hps.model_name)
        base_model = transformers.AutoModel.from_pretrained(hps.model_name)
        base_model_config = transformers.AutoConfig.from_pretrained(hps.model_name)


        phase_param = {
            "df":{'train': train_df, 'val': valid_df, 'test': test_df},
            "argument":{'train': hps.train_argument, 'val': False, 'test': False},
            "batch_size":{'train':hps.batch_size, 'val':hps.batch_size*2, 'test':hps.batch_size*2},
            "shuffle":{'train': True, 'val': False, 'test': False},
            "upsample_pos_n":{'train': hps.upsample_pos_n, 'val': 1, 'test': 1},
        }


        datasets = {phase:TextClassificationDataset(df=phase_param['df'][phase], tokenizer=base_tokenizer, use_col=hps.use_col, \
                                                    token_max_length=hps.token_max_length, argument=phase_param['argument'][phase], \
                                                    upsample_pos_n=phase_param['upsample_pos_n'][phase]) for phase in ['train', 'val', 'test']}
        dataloaders = {phase: DataLoader(datasets[phase], batch_size=phase_param['batch_size'][phase], \
                                        shuffle=phase_param['shuffle'][phase]) for phase in ['train', 'val', 'test']}
        print(len(datasets['train']), len(datasets['val']), len(datasets['test']))
        print(len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['test']))


        if hps.model_type == 'cnn':
            print(f"Choosed BertLstmModel")
            model = BertLstmModel(base_model=base_model, hidden_size=base_model_config.hidden_size)
        elif hps.model_type == 'lstm':
            print(f"Choosed BertLstmModel")
            model = BertLstmModel(base_model=base_model, hidden_size=base_model_config.hidden_size)


        model = model.to(device)
        optimizer = optim.AdamW(
            params=[
                {'params': model.base_model.parameters(), 'lr': 2e-5},
                {'params': model.lstm.parameters(), 'lr': 2e-4},
                {'params': model.regressor.parameters(), 'lr': 2e-4}
            ]
        )
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        device_num = torch.cuda.device_count()
        if device_num > 1:
            print(f"Use {device_num} GPUs")
            model = nn.DataParallel(model)


    
        
        hps.class_1_weight = w
        print(f"Set class 1 weight : {hps.class_1_weight}")

        model = fit(dataloaders=dataloaders, model=model,
                optimizer=optimizer, num_epochs=hps.num_epochs, device=device, batch_size=hps.batch_size, lr_scheduler=exp_lr_scheduler)

        preds_labels_dict = inference(model, dataloader=dataloaders['test'], device=device)

        fb_score = fbeta_score(y_true=preds_labels_dict['labels'], y_pred=preds_labels_dict['preds'], beta=7.0)
        
        del model
        torch.cuda.empty_cache()


        with open('logs/class_1_weight.csv', 'a') as f:
            output_txt = f"{w},{fb_score}\n"
            f.write(output_txt)

if __name__ == '__main__':
    main()
