import os
import sys
# coding=utf-8
from PD_data_process import EEProcessor
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# from sklearn import model_selection
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchcrf import CRF
import csv
from conlleval import PD_acc_evaluate
from transformers import (
    BertTokenizer, BertModel,
    get_linear_schedule_with_warmup
)
import numpy as np
import re
import json
#from pytorch_lightning.metrics import Accuracy
# import nltk
from torchmetrics import BLEUScore 



class EEModel(pl.LightningModule):
    def __init__(self, config, k):
        # 1. Init parameters
        super(EEModel, self).__init__()
        
        self.config=config

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.crf_lr = config.crf_lr
        self.dropout = config.dropout
        self.optimizer = config.optimizer
        self.num_labels = 5
        self.hidden_size = 768
        self.threshold = 0.5

        self.use_bart = config.use_bart
        self.use_crf = config.use_crf
        
        self.addition_vocab_path = config.addition_vocab_path
        with open(self.addition_vocab_path, 'r') as f:
            addition_vocab = json.load(f)
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)  # "hfl/chinese-roberta-wwm-ext"

        self.processor = EEProcessor(config, k, self.tokenizer)
        self.model = BertModel.from_pretrained(config.pretrained_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print('num tokens:', len(self.tokenizer))

        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        # 2. Init crf model and loss
        self.criterion = nn.BCELoss()
        # self.criterion = nn.NLLLoss(ignore_index=0, size_average=True)
        
        print("EE model init: done.")
    
    def prepare_data(self):
        [train_data,train_label] = self.processor.get_train_data()
        [dev_data,dev_label]=self.processor.get_dev_data()
        # if self.config.train_num>0:
        #     train_data=train_data[:self.config.train_num]
        # if self.config.dev_num>0:
        #     dev_data=dev_data[:self.config.dev_num]

        # print("train_length:", len(train_data))
        # print("valid_length:", len(dev_data))

        if self.use_bart:
            self.train_loader = self.processor.create_dataloader(
                1, train_data,train_label,batch_size=self.batch_size, shuffle=False)
            self.valid_loader = self.processor.create_dataloader(
                2, dev_data,dev_label,batch_size=self.batch_size, shuffle=False)
            # self.test_loader = self.processor.create_dataloader(0, dev_data,dev_label,batch_size=self.batch_size, shuffle=False)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids = None, decoder_attention_mask = None,return_dict = None, labels = None):
        # 训练时模型会自动从labels参数右移得到decoder_input_ids 
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]  # [bs,len,numlabels]
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        logits = self.sigmoid(logits)

        return logits

    def training_step(self, batch, batch_idx):
        if self.use_bart:
            input_ids, attention_mask, labels = batch
            outputs = self(input_ids = input_ids,attention_mask=attention_mask,labels = labels)

        loss = self.criterion(outputs, labels)

        self.log('train_loss', loss.item())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        
        if self.use_bart:
            input_ids, attention_mask, labels = batch
            outputs = self(input_ids = input_ids,attention_mask=attention_mask)

        return labels.cpu().detach().numpy(),outputs.cpu().detach().numpy()

    def validation_epoch_end(self, outputs):

        gold,pre=zip(*outputs)
        golds = []
        for batch in gold:
            for idx in batch:
                golds.append(idx)
        pres = []
        for batch in pre:
            for idx in batch:
                pres.append(idx)
        #筛选出大于阈值的
        for i in range(len(pres)):
            for j in range(len(pres[i])):
                if pres[i][j] > self.threshold:
                    pres[i][j] = 1
                else:
                    pres[i][j] = 0

        acc = PD_acc_evaluate(pres, golds)
        print("acc:",acc)
        print("avg_acc:", sum(acc)/len(acc))
        self.log('avg_acc', sum(acc)/len(acc))
        #print("prediction:", pre[0:10])

    def configure_optimizers(self):
        if self.use_crf:
            crf_params_ids = list(map(id, self.crf.parameters()))
            base_params = filter(lambda p: id(p) not in crf_params_ids, [
                                 p for p in self.parameters() if p.requires_grad])

            arg_list = [{'params': base_params}, {'params': self.crf.parameters(), 'lr': self.crf_lr}]
        else:
            # label_embed_and_attention_params = list(map(id, self.label_embedding.parameters())) + list(map(id, self.self_attention.parameters()))
            # arg_list = [{'params': list(self.label_embedding.parameters()) + list(self.self_attention.parameters()), 'lr': self.lr}]
            arg_list = [p for p in self.parameters() if p.requires_grad]

        print("Num parameters:", len(arg_list))
        if self.optimizer == 'Adam':
            return torch.optim.AdamW(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
    
    def test_dataloader(self):
        return self.test_loader


class EEPredictor:
    def __init__(self, checkpoint_path, config, k):
        self.use_bart = config.use_bart
        self.use_crf = config.use_crf

        self.model = EEModel.load_from_checkpoint(checkpoint_path, config=config, k=k)

        self.test_data = self.model.processor.get_test_data()
        if config.use_bart:
            self.tokenizer = self.model.tokenizer
            self.dataloader = self.model.processor.create_dataloader(
                0, self.test_data[0],self.test_data[1], batch_size=config.batch_size, shuffle=False)


        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

    def generate_sample(self):
        pred = self.tokenizer.batch_decode(self.model.generate(input_ids=self.test_data,num_return_sequences=1))
    def generate_result(self, outfile_txt):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        with open(outfile_txt, 'w') as fout:
            for batch in tqdm.tqdm(self.dataloader):
                for i in range(len(batch)):
                    batch[i] = batch[i].to(device)
                if self.use_bart:
                    input_ids, attention_mask,labels,label_attention_mask = batch

                feats = self.model.model.generate(input_ids,attention_mask = attention_mask)
                preds = self.tokenizer.batch_decode(feats, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for pred in preds:
                    fout.write(json.dumps(pred,ensure_ascii=False)+"\n")

        print('done--all tokens.')
    
    def generate_points(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        golds,pres = [],[]
        threshold = 0.5
        for batch in tqdm.tqdm(self.dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
            if self.use_bart:
                input_ids, attention_mask,labels = batch

            outputs = self.model(input_ids = input_ids,attention_mask=attention_mask)
            gold = labels.cpu().detach().numpy()
            pre = outputs.cpu().detach().numpy()
        
            for i in range(len(batch)):
                golds.append(gold[i])
                pres.append(pre[i])
            #筛选出大于阈值的
            for i in range(len(pres)):
                for j in range(len(pres[i])):
                    if pres[i][j] > threshold:
                        pres[i][j] = 1
                    else:
                        pres[i][j] = 0

        acc = PD_acc_evaluate(pres, golds)

        return sum(acc)/len(acc)


def predict_ckpt(config,checkpoint_path):
    dm = EEModel(config)
    model = EEModel.load_from_checkpoint(checkpoint_path, config=config)
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    x = trainer.test(dm, dm.test_dataloader,ckpt_path=checkpoint_path)  # 预测结果已经在on_predict_epoch_end中保存了
    print(type(x))