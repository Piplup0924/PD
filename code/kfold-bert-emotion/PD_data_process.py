import sys
import os
from turtle import position
import tqdm

# 添加src目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

from transformers import (
    DataProcessor,
    BertTokenizer,
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from transformers import default_data_collator
from collections import defaultdict
import torch
import csv
import pandas as pd
import numpy as np
import json
import numbers
from sklearn.model_selection import KFold
from typing import Optional
from torch.utils.data.dataset import Dataset

from tqdm.contrib import tzip


class EEProcessor(DataProcessor):
    """
       从数据文件读取数据，生成训练数据dataloader，返回给模型
    """

    def __init__(self, config, k, tokenizer=None):
        self.train_path = config.train_path
        self.dev_path=config.dev_path
        self.test_path=config.test_path
        if tokenizer==None:
            self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)  # 
        else:
            self.tokenizer = tokenizer
        self.k = k
        self.data_train = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.dataset = None
        self.config = config

        self.emotion_tokenizer = AutoTokenizer.from_pretrained(config.emotion_pretrain_path)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(config.emotion_pretrain_path)
    
    def read_csv_data(self,path):
        df = pd.read_csv(path)

        texts = df['text'].tolist()
        characters = df['character'].tolist()
        cAGR,cCON,cEXT,cOPN,cNEU = df['cAGR'].tolist(),df['cCON'].tolist(),df['cEXT'].tolist(),df['cOPN'].tolist(),df['cNEU'].tolist()
        
        dialogues = []
        for text in texts:
            dialogue = text.split("<br><br>")
            target_user = dialogue[0].split(" for ")[1][:-4]
            speakers = {}
            processed_dialogue = []
            for utterance in dialogue[1:-1]:
                if utterance[:3] != "<b>":
                    continue
                speaker, utterance = utterance.split("</b>: ")
                speaker = speaker[3:]
                # if speaker == target_user:
                #     speaker = 0
                # else:
                #     if speaker not in speakers:
                #         speakers[speaker] = len(speakers) + 1
                #     speaker = speakers[speaker]
                processed_dialogue.append((speaker, utterance))
            dialogues.append(processed_dialogue)
        
        #组合cAGR,cCON,cEXT,cOPN,cNEU为向量
        labels = []
        for i in range(len(cAGR)):
            labels.append([float(cAGR[i]),float(cCON[i]),float(cEXT[i]),float(cOPN[i]),float(cNEU[i])])
        return dialogues, characters, labels
    
    def add_utterance_emotion(self, utterance):
        input_ids = self.emotion_tokenizer.encode(utterance, return_tensors="pt")
        outputs = self.emotion_model(input_ids)
        logits = outputs[0]
        label = torch.argmax(logits, dim=1)
        # label_text = self.label_dict[str(label.item())]
        # utterance = label_text + " : " + utterance
        return label.detach().numpy().tolist()[0]
    
    #仅选择target_user的对话
    def choose_target_only(self, dialogues, characters):
        texts = []
        emo_labels = []
        for dialogue, character in tzip(dialogues, characters):
            text = ''
            emo_label = []
            for utterance in dialogue:
                if utterance[0] == character:
                    # text = text + " " +self.add_utterance_emotion(utterance[1])
                    text += utterance[1]
                    emo_label.append(self.add_utterance_emotion(utterance[1]))
            texts.append(text)
            emo_labels.append(emo_label)
        return texts, emo_labels

    
    def get_train_data(self):
        dialogues, characters, labels = self.read_csv_data(self.train_path)
        data, emo_labels = self.choose_target_only(dialogues, characters)
        return data,emo_labels,labels
    
    def get_dev_data(self):
        dialogues, characters, labels = self.read_csv_data(self.train_path)
        data, emo_labels = self.choose_target_only(dialogues, characters[640:])
        return data,emo_labels,labels[640:]
    
    def get_test_data(self):
        dialogues, characters, labels = self.read_csv_data(self.train_path)
        data = self.choose_target_only(dialogues, characters)
        return data[:150],labels[:150]
    
    def get_data(self):
        dialogues, characters, labels = self.read_csv_data(self.train_path)
        data = self.choose_target_only(dialogues, characters)
        return data,labels

    def create_dataset(self, dialog ,emo_labels, label , batch_size, shuffle=False, max_length=512):
        tokenizer = self.tokenizer

        dialogs = tokenizer(     # 得到文本的编码表示
            dialog,
            padding=True,
            truncation = True,
            return_tensors="pt",
            max_length=512
        )
        emolabel_max_length = 5
        for i in range(len(emo_labels)):
            if len(emo_labels[i]) < emolabel_max_length:
                emo_labels[i] = emo_labels[i] + [-1]*(emolabel_max_length-len(emo_labels[i]))
            else:
                emo_labels[i] = emo_labels[i][:emolabel_max_length]

        self.dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(dialogs["input_ids"]),          # 句子字符id
            torch.LongTensor(dialogs["attention_mask"]),     # 区分是否是pad值。句子内容为1，pad为0 
            torch.LongTensor(emo_labels),
            # torch.FloatTensor(emo_attention_mask),
            torch.FloatTensor(label)
        )

        #k折拆分
        seed = 20220924
        kf = KFold(n_splits=self.config.num_folds, shuffle=True, random_state=seed)
        all_splits = [k for k in kf.split(self.dataset)]

        train_indexes, val_indexes = all_splits[self.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
        self.data_train, self.data_val = self.dataset[train_indexes], self.dataset[val_indexes]


        # _, test_indexes = all_splits[0]
        # test_indexes = test_indexes.tolist()
        # self.data_test = self.dataset[test_indexes]
        # data_collator = DataCollatorForSeq2Seq(
        #     tokenizer,
        #     model = BartForConditionalGeneration,
        #     label_pad_token_id=tokenizer.pad_token_id
        # )
    def create_dataloader(self, is_train, dialog ,emo_labels, label , batch_size, shuffle=False, max_length=512):

        self.create_dataset(dialog ,emo_labels, label , batch_size, shuffle, max_length)

        if is_train == 1:
            self.dataset.tensors = self.data_train
        elif is_train == 2:
            self.dataset.tensors = self.data_val
        # elif is_train == 0:
        #     self.dataset.tensors = self.data_test

        dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            shuffle=shuffle,
            # collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=0,
        )
        
        return dataloader
