import os
import sys
# coding=utf-8
from PD_data_process import EEProcessor
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningDataModule
from dataclasses import dataclass

# from sklearn import model_selection
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
#from torchcrf import CRF
import csv
from conlleval import PD_acc_evaluate
from transformers import (
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.trainer.states import TrainerFn
import os.path as osp

import numpy as np
import re
import json
#from pytorch_lightning.metrics import Accuracy
# import nltk
from torchmetrics import BLEUScore 

class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass

@dataclass
class KFoldDataModule(BaseKFoldDataModule):
#############################################################################################
#                           Step 2 / 5: Implement the KFoldDataModule                       #
# The `KFoldDataModule` will take a train and test dataset.                                 #
# On `setup_folds`, folds will be created depending on the provided argument `num_folds`    #
# Our `setup_fold_index`, the provided train dataset will be split accordingly to        #
# the current fold split.                                                                   #
#############################################################################################
    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def __init__(self, config) -> None:
        super().__init__()
        self.config=config
        self.batch_size = config.batch_size

        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)
        self.processor = EEProcessor(config, self.tokenizer)

    def setup(self, stage: str) -> None:
        # load the data
        # tokenizer = BertTokenizer.from_pretrained("/home/hutu/PersonaDataset/bert-base-uncased")
        # processor = EEProcessor(config, self.tokenizer)

        [data,label] = self.processor.get_data()
        dataset = self.processor.create_dataset(data,label,batch_size=self.batch_size, shuffle=False)
        self.train_dataset, self.test_dataset = random_split(dataset, [640, 71])

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)
    
    def __post_init__(cls):
        super().__init__()
    


class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.criterion = nn.BCELoss()


    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = self.criterion(logits, batch[1])
        acc = PD_acc_evaluate(logits, batch[1])
        self.log("test_acc", self.acc)
        self.log("test_loss", loss)

class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.

        # the test loop normally expects the model to be the pure LightningModule, but since we are running the
        # test loop during fitting, we need to temporarily unpack the wrapped module
        wrapped_model = self.trainer.strategy.model
        self.trainer.strategy.model = self.trainer.strategy.lightning_module
        self.trainer.test_loop.run()
        self.trainer.strategy.model = wrapped_model
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


class EEModel(pl.LightningModule):
    def __init__(self, config):
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

        self.model = BertModel.from_pretrained(config.pretrained_path)

        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        # 2. Init crf model and loss
        self.criterion = nn.BCELoss()
        # self.criterion = nn.NLLLoss(ignore_index=0, size_average=True)
        
        print("EE model init: done.")
    

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
            return torch.optim.Adam(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)



class EEPredictor:
    def __init__(self, checkpoint_path, config):
        self.use_bart = config.use_bart
        self.use_crf = config.use_crf

        self.model = EEModel.load_from_checkpoint(checkpoint_path, config=config)

        self.test_data = self.model.processor.get_test_data()
        if config.use_bart:
            self.tokenizer = self.model.tokenizer
            self.dataloader = self.model.processor.create_dataloader(
                self.test_data[0],self.test_data[1], batch_size=config.batch_size, shuffle=False)


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

def predict_ckpt(config,checkpoint_path):
    dm = EEModel(config)
    model = EEModel.load_from_checkpoint(checkpoint_path, config=config)
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    x = trainer.test(dm, dm.test_dataloader,ckpt_path=checkpoint_path)  # 预测结果已经在on_predict_epoch_end中保存了
    print(type(x))