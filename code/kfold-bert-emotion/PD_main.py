# coding=utf-8
import sys
import os
import torch
import argparse
import glob
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

# 添加src目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
sys.path.append(os.path.dirname(BASE_DIR))              # 将src目录添加到环境

from PD_model import EEModel, EEPredictor
import utils
print(torch.__version__)
utils.set_random_seed(20220924)
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
wandb_logger = WandbLogger(project="PD-kfold", name= "bert-10fold-emotion")


if __name__ == '__main__':

    WORKING_DIR = BASE_DIR

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=utils.str2bool, default=True, help="train the EE model or not (default: True)")
    parser.add_argument("--batch_size", type=int, default=16, help="input batch size for training and test (default: 4)")
    parser.add_argument("--max_epochs", type=int, default=30, help="the max epochs for training and test (default: 20)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (default: 2e-5)")
    parser.add_argument("--crf_lr", type=float, default=0.1, help="crf learning rate (default: 0.1)")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout (default: 0.2)")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="optimizer")
    parser.add_argument("--num_folds", type=int, default=10)

    parser.add_argument("--use_bart", type=utils.str2bool, default=True,
                        help="whether to use bert training or not (default: True)")
    parser.add_argument("--use_crf", type=utils.str2bool, default=False,
                        help="whether to use crf layer training or not (default: True)")

    # 下面参数基本默认
    parser.add_argument("--train_path", type=str, default="/home/hutu/PersonaDataset/PD/friends-personality/CSV/friends-personality.csv".format(WORKING_DIR),
                        help="train_path")
    parser.add_argument("--dev_path", type=str, default="/home/hutu/PersonaDataset/PE/data/data_augment/distill_augment_data/valid_augment_after_process.json".format(WORKING_DIR),
                        help="dev_path")
    parser.add_argument("--train_num", type=int, default=-1,help="train data number")
    parser.add_argument("--dev_num", type=int, default=-1,help="train data number")
    parser.add_argument("--addition_vocab_path", type = str, default="/home/hutu/PersonaDataset/PE/data/addition_vocab.json", help="addition_vocab_path")
    # parser.add_argument("--schema_path", type=str, default="{}/data/duee_event_schema.json".format(WORKING_DIR),
    #                     help="schema_path")
    parser.add_argument("--test_path", type=str, default="/home/hutu/PersonaDataset/PE/data/data_augment/test_human.json".format(WORKING_DIR),
                        help="test_path")
    parser.add_argument("--ee_result_path", type=str, default="{}/result".format(WORKING_DIR),
                        help="ee_result_path")
    parser.add_argument("--ckpt_save_path", type=str,
                        default="/home/hutu/PersonaDataset/PD/code/kfold-bert-emotion/weight/".format(WORKING_DIR), help="ckpt_save_path")
    parser.add_argument("--emotion_pretrain_path", type=str,
                        default="/home/hutu/PersonaDataset/PD/emotion_predict/distilbert-base-uncased-go-emotions-student".format(WORKING_DIR), help="emotion_pretrain_path")
    parser.add_argument("--resume_ckpt", type=str,
                        default=None, help="checkpoint file name for resume")
    parser.add_argument("--pretrained_path", type=str,
                        default="/home/hutu/PersonaDataset/bert-base-uncased".format(WORKING_DIR), help="pretrained_path")
    parser.add_argument("--ckpt_name",  type=str, default="base", help="ckpt save name")
    parser.add_argument("--test_ckpt_name",  type=str, default="base_epoch=7_pos_f1=61.6269.ckpt", help="ckpt name for test")
    

    args = parser.parse_args()



    print('--------config----------')
    print(args)
    print('--------config----------')

    if args.is_train == True:
        # ============= train 训练模型==============
        print("start train model ...")

        results = []

        for k in range(args.num_folds):
            
            model = EEModel(args,k)

            # 设置保存模型的路径及参数
            ckpt_callback = ModelCheckpoint(
                dirpath=args.ckpt_save_path,    
                filename=str(k) + "_" + args.ckpt_name + "_{epoch}_{avg_acc:.4f}",   
                # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
                monitor='avg_acc',   
                mode='max',
                save_top_k=1,  
                verbose=True,
            )

            resume_checkpoint=None
            if args.resume_ckpt:
                resume_checkpoint=os.path.join(args.ckpt_save_path ,args.resume_ckpt)

            # 设置训练器
            trainer = pl.Trainer(
                progress_bar_refresh_rate=1,
                resume_from_checkpoint = resume_checkpoint,
                max_epochs=args.max_epochs,
                callbacks=[ckpt_callback],
                checkpoint_callback=True,
                gpus=1,
                logger=wandb_logger
            )

            # 开始训练模型
            trainer.fit(model)

        #预测
        #     ckpts = glob.glob(args.ckpt_save_path +str(k)+"*")

        #     # test_ckpt_name = ckpt_callback.filename
        #     checkpoint_path = ckpts[0]
        #     predictor = EEPredictor(checkpoint_path, args, k)
        #     wandb.log({"turn_acc":predictor.generate_points()})
        #     wandb_logger.log_metrics({"turn_acc" + str(k):predictor.generate_points()})
        #     results.append(predictor.generate_points())
        # wandb_logger.log_metrics({"mean_acc":sum(results)/args.num_folds})

            

        # 只训练CRF的时候，保存最后的模型
        # if config.use_crf and config.first_train_crf == 1:
        #     trainer.save_checkpoint(os.path.join(config.ner_save_path, 'crf_%d.ckpt' % (config.max_epochs)))
    else:
        # ============= test 测试模型==============
        print("\n\nstart test model...")

        # outfile_txt = os.path.join(args.ee_result_path, args.test_ckpt_name[:-5] + "train.json")
        # outfile_txt = "/home/hutu/PersonaDataset/PE/code_history/prompt/best_result/distill_augment_result.json"
        results = []
        for k in range(1, args.num_folds):
            ckpts = glob.glob(args.ckpt_save_path +str(k)+"*")
            checkpoint_path = ckpts[0]
            predictor = EEPredictor(checkpoint_path, args, k)
            results.append(predictor.generate_points())
        print(results)
        print(sum(results)/args.num_folds)
        # 开始测试，将结果保存至输出文件
        # checkpoint_path = os.path.join(args.ckpt_save_path, args.test_ckpt_name)
        # # predict_ckpt(args,"/home/hutu/PersonaDataset/PE/code_history/t5-triple/best_model/base_epoch=5_val_f1=0.0-v2.ckpt")
        # predictor = EEPredictor(checkpoint_path, args)
        # predictor.generate_result(outfile_txt)
        # print('\n', 'outfile_txt name:', outfile_txt)  