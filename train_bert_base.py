#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os, sys
import argparse
import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn
from transformers import get_scheduler
from torch.optim import AdamW
from datasets import SJTUDataSet
from transformers import AutoConfig, AutoModelForSequenceClassification,BertTokenizer
import transformers
from adversarial_training import FreeLB, FGM
from config import opt
from adv_trainer.freelb import FreeLBTrainer
transformers.logging.set_verbosity_error()



def train_bert_base(model,tokenizer,train_dataloader,test_dataloader,lr,epochs, log_dir):
    best_acc = 0
    training_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_steps
    )
    # 单卡训练
    model.cuda()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(log_dir, 'training.log'))])

    for epoch in tqdm(range(epochs), desc="Training Process"):
        model.train()
        train_acc_list, train_loss_list = [], []

        for _, sentences1, sentences2, label in tqdm(train_dataloader):
            encoded_input = tokenizer(sentences1, sentences2, max_length=128, padding="max_length", truncation=True,return_tensors='pt')
            input_ids = encoded_input['input_ids'].cuda()
            attention_mask = encoded_input['attention_mask'].cuda()
            label = torch.LongTensor(label).cuda()
            token_type_ids = encoded_input['token_type_ids'].cuda()

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label)
            loss = outputs.loss
            train_loss_list.append(loss.item())
            accuracy = (outputs.logits.argmax(dim=-1) == label.cuda()).float().mean()
            train_acc_list.append(accuracy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        avg_train_acc = sum(train_acc_list) / len(train_acc_list)
        logging.info(f"[ Train | {epoch + 1:03d}/{epochs:03d}], loss={avg_train_loss:.5f}, train accuracy={avg_train_acc:.5f}")


        # 每个epoch测试一次
        model.eval()
        valid_loss_list, valid_acc_list = [], []

        for _, sentences1, sentences2, label in tqdm(test_dataloader):


            encoded_input = tokenizer(sentences1, sentences2, max_length=128, padding="max_length", truncation=True,
                                      return_tensors='pt')
            input_ids = encoded_input['input_ids'].cuda()
            attention_mask = encoded_input['attention_mask'].cuda()
            token_type_ids = encoded_input['token_type_ids'].cuda()
            label = torch.LongTensor(label).cuda()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=label)
                loss = outputs.loss
                valid_loss_list.append(loss.item())
                accuracy = (outputs.logits.argmax(dim=-1) == label.cuda()).float().mean()
                valid_acc_list.append(accuracy)

        valid_loss = sum(valid_loss_list) / len(valid_loss_list)
        valid_acc = sum(valid_acc_list) / len(valid_acc_list)
        logging.info(f"[ Valid | {epoch + 1:03d}/{epochs:03d}], loss={valid_loss:.5f}, accuracy={valid_acc:.5f}")

        if valid_acc > best_acc:
            logging.info(f"Best model found at epoch {epoch}, saving model")
            logging.info(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            best_acc = valid_acc
            save_path = os.path.join(f"./bert_base/acc{valid_acc:.5f}_model_only_TNLI")
            model.save_pretrained(save_path)


# 有点问题
def freelb_train_bert_base(model,tokenizer,train_dataloader,test_dataloader,lr,epochs, log_dir):
    best_acc = 0
    training_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_steps
    )
    trainer = FreeLBTrainer(model, optimizer, lr_scheduler, max_train_steps=10000)
    # 单卡训练
    model.cuda()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(log_dir, 'training.log'))])

    for epoch in tqdm(range(epochs), desc="Training Process"):
        model.train()
        train_acc_list, train_loss_list = [], []

        train_loss, train_step = trainer.step(train_dataloader)
        global_step = trainer.global_step

        # for _, sentences1, sentences2, label in tqdm(train_dataloader):
        #     encoded_input = tokenizer(sentences1, sentences2, max_length=128, padding="max_length", truncation=True,return_tensors='pt')
        #     input_ids = encoded_input['input_ids'].cuda()
        #     attention_mask = encoded_input['attention_mask'].cuda()
        #     label = torch.LongTensor(label).cuda()
        #     token_type_ids = encoded_input['token_type_ids'].cuda()

        #     inputs = {
        #         "input_ids": input_ids,
        #         "token_type_ids": token_type_ids,
        #         "attention_mask": attention_mask,
        #     }
        #     loss, prediction_scores = freelb.attack(model, inputs, label)
        #     train_loss_list.append(loss.item())
        #     accuracy = (prediction_scores.argmax(dim=-1) == label.cuda()).float().mean()
        #     train_acc_list.append(accuracy)


        #     loss.backward()
        #     optimizer.step()
        #     lr_scheduler.step()
        #     model.zero_grad()

        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        avg_train_acc = sum(train_acc_list) / len(train_acc_list)
        logging.info(f"[ Train | {epoch + 1:03d}/{epochs:03d}], loss={avg_train_loss:.5f}, train accuracy={avg_train_acc:.5f}")


        # 每个epoch测试一次
        model.eval()
        valid_loss_list, valid_acc_list = [], []

        for _, sentences1, sentences2, label in tqdm(test_dataloader):


            encoded_input = tokenizer(sentences1, sentences2, max_length=128, padding="max_length", truncation=True,
                                      return_tensors='pt')
            input_ids = encoded_input['input_ids'].cuda()
            attention_mask = encoded_input['attention_mask'].cuda()
            token_type_ids = encoded_input['token_type_ids'].cuda()
            label = torch.LongTensor(label).cuda()
            with torch.no_grad():
                inputs = {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                }
                loss, prediction_scores = freelb.attack(model, inputs, label)
                valid_loss_list.append(loss.item())
                accuracy = (prediction_scores.argmax(dim=-1) == label.cuda()).float().mean()
                valid_acc_list.append(accuracy)

        valid_loss = sum(valid_loss_list) / len(valid_loss_list)
        valid_acc = sum(valid_acc_list) / len(valid_acc_list)
        logging.info(f"[ Valid | {epoch + 1:03d}/{epochs:03d}], loss={valid_loss:.5f}, accuracy={valid_acc:.5f}")

        if valid_acc > best_acc:
            logging.info(f"Best model found at epoch {epoch}, saving model")
            logging.info(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            best_acc = valid_acc
            save_path = os.path.join("./bert_base/best_model_freelb")
            model.save_pretrained(save_path)

def fgm_train_bert_base(model,tokenizer,train_dataloader,test_dataloader,lr,epochs, log_dir):
    best_acc = 0
    training_steps = len(train_dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_steps
    )
    fgm = FGM(model)
    # 单卡训练
    model.cuda()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(log_dir, 'training.log'))])

    for epoch in tqdm(range(epochs), desc="Training Process"):
        model.train()
        train_acc_list, train_loss_list = [], []

        for _, sentences1, sentences2, label in tqdm(train_dataloader):
            encoded_input = tokenizer(sentences1, sentences2, max_length=128, padding="max_length", truncation=True,return_tensors='pt')
            input_ids = encoded_input['input_ids'].cuda()
            attention_mask = encoded_input['attention_mask'].cuda()
            label = torch.LongTensor(label).cuda()
            token_type_ids = encoded_input['token_type_ids'].cuda()

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label)
            accuracy = (outputs.logits.argmax(dim=-1) == label.cuda()).float().mean()

            loss = outputs.loss
            loss.backward()
            fgm.attack()
            loss_adv = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label).loss
            loss_adv.backward()

            fgm.restore()
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()

            train_loss_list.append(loss.item())
            train_acc_list.append(accuracy)

        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        avg_train_acc = sum(train_acc_list) / len(train_acc_list)
        logging.info(f"[ Train | {epoch + 1:03d}/{epochs:03d}], loss={avg_train_loss:.5f}, train accuracy={avg_train_acc:.5f}")


        # 每个epoch测试一次
        model.eval()
        valid_loss_list, valid_acc_list = [], []

        for _, sentences1, sentences2, label in tqdm(test_dataloader):


            encoded_input = tokenizer(sentences1, sentences2, max_length=128, padding="max_length", truncation=True,
                                      return_tensors='pt')
            input_ids = encoded_input['input_ids'].cuda()
            attention_mask = encoded_input['attention_mask'].cuda()
            token_type_ids = encoded_input['token_type_ids'].cuda()
            label = torch.LongTensor(label).cuda()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label)
                loss = outputs.loss

                valid_loss_list.append(loss.item())
                accuracy = (outputs.logits.argmax(dim=-1) == label.cuda()).float().mean()
                valid_acc_list.append(accuracy)

        valid_loss = sum(valid_loss_list) / len(valid_loss_list)
        valid_acc = sum(valid_acc_list) / len(valid_acc_list)
        logging.info(f"[ Valid | {epoch + 1:03d}/{epochs:03d}], loss={valid_loss:.5f}, accuracy={valid_acc:.5f}")

        if valid_acc > best_acc:
            logging.info(f"Best model found at epoch {epoch}, saving model")
            logging.info(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            best_acc = valid_acc
            save_path = os.path.join(f"./bert_base/acc{valid_acc:.5f}_model_only_TNLI")
            model.save_pretrained(save_path)


def main(args):
    train_set = SJTUDataSet(path_data=args.path_data, is_train=0)# train
    test_set = SJTUDataSet(path_data=args.path_data, is_train=1)# valid
    train_dataloader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset=test_set, num_workers=4, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3,problem_type="single_label_classification")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if args.do_adv == 0:
        path_log = os.path.join(args.log_dir, args.exp_desc, "lr{}-epochs{}-noadv".format(args.lr, args.epochs))
        os.makedirs(path_log, exist_ok=True)
        train_bert_base(model=model,tokenizer=tokenizer,train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,lr=args.lr,epochs=args.epochs,log_dir=path_log)
    else:
        if args.adv_train_type == "freelb":
            path_log = os.path.join(args.log_dir, args.exp_desc, "lr{}-epochs{}-adv_type{}".format(args.lr, args.epochs, args.adv_train_type))
            os.makedirs(path_log, exist_ok=True)
            freelb_train_bert_base(model=model,tokenizer=tokenizer,train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,lr=args.lr,epochs=args.epochs,log_dir=path_log)

        elif args.adv_train_type == "fgm":
            path_log = os.path.join(args.log_dir, args.exp_desc, "lr{}-epochs{}-adv_type:{}".format(args.lr, args.epochs, args.adv_train_type))
            os.makedirs(path_log, exist_ok=True)
            fgm_train_bert_base(model=model,tokenizer=tokenizer,train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,lr=args.lr,epochs=args.epochs,log_dir=path_log)


def parse_args():
    parser = argparse.ArgumentParser(description="bert_base model")

    parser.add_argument("--path_data",
                        type=str,
                        default="./data")

    parser.add_argument("--log_dir",
                        type=str,
                        default="./log")

    parser.add_argument("--exp_desc",
                        type=str,
                        default="only_TNLI")

    # 明天起来写一下对抗训练
    parser.add_argument("--do_adv",
                        type=int,
                        default=0,
                        help="use adversarial training strategy")

    parser.add_argument("--adv_train_type",
                        type=str,
                        choices=["freelb","pgd","fgm"],
                        default="fgm")

    parser.add_argument("--lr",
                        type=float,
                        default=5e-5)

    parser.add_argument("--epochs",
                        type=int,
                        default=20)

    parser.add_argument("--batch_size",
                        type=int,
                        default=64)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)







