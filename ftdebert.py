# -*- coding: utf-8 -*-


from tqdm import tqdm, trange
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
from transformers import AutoConfig, AutoModelForSequenceClassification,AutoTokenizer
import transformers
from config import opt
transformers.logging.set_verbosity_error()


def finetuning_teacher(model,tokenizer,train_dataloader,test_dataloader,lr,epochs, log_dir, partial_ft=True):
    best_acc = 0
    training_steps = len(train_dataloader) * epochs
    if partial_ft:
        optimizer = AdamW(model.classifier.parameters(), lr=lr)
    else:
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

    for epoch in tqdm(range(epochs), desc="Training Process: 修改了label2id"):
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
            save_path = os.path.join(f"./sileod_deberta_large_best/acc{valid_acc:.5f}_model_only_TNLI")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)


def main(args):
    train_set = SJTUDataSet(path_data=args.path_data, is_train=0)# train
    test_set = SJTUDataSet(path_data=args.path_data, is_train=1)# valid
    train_dataloader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset=test_set, num_workers=4, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained("sileod/deberta-v3-large-tasksource-nli", num_labels=3, problem_type="single_label_classification")
    tokenizer = AutoTokenizer.from_pretrained("sileod/deberta-v3-large-tasksource-nli")

    path_log = os.path.join(args.log_dir, args.exp_desc, "lr{}-epochs{}-partial_ft{}".format(args.lr, args.epochs, args.partial_ft))
    os.makedirs(path_log, exist_ok=True)
    finetuning_teacher(model=model,tokenizer=tokenizer,train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,lr=args.lr,epochs=args.epochs,log_dir=path_log, partial_ft=args.partial_ft)



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
                        default="sileod_deberta_large_only_TNLI")

    parser.add_argument("--lr",
                        type=float,
                        default=1e-6)

    parser.add_argument("--epochs",
                        type=int,
                        default=10)

    parser.add_argument("--batch_size",
                        type=int,
                        default=16)
    
    parser.add_argument("--partial_ft",
                        type=bool,
                        default=False)

    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = parse_args()
    main(args)
    
