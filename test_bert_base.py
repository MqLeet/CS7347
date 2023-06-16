#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification,BertTokenizer
from datasets import SJTUDataSet
import logging
import transformers

num_test_sample = 4000
Num2name = {0:"entailment", 1:"neutral", 2:"contradiction"}


def label2category(label_list):
    category_list = []
    for _, label in enumerate(label_list):
        category = Num2name[label]
        category_list.append(category)

    return category_list

def test_model(model, tokenizer,test_dataloader, output_file):

    model.cuda()
    model.eval()
    test_acc = []
    ids = [i for i in range(1,  num_test_sample + 1)]
    res = [0] * num_test_sample
    logging.disable(logging.WARNING)

    for id, sentences1, sentences2 in tqdm(test_dataloader):
        encoded_input = tokenizer(sentences1, sentences2, max_length=128, padding="max_length", truncation=True,
                                  return_tensors='pt')
        input_ids = encoded_input['input_ids'].cuda()
        attention_mask = encoded_input['attention_mask'].cuda()
        token_type_ids = encoded_input['token_type_ids'].cuda()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pred_category = outputs.logits.argmax(dim=-1).tolist()
            id = id.tolist()
            for i in range(len(id)):
                res[id[i] - 1] = pred_category[i]
    categories = label2category(res)
    df = pd.DataFrame({"Id": ids, "Category": categories})
    df.to_csv(output_file, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="bert_base model")

    parser.add_argument("--path_data",
                        type=str,
                        default="./data")

    parser.add_argument("--path_output_csv",
                        type=str,
                        default="./output")

    parser.add_argument("--desc_output_csv",
                        type=str,
                        default="student_bert_base_only_TNLI")

    parser.add_argument("--batch_size",
                        type=int,
                        default=64)
    parser.add_argument("--path_trained_model",
                        type=str,
                        default="./student_bert_base/best_model")

    return parser.parse_args()

def main(args):
    test_set = SJTUDataSet(path_data=args.path_data, is_train=2)
    test_dataloader = DataLoader(dataset=test_set, num_workers=4, batch_size=args.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(args.path_trained_model)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    transformers.logging.set_verbosity_error()
    os.makedirs(args.path_output_csv, exist_ok=True)
    path_outcsv = os.path.join(args.path_output_csv, args.desc_output_csv)
    path_outcsv += ".csv"
    test_model(model=model,tokenizer=tokenizer,test_dataloader=test_dataloader, output_file=path_outcsv)



if __name__ == "__main__":
    args = parse_args()
    main(args)