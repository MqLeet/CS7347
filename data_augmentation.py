#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import pandas as pd
import csv
import os
import time
import numpy as np
from config import opt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)



tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", padding="max_length")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
device = torch.device("cuda:2")
model = model.to(device)


def reg_str(text):
    text = str(text).replace('\n', ' ').replace('\t', ' ').replace('.', '')
    # 转换为小写
    text = text.lower()
    text = text.strip()

    return text

def generate_paraphrase(prompt):
    prompt = reg_str(prompt)
    max_new_tokens = len(prompt) + 10
    text = "paraphrase: " + prompt + " </s>"
    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=max_new_tokens,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=5
    )

    final_outputs = []
    for output in outputs:
        sent = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        sent = reg_str(sent)

        if sent.lower() != prompt.lower() and sent.lower() not in final_outputs:
            final_outputs.append(sent.lower())


    if len(final_outputs) >= 2:
        return final_outputs[0], final_outputs[1]
    elif len(final_outputs) == 1:
        return final_outputs[0], None
    else:
        return None, None




def data_augmentation(path_origin_trainset, path_augmented_trainset, st, ed):
    data = pd.read_csv(path_origin_trainset, sep='\t', usecols=['id','sentence1', 'sentence2', 'label'], quoting=csv.QUOTE_NONE)
    sentences1 = data['sentence1'][st:ed]
    sentences2 = data['sentence2'][st:ed]
    id = data['id'][st:ed]
    label = data['label'][st:ed]

    augmented_sentences1 = [[generate_paraphrase(text)[0], generate_paraphrase(text)[1]] for text in sentences1]
    augmented_sentences2 = [[generate_paraphrase(text)[0], generate_paraphrase(text)[1]] for text in sentences2]

    augmented_sentences1 = np.array(augmented_sentences1)
    augmented_sentences2 = np.array(augmented_sentences2)


    augmented_data_1 = pd.DataFrame({
        'id': id,
        'sentence1': augmented_sentences1[:, 0],
        'sentence2': augmented_sentences2[:, 0],
        'label': label
    })

    augmented_data_2 = pd.DataFrame({
        'id': id,
        'sentence1': augmented_sentences1[:, 1],
        'sentence2': augmented_sentences2[:, 1],
        'label': label
    })

    augmented_data_3 = pd.DataFrame({
        'id': id,
        'sentence1': augmented_sentences1[:, 0],
        'sentence2': augmented_sentences2[:, 1],
        'label': label
    })

    augmented_data_4 = pd.DataFrame({
        'id': id,
        'sentence1': augmented_sentences1[:, 1],
        'sentence2': augmented_sentences2[:, 0],
        'label': label
    })

    augmented_data_1 = augmented_data_1.dropna()
    augmented_data_2 = augmented_data_2.dropna()
    augmented_data_3 = augmented_data_3.dropna()
    augmented_data_4 = augmented_data_4.dropna()

    path_augmented_trainset_1 = path_augmented_trainset + "_1.tsv"
    path_augmented_trainset_2 = path_augmented_trainset + "_2.tsv"
    path_augmented_trainset_3 = path_augmented_trainset + "_3.tsv"
    path_augmented_trainset_4 = path_augmented_trainset + "_4.tsv"

    augmented_data_1.to_csv(path_augmented_trainset_1, sep='\t', index=False)
    augmented_data_2.to_csv(path_augmented_trainset_2, sep='\t', index=False)
    augmented_data_3.to_csv(path_augmented_trainset_3, sep='\t', index=False)
    augmented_data_4.to_csv(path_augmented_trainset_4, sep='\t', index=False)


def main(args):

    path_origin_trainset = "./data/train.tsv"
    path_augmented_trainset = f"./data/augmented_train_{args.augmented_des}"

    data_augmentation(path_origin_trainset=path_origin_trainset, path_augmented_trainset=path_augmented_trainset,st=args.st_point,ed=args.ed_point)
    print("Enjoy your augmented data")


def parse_args():
    parse = argparse.ArgumentParser(description="Augmentation param")

    parse.add_argument("--st_point",
                        type=int,
                        default=40000)

    parse.add_argument("--ed_point",
                        type=int,
                        default=50000)
    
    parse.add_argument("--augmented_des",
                        type=int,
                        default=4)

    return parse.parse_args()



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    script_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_path)
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    running_time = end_time - start_time
    print(f"程序运行时间为: {running_time:.2f} 秒")

