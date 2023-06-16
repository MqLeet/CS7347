import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, BertTokenizer, DebertaTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertForSequenceClassification, DebertaForSequenceClassification
from datasets import SJTUDataSet
from tqdm import tqdm, trange
import argparse
import logging
import os, sys
from torch.utils.data import DataLoader

# ==============================蒸馏损失=============================== 
class DistillLoss(nn.Module):
    def __init__(self, alpha, T):
        super(DistillLoss, self).__init__()
        self.alpha = alpha
        self.T = T
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, s_logits, t_logits, labels):
        # 计算分类损失
        ce_loss = self.ce_loss(s_logits, labels)

        # 计算蒸馏损失
        # student网络输出软化后结果
        # log_softmax与softmax没有本质的区别，只不过log_softmax会得到一个正值的loss结果。
        s_logits = F.log_softmax(s_logits/self.T, dim=-1)
        # teacher网络输出软化后结果
        t_logits = F.softmax(t_logits/self.T, dim=-1)
        kl_loss = self.kl_loss(s_logits, t_logits) * self.T * self.T

        # 总损失
        loss = self.alpha * ce_loss + (1-self.alpha) * kl_loss
        return loss




# 用bert-base-uncased做学生模型，deberta-v3做教师模型，通过知识蒸馏来完成一个Natural Language Inference任务
class Teacher(object):
    def __init__(self, cache_dir, max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(cache_dir)
        self.model.eval()


    def predict(self, sentences1, sentences2):
        self.device = self.model.device

        encoded_input = self.tokenizer(sentences1, sentences2, max_length=128, padding="max_length", truncation=True,
                            return_tensors='pt')
        encoded_input = encoded_input.to(self.device)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        token_type_ids = encoded_input['token_type_ids']

        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs


class Student(object):
    def __init__(self, model_name='bert-base-uncased', max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3,problem_type="single_label_classification")


    def predict(self, sentences1, sentences2):
        self.device = self.model.device
        encoded_input = self.tokenizer(sentences1, sentences2, max_length=128, padding="max_length", truncation=True,
                            return_tensors='pt')
        encoded_input = encoded_input.to(self.device)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        token_type_ids = encoded_input['token_type_ids']

        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)     
        return outputs


def distillation(train_dataloader, test_dataloader, lr, epochs, log_dir, teacher_cache_dir, temperature=2.0, alpha=0.2):

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(log_dir, 'training.log'))])

    teacher = Teacher(cache_dir=teacher_cache_dir)
    student = Student()
    

    student_model = student.model
    student_tokenizer = student.tokenizer
    student_model.cuda()
    teacher.model.cuda()

    best_acc = 0
    training_steps = len(train_dataloader) * epochs
    optimizer = AdamW(student_model.parameters(), lr=lr)
    loss_fn_distill = DistillLoss(alpha=alpha, T=temperature)
    loss_fn_cls = nn.CrossEntropyLoss()
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=training_steps)

    # training loop
    for epoch in trange(epochs, desc="Training Process"):
        student_model.train()
        student_train_acc_list, teacher_train_acc_list, train_loss_list = [], [], []


        for _, sentences1, sentences2, label in tqdm(train_dataloader):
            label = torch.LongTensor(label).to(student_model.device)
            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher.predict(sentences1, sentences2)
                teacher_logits = teacher_outputs.logits
            
            # Get student predictions
            student_outputs = student.predict(sentences1, sentences2)
            student_logits = student_outputs.logits
            
            # Compute the distillation loss
            loss = loss_fn_distill(student_logits, teacher_logits, label)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            train_loss_list.append(loss.item())
            student_accuracy = (student_outputs.logits.argmax(dim=-1) == label).float().mean()
            teacher_accuracy = (teacher_outputs.logits.argmax(dim=-1) == label).float().mean()
            teacher_train_acc_list.append(teacher_accuracy)
            student_train_acc_list.append(student_accuracy)
    
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        avg_teacher_train_acc = sum(teacher_train_acc_list) / len(teacher_train_acc_list)
        avg_student_train_acc = sum(student_train_acc_list) / len(student_train_acc_list)
        logging.info(f"[ Train | {epoch + 1:03d}/{epochs:03d}], loss={avg_train_loss:.5f}, teacher train accuracy={avg_teacher_train_acc:.5f}, student train accuracy={avg_student_train_acc:.5f}")
    

        # 每个epoch测试一次
        student_model.eval()
        student_valid_acc_list, valid_loss_list = [], []

        for _, sentences1, sentences2, label in tqdm(test_dataloader):
            with torch.no_grad():
                student_outputs = student.predict(sentences1, sentences2)
                label = torch.LongTensor(label).to(student_model.device)
                loss = loss_fn_cls(student_outputs.logits, label)
                valid_loss_list.append(loss.item())

                student_accuracy = (student_outputs.logits.argmax(dim=-1) == label.cuda()).float().mean()
                student_valid_acc_list.append(student_accuracy)
        
        avg_valid_loss = sum(valid_loss_list) / len(valid_loss_list)
        avg_student_valid_acc = sum(student_valid_acc_list) / len(student_valid_acc_list)
        logging.info(f"[ Valid | {epoch + 1:03d}/{epochs:03d}], loss={avg_valid_loss:.5f}, student valid accuracy={avg_student_valid_acc:.5f}")

        if avg_student_valid_acc > best_acc:
            logging.info(f"Save model at epoch {epoch + 1} with best acc {avg_student_valid_acc:.5f}")
            logging.info(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {avg_valid_loss:.5f}, acc = {avg_student_valid_acc:.5f} -> best")
            best_acc = avg_student_valid_acc

            save_path = os.path.join(f"./student_bert_base/acc{best_acc:.5f}_model_only_TNLI")
            student_model.save_pretrained(save_path)
            student_tokenizer.save_pretrained(save_path)

def main(args):
    train_set = SJTUDataSet(path_data=args.path_data, is_train=0) # train
    test_set = SJTUDataSet(path_data=args.path_data, is_train=1)# valid
    train_dataloader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset=test_set, num_workers=4, batch_size=args.batch_size)



    path_log = os.path.join(args.log_dir, args.exp_desc, "lr{}-epochs{}-temperature{}-alpha{}".format(args.lr, args.epochs, args.temperature, args.alpha))
    os.makedirs(path_log, exist_ok=True)
    distillation(train_dataloader=train_dataloader, test_dataloader=test_dataloader, lr=args.lr, epochs=args.epochs, log_dir=path_log, temperature=args.temperature, teacher_cache_dir=args.teacher_cache_dir, alpha=args.alpha)


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
                        default="distillation_teacher_deberta_base_only_TNLI")

    parser.add_argument("--lr",
                        type=float,
                        default=5e-5)

    parser.add_argument("--epochs",
                        type=int,
                        default=25)

    parser.add_argument("--batch_size",
                        type=int,
                        default=64)
    parser.add_argument("--temperature",
                        type=float,
                        default=2.0)
    parser.add_argument("--alpha",
                        type=float,
                        default=0.2)
    
    parser.add_argument("--teacher_cache_dir",
                        type=str,
                        default="./sileod_deberta_base_best/acc0.91195_model_only_TNLI/")

    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = parse_args()
    main(args)









