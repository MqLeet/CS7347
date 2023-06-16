# CS7347 course project
Final project for SJTU graduate course:CS7347,NLU,2023 - [A NLI task](https://www.kaggle.com/competitions/sjtu-nlu2023)

## Preparation
- Split the train set into train set and valid set
- Create an enve
```shell
conda create -n nlu_project
pip install -r requirements.txt
```

## Methods

### 1. Fine-tuning the bert-base model
```shell
python train_bert_base.py --exp_desc "only_TNLI" \
                          --do_adv 0 \
                          --lr 5e-5 \
                          --epochs 10 \
                          --batch_size 64 \
```

### 2. Fine-tuning the bert-base model with adversarial training strategy
```shell
python train_bert_base.py --exp_desc "only_TNLI" \
                          --do_adv 0 \
                          --lr 5e-5 \
                          --epochs 10 \
                          --batch_size 64 \
                          --adv_train_type "fgm"
```
`do_adv` is the switch of whether use adversarial training, `adv_train_type` is the type of adversarial training.

### 3. Data augmentation
Use T5 model to paraphrase in order to get more sentences pairs.
```shell
python data_augmentation.py --st_point 0 \
                        --ed_point 10000 \
                        --augmented_des 0

# st_point, ed_point means the start and end of the part in the dataset, which is going to be dealed.
# augmented_des means the description of this process.
```

Then, train the bert-base model as described in 1 and 2.

### 4. Knowledge Distillation
- Fine-tuning the teacher model:
```shell
python ftdebert.py
```

- Knowledge Distillation
```shell
python distillation.py --teacher_cache_dir $HOME/sileod_deberta_base_best/$finetuned_teacher_model \
                    --alpha 0.2 \
                    --temperature 2.0 \
                    --lr 5e-5 \
                    --epochs 25
```

## Inference and get the submission.csv
```shell
python test_bert_base.py --path_trained_model $HOME/student_bert_base/best_model
```


