export CUDA_VISIBLE_DEVICES=1
python train_bert_base.py --exp_desc "only_TNLI" \
                          --do_adv 0 \
                          --lr 5e-5 \
                          --epochs 10 \
                          --batch_size 64 \
                          --adv_train_type "fgm"
