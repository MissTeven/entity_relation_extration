#!/usr/bin/env bash

#修改gpu编号
export CUDA_VISIBLE_DEVICES=0
python3 train.py \
--pretrained_model_path="pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large" \
--bert_vocab_path="pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large/vocab.txt"





