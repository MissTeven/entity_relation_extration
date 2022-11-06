import argparse
from bert4keras.tokenizers import Tokenizer

from result_generator import resultGenerator
from model import GRTE
from util import *
from tqdm import tqdm
import os
import json
from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch


def evaluate(args, tokenizer, id2predicate, id2label, label2id, model, dataloader, fold):
    model.to("cuda")
    test_pred_path = os.path.join(args.result_path, f"{fold}.json")
    f = open(test_pred_path, 'w', encoding='utf-8')
    total = {}
    for batch in tqdm(dataloader):
        batch_ex = batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch
        batch_spo = extract_spo_list(args, tokenizer, id2predicate, id2label, label2id, model, batch_ex, batch_token_ids,
                                     batch_mask)
        for i, ex in enumerate(batch_ex):
            R = batch_spo[i]
            id = ex['id']
            spo_list = list(R)
            triples = []
            for spo in spo_list:
                s = spo[0]
                p = spo[1]
                o = spo[2]
                triple = {"s": tuple(s), "p": p, "o": tuple(o)}
                triples.append(triple)
            total[id] = triples
    json.dump(total, f, ensure_ascii=False, indent=2)


def predict():
    output_path = os.path.join(args.output_path)
    test_path = os.path.join(args.base_path, args.dataset, "test.json")
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # label
    label_list = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH", "MST"]
    id2label, label2id = {}, {}
    for i, l in enumerate(label_list):
        id2label[str(i)] = l
        label2id[l] = i

    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))
    tokenizer = Tokenizer(args.bert_vocab_path)

    test_dataloader = data_generator(args, test_data, tokenizer, [predicate2id, id2predicate], [label2id, id2label],
                                     args.test_batch_size, random=False, is_train=False)
    for fold in range(1, 4):
        config = BertConfig.from_pretrained(args.pretrained_model_path)
        config.num_p = len(id2predicate)
        config.num_label = len(label_list)
        config.rounds = args.rounds
        config.fix_bert_embeddings = args.fix_bert_embeddings
        train_model = GRTE.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path, config=config)

        train_model.to("cuda")
        save_dir = f"{args.output_path}"
        train_model.load_state_dict(
            torch.load(f"{save_dir}/model_{fold}.pth", map_location="cuda"))
        evaluate(args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader, fold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--dataset', default='bdci', type=str)
    parser.add_argument('--rounds', default=4, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--max_len', default=200, type=int)
    parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
    parser.add_argument('--bert_vocab_path',
                        default="pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large/vocab.txt", type=str)
    parser.add_argument('--pretrained_model_path',
                        default="pretrain_models/chinese_pretrain_mrc_roberta_wwm_ext_large", type=str)
    parser.add_argument('--base_path', default="data", type=str)
    parser.add_argument('--output_path', default="output", type=str)
    parser.add_argument('--result_path', default="result", type=str)
    args = parser.parse_args()

    predict()

    rg = resultGenerator()
    rg.merge_k_fold()
    rg.merge_text()
