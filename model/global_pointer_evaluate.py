import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras
import json
from bert4keras.snippets import open, to_array


class GlobalPointerEvaluate(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, model, valid_data, tokenizer, max_token_length, model_weight_file_path):
        self.best_val_f1 = 0.
        self.model = model
        self.valid_data = valid_data
        self.tokenizer = tokenizer
        self.model_weight_file_path = model_weight_file_path
        self.max_token_length = max_token_length

    def on_epoch_end(self, epoch, logs=None):
        # optimizer.apply_ema_weights()
        f1, precision, recall = self.evaluate(self.valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.model_weight_file_path)
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

    def evaluate(self, data):
        """评估函数，计算f1、precision、recall
        """
        X, Y, Z = 1e-10, 1e-10, 1e-10
        f = open('dev_pred.json', 'w', encoding='utf-8')
        pbar = tqdm()
        for d in data:
            R = set([SPO(spo=spo, tokenizer=self.tokenizer) for spo in self.extract_spoes(d['text'])])
            T = set([SPO(spo=spo, tokenizer=self.tokenizer) for spo in d['spo_list']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
            s = json.dumps({
                'text': d['text'],
                'spo_list': list(T),
                'spo_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            },
                ensure_ascii=False,
                indent=4)
            f.write(s + '\n')
        pbar.close()
        f.close()
        return f1, precision, recall

    def extract_spoes(self, text, threshold=0):
        """抽取输入text所包含的三元组
        """
        tokens = self.tokenizer.tokenize(text, maxlen=self.max_token_length)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_token_length)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        outputs = self.model.predict([token_ids, segment_ids])
        outputs = [o[0] for o in outputs]
        # 抽取subject和object
        subjects, objects = set(), set()
        outputs[0][:, [0, -1]] -= np.inf
        outputs[0][:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(outputs[0] > threshold)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))
        # 识别对应的predicate
        spoes = set()
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add((
                        text[mapping[sh][0]:mapping[st][-1] + 1], self.id2predicate[p],
                        text[mapping[oh][0]:mapping[ot][-1] + 1]
                    ))
        return list(spoes)


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """

    def __init__(self, spo, tokenizer):
        self.spo = spo
        self.tokenizer = tokenizer
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox
