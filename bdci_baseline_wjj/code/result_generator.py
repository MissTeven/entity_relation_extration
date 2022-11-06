import json
import os
from collections import Counter


class resultGenerator:
    def __init__(self):
        self.result = {}

    def merge_k_fold(self):
        base_path = 'result'
        files = os.listdir(base_path)
        total = {}
        for file in files:
            if "json" not in file:
                continue
            print("cuurent file:", file)
            with open(os.path.join(base_path, file), 'r', encoding='utf8') as f:
                datas = eval(f.read())
                for k, vv in datas.items():
                    id = k
                    spos = []
                    for v in vv:
                        s = tuple(v['s'])
                        p = v['p']
                        o = tuple(v['o'])
                        spo = (s, p, o)
                        spos.append(spo)
                    if id in total:
                        total[id].extend(spos)
                    else:
                        total[id] = spos

        origin_dataset = open('data/bdci/test.json', 'r', encoding='utf8')
        origins = json.load(origin_dataset)
        self.id2text = {}
        for data in origins:
            id = data['id']
            text = data['text']
            self.id2text[id] = text

        for k, vv in total.items():
            spos = dict(sorted(dict(Counter(vv)).items(), key=lambda x:x[1], reverse=True))
            triples = []
            for spo, v in spos.items():
                #此阈值可调
                if v >= 2:
                    triple = {"s": spo[0], "p": spo[1], "o": spo[2]}
                    triples.append(triple)
                else:
                    break
            self.result[k] = triples

    def merge_text(self):
        # 将切分的句子进行合并
        final = {}
        for k, v in self.result.items():
            text = self.id2text[k]
            real_id = k.split("_")[0]
            if real_id in final:
                final[real_id].append({"text": text, "spos": v})
            else:
                final[real_id] = [{"text": text, "spos": v}]

        fout = open('res.json', 'w', encoding='utf8')
        for k, vv in final.items():
            if len(vv) == 1:
                text = vv[0]['text']
                spo_list = []
                spos = vv[0]['spos']
                for spo in spos:
                    s_sidx, s_eidx, s_entity = spo['s']
                    p = spo['p']
                    o_sidx, o_eidx, o_entity = spo['o']

                    one = {"h": {"name": s_entity, "pos": [s_sidx, s_eidx]}, "t": {"name": o_entity, "pos": [o_sidx, o_eidx]}, "relation": p}
                    spo_list.append(one)
                line = {"ID": k, "text": text, "spo_list": spo_list}
                fout.write(json.dumps(line, ensure_ascii=False)+"\n")
            elif len(vv) > 1:
                spo_list = []
                total_text = ""
                for v_idx, v in enumerate(vv):
                    text = v['text']
                    spos = v['spos']
                    if v_idx == 0:
                        for spo in spos:
                            s_sidx, s_eidx, s_entity = spo['s']
                            p = spo['p']
                            o_sidx, o_eidx, o_entity = spo['o']

                            one = {"h": {"name": s_entity, "pos": [s_sidx, s_eidx]},
                                   "t": {"name": o_entity, "pos": [o_sidx, o_eidx]}, "relation": p}
                            spo_list.append(one)
                    else:
                        for spo in spos:
                            s_sidx, s_eidx, s_entity = spo['s']
                            p = spo['p']
                            o_sidx, o_eidx, o_entity = spo['o']

                            one = {"h": {"name": s_entity, "pos": [s_sidx+len(total_text), s_eidx+len(total_text)]},
                                   "t": {"name": o_entity, "pos": [o_sidx+len(total_text), o_eidx+len(total_text)]}, "relation": p}
                            spo_list.append(one)
                    total_text += text
                line = {"ID": k, "text": total_text, "spo_list": spo_list}
                fout.write(json.dumps(line, ensure_ascii=False)+"\n")


