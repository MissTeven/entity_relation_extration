import json
import random
import re
import string
import threading


def train_generator():
    fr = open('bdci/train_bdci.json').readlines()
    fw = open('bdci/train.json', 'w', encoding='utf8')

    arr_all = []

    for i in fr:
        i = i.strip()
        if i == "":
            continue

        dic_single = {}
        arr_single = []

        data = json.loads(i)
        id = data['ID']
        text = data['text']
        spo_list = data['spo_list']

        # id_input = int(id.replace('AT', '').lstrip('0'))
        dic_single['id'] = id
        dic_single['text'] = text
        dic_single['spos'] = []

        if text in arr_all:
            continue

        if len(text) > 200:
            for spo in spo_list:
                h = spo['h']
                t = spo['t']
                relation = spo['relation']
                line = [(h['pos'][0], h['pos'][1], h['name']), relation, (t['pos'][0], t['pos'][1], t['name'])]
                arr_single.append(line)

            # dict_all[text] = arr_single
            spos = sorted(arr_single)

            split_blocks = cut_pattern.split(text)
            split_blocks.append("")
            split_blocks = ["".join(i) for i in zip(split_blocks[0::2], split_blocks[1::2])]
            current_text = ""
            total_blocks = []
            for block in split_blocks:
                if len(current_text + block) > 200:
                    total_blocks.append(current_text)
                    current_text = block
                else:
                    current_text += block

            if len(current_text) > 0:
                total_blocks.append(current_text)

            start_idx = 0
            end_idx = 0
            for t_idx, block_text in enumerate(total_blocks):

                end_idx += len(block_text)
                new_spos = []
                for spo in spos:

                    h_sidx, h_eidx, h_name = spo[0]
                    t_sidx, t_eidx, t_name = spo[2]

                    if start_idx <= h_eidx < end_idx and start_idx <= t_eidx <= end_idx:
                        new_spos.append(spo)

                if t_idx == 0:
                    line = {"id": id, "text": block_text, "spos": new_spos}
                    arr_all.append(line)

                else:
                    new_spos2 = []
                    for spo in new_spos:
                        h_sidx, h_eidx, h_name = spo[0]
                        relation = spo[1]
                        t_sidx, t_eidx, t_name = spo[2]
                        tmp = []
                        tmp.append((h_sidx - start_idx, h_eidx - start_idx, h_name))
                        tmp.append(relation)
                        tmp.append((t_sidx - start_idx, t_eidx - start_idx, t_name))
                        new_spos2.append(tmp)

                    line = {"id": id, "text": block_text, "spos": new_spos2}
                    arr_all.append(line)
                start_idx += end_idx

        else:
            for spo in spo_list:
                h = spo['h']
                t = spo['t']
                relation = spo['relation']

                arr_h = []
                arr_h.append(h['pos'][0])
                arr_h.append(h['pos'][1])
                arr_h.append(h['name'])

                arr_t = []
                arr_t.append(t['pos'][0])
                arr_t.append(t['pos'][1])
                arr_t.append(t['name'])

                arr_spo = []
                arr_spo.append(arr_h)
                arr_spo.append(relation)
                arr_spo.append(arr_t)
                dic_single['spos'].append(arr_spo)

            arr_all.append(dic_single)

    fw.writelines(json.dumps(arr_all, ensure_ascii=False, indent=2))


def test_generator():
    fr = open('bdci/evalA.json', 'r', encoding='utf8').readlines()
    fw = open('bdci/test.json', 'w', encoding='utf8')

    datas = []
    for line in fr:
        case_data = json.loads(line)
        idx = case_data['ID']
        txt = case_data['text']
        if len(txt) > 200:
            split_blocks = cut_pattern.split(txt)
            split_blocks.append("")

            split_blocks = ["".join(i) for i in zip(split_blocks[0::2], split_blocks[1::2])]
            current_text = ""
            total_blocks = []
            for block in split_blocks:
                if len(current_text + block) > 200:
                    total_blocks.append(current_text)
                    current_text = block
                else:
                    current_text += block

            if len(current_text) > 0:
                total_blocks.append(current_text)

            for sub_idx, block in enumerate(total_blocks):
                line = {"id": str(idx) + "_{}".format(sub_idx), "text": block}
                datas.append(line)
        else:
            line = {"id": str(idx), "text": txt}
            datas.append(line)

    json.dump(datas, fw, ensure_ascii=False, indent=2)


def enhance():
    synonyms = synonym()
    arr_all__text_set = set([])
    layered_synonyms = []
    for word, wordSynonyms in synonyms.items():
        if len(layered_synonyms) < len(word):
            for i in range(len(layered_synonyms), len(word)):
                layered_synonyms.append({})
        layered_synonyms[len(word) - 1][word] = wordSynonyms

    for wordSynonyms in layered_synonyms:
        if len(wordSynonyms) > 0:
            maxLen = layered_synonyms.index(wordSynonyms) + 1
            res = replaceBySynonym(wordSynonyms, maxLen)
            print("res count:{}".format(len(res)))
            arr_all = []
            for item in res:
                if item["text"] not in arr_all__text_set:
                    arr_all.append(res)
                    arr_all__text_set.add(item["text"])
            if len(arr_all) > 0:
                threading.Thread(target=save(arr_all, 'bdci/enhance{}.json'.format(maxLen))).start()


def save(data, path):
    print("data count:{} save:{}".format(len(data), path))
    json.dump(data, open(path, 'w', encoding='utf8'), ensure_ascii=False, indent=2)


def replaceBySynonym(synonym, maxLen):
    print("replaceBySynonym maxLen:{}".format(maxLen))
    fr = open('bdci/train_bdci.json').readlines()
    arr_all = []
    for line in fr:
        line = line.strip()
        if line == "":
            continue
        data = json.loads(line)
        id = data['ID']
        text = data['text']
        spo_list = data['spo_list']
        entities = {}
        for spo in spo_list:
            entities[spo['h']['name']] = spo['h']['pos']
            entities[spo['t']['name']] = spo['t']['pos']

        # for (entity, entityPos) in entities.items():
        #     print("{}:{}".format(entity, entityPos))

        token_start_index = 0
        temp = ""
        # 从第一个字遍历文本
        while token_start_index < len(text):
            # print("token_start_index:{}".format(token_start_index))
            token_start_index_temp = token_start_index
            token_end_index = token_start_index + 1
            # 当有实体是以当前词为起始的，则跳过该实体词汇
            for entity in entities.keys():
                if text[token_start_index:token_start_index + len(entity)] == entity:
                    temp = temp + entity
                    token_start_index = token_start_index + len(entity)
                    continue
            while token_start_index == token_start_index_temp and token_end_index < min(len(text),
                                                                                        token_start_index + maxLen + 1):
                # 查看当前字是否跨入实体内，跨入实体内
                for entityPos in entities.values():
                    entity_start_index = entityPos[0]
                    if token_end_index > entity_start_index > token_start_index:
                        temp = temp + text[token_start_index]
                        token_start_index = token_start_index + 1
                        break
                # 获取以当前字为头的词
                word = text[token_start_index:token_end_index]
                # 如果当前词拥有一个同义词
                if synonym.get(word) is not None:
                    # 如果当前词具备被替换的条件：被随机选中
                    if random.randint(0, 1) == 1:
                        word_synos = synonym.get(word)
                        syno = word_synos[random.randint(0, len(word_synos) - 1)]
                        print("{}->{}".format(word, syno))
                        # 如果替换的词和原来的词长度不一样并且实体位于替换词的右侧则需要更新实体的坐标
                        if len(word) != len(syno):
                            pos_diff = len(syno) - len(word)
                            for (entity, entityPos) in entities.items():
                                entity_start_index = entityPos[0]
                                entity_end_index = entityPos[1]
                                if entity_start_index >= token_end_index:
                                    entity_start_index = entity_start_index + pos_diff
                                    entity_end_index = entity_end_index + pos_diff
                                    entities[entity] = [entity_start_index, entity_end_index]
                        temp = temp + syno
                    else:
                        temp = temp + word
                    token_start_index = token_start_index + len(word)
                    break
                else:
                    token_end_index = token_end_index + 1
            if token_start_index == token_start_index_temp and token_end_index == min(len(text),
                                                                                      token_start_index + maxLen + 1):
                temp = temp + text[token_start_index]
                token_start_index = token_start_index + 1
        text = temp
        # for (entity, entityPos) in entities.items():
        #     print("{}:{}".format(entity, entityPos))
        for spo in spo_list:
            spo['h']['pos'] = entities[spo['h']['name']]
            spo['t']['pos'] = entities[spo['t']['name']]
        dic_single = {'id': id, 'text': text, 'spos': spo_list}
        arr_all.append(dic_single)
    return arr_all


def synonym():
    fr = open('synonym.txt').readlines()
    synonyms = {}
    for line in fr:
        line = line.replace("\n", "")
        words = line.split(" ")
        if words == "" or len(words) < 2:
            continue
        if synonyms.get(words[0]) is None:
            synonyms[words[0]] = [words[1]]
        else:
            synonyms[words[0]].append(words[1])

        if synonyms.get(words[1]) is None:
            synonyms[words[1]] = [words[0]]
        else:
            synonyms[words[1]].append(words[0])
    return synonyms


if __name__ == '__main__':
    cut_pattern = re.compile(r'([，。！？、])')
    enhance()
