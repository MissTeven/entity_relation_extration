# _*_ coding:utf-8 _*_
import json
import os
import sys

#将json文件中的内容转换为字典，key为ID，value为每一行的内容
def convert_str_to_dict(path):
    f = open(path, 'r',encoding='UTF-8')
    id_to_content_dict = {}
    lines = f.readlines()
    for line in lines:
        try:
            s = json.loads(line)
            id = s["ID"]
            id_to_content_dict[id] = s #放入到字典中
        except Exception as e:
            pass

    return id_to_content_dict

#将两个字典merge
def merge_dict(json_path1, json_path2):
    dict1 = convert_str_to_dict(json_path1)
    dict2 = convert_str_to_dict(json_path2)
    dict_merge = {}  # 并集
    for key, value in dict1.items():
        value_merge = value
        spo_list_merge = []
        spo_list1 = value["spo_list"] #第一个文件的spo_list
        spo_list2 = [] #第二个文件的spo_list
        if key in dict2:
            value2 = dict2[key]
            spo_list2 = value2["spo_list"]

        for item in spo_list1:
            if item not in spo_list_merge:
                spo_list_merge.append(item)

        for item in spo_list2:
            if item not in spo_list_merge:
                spo_list_merge.append(item)

        value_merge["spo_list"] = spo_list_merge
        dict_merge[key] = value_merge
    return dict_merge

def mix_dict(json_path1, json_path2):
    dict1 = convert_str_to_dict(json_path1)
    dict2 = convert_str_to_dict(json_path2)
    dict_mix = {}  # 交集
    for key, value in dict1.items():
        value_mix = value
        spot_list_mix = []
        spo_list1 = value["spo_list"]  # 第一个文件的spo_list
        spo_list2 = []  # 第二个文件的spo_list
        if key in dict2:
            value2 = dict2[key]
            spo_list2 = value2["spo_list"]

        #取交集，只有两个都有，才放入进去
        for item in spo_list1:
            if item in spo_list2:
                spot_list_mix.append(item)

        value_mix["spo_list"] = spot_list_mix
        dict_mix[key] = value_mix
    return dict_mix

if __name__ == "__main__":

    if len(sys.argv) == 3:
        json_path1 = sys.argv[1]
        json_path2 = sys.argv[2]
        # 将两个map合并，取并集
        dict_merge = merge_dict(json_path1, json_path2)
        #取交集
        dict_mix = mix_dict(json_path1, json_path2)

        #如果有文件就删除，重新生成
        if os.path.exists("spo_mix.json"):
           os.remove("spo_mix.json")

        if os.path.exists("spo_merge.json"):
            os.remove("spo_merge.json")

        fp = open("spo_mix.json", "w",encoding='UTF-8')
        for key , value in dict_mix.items():
            s = str(value)
            s = s.replace('\'', '"')
            fp.write(s)
            fp.write("\n")

        fp2 = open("spo_merge.json", "w",encoding='UTF-8')
        for key, value in dict_merge.items():
            s = str(value)
            s = s.replace('\'', '"')
            fp2.write(s)
            fp2.write("\n")
