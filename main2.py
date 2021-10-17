# step1: text reading and pre-processing
from math import log10
import multiprocessing
import re
import time
import pickle
from copy import deepcopy
import cul_cos


words = {}
with open("raw.txt", encoding='gbk') as f:
    line = f.readline()
    while line:
        label = line[0:19]
        if not label:
            continue
        symbols = "[A-Za-z0-9\!\%\[\]\,\。\.\，\、\~\?\(\)\（\）\？\！\“\”\:\：\;\"\"\；\……&\-\_\|\．\Ａ．Ｂ．Ｃ\*\^/\n]"
        line_n = re.sub(symbols, '', line)
        word_list = [x for x in line_n.split(" ") if x != '']
        words[label] = word_list
        line = f.readline()
    f.close()

# print(words)
# '19980131-04-013-025': ['惊', '溅', '起', '不可言传', '的', '天籁'],
# '19980131-04-013-026': ['怀', '揣', '这', '如泣如诉', '的', '呵护'],
# '19980131-04-013-027': ['才', '发觉', '已', '迷失', '了', '来路']
print("Pre-processing finished!")

# step2: convert words to vector
# build a dictionary, compute the TF and IDF
time_start = time.time()

TF_IDF_dict = {}
IDF_dict = {}
TF_dict = {}
idf_compute = set()

for label in words:
    # compute TF
    tf_dict = {}
    length_sentence = float(len(words[label]))
    for w in words[label]:
        tf_dict[w] = tf_dict.get(w, 0.) + (1.0 / length_sentence)
        # mark the word existing in the file
        idf_compute.add(w)
    TF_dict[label] = tf_dict
    # compute the intermediate IDF result
    for w in idf_compute:
        IDF_dict[w] = IDF_dict.get(w, 0.) + 1.0
    idf_compute.clear()
print("TF computing finished!")


# compute IDF
length_file = float(len(words))
for w in IDF_dict:
    IDF_dict[w] = log10(length_file / IDF_dict[w])
print("IDF computing finished!")


# compute TF-IDF
for label in words:
    temp_TD_IDF = []
    for w in words[label]:
        tfidf = IDF_dict[w] * TF_dict[label][w]
        #np.float32(tfidf)
        temp_TD_IDF.append(tfidf)
    TF_IDF_dict[label] = temp_TD_IDF
print("TF-IDF computing finished")

# print(TF_IDF_dict)
# '19980131-04-013-025': [0.5413809059569179, 0.6146030215952949, 0.22748035418363668, 0.7149463534832887, 0.0311159817302449, 0.5984513527606189],
# '19980131-04-013-026': [0.5142596897073012, 0.5852544784193481, 0.14736016898722598, 0.7149463534832887, 0.0311159817302449, 0.6354261443633449],
# '19980131-04-013-027': [0.27178554637254954, 0.6146030215952949, 0.19215959229055427, 0.6354261443633449, 0.08359674791990582, 0.7149463534832887]


# pre-processing for numtiprocessing: split the dict[0:n1][n1:n2][n2:n3][n3:end]
keys = list(TF_IDF_dict.keys())
splitPoints = [0, 2500, 6000, 10000, len(keys)]
TF_IDF_dict_list = []

for i in range(4):
    tmp_dict = {}
    for k in keys[splitPoints[i]:splitPoints[i+1]]:
        tmp_dict[k] = TF_IDF_dict[k]
    TF_IDF_dict_list.append(deepcopy(tmp_dict))


def compute(Dict, fileorder):
    global r
    if fileorder == 0:
        r = open("result0.pickle", "wb")
    elif fileorder == 1:
        r = open("result1.pickle", "wb")
    elif fileorder == 2:
        r = open("result2.pickle", "wb")
    elif fileorder == 3:
        r = open("result3.pickle", "wb")

    for label1 in Dict:
        for label2 in TF_IDF_dict:
            if label1 >= label2:
                continue
            pickle.dump(label1 + " " + label2 + " " +
                        str(cul_cos.cul_cos(words[label1], words[label2],
                                            TF_IDF_dict[label1],TF_IDF_dict[label2]))
                        + "\n", r)
    f.close()


print("start cosine computing")
p0 = multiprocessing.Process(target= compute, args= (TF_IDF_dict_list[0], 0))
p1 = multiprocessing.Process(target= compute, args= (TF_IDF_dict_list[1], 1))
p2 = multiprocessing.Process(target= compute, args= (TF_IDF_dict_list[2], 2))
p3 = multiprocessing.Process(target= compute, args= (TF_IDF_dict_list[3], 3))
p0.start()
p1.start()
p2.start()
p3.start()
p0.join()
p1.join()
p2.join()
p3.join()

time_end = time.time()
print('time cost', time_end - time_start, 's')

