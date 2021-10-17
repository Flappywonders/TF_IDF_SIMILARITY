# step1: text reading and pre-processing
import math
import re
import time

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
        tf_dict[w] = tf_dict.get(w, 0.) + (1.0/ length_sentence)
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
    IDF_dict[w] = math.log10(length_file / IDF_dict[w])
print("IDF computing finished!")


# compute TF-IDF
for label in words:
    temp_TD_IDF = []
    for w in words[label]:
        temp_TD_IDF.append(IDF_dict[w] * TF_dict[label][w])
    TF_IDF_dict[label] = temp_TD_IDF
print("TF-IDF computing finished")


#print(TF_IDF_dict)
# '19980131-04-013-025': [0.5413809059569179, 0.6146030215952949, 0.22748035418363668, 0.7149463534832887, 0.0311159817302449, 0.5984513527606189],
# '19980131-04-013-026': [0.5142596897073012, 0.5852544784193481, 0.14736016898722598, 0.7149463534832887, 0.0311159817302449, 0.6354261443633449],
# '19980131-04-013-027': [0.27178554637254954, 0.6146030215952949, 0.19215959229055427, 0.6354261443633449, 0.08359674791990582, 0.7149463534832887]

doc_len = 19484
cnt = 0

# compute the similarities between label1 and label2
with open("result.txt", "w") as r:
    for label1 in TF_IDF_dict:
        cnt = cnt + 1
        if cnt % 20 == 0:
            print("cul " + str(cnt/20) + " ‰\n")
            time_end = time.time()
            print('time cost',time_end - time_start,'s')
        for label2 in TF_IDF_dict:
            if label1 >= label2:
                continue
            result_dot = 0.0
            result_len_l1 = 0.0
            result_len_l2 = 0.0
            index1 = 0
            for w in words[label1]:
                index2 = 0
                for w2 in words[label2]:
                    if w == w2:
                        result_dot = \
                            result_dot + \
                            TF_IDF_dict[label1][index1]*\
                            TF_IDF_dict[label2][index2]
                    index2 = index2 + 1
                index1 = index1 + 1
            if result_dot == 0.0:
                r.write(label1 + " " + label2 + " " + "0.0\n")
                continue
            for i in range(len(TF_IDF_dict[label1])):
                result_len_l1 = result_len_l1 + pow(TF_IDF_dict[label1][i], 2)
            for i in range(len(TF_IDF_dict[label2])):
                result_len_l2 = result_len_l2 + pow(TF_IDF_dict[label2][i], 2)
            result = result_dot/ pow((result_len_l1 * result_len_l2), 0.5)
            r.write(label1 + " " + label2 + " " + str(result) + "\n")
    r.close()

time_end = time.time()
print('time cost',time_end - time_start,'s')

