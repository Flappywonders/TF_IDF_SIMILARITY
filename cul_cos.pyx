def cul_cos(wordlist1, wordlist2, tfidflist1, tfidflist2):
    result_dot = 0.0
    result_len_l1 = 0.0
    result_len_l2 = 0.0
    index1 = 0
    for w in wordlist1:
        index2 = 0
        for w2 in wordlist2:
            if w == w2:
                result_dot = \
                    result_dot + \
                    tfidflist1[index1] * \
                    tfidflist2[index2]
            index2 = index2 + 1
        index1 = index1 + 1
    if result_dot == 0.0:
        return 0
    length1 = len(tfidflist1)
    length2 = len(tfidflist2)
    for i in range(length1):
        result_len_l1 = result_len_l1 + tfidflist1[i] * \
                        tfidflist1[i]
    for i in range(length2):
        result_len_l2 = result_len_l2 + tfidflist2[i] * \
                        tfidflist2[i]
    return result_dot / pow(result_len_l1 * result_len_l2, 0.5)
