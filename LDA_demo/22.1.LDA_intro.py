# !/usr/bin/python
# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
from pprint import pprint
import pandas as pd

import jieba
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# import sys
# default_encoding = 'utf-8'
# if sys.getdefaultencoding() != default_encoding:
#     reload(sys)
#     sys.setdefaultencoding(default_encoding)

if __name__ == '__main__':
    f = open('1.txt', encoding='utf-8')
    # words= pd.read_csv('300text.csv', encoding='utf-8')
    stop_list = set('for a of the and to in'.split())
    # texts = [line.strip().split() for line in f]
    # print(words)
    texts = [[word for word in line.split() if word not in stop_list] for line in f]
    split_text = ['/'.join(text) for text in texts]

    print ('Text = ')
    pprint(texts)

    dictionary = corpora.Dictionary(texts) #把所有词语取一个set(),并对set中每个单词分配一个Id号的map;
    V = len(dictionary)
    print (dictionary.get(0))
    corpus = [dictionary.doc2bow(text) for text in texts]
    print (corpus)
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    #
    # print ('TF-IDF:')
    # for c in corpus_tfidf:
    #     print (c)

    # print ('\nLSI Model:')
    # lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word=dictionary)
    # topic_result = [a for a in lsi[corpus_tfidf]]
    # pprint(topic_result)
    # print ('LSI Topics:')
    # pprint(lsi.print_topics(num_topics=2, num_words=5))
    # similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])   # similarities.Similarity()
    # print ('Similarity:')
    # pprint(list(similarity))

    print ('\nLDA Model:')
    num_topics = 3
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001)
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print ('Document-Topic:\n')
    pprint(doc_topic)
    for doc_topic in lda.get_document_topics(corpus_tfidf):
        print (doc_topic)
    for topic_id in range(num_topics):
        print ('Topic', topic_id)
        pprint(lda.get_topic_terms(topicid=topic_id))
        pprint(lda.show_topic(topic_id))
    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    # print ('Similarity:')
    # pprint(list(similarity))

    print ('\nHDP Model:')
    hdp = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hdp[corpus_tfidf]]
    # print ('\ndoc topic:')
    # for doc_topic1 in hdp.get_document_topics(corpus_tfidf):
    #     print (doc_topic1)
    # topic_result = hdp[corpus_tfidf]

    print ('\nDocument-Topic--HDP Model:')
    pprint(topic_result)
    topic_result_x = None
    for topic_result_item in topic_result:
        topic_result_x = [x for x,y in topic_result_item]
        topic_result_y = [y for x,y in topic_result_item]
        max_topic_result_y = max(topic_result_y)
        max_topic_result_y_pos = topic_result_y.index(max_topic_result_y)
        print ('HDP max Topics in :')
        print (topic_result_item[max_topic_result_y_pos])
        print ('\n')

    topic = [[x for x,y in topic_result_item]for topic_result_item in topic_result]
    topic_content = [[','.join([x for x,y in hdp.show_topic(topic_id=topic_id)]) for topic_id in item]for item in topic]
    print ('HDP Topics:')
    print (hdp.print_topics(num_topics=-1, num_words=5))
    print (hdp.show_topic(topic_id=0))
    output = pd.DataFrame({'splited_text': split_text, 'topic': topic, 'topin_content': topic_content})

    output.to_csv('output.csv', encoding='gbk')
