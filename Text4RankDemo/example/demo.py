#-*- encoding:utf-8 -*-
from __future__ import print_function
import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence

import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

text = codecs.open('../test/test/2.txt', 'r', 'utf-8').read()
#print(temp)
#text = "有些视频更新得比较慢"
tr4w = TextRank4Keyword()

tr4w.analyze(text=text, lower=True, window=2)

print( '关键词：' )
for item in tr4w.get_keywords(20, word_min_len=1):
    print(item.word, item.weight)

print()
print( '关键短语：' )
for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num= 2):
    print(phrase)

# print()
# print('sentences:')
# for s in tr4w.sentences:
#     print(s)                 # py2中是unicode类型。py3中是str类型。
#
# print()
# print('words_no_filter')
# for words in tr4w.words_no_filter:
#     print('/'.join(words))   # py2中是unicode类型。py3中是str类型。
#
# print()
# print('words_no_stop_words')
# for words in tr4w.words_no_stop_words:
#     print('/'.join(words))   # py2中是unicode类型。py3中是str类型。

# print()
# print('words_all_filters')
# for words in tr4w.words_all_filters:
#     print('/'.join(words))   # py2中是unicode类型。py3中是str类型。