# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:54:08 2019

@author: 64191
"""
import numpy as np
def loadDataSet():
    '''创建一些实验样本'''
    postingList = [['my','dog','has','flea','problems','help','please'],
                  ['maybe','not','take','him','to','dog','park','stupid'],
                  ['my','dalmation','is','so','cute','I','love','him'],
                  ['stop','posting','stupid','worthless','garbage'],
                  ['mr','licks','ate','my','steak','how','to','stop','him'],
                  ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]  #0代表正常言论   1表示侮辱性
    return postingList,classVec
def createVocabList(dataSet):
    '''返回一个包含所有文档中出现的不重复的词条集合'''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)   #创建两个集合的并集
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    '''接受词汇表和某个文档，返回该文档向量'''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:{} is not in my Vocabulary".format(word))
    return returnVec
def setOfWords2Vec(vocabList,inputSet):
    '''接受词汇表和某个文档，返回该文档向量'''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:{} is not in my Vocabulary".format(word))
    return returnVec

