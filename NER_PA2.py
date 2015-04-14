# nltk stanford NER -http://stackoverflow.com/questions/18371092/stanford-named-entity-recognizer-ner-functionality-with-nltk
# python interface for stanford NER - https://github.com/dat/stanford-ner
# using pyner - python interface to stanford NER

#----------

import ner
import nltk 
from nltk import word_tokenize


#or using nltk stanford NER
# Use this for now - take a few seconds (a little slow)

from nltk.tag.stanford import NERTagger
def queryForEntity2(expectedEntity,passage):
    st = NERTagger('/Users/srinisha/Downloads/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz','/Users/srinisha/Downloads/stanford-ner-2014-06-16/stanford-ner.jar') 
    answer=st.tag(passage.split()) 
    print answer
    answers=[]
    for j,currentExpectedEntity in enumerate(expectedEntity):
        for i,pair in enumerate(answer):
            if(pair[1]==currentExpectedEntity):
                answers.append(answer[i])   
    return answers

expectedEntity=["PERSON","LOCATION"]
answersList=queryForEntity2(expectedEntity,"Ezra is great in Ithaca")
print "answer using nltk.tag.stanford NERTagger" 
print answersList

if answersList==[]:
    # Using POS tagging to get noun phrase
    print "ner is of no use"
    answer = word_tokenize("Ezra is great")
    answers=nltk.pos_tag(answer)
    print answers
    for i,pair in enumerate(answers):
        if(pair[1]=="NNP"):
            answersList.append(answer[i]) 
    print "noun pharases"
    print answersList

# !!!!!!!!!!OR!!!!!! 



#def queryForEntity1(expectedEntity,passage):
#    tagger = ner.SocketNER(host='localhost', port=8081) # requires server to be started
#    answer=tagger.get_entities(passage)
#    print answer
#    answers=[]
#    for j,currentExpectedEntity in enumerate(expectedEntity):
#        for i,pair in enumerate(answer):
         # pair is not working properly for some entities like location
#            if(pair[1]==currentExpectedEntity):
#                answers.append(answer[i]) 
#    return answers
#    
#expectedEntity=["PERSON","LOCATION"]
#answersList=queryForEntity1(expectedEntity,"Ezra is great in Ithaca")
#print "answer using python interface & NER" 
#print answersList
#if answersList==[]:
#    # Using POS tagging to get noun phrase
#    print "ner is of no use"
#    answer = word_tokenize("Ezra is great")
#    answers=nltk.pos_tag(answer)
#    print answers
#    for i,pair in enumerate(answers):
#        if(pair[1]=="NNP"):
#            answersList.append(answer[i]) 
#    print "noun pharases"
#    print answersList



