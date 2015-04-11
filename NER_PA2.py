# nltk stanford NER -http://stackoverflow.com/questions/18371092/stanford-named-entity-recognizer-ner-functionality-with-nltk
# python interface for stanford NER - https://github.com/dat/stanford-ner
# using pyner - python interface to stanford NER
import ner

def queryForEntity1(expectedEntity,passage):
    tagger = ner.SocketNER(host='localhost', port=8081) # requires server to be started
    answer=tagger.get_entities(passage)
    answers=[]
    for i,pair in enumerate(answer):
        if(pair[1]==expectedEntity):
            answers.append(answer[i])       
    return answers

answersList=queryForEntity1("PERSON","Rami Eid is studying at Stony Brook University in NY") # person is not tagged while using pyner
print "answer using python interface & NER" 
print answersList

#or  using nltk stanford NER

from nltk.tag.stanford import NERTagger
def queryForEntity2(expectedEntity,passage):
    st = NERTagger('/Users/srinisha/Downloads/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz','/Users/srinisha/Downloads/stanford-ner-2014-06-16/stanford-ner.jar') 
    answer=st.tag(passage.split()) 
    answers=[]
    for i,pair in enumerate(answer):
        if(pair[1]==expectedEntity):
            answers.append(answer[i])       
    return answers

