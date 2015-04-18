import math
import re
import os
import datetime
from operator import itemgetter
from nltk.corpus import stopwords
import ner
import nltk 
import collections
from nltk import word_tokenize
from nltk.tag.stanford import NERTagger
from nltk.stem import WordNetLemmatizer

entity_labels = {"How": ["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"], "What": ["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"],"Where": ["LOCATION"], "Who": ["PERSON", "ORGANIZATION"], "When": ["TIME", "DATE"], "How many": ["COUNT","MONEY","PERCENT"],"Which":["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"]}
dict_phrases = {}

WL = WordNetLemmatizer()

def similarity(query_dict,top_docs_dict):
    query_no = query_dict.keys()

    for i in range(len(query_no)):
        list_scores = []
        query = query_no[i]
        doc_dict = top_docs_dict[query]
        query_text = query_dict[query]
        idf_values = find_idf(query_text,doc_dict)
        query_tf = find_tf(query_text)
        query_tf_idf = get_tf_idf(query_tf,idf_values)
        query_normalized = cosine_normalize(query_tf_idf)
        #return list of ngrams here and iterate over each n-gram
        keys_list_docs = doc_dict.keys()
        fwrite = open("tempfile", "w+")
            
        
        for l in range(len(doc_dict)):
            n_grams = doc_dict[keys_list_docs[l]]
            for n in range(len(n_grams)):
                dict_scores = {}
                doc_tf_dict = find_tf(n_grams[n])
                doc_normalized = cosine_normalize(doc_tf_dict)
                score = calculate_dot_product(query_normalized,doc_normalized)
                #print n_grams[n],score 
                dict_scores['phrase'] = n_grams[n]
                dict_scores['score'] = score
                list_scores.append(dict_scores)
        newlist = sorted(list_scores, key=itemgetter('score'), reverse=True) 
        dict_phrases[query] = newlist
        for item in newlist:
            fwrite.write("%s\n" % item)      
        fwrite.close()
        
def get_tf_idf(query_tf,idf_values):
    keys = query_tf.keys()
    for k in range(len(keys)):
        value = query_tf.get(keys[k]) 
        if(idf_values.has_key(keys[k])):
            tf_idf = value * idf_values.get(keys[k]) 
        else:
            tf_idf = 0
        query_tf[keys[k]] = tf_idf
    return query_tf
    
def find_idf(query,doc_dict):
    words = query.split(" ")
    idf_map = {}
    doc_size = len(doc_dict)
    keys = doc_dict.keys()
    
    for w in range(len(words)):
        word = words[w]
        count = 0
        for i in range(len(doc_dict)):
            list_ngrams = doc_dict[keys[i]]
            for j in range(len(list_ngrams)):
                n_gram = list_ngrams[j]
                if word in n_gram:
                    count = count+1
                    break;
        idf_map[word] = count
        
            
    keys = idf_map.keys()
    for k in range(len(keys)):
        value = idf_map.get(keys[k])
        if( value != 0):
            idf_map[keys[k]] = math.log10(float(doc_size)/float(value))
        else:
            idf_map[keys[k]] = 0
    return idf_map

def find_tf(text):
    words = text.split(" ")
    dict_terms = {}
    for j in range(len(words)):
        if(entity_labels.has_key(words[j])):
            continue;
        else:
            if(dict_terms.has_key(words[j])):
                count = dict_terms.get(words[j])
                dict_terms[words[j]] = count+1
            else:
                dict_terms[words[j]] = 1

    keys = dict_terms.keys()
    for k in range(len(keys)):
        value = dict_terms.get(keys[k])
        dict_terms[keys[k]] = 1 + math.log10(value)
        
    return dict_terms
            
def cosine_normalize(dict_tf):
    sum_denominator = 0
    keys = dict_tf.keys()
    for k in range(len(keys)):
        sum_denominator = sum_denominator + math.pow(dict_tf.get(keys[k]),2)
    denominator = math.sqrt(sum_denominator)
    
 
    
    for k in range(len(keys)):
        value = dict_tf.get(keys[k])
#        print keys[k]
#        print value
#        print denominator
        dict_tf[keys[k]] = value/denominator
        
    return dict_tf
        
def calculate_dot_product(query,document):
    keys = query.keys()
    sum = 0
    for k in range(len(keys)):
        if(document.has_key(keys[k])):
            sum = sum + (query[keys[k]] * document[keys[k]] )
    return sum

def getquerydict(questions_filename):
    questions_file = open(questions_filename, "r")
    
    stop = stopwords.words('english')
    
    query_dict = dict()
    
    for line in questions_file:
        matchQueryNumber = re.match(r'Number: [0-9]+', line, re.M|re.I)
        if matchQueryNumber:
            qnum = matchQueryNumber.group(0).split()
        else:
            if line!="\n":
                line=" ".join([w for w in line.split(" ") if not w in stop])
                newline = re.compile(r'(\n)', re.UNICODE)
                line = newline.sub('',line)
                punctuation = re.compile(r'[\?."\',\(\)&/:]+', re.UNICODE)                
                line = punctuation.sub('',line)
                whitespace = re.compile(r'(\s)+', re.UNICODE)
                line = whitespace.sub(' ', line)
                line=" ".join([WL.lemmatize(i) for i in line.split()])
                line = line.encode()
                query_dict.update({qnum[1]:line})          
    return query_dict

def preprocessing(finalstr):
    newline = re.compile(r'(\n)', re.UNICODE)
    finalstr = newline.sub(' ',finalstr)
    
    multlinesMatch = re.compile(r'(\n)+', re.UNICODE)
    finalstr = multlinesMatch.sub('',finalstr)
    
    punctuation = re.compile(r'[#,^&$\(\)\?";:\'`\[\]{}@\*\-+~=_]+', re.UNICODE)
    finalstr = punctuation.sub('', finalstr)
    
    whitespace = re.compile(r'(\s)+', re.UNICODE)
    finalstr = whitespace.sub(' ', finalstr)
    
    fullstop = re.compile(r'[.]+', re.UNICODE)
    finalstr = fullstop.sub(' .', finalstr)
    
    return finalstr

def getngrams(ngramterms):
    ngram_list = []
    for eachTerm in ngramterms :
        tTerm = eachTerm.split()
        tempTerm = list()
        for eachTempTerm in tTerm :
            eachTempTerm = eachTempTerm.decode('ascii',errors='ignore')
            eachTempTerm = WL.lemmatize(eachTempTerm)
            eachTempTerm = eachTempTerm.encode()
            tempTerm.append(eachTempTerm)
        if len(tempTerm)> 9  :
            for i in range(0, len(tempTerm) - 9) :
                ngram_list.append(tempTerm[i] + " " + tempTerm[i + 1] + " " + tempTerm[i + 2] + " " + tempTerm[i + 3] + " " + tempTerm[i + 4] + " " + tempTerm[i + 5] + " " + tempTerm[i + 6] + " " + tempTerm[i + 7] + " " + tempTerm[i + 8] + " " + tempTerm[i + 9])
        else :
            if len(tempTerm) != 0 :
                temp = tempTerm[0]
                for i in range(1, len(tempTerm)) :
                    temp += " " + tempTerm[i]
                ngram_list.append(temp)
    return ngram_list

def getTopDocsDict(pathTopDocs):
    top_docs_dict = dict()
    for name in os.listdir(pathTopDocs) :
        docs_dict = dict()
        
        filename = pathTopDocs+name
        parseDocNo = name.split(".")
        docNo = parseDocNo[1]
        
        topDocsFile = open(filename, "r")
        whole_data = topDocsFile.read()
    
        textMatch = re.compile(r'<TEXT>(.*?)</TEXT>', re.DOTALL)
        completeText = textMatch.findall(whole_data)
        
        PTagsMatch = re.compile(r'<P>', re.DOTALL)
        PEndTagsMatch = re.compile(r'</P>', re.DOTALL)
        
        index = 0;
        for eachText in completeText:
            eachText = PTagsMatch.sub('', eachText)
            eachText = PEndTagsMatch.sub('', eachText)
    
            tempCompText = ""
    
            for line in eachText.splitlines():
                tempText = preprocessing(line)
                tempCompText += " "+tempText
            
            whitespace = re.compile(r'(\s)+', re.UNICODE)
            tempCompText = whitespace.sub(' ', tempCompText)
    
            ngramterms = tempCompText.split(". ")
            ngram_list = getngrams(ngramterms)
    
            docs_dict.update({index:ngram_list})
            index = index+1
            
        top_docs_dict.update({docNo:docs_dict})
    
    return top_docs_dict


# nltk stanford NER
# Use this for now -- take a few seconds (a little slow)

# nltk stanford NER -http://stackoverflow.com/questions/18371092/stanford-named-entity-recognizer-ner-functionality-with-nltk


def queryForEntity(expectedEntity,passage,pathtoClassifier,pathtoNerjar):
    tagger = ner.SocketNER(host='localhost', port=8081) # requires server to be started
    answer=tagger.get_entities(passage)
    #print answer
    answers=[]
    for j,currentExpectedEntity in enumerate(expectedEntity):
        for key in answer:
            #pair is not working properly for some entities like location
            if(key==currentExpectedEntity):
                for eachAnswer in answer[key]:
                    answerString=eachAnswer.encode()
                    answers.append(answerString) 
    return answers
    
#    st = NERTagger(pathtoClassifier,pathtoNerjar) 
#    answer=st.tag(passage.split()) 
#    answers=[]
#    for j,currentExpectedEntity in enumerate(expectedEntity):
#        for i,pair in enumerate(answer):
#            if(pair[1]==currentExpectedEntity):
#                answerString=pair[0].encode()
#                answers.append(answerString)  
#    return answers


def getAnswers(pathtoClassifier,pathtoNerjar,pathToAnswerFile,query_dict):
    if os.path.exists(pathToAnswerFile):
        f = file(pathToAnswerFile,"w")
    else:
        f = file(pathToAnswerFile,"a")
    query_dict=collections.OrderedDict(sorted(query_dict.items()))
    for query in query_dict: # for every query
        cnt=0
        # entity_labels doesn't yet have entries like What's  
        Query=query_dict[query]
        QueryNo= query;
        #print "QueryNo"
        #print QueryNo
        #print "Current Query"
        #print Query
        f.write("qid"+" "+str(QueryNo)+"\n")

        testHowMany = re.compile("How many") 
        testHow=re.compile("How") 
        testWhen=re.compile("When")
        testWhich=re.compile("Which")
        testWho=re.compile("Who")
        testWhat=re.compile("What")
        testWhere=re.compile("Where")
        if testHow.match(Query):
            expectedEntity=entity_labels["How"]
        if testHowMany.match(Query):
            expectedEntity=entity_labels["How many"]        
        if testWhen.match(Query):
            expectedEntity=entity_labels["When"]
        if testWhich.match(Query):
            expectedEntity=entity_labels["Which"]
        if testWho.match(Query):
            expectedEntity=entity_labels["Who"]
        if testWhat.match(Query):
            expectedEntity=entity_labels["What"]
        if testWhere.match(Query):
            expectedEntity=entity_labels["Where"]
        #print "expectedEntity"
        #print expectedEntity
        #print "New query "
        list_scores=dict_phrases[QueryNo]
        
        for currDict in list_scores: 
            if cnt>=10:
                break
            else:
                currpassage=currDict['phrase']
                #print "current passage (ngram) "
                #print currpassage
                answersList=[]
                if expectedEntity!=[]:
                    answersList=queryForEntity(expectedEntity,currpassage,pathtoClassifier,pathtoNerjar)
                    #if len(answersList)!=0:
                        #print "answer found shouldn't go to POS"
                    #else:
                        #print "Couldnt find any matching entries in entity_labels dictory so didnt go for queryForEntity function"
                    #print cnt
                    #print "answer using nltk.tag.stanford NERTagger" 
                    #print answersList
                    for answer in answersList:
                        if cnt<10:
                            cnt=cnt+1
                            #print cnt
                            f.write(str(cnt)+" "+answer+"\n")
                        else:
                            break
        for currDict in list_scores: 
            if cnt>=10:
                break
            else:
                currpassage=currDict['phrase']
                #print "current passage (ngram) "
                #print currpassage
                answersList=[]
                # since no answer was found from any of the exoected entities noun phrase extraction was done
                # Using POS tagging to get noun phrase
                #print "when answer list should be empty to come here"
                #print "ner is of no use doing POS tagging to get noun phrase"
                answer = word_tokenize(currpassage)
                answers=nltk.pos_tag(answer)
                for i,pair in enumerate(answers):
                    if(pair[1]=="NNP"):
                        answersList.append(answer[i])
                #print "answer found from noun pharases"
                #print answersList
                for answer in answersList:
                    if cnt<10:
                        cnt=cnt+1
                        #print cnt
                        f.write(str(cnt)+" "+answer+"\n")
                    else:
                        break
                                    
                    

        
        
    

def main():
    print "Process started", datetime.datetime.now().time()
    questions_filename = "questions.txt"
    pathTopDocs = "dev1/"  
    
    query_dict = getquerydict(questions_filename)
    print "Query Dictionary Generated", datetime.datetime.now().time()
    
    top_docs_dict = getTopDocsDict(pathTopDocs)
    print "Top Docs Dictionary Generated", datetime.datetime.now().time()
    
    similarity(query_dict,top_docs_dict)
    print "Process completed", datetime.datetime.now().time()
    '''
    pathToAnswerFile="/Users/srinisha/Downloads/pa2-release/answer.txt"

    
    pathtoClassifier='/Users/srinisha/Downloads/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz'
    pathtoNerjar='/Users/srinisha/Downloads/stanford-ner-2014-06-16/stanford-ner.jar'
    
    getAnswers(pathtoClassifier,pathtoNerjar,pathToAnswerFile,query_dict)
    '''
if __name__ == "__main__":
    main()   
