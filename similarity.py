import math
import re
import os
import datetime
from operator import itemgetter
import ner
import nltk 
import collections
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

entity_labels = {"How": ["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"],"Where": ["LOCATION"], "Who": ["PERSON", "ORGANIZATION"], "When": ["TIME", "DATE"],"Which":["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"],"What": ["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT","ORGANIZATION"],"NAME":["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT","ORGANIZATION"]}
dict_phrases = {}
WL = WordNetLemmatizer()
uniqueAnswers=[]

def similarity(query_dict,top_docs_dict):
    query_no = query_dict.keys()
    
    for i in range(len(query_no)):
        list_scores = []
        query = query_no[i]
        doc_dict = top_docs_dict[query]
        query_text = query_dict[query]        
        keys_list_docs = doc_dict.keys()
        matr = {}
        n_grams =[]
        for l in range(len(doc_dict)):
            list_doc_dict = doc_dict[keys_list_docs[l]]
            for j in range(len(list_doc_dict)):
                n_grams.append(list_doc_dict[j])
        dict_scores = {}
        train_set = []
        train_set.append(query_text)
        for n in range(len(n_grams)):
            train_set.append(n_grams[n])
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)  #finds the tfidf score with normalization
        matr[int(query)] = cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)
        tup = matr[int(query)]
        
        for k in range(len(n_grams)):
            dict_scores = {}
            if k==0 :
                continue
            score = tup[0][k]
            if(score != 0):
                dict_scores['phrase'] = n_grams[k-1]
                dict_scores['score'] = score
                list_scores.append(dict_scores)
                
        newlist = sorted(list_scores, key=itemgetter('score'), reverse=True) 
        dict_phrases[query] = newlist
        
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
                n_gram_lower=n_gram.lower()
                if word in n_gram or word in n_gram_lower:
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
        try:
            dict_tf[keys[k]] = value/denominator
        except:
            print "no words matching"
            
        
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
    
    
    query_dict = collections.OrderedDict()
    
    for line in questions_file:
        matchQueryNumber = re.match(r'Number: [0-9]+', line, re.M|re.I)
        if matchQueryNumber:
            qnum = matchQueryNumber.group(0).split()
        else:
            if line!="\n":
                line=" ".join([w for w in line.split(" ")])
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

# nltk stanford NER -http://stackoverflow.com/questions/18371092/stanford-named-entity-recognizer-ner-functionality-with-nltk
# pyner https://github.com/dat/pyner


def queryForEntity(expectedEntity,passage):
    tagger = ner.SocketNER(host='localhost', port=8081) # requires server to be started
    answer=tagger.get_entities(passage)
    answers=[]
    for j,currentExpectedEntity in enumerate(expectedEntity):
        for key in answer:
            if(key==currentExpectedEntity):
                for eachAnswer in answer[key]:
                    answerString=eachAnswer.encode()
                    answers.append(answerString) 
    return answers


def getAnswers(pathToAnswerFile,query_dict):
    if os.path.exists(pathToAnswerFile):
        f = file(pathToAnswerFile,"w")
    else:
        f = file(pathToAnswerFile,"a")
    for query in query_dict: # for every query
        cnt=0
        # What, Name uses POS tagging to extract NNP, we are not using POS tagging for these questions
        Query=query_dict[query]
        QueryNo= query;
        list_scores=dict_phrases[QueryNo]       
        f.write("qid"+" "+str(QueryNo)+"\n")
        expectedEntity=[]
        testHow=re.compile("How") 
        testWhen=re.compile("When")
        testWhich=re.compile("Which")
        testWho=re.compile("Who")
        testWhat=re.compile("What")
        testWhere=re.compile("Where")
        testName=re.compile("Name")
        Qwords=Query.split(" ")
        if testHow.match(Query):
            temp=["many","long","much"] # "How many? How much? How long?
            if Qwords[1] in temp:
                for currDict in list_scores: 
                    if cnt>=10:
                        break
                    else:
                        currpassage=currDict['phrase']
                        answersList=[]
                        getNumber=re.compile(r'(([0-9]+)|((one|twe|three|four|five|six|seven|eight|nine)+|(ten*|\shundred*|\sthousand*)))')
                        answersList=getNumber.findall(currpassage)
                        if answersList!=[]:
                            for answer in answersList:
                                if answer[0] not in uniqueAnswers:
                                    if cnt<10:
                                        cnt=cnt+1
                                        f.write(str(cnt)+" "+answer[0]+"\n")
                                        uniqueAnswers.append(answer[0])
                                    else:
                                        break             
            else:
                expectedEntity=entity_labels["How"]     
        if testWhen.match(Query):
            expectedEntity=entity_labels["When"]
        if testWhich.match(Query):
            expectedEntity=entity_labels["Which"]
        if testWho.match(Query): 
            temp=["is", "are", "wa"]
            if Qwords[1] in temp: # Who is|was|are wont expect person or oragnization so we have to do POS
                expectedEntity =[]
            else:
                expectedEntity=entity_labels["Who"]
        if testWhat.match(Query): # POS
            expectedEntity=[]
        if testName.match(Query): # POS
             expectedEntity=[]
        if testWhere.match(Query):
            expectedEntity=entity_labels["Where"]
            
        for currDict in list_scores: 
            if cnt>=10:
                break
            else:
                currpassage=currDict['phrase']
                answersList=[]
                if expectedEntity!=[]:
                    answersList=queryForEntity(expectedEntity,currpassage)
                    for answer in answersList:
                        if answer not in uniqueAnswers:
                            if cnt<10:
                                cnt=cnt+1
                                f.write(str(cnt)+" "+answer+"\n")
                                uniqueAnswers.append(answer)
                            else:
                                break
                else:
                    currpassage=currDict['phrase']
                    answersList=[]
                    answer = word_tokenize(currpassage)
                    answers=nltk.pos_tag(answer)
                    for i,pair in enumerate(answers):
                        if(pair[1]=="NNP"):
                          answersList.append(answer[i])
                    for answer in answersList:
                        if answer not in uniqueAnswers:                    
                            if cnt<10:
                                cnt=cnt+1
                                f.write(str(cnt)+" "+answer+"\n")
                                uniqueAnswers.append(answer)
                            else:
                                break
        if cnt<10:

            for currDict in list_scores: 
                if cnt>=10:
                    break
                else:
                    currpassage=currDict['phrase']
                    answersList=[]
                    answer = word_tokenize(currpassage)
                    answers=nltk.pos_tag(answer)
                    for i,pair in enumerate(answers):
                        if(pair[1]=="NNP"):
                            answersList.append(answer[i])
                    for answer in answersList:
                        if answer not in uniqueAnswers:
                            if cnt<10:
                                cnt=cnt+1
                                f.write(str(cnt)+" "+answer+"\n")
                                uniqueAnswers.append(answer)
                            else:
                                break
        del uniqueAnswers[:]

                                    
        
    

def main():
    print "Process started", datetime.datetime.now().time()
    questions_filename = "C:/Users/Shubhangi/Desktop/CORNELL COURSES/Spring 2015/NLP/Project2/pa2_data/pa2-release/qadata/dev/questions.txt"
    pathTopDocs = "C:/Users/Shubhangi/Desktop/CORNELL COURSES/Spring 2015/NLP/Project2/pa2_data/pa2-release/topdocs/dev/"  
    
    query_dict = getquerydict(questions_filename)
    print "Query Dictionary Generated", datetime.datetime.now().time()
    
    top_docs_dict = getTopDocsDict(pathTopDocs)
    print "Top Docs Dictionary Generated", datetime.datetime.now().time()
    
    similarity(query_dict,top_docs_dict)

    pathToAnswerFile="C:/Users/Shubhangi/Desktop/CORNELL COURSES/Spring 2015/NLP/Project2/pa2_data/pa2-release/answer.txt"

    getAnswers(pathToAnswerFile,query_dict)
    print "Process completed", datetime.datetime.now().time()

    
if __name__ == "__main__":
    main()   