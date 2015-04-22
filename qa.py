'''
@authors: Srinisha, Shubhangi, Nivedhitha, Anisha
'''

import re
import os
import datetime
from operator import itemgetter
import ner
import nltk 
import collections
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

questions_filename = "questions.txt"
pathTopDocs = "test/" 
pathToAnswerFile="answers.txt"

entity_labels = {"How": ["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"],"Where": ["LOCATION"], "Who": ["PERSON", "ORGANIZATION"], "When": ["TIME", "DATE"],"Which":["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"],"What": ["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT","ORGANIZATION"],"NAME":["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT","ORGANIZATION"]}
ques_words = ["who","where","when","what","why","how","which","whom"]
dict_phrases = {}
WL = WordNetLemmatizer()
uniqueAnswers=[]
stop = stopwords.words('english')

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
                #Strip any punctuations in the query
                punctuation = re.compile(r'[\?."\',\(\)&/:]+', re.UNICODE)                
                line = punctuation.sub('',line)
                whitespace = re.compile(r'(\s)+', re.UNICODE)
                line = whitespace.sub(' ', line)
                #Lemmatize the query
                line=" ".join([WL.lemmatize(i) for i in line.split()])
                line = line.encode()
                query_dict.update({qnum[1]:line})          
    return query_dict

#This function is to preprocess the top-docs
def preprocessing(finalstr):
    #Replace any new line with a blank space
    newline = re.compile(r'(\n)', re.UNICODE)
    finalstr = newline.sub(' ',finalstr)
    #Strip multiple new lines
    multlinesMatch = re.compile(r'(\n)+', re.UNICODE)
    finalstr = multlinesMatch.sub('',finalstr)
    #Remove the punctuations
    punctuation = re.compile(r'[#,^&$\(\)\?";:\'`\[\]{}@\*\-+~=_]+', re.UNICODE)
    finalstr = punctuation.sub('', finalstr)
    #Strip extra multiple whitespaces
    whitespace = re.compile(r'(\s)+', re.UNICODE)
    finalstr = whitespace.sub(' ', finalstr)
    
    fullstop = re.compile(r'[.]+', re.UNICODE)
    finalstr = fullstop.sub(' .', finalstr)
    
    return finalstr

#This function will process the top-docs and generate 10-grams
def getngrams(ngramterms):
    ngram_list = []
    for eachTerm in ngramterms :
        tTerm = eachTerm.split()
        tempTerm = list()
        for eachTempTerm in tTerm :
            eachTempTerm = eachTempTerm.decode('ascii',errors='ignore')
            #Lemmatize each term
            eachTempTerm = WL.lemmatize(eachTempTerm)
            eachTempTerm = eachTempTerm.encode()
            tempTerm.append(eachTempTerm)
        if len(tempTerm)> 9  :
            for i in range(0, len(tempTerm) - 9) :
                ngram_list.append(tempTerm[i] + " " + tempTerm[i + 1] + " " + tempTerm[i + 2] + " " + tempTerm[i + 3] + " " + tempTerm[i + 4] + " " + tempTerm[i + 5] + " " + tempTerm[i + 6] + " " + tempTerm[i + 7] + " " + tempTerm[i + 8] + " " + tempTerm[i + 9])
        else :
            #If the sentence is less than 10 terms, treat the complete sentence as 10-gram
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
        
        #Match the content between the <TEXT></TEXT> tags
        textMatch = re.compile(r'<TEXT>(.*?)</TEXT>', re.DOTALL)
        completeText = textMatch.findall(whole_data)
        
        #Match the content between the <P></P> tags
        PTagsMatch = re.compile(r'<P>', re.DOTALL)
        PEndTagsMatch = re.compile(r'</P>', re.DOTALL)
        
        index = 0;
        for eachText in completeText:
            #Remove the <P> and </P> tags
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

#Calculate similarity between query and n-grams
def similarity(query_dict,top_docs_dict):
    query_no = query_dict.keys()
    
    for i in range(len(query_no)):
        list_scores = []
        query = query_no[i]
        #Get the n-grams and query text for the given query
        doc_dict = top_docs_dict[query]
        query_text = query_dict[query]  
        
        #Remove stopwords from the query
        query_text=" ".join([w for w in query_text.split(" ") if (w.lower() not in stop)])
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
        #finds the tfidf score with normalization
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)  
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
        #Sort the n-grams by scores        
        newlist = sorted(list_scores, key=itemgetter('score'), reverse=True) 
        dict_phrases[query] = newlist
        
# nltk stanford NER
# nltk stanford NER -http://stackoverflow.com/questions/18371092/stanford-named-entity-recognizer-ner-functionality-with-nltk
# pyner https://github.com/dat/pyner


def queryForEntity(expectedEntity,passage):
    tagger = ner.SocketNER(host='localhost', port=8081) # requires server to be started
    #Tag the passage with the named entities
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
        testHow=re.compile(r'How|how') 
        testWhen=re.compile(r'When|when')
        testWhich=re.compile(r'Which|which')
        testWho=re.compile(r'Who|who')
        testWhat=re.compile(r'What|what')
        testWhere=re.compile(r'Where|where')
        testName=re.compile(r'Name|name')
        Qwords=Query.split(" ")
        if testHow.findall(Query):
            temp=["many","long","much"] # "How many? How much? How long?
            if Qwords[1] in temp:
                for currDict in list_scores: 
                    if cnt>=10:
                        break
                    else:
                        currpassage=currDict['phrase']
                        answersList=[]
                        #Regex match for numbers
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
        if testWhen.findall(Query):
            expectedEntity=entity_labels["When"]
        if testWhich.findall(Query):
            expectedEntity=entity_labels["Which"]
        if testWho.findall(Query): 
            temp=["is", "are", "wa"]#wa - since "was" is lemmatized to "wa"
            if Qwords[1] in temp: # Who is|was|are wont expect person or organization so we have to do POS
                expectedEntity =[]
            else:
                expectedEntity=entity_labels["Who"]
        if testWhat.findall(Query):
            testtemp=re.compile(r'(state|country|place|city|continent|located)')
            match=testtemp.findall(Query)
            if match!=[]:
                expectedEntity=["LOCATION"]
            else:
                expectedEntity=[] # Do POS tagging
        if testName.findall(Query): 
            testtemp=re.compile(r'(state|country|place|city|continent|located)')
            match=testtemp.findall(Query)
            if match!=[]:
                expectedEntity=["LOCATION"]
            else:
                expectedEntity=[] # Do POS tagging
        if testWhere.findall(Query):
            expectedEntity=entity_labels["Where"]
            
        for currDict in list_scores: 
            if cnt>=10:
                break
            else:
                currpassage=currDict['phrase']
                answersList=[]
                if expectedEntity!=[]:
                    #NER Tagging
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
                    #POS Tagging
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
                    #POS Tagging
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

    query_dict = getquerydict(questions_filename)
    print "Query Dictionary Generated", datetime.datetime.now().time()
    
    top_docs_dict = getTopDocsDict(pathTopDocs)
    print "Top Docs Dictionary Generated", datetime.datetime.now().time()
    
    similarity(query_dict,top_docs_dict)
    print "Similarity Computed", datetime.datetime.now().time()
    
    print("Retrieving Answers...Please wait")
    getAnswers(pathToAnswerFile,query_dict)
    print "Process completed", datetime.datetime.now().time()
    
if __name__ == "__main__":
    main()   


'''
##Experiments on tf-idf similarity measures and n-gram tiling##

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

def ngramtiling(answer_phrases):
        eachPhrases = []
        for eachItem in answer_phrases :
            eachPhrases.append(eachItem)
        #eachPhrases = answer_phrases
        index = 0
        #print "eachPhrase :", eachPhrases
        for phrase in eachPhrases :
            maintPhrase = phrase
            splitmaintPhrase = maintPhrase.split()
            index1 = 0
            for i in range(len(eachPhrases)) :
                if i >= len(eachPhrases) :
                    break
                tPhrase = eachPhrases[i]
                if maintPhrase==tPhrase :
                    index1 = index1+1
                    continue
                else :
                    cnt = 0
                    splittPhrase = tPhrase.split()
                    for j in range(len(splitmaintPhrase)) :
                        cnt = 0
                        flag=False
                        #j=j+1
                        for k in range(len(splittPhrase)) :
                            if j!=len(splitmaintPhrase) :
                                if(splitmaintPhrase[j] == splittPhrase[k]) :
                                    l1 = j
                                    l2 = k
                                    cnt = cnt+1
                                    while(l1 < len(splitmaintPhrase) and l2 < len(splittPhrase)) :
                                        l1 = l1+1
                                        l2 = l2+1
                                        if(l1!=len(splitmaintPhrase) and l2!=len(splittPhrase)) :
                                            t1 = splitmaintPhrase[l1]
                                            t2 = splittPhrase[l2]
                                            if(t1==t2) :
                                                flag = True
                                                cnt = cnt+1
                                                if(l1+1 == len(splitmaintPhrase)) :
                                                    break
                                            else :
                                                flag = False
                                                break
                                        else :
                                            flag = True
                                            break
                                    if flag== True :
                                        break
                                else :
                                    break
                        
                        if flag == True :
                            break
                        
                    if(flag==True) :
                        for x in range(len(splitmaintPhrase)) :
                            if(splittPhrase[0] == splitmaintPhrase[x]) :
                                y = x
                                break
                        finalstr =""
                        
                        if (y!=0) :
                            z=0
                            while(z<y) :
                                finalstr = finalstr +" " + splitmaintPhrase[z]
                                z = z+1 
                            z=0
                            while(z<len(splittPhrase)) :
                                finalstr = finalstr +" " + splittPhrase[z]
                                z = z+1
                        else :
                            if(len(splitmaintPhrase) > len(splittPhrase)) :
                                z=0
                                while(z<len(splitmaintPhrase)) :
                                    finalstr = finalstr +" " + splitmaintPhrase[z]
                                    z = z+1
                            else :
                                z=0
                                while(z<len(splittPhrase)) :
                                    finalstr = finalstr +" " + splittPhrase[z]
                                    z = z+1
                        finalstr = finalstr.strip()
                        eachPhrases[index] = finalstr
                        eachPhrases.pop(index1)
                        maintPhrase = finalstr
                index1 = index1+1    
            index = index + 1
        return eachPhrases  
'''