import math
import re
import os
import datetime
from operator import itemgetter
from nltk.corpus import stopwords


entity_labels = {"HOW": ["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"], "WHAT": ["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"],"WHERE": ["LOCATION"], "WHO": ["PERSON", "ORGANIZATION"], "WHEN": ["TIME", "DATE"], "HOW MANY": ["COUNT","MONEY","PERCENT"],"WHICH":["LOCATION","PERSON", "TIME", "DATE", "MONEY", "PERCENT"]}


def similarity(query_dict,top_docs_dict):
    query_no = query_dict.keys()
    dict_phrases = {}
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
                print n_grams[n],score 
                dict_scores['phrase'] = n_grams[n]
                dict_scores['score'] = score
                list_scores.append(dict_scores)
        newlist = sorted(list_scores, key=itemgetter('score'), reverse=True) 
        dict_phrases[i] = newlist
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
    for i in range(0, len(ngramterms) - 9):
            if ngramterms[i] != "." and ngramterms[i+1] !="." and ngramterms[i+2] != "." and ngramterms[i+3] != "." and ngramterms[i+4] != "." and ngramterms[i+5] != "." and ngramterms[i+6] != "." and ngramterms[i+7] != "." and ngramterms[i+8] != ".":
                ngram_list.append(ngramterms[i] + " " + ngramterms[i + 1] + " " + ngramterms[i + 2] + " " + ngramterms[i + 3] + " " + ngramterms[i + 4] + " " + ngramterms[i + 5] + " " + ngramterms[i + 6] + " " + ngramterms[i + 7] + " " + ngramterms[i + 8] + " " + ngramterms[i + 9])
            '''
            else :
                if ngramterms[i] != "." :
                    if ngramterms[i+1] == "." :
                        ngram_list.append(ngramterms[i] + " " + ngramterms[i + 1])
                    elif ngramterms[i+2] == "." :
                        ngram_list.append(ngramterms[i] + " " + ngramterms[i + 1] + " " + ngramterms[i + 2])
                    elif ngramterms[i+3] == "." :
                        ngram_list.append(ngramterms[i] + " " + ngramterms[i + 1] + " " + ngramterms[i + 2] + " " + ngramterms[i + 3])
                    elif ngramterms[i+4] == "." :
                        ngram_list.append(ngramterms[i] + " " + ngramterms[i + 1] + " " + ngramterms[i + 2] + " " + ngramterms[i + 3] + " " + ngramterms[i + 4])
                    elif ngramterms[i+5] == "." :
                        ngram_list.append(ngramterms[i] + " " + ngramterms[i + 1] + " " + ngramterms[i + 2] + " " + ngramterms[i + 3] + " " + ngramterms[i + 4] + " " + ngramterms[i + 5])
                    elif ngramterms[i+6] == "." :
                        ngram_list.append(ngramterms[i] + " " + ngramterms[i + 1] + " " + ngramterms[i + 2] + " " + ngramterms[i + 3] + " " + ngramterms[i + 4] + " " + ngramterms[i + 5] + " " + ngramterms[i + 6])
                    elif ngramterms[i+7] == "." :
                        ngram_list.append(ngramterms[i] + " " + ngramterms[i + 1] + " " + ngramterms[i + 2] + " " + ngramterms[i + 3] + " " + ngramterms[i + 4] + " " + ngramterms[i + 5] + " " + ngramterms[i + 6] + " " + ngramterms[i + 7])
                    elif ngramterms[i+8] == "." :
                        ngram_list.append(ngramterms[i] + " " + ngramterms[i + 1] + " " + ngramterms[i + 2] + " " + ngramterms[i + 3] + " " + ngramterms[i + 4] + " " + ngramterms[i + 5] + " " + ngramterms[i + 6] + " " + ngramterms[i + 7] + " " + ngramterms[i + 8])
            '''
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
    
            ngramterms = tempCompText.split()
            ngram_list = getngrams(ngramterms)
    
            docs_dict.update({index:ngram_list})
            index = index+1
            
        top_docs_dict.update({docNo:docs_dict})
    
    return top_docs_dict

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
    
if __name__ == "__main__":
    main()   