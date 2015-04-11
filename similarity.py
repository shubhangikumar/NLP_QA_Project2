# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 19:13:07 2015

@author: Shubhangi
"""
import math
from operator import itemgetter

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
    
def main():
    query_words_list = "Who found Cornell"
    query_dict = {'0':query_words_list}
    n_grams1 = ["Ezra Cornell found","Cornell found Cornell", "found Cornell University", "Cornell University in", "University in 1857" ]
    n_grams2 = ["I am living","am living in" , "living in Ithaca"]
    dict_test = {"Doc1" : n_grams1, "Doc2" : n_grams2}    
    top_docs_dict = {'0': dict_test}
    similarity(query_dict,top_docs_dict)
    
if __name__ == "__main__":
    main()      
        
        
