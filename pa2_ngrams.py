__authors__ = 'Anisha,Nivedhitha,Srinisha,Shubhangi'
 
import re
import os

pathTopDocs = "dev1/"
top_docs_dict = dict()

def preprocessing(finalstr):
    newline = re.compile(r'(\n)', re.UNICODE)
    finalstr = newline.sub(' ',finalstr)
    
    multlinesMatch = re.compile(r'(\n)+', re.UNICODE)
    finalstr = multlinesMatch.sub('',finalstr)
    
    punctuation = re.compile(r'[#,^&$\(\)\?";:\'\[\]{}@\*\-+~=_]+', re.UNICODE)
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
    return ngram_list

def getTopDocsDict(pathTopDocs):
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

top_docs_dict1 = getTopDocsDict(pathTopDocs)
print top_docs_dict1
print "Completed"