# NLP_QA_Project2

similarity.py:

Give paths to the following:

  1. questions_filename = ""
 
   Should be in pa2-release/qadata/dev/questions.txt
   
  2. pathTopDocs = ""  
   
   Should be in pa2-release/topdocs/dev/
  
  Install stanford NER     # (not the latest version unless u have java 1.8)
  
  install pyner https://github.com/dat/pyner

 start server of inside stanford NER
 
 java -mx1000m -cp stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier classifiers/english.muc.7class.distsim.crf.ser.gz -port 8081 -outputFormat inlineXM
  
  3. pathtoClassifier=""
      
     Look for english.all.3class.distsim.crf.ser.gz in stanford-ner-2014-06-16/classifiers/ and give the path

   4. pathtoNerjar='
  
  Give path to stanford-ner.jar in stanford-ner-2014-06-16 folder.


