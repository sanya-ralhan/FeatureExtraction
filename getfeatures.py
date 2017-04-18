from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import csv, math 
import numpy as np
from sklearn.externals import joblib
import gib_detect_train
import pickle
import sys

model_data = pickle.load(open('gib_model.pki', 'rb'))

para= sys.argv[1]

csvfile = file(str(para)+'.csv', 'rb')
reader = csv.reader(csvfile)
content=[] ;
for line in reader:
        leng=2;
        ngram0=[];
        newline=[];
        while(leng<( len(str(line))-2)):
            ngram0.append(str(line)[leng]);
            leng=leng+1;
            if(str(line)[leng]=="."):
                leng=leng+1;
                newline.append(''.join(ngram0))
                ngram0=[];
        if(len(newline)==1):
            content.append(newline[0]);
        if(len(newline)==2): 
            if((newline[0]!='www') and (newline[0]!='www2') and (newline[0]!='www3')):
                    #print (newline[0]);
                    content.append(newline[0]+newline[1]);
            else:      
                    content.append(newline[1]);
        if(len(newline)>2):
                content.append(newline[1]);
                    
        
csvfile.close()


#-----------------------------ngram--------------------------------
gramdict={};
csvfile0 = file('ngramdictset.csv', 'rb') 
reader0 = csv.reader(csvfile0)
for line0 in reader0: 
    gramdict[line0[0]] = line0[1];
csvfile0.close()

gramwhite={};
csvfile0 = file('ngramwhiteset.csv', 'rb') 
reader0 = csv.reader(csvfile0)
for line0 in reader0: 
    gramwhite[line0[0].lstrip('|')] = line0[1]; 
csvfile0.close()
 

ngramdict=[];
ngramwhite=[];
for line in content:  
    leni=0;
    lenj=0; 
#------ngram---------------------------
    leng=0;
    ngram=[] ;
    ngram0=[] ;
    while(leng<( len(str(line)))):
        ngram0.append(str(line)[leng]);
        leng=leng+1; 
    leng1=0; 
    while(leng1<len(ngram0)):
          leng2=1+leng1;
          while(leng2<len(ngram0)):
                ngram.append(ngram0[leng1]);
                ngram0[leng1]=ngram0[leng1]+ngram0[leng2]; 
                #print(ngram0[leng1]);
                leng2=leng2+1;
          ngram.append(ngram0[leng1]);
          leng1=leng1+1; 
    ngramelement=0;
    for item in ngram: 
        if(gramdict.has_key(item)==True):
          ngramelement=ngramelement+(float(ngram.count(item))/float(len(ngram)))*float(gramdict[item]);
    ngramdict.append(ngramelement)
    ngramelement=0;
    for item in ngram: 
        if(gramwhite.has_key(item)==True):
          ngramelement=ngramelement+(float(ngram.count(item))/float(len(ngram)))*float(gramwhite[item]);
    ngramwhite.append(ngramelement)
 

#-----------------------------ngram--------------------------------
name = [] 
length = [] 
entropy = []
vowelratio = []
digitratio = []
consdigitratio = []
consletterratio = []
consonantratio = []
pronounciation = []

for line in content: 
    name.append(line)
    leng = len(str(line)) 
    length.append(leng)
    leni = 0
    lenj = 0
    entro = 0
    vowelrat = 0
    digit = 0
    repeat = 0
    consdig = 0
    consletter = 0
    conscon = 0
    while(leni<( len(str(line)))):
        freq = 0
        lenj =0
        while( lenj<(len(str(line)))):
            if(str(line)[leni]==str(line)[lenj]):
                freq = freq+1;
            lenj = lenj+1;
        entro=entro-(1/float(freq))*math.log(1/float(freq));
        leni = leni+1;
    
    entropy.append( entro );
    #print line

   
    for char in str(line):
        #print char
        if char in 'aeiouAEIOU':
            vowelrat += 1
        if char in '0123456789':
            digit += 1
        
    digitratio.append(digit/float(leng))            
    vowelratio.append(vowelrat/float(leng))         
    
    for i in range(len(str(line))-1):
        if  str(line)[i].isalpha():
              if  str(line)[i+1] == str(line)[i]:
                     consletter += 1
        if  str(line)[i].isdigit():
              if  str(line)[i+1].isdigit():
                        consdig += 1
        if str(line)[i].isalpha() and str(line)[i] not in 'aeiouAEIOU':
            if str(line)[i+1].isalpha() and str(line)[i+1] not in 'aeiouAEIOU':
                   conscon += 1

                   
                   
    consonantratio.append(conscon/float(leng))
    consletterratio.append(consletter/float(leng))
    consdigitratio.append(consdig/float(leng))
 
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    if gib_detect_train.avg_transition_prob(str(line)[2:leng+2], model_mat) > threshold:
        pronounciation.append('1')
    else:
        pronounciation.append('0')
 
csvfile2 = file('result_train_'+str(para)+'.csv', 'wb')
if str(para) == 'black':
   x= 0
else:
   x = 1
writer = csv.writer(csvfile2)
#writer.writerow(['domain', 'length', 'entrophy','Pronunciation','Vowel ratio','Digit ratio','Repeat letter', 'Consecutive digit ratio', 'Consecutive consonant ratio','N-gram score w','N-gram score d','Q']) 
i=0;
while(i<len(name)): 
   writer.writerow([ name[i], length[i], entropy[i],pronounciation[i], vowelratio[i],digitratio[i],consletterratio[i],consdigitratio[i],consonantratio[i],float(ngramwhite[i]/10000),float(ngramdict[i]/10000),x])
   i=i+1;
csvfile2.close() 

 
