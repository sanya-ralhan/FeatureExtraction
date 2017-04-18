from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import csv, math 
import numpy as np
from sklearn.externals import joblib
import gib_detect_train
import pickle

model_data = pickle.load(open('pron_model.pki', 'rb'))

#read input domain list and cleaning 
csvfile = file('testlist.csv', 'rb')
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
                   
                    content.append(newline[0]+newline[1]);
            else:      
                    content.append(newline[1]);
        if(len(newline)>2):
                content.append(newline[1]);
                    
        
csvfile.close()

#calculate n gram score 
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
 
csvfile2 = file('result_test.csv', 'wb')
writer = csv.writer(csvfile2)
#writer.writerow(['domain', 'length', 'entrophy','Pronunciation','Vowel ratio','Digit ratio','Repeat letter', 'Consecutive digit ratio', 'Consecutive consonant ratio','N-gram score w','N-gram score d','Q']) 
i=0;
while(i<len(name)): 
   writer.writerow([ name[i], length[i], entropy[i],pronounciation[i], vowelratio[i],digitratio[i],consletterratio[i],consdigitratio[i],consonantratio[i],float(ngramwhite[i]/10000),float(ngramdict[i]/10000),2])
   i=i+1;
csvfile2.close() 


data1 = []
target1 = []

with open('result_test.csv') as f:
        for line in csv.reader(f, delimiter = ","):
            data1.append([float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10])])
            target1.append(str(line[0]))


data2 = []
target2 = []

with open("result_train_mix.csv") as f:
        for line in csv.reader(f, delimiter = ","):
            data2.append([float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10])])
            target2.append(int(line[-1]))
            



splitnum=10
treenum=10 #set number of trees 
samplenum=1000; #set sample input domain size 
targetnum=0; 

splitsizw= (len(data2)-len(data2)%splitnum)/splitnum #get training data 
iterate=0
treedata=[[] for i in range(splitnum)]
treetarget=[[] for i in range(splitnum)]
i=0; 
while(i<len(data2)): 
      treedata[iterate].append(data2[i])
      treetarget[iterate].append(target2[i])
      if(((i+1)%splitsizw==0) ):
          if(iterate<(splitnum-1)):
             iterate=iterate+1; 
      i=i+1;
i=0
while(i<treenum): 
       rf2 = tree.DecisionTreeClassifier()
       i1=(i+1)%10 
       i2=(i+2)%10 
       i3=(i+3)%10  
       i4=(i+4)%10 
       i5=(i+5)%10  
       rf2.fit(treedata[i1]+treedata[i2]+treedata[i3]+treedata[i4]+treedata[i5], treetarget[i1]+treetarget[i2]+treetarget[i3]+treetarget[i4]+treetarget[i5])
       #print (i) 
       joblib.dump(rf2, "train_model_tree"+str(i)+".m") 
       i=i+1;

csvfile2 = file("prediction_result.csv", 'wb') #write to prediction file 
writer = csv.writer(csvfile2)
i=0;
while(i<len(data1)): 
      j=0
      prediction=0;
      while(j<treenum):
             clf = joblib.load("train_model_tree"+str(j)+".m")
             treepredict=float(clf.predict(data1[i]));
             prediction=prediction+treepredict;
             j=j+1;
      prediction=float(prediction)/float(treenum);
      if  prediction>0.45: 
             writer.writerow([target1[i], 'legal']) 
      if  prediction<=0.45:
             writer.writerow([target1[i], 'malicious']) 
      i=i+1;
 
csvfile2.close()  

