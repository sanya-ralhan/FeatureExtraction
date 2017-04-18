from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import csv, math 
import numpy as np
from sklearn.externals import joblib
import gib_detect_train
import pickle
import sys

model_data = pickle.load(open('pron_model.pki', 'rb'))

'''
csvfile = file('testlist.csv', 'rb')
reader = csv.reader(csvfile)
content=[] ;
#Stripping the www,www2,www3
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
    
#pronunciation score usind previosuly generated thresholds 
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    if gib_detect_train.avg_transition_prob(str(line)[2:leng+2], model_mat) > threshold:
        pronounciation.append('1')
    else:
        pronounciation.append('0')
#Write features of test data to file 
csvfile2 = file('result_test.csv', 'wb')
writer = csv.writer(csvfile2)
#writer.writerow(['domain', 'length', 'entrophy','Pronunciation','Vowel ratio','Digit ratio','Repeat letter', 'Consecutive digit ratio', 'Consecutive consonant ratio','N-gram score w','N-gram score d','Q']) 
i=0;
while(i<len(name)): 
   writer.writerow([ name[i], length[i], entropy[i],pronounciation[i], vowelratio[i],digitratio[i],consletterratio[i],consdigitratio[i],consonantratio[i],float(ngramwhite[i]/10000),float(ngramdict[i]/10000),2])
   i=i+1;
csvfile2.close() 
'''


#Opening trained data for generating decision trees and forest

data2 = []
target2 = []

with open("result_train_mix.csv") as f:
        for line in csv.reader(f, delimiter = ","):
            data2.append([float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10])])
            target2.append(int(line[-1]))
            

#Create user input number of trees and combine find accuracy results before and after combining averages

splitnum=10
treenum= float(sys.argv[1])
samplenum=1000;
targetnum=0; 
csvfile2 = file("accuracy_"+str(treenum)+"_trees.csv", 'wb')
writer = csv.writer(csvfile2)
splitsizw= (len(data2)-len(data2)%splitnum)/splitnum
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
while(targetnum<10):   
  i=0
  while(i<treenum): 
       rf2 = tree.DecisionTreeClassifier()
       i1=(i+1)%10
       if(i1==targetnum):
           i1=(i+8)%10
       i2=(i+2)%10
       if(i2==targetnum):
           i2=(i+8)%10
       i3=(i+3)%10 
       if(i3==targetnum):
           i3=(i+8)%10
       i4=(i+4)%10
       if(i4==targetnum):
           i4=(i+8)%10
       i5=(i+5)%10 
       if(i5==targetnum):
           i5=(i+8)%10
       rf2.fit(treedata[i1]+treedata[i2]+treedata[i3]+treedata[i4]+treedata[i5], treetarget[i1]+treetarget[i2]+treetarget[i3]+treetarget[i4]+treetarget[i5])
        
       joblib.dump(rf2, "train_model_tree"+str(i)+".m") 
       i+=1;
#to store accuracy results 
  correct=[0,0,0,0,0,0,0,0,0,0,0];
  error=[0,0,0,0,0,0,0,0,0,0,0];
  total=[0,0,0,0,0,0,0,0,0,0,0];
  falsepos=[0,0,0,0,0,0,0,0,0,0,0];
  trupos=[0,0,0,0,0,0,0,0,0,0,0];
  falseneg=[0,0,0,0,0,0,0,0,0,0,0];
  truneg=[0,0,0,0,0,0,0,0,0,0,0];
  fn,tn,fp,tp =0,0,0,0;
  i=0;
  thres= 0.45 #change the threshold for classification decision 
  while(i<samplenum): 
      j=0
      prediction=0;
      while(j<treenum):
             clf = joblib.load("train_model_tree"+str(j)+".m")
             treepredict=float(clf.predict(treedata[targetnum][i]));
             if  treepredict>thres:  
               if treetarget[targetnum][i]>thres:
                 correct[j]+=1;
                 truneg[j]+=1
               else:
                 error[j]+=1;
                 falseneg[j]+=1
             else:
               if treetarget[targetnum][i]<=thres: 
                 correct[j]+=1;
                 trupos[j]+=1
               else:
                 error[j]+=1;
                 falsepos[j]+=1
             prediction=prediction+treepredict;
             j=j+1;
      prediction=float(prediction)/float(treenum);
      if  prediction>thres: 
             #writer.writerow("domain", 'legal']) 
             if treetarget[targetnum][i]>thres:
                 correct[10]+=1;
                 tn+=1
                 
             else:
                 error[10]=error[10]+1;
                 fn+=1
      if  prediction<=thres:
             #writer.writerow(["domain", 'malicious']) 
             if treetarget[targetnum][i]<=thres: 
                 correct[10]+=1
                 tp+=1
             else:
                 error[10]+=1;
                 fp+=1
      i=i+1;
  writer.writerow([ 'target block',targetnum])
  j=0
  while(j<treenum):
       writer.writerow([ 'tree',j,'correct:',correct[j],'wrong:',error[j],'true positive',trupos[j],'false positive',falsepos[j],'true negative',truneg[j],'false negative',falseneg[j],'total:',samplenum,'accuracy:',correct[j]/float(samplenum), 'sensitivity',trupos[j]/float(trupos[j]+falseneg[j]), 'specificity',truneg[j]/float(truneg[j]+falsepos[j]) ])
       j=j+1
  writer.writerow([ 'forest',targetnum,'correct:',correct[10],'wrong:',error[10],'true positive',tp,'false positive',fp,'true negative',tn,'false negative',fn,'total:',samplenum,'accuracy:',correct[10]/float(samplenum), 'sensitivity',tp/float(tp+fn), 'specificity',tn/float(tn+fp) ])
  targetnum=targetnum+1;  
csvfile2.close()  
