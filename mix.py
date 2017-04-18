from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import csv, math 
import numpy as np
from sklearn.externals import joblib
import gib_detect_train
import pickle
#os.chdir("workspace/model_save")
model_data = pickle.load(open('gib_model.pki', 'rb'))
data2=[]
target2=[]
name2=[]
data1=[]
target1=[]
name1=[]
with open("result_train_black.csv") as f:
        for line in csv.reader(f, delimiter = ","):
            if(int(line[-1])==0):
              name1.append(str(line[0]))
              data1.append([float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10])])
              target1.append(int(line[-1]))
            if(int(line[-1])==1):
              name2.append(str(line[0]))
              data2.append([float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10])])
              target2.append(int(line[-1])) 
with open("result_train_white.csv") as f:
        for line in csv.reader(f, delimiter = ","):
            if(int(line[-1])==0):
              name1.append(str(line[0]))
              data1.append([float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10])])
              target1.append(int(line[-1]))
            if(int(line[-1])==1):
              name2.append(str(line[0]))
              data2.append([float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10])])
              target2.append(int(line[-1])) 
num2=100;              
num1= 100*len(data1)/len(data2)
csvfile2 = file('result_train_mix.csv', 'wb')
writer = csv.writer(csvfile2)
#writer.writerow(['domain', 'length', 'entrophy','Pronunciation','Vowel ratio','Digit ratio','Repeat letter', 'Consecutive digit ratio', 'Consecutive consonant ratio','N-gram score w','N-gram score d','Q']) 
i=0;
i1=0;
i2=0;
n1=num1;
n2=num2;

while(i<(len(target2)+len(target1))): 
   if((i1<n1)):
       writer.writerow([name1[i1],data1[i1][0],data1[i1][1],data1[i1][2],data1[i1][3],data1[i1][4],data1[i1][5],data1[i1][6],data1[i1][7],data1[i1][8],data1[i1][9],target1[i1]])
       i=i+1;
       i1=i1+1;
   if((i2<n2) ):
       writer.writerow([name2[i2],data2[i2][0],data2[i2][1],data2[i2][2],data2[i2][3],data2[i2][4],data2[i2][5],data2[i2][6],data2[i2][7],data2[i2][8],data2[i2][9],target2[i2]])
       i=i+1;
       i2=i2+1;
   if((i1==n1) and (i2==n2)):
        n1=num1+n1;
        n2=num2+n2; 
        if(n1>len(target1)):
            n1=len(target1);
        if(n2>len(target2)):
            n2=len(target2);
        
csvfile2.close()
print "completed"
