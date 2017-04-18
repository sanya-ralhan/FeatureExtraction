Steps to run machine learning code for CS6262 project 

Requirements 
python installed 

1. Run ‘python pron_train.py’ to get Markov Model probabilities and thresholds for pronunciation 

2. Run ‘python getfeatures.py black’ for feature extraction for black.csv ( blacklisted known training dataset 

2. Run ‘python getfeatures.py white’ for feature extraction for white.scv ( whitelisted known training dataset 

3.Run ‘python mix.py’ this step mixes both the datasets into one trained model

4. Run ‘python forest.py’ for predicting domain names output from classification for testlist.csv and result can be seen in prediciton_result.csv 

5. Edit crossvalidation.py for editing the number of trees (default is 10)

6.Run ‘python crossvalidation.py N’ for decision tree and decision forest creation and see accuracy results in accuracy_n_trees.csv (N= number of individual trees)



