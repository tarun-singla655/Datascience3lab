import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score  # importing confusion matrix and accuracy_score
from sklearn.model_selection import train_test_split   # import train_test_split
from sklearn.neighbors import KNeighborsClassifier # importing knn classifier
def common(a,b): 
    c = [value for value in a if value in b] 
    return len(c)
def findMaxliklhood(Meanv1 , Covar1 , x):   # function to calcuate maximum likelihood
    determinate = np.linalg.det(Covar1)    # determinent of covariance matrix
    val1 = 1/(((2*np.pi)**5.)*determinate**0.5)    
    signmainv = np.linalg.inv(Covar1);    # invariance of covariance matrix
    firstparmeter = np.transpose(x-Meanv1);                   
    firstparmeter = np.dot(firstparmeter,signmainv)
    finalparmeter = np.dot(firstparmeter,(x-Meanv1))  # mahalnobis distance
    finalparmeter = -finalparmeter/2;
    finalans = val1*np.exp(finalparmeter);   # final answer of 
    return finalans
    



Remove_attributes = ['nbumps', 'nbumps2', 'nbumps3', 'nbumps4', 'nbumps5', 'nbumps6', 'nbumps7', 'nbumps89']  # removing columns with most zero values

l1 = [1, 3, 5]

FileData = pd.read_csv('seismic_bumps1.csv')  # reading file
accuracy = {'KNN': 0, 'KNN-normalized': 0, 'Bayes': 0}  #dictionary to store maximum value of KNN with and without normalizd data

FileData.drop(columns=Remove_attributes, inplace=True)  # droping columns mentioned above

# Question 1
FileData0 = FileData[FileData['class'] == 0]  # class0 Data
FileData1 = FileData[FileData['class'] == 1]  # class1 Data 


X0 = FileData0.drop(columns=['class'])    # droping calss attribute as we will determine it using our classification model 
X0_label = FileData0['class']   # storing class
X1 = FileData1.drop(columns=['class'])
X1_label = FileData1['class']    # storing of class1
# in these two lines we are splitting data into training and testing datasets with 70 and 30% respectively
X0_train, X0_test, X0_label_train, X0_label_test = train_test_split(X0, X0_label, test_size=0.3,
                                                                  random_state=42, shuffle=True) 
X1_train, X1_test, X1_label_train, X1_label_test = train_test_split(X1, X1_label, test_size=0.3, random_state=42,
                                                                    shuffle=True)

X_train, X_test, X_label_train, X_label_test = X0_train.append(X1_train), X0_test.append(
    X1_test), X0_label_train.append(X1_label_train), X0_label_test.append(X1_label_test)  # combining both test and training samples of both example

X_train.merge(X_label_train, left_index=True, right_index=True).to_csv('seismic-bumps-train.csv', index=False) # making the pdf of it
X_test.merge(X_label_test, left_index=True, right_index=True).to_csv('seismic-bumps-test.csv', index=False) # making pdf of it

print("........................QUESTION1...................................")
for k in l1: # for k = 1,3,5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, X_label_train) # fitting model
    X_label_pred = knn.predict(X_test)  # pridicting 
    print('For K = {}'.format(k)) # printing k
    accuracy['KNN'] = max(  accuracy['KNN'],accuracy_score(X_label_test, X_label_pred) * 100) # calculate accuracy
    print('Confusion matrix\n', confusion_matrix(X_label_test, X_label_pred))   # confusion matrix
    print('Accuracy using kNN = {:.3f}\n'.format(accuracy_score(X_label_test, X_label_pred)*100))  #printing accuracy

# Question 2
X_train_normalised = (X_train - X_train.min()) / (X_train.max() - X_train.min())  #normalizing data
X_test_normalised = (X_test - X_train.min()) / (X_train.max() - X_train.min())  #normalizing data using min-max of train-data
X_train_normalised.merge(X_label_train, left_index=True, right_index=True).to_csv(
    'seismic-bumps-train-normalised.csv', index=False)   # mereging to make save csv file of training data
X_test_normalised.merge(X_label_test, left_index=True, right_index=True).to_csv(  
    'seismic-bumps-test-normalised.csv', index=False)  # merging test data to save as csv

print("........................QUESTION2...................................\n")
print("Normalizing the test and train Data\n")
for k in l1:
    knn = KNeighborsClassifier(n_neighbors=k)   # classifying data
    knn.fit(X_train_normalised, X_label_train)   # fitting curve
    X_label_pred = knn.predict(X_test_normalised)
    print('For K = {}'.format(k))
    print('Confusion matrix\n', confusion_matrix(X_label_test, X_label_pred)) #printing confusion matrix
    accuracy['KNN-normalized'] = max(accuracy['KNN-normalized'],accuracy_score(X_label_test, X_label_pred) * 100) # saving maximum accuracy of knn-normalized
    print('Accuracy using KNN on normalized data = {:.3f}\n'.format(accuracy_score(X_label_test, X_label_pred)*100)) #printing accuracy of knn
    
    
Mean0 = X0_train.mean();  #calculating mean of class0
Mean1 = X1_train.mean();   # calcuating mean of class1
Covariance0 = np.cov(X0_train.T); # calculating covariance class0
Covariance1 = np.cov(X1_train.T);  #calculating covariance of class1


Class1Predicted = [];
Class0Predicted = [];
Actual0Class = []
Actual1Class = []
for i in X_test.index:
   
    M0 = findMaxliklhood(Mean0,Covariance0,X_test.loc[i]);  #finding maximum likelihood with mean and covariance vector as class0
    M1 = findMaxliklhood(Mean1,Covariance1,X_test.loc[i]);  #finding maximum likelihood with mean and covariance vector as class1
    M0 = M0*X0_train.size;
    M1 = M1*X1_train.size;
    if(M0>M1):
        Class0Predicted.append(i);  # appending class according to maximum likelihood value
    else:
        Class1Predicted.append(i);
for index,items in X_label_test.iteritems():
    if items==1:
        Actual1Class.append(index)  # appending indics with actuall class1
    else:
        Actual0Class.append(index);  # appending indics with actuall class0
    
print("........................QUESTION3...................................\n")
print("Using bayes classifier\n")
Class0Correct =   comClass0Predictmon(Actual0Class,ed)   # classes that were correctly predicted as 0
Class1Correct =   common(Actual1Class,Class1Predicted)  # classes that were correctly predicted
Class1Class0  =   len(Actual1Class)-Class1Correct  # classes that were wrongly predicted as 0
Class0Class1  =   len(Actual0Class)-Class0Correct   # classes that were wrongly predicted as 1
matrix = np.array([[Class0Correct,Class0Class1],[Class1Class0,Class1Correct]]);  # confusion matrix
print(matrix)
Accuracy = (matrix[0][0] + matrix[1][1])/(matrix[1][1] + matrix[0][1] + matrix[1][0] + matrix[0][0]); # calculating accuracy
print('Accuracy using bayes = {:.3f}\n'.format(Accuracy))   
accuracy['Bayes'] = max(Accuracy*100 , 0)
print("...........................QUESTION4............................")
print("Maximum accuracy of all three methods are")
for x in accuracy.keys():
    print(x,end = " ");
    print('= {:.3f}\n'.format(accuracy[x]));  #printing maximum accuracy from all methods
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    