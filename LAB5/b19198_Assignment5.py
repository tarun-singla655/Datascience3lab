#Tarun Singla
#Roll no - B19198
#Mobile no - 8872526396

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error    


#Question 1
print("---------------------------Question A-------------------------------")
print("")
print("...........................question 1.....................................")
train = pd.read_csv('seismic-bumps-train.csv')#importing the train data
test = pd.read_csv('seismic-bumps-test.csv')#importing the test data

X_train0 = train[train['class']==0] 
X_train1 = train[train['class']==1]
X_train0 = X_train0.drop(['class'],axis = 1)#training input set for class 0
X_train1 = X_train1.drop(['class'],axis = 1)#training input set for class 1

X_test = test.drop(['class'],axis = 1)#test input set
X_label_test = test['class']#actual test output set
p0 = X_train0.size/(X_train0.size+X_train1.size)
p1 = X_train1.size/(X_train0.size+X_train1.size)
for i in range(1,5):
    gauss_pred = []
    GMM0 = GaussianMixture(n_components=2**i, covariance_type='full') #gaussian mixture
    GMM0.fit(X_train0)#fitting train data on GMM
    score_samples_0 = GMM0.score_samples(X_test)#weighted log probabilities for class 0

    GMM1 = GaussianMixture(n_components=2**i, covariance_type='full')#gaussian mixture
    GMM1.fit(X_train1)#fitting test data on GMM
    score_samples_1 = GMM1.score_samples(X_test)#weighted log probabilities for class 1
    
    for j in range(len(score_samples_1)):
        if p0*np.exp(score_samples_0[j])>p1*np.exp(score_samples_1[j]):
            gauss_pred.append(0)
        else:
            gauss_pred.append(1)
    print("\nFor Q = ",2**i)
    print("\nConfusion matrix is : ")
    print(confusion_matrix(gauss_pred,X_label_test))#calculating confusion matrix 
    accuracy = metrics.accuracy_score(X_label_test,gauss_pred)#calculating the accuracy
    print("Accuracy is : ",accuracy)
print("")
print("...........................question 2.....................................")
print("Maxmimum accuracy for KNN is 93.170")
print("Maxmimum accuracy for KNN on Normalized is 92.913")
print("Maxmimum accuracy for Bayes using unimodel gaussian density 87.500")
print("Maxmimum accuracy for KNN is 93.556")
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

 
def findPredictionAcc(Y_pred,Y):   # for calculating prediction Accuracy RMSE
        length = Y.shape[0]   
        ans = 0
        for i in range(length):
            ans += (Y[i]-Y_pred[i])**2
        ans/=length
        ans = math.sqrt(ans)
        return ans
print("")
print(".......................Question B....................................")
print(".........1a...........................")
df = pd.read_csv("atmosphere_data.csv")
x = df['pressure']
y = df['temperature']
X_train , X_test, X_label_train, X_label_test = train_test_split(x, y, test_size=0.3,
                                                                  random_state=42, shuffle=True) # splitting data
x = np.array(X_train)    # converting into array
y = np.array(X_label_train)
x = x.reshape(-1,1)   # reshape into -1,-1 for function in sklearn
y = y.reshape(-1,1)
reg = LinearRegression().fit(x, y)  # fitting linear model
y_pred = reg.predict(x)
plt.plot(x,y_pred,color = "r")  # plotting regression line 
plt.scatter(x,y)  # scatter plot of actual data
plt.title("best Fit line")   
plt.xlabel("Pressure")
plt.ylabel("Temperature")   
plt.show();
print(".........1b...........................")
print("Prediction Accuracy on training dataset")
ans = findPredictionAcc(y_pred,y)  # calculating prediction accuracy
print(ans)
X_test = np.array(X_test)              
X_test = X_test.reshape(-1,1)
X_label_test = np.array(X_label_test)
X_label_test = X_label_test.reshape(-1,1)
y_pred1 = reg.predict(X_test)     # predicting test data based on linear plot

print(".........1c...........................")
print("Prediction Accuracy on testing dataset")
ans = findPredictionAcc(y_pred1,X_label_test)     # calculating prediction accuracy 
print(ans)
plt.scatter(X_label_test,y_pred1)  # scattel plot of actual temperature and predicted temperature
print(".........1d...........................")
plt.title("Actual temperature v/s predicted temperature")
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.show();
p = [2,3,4,5]  # plotting for different p values
X1 = []
Y1 = []
print("")
print("........................2a.........................")
for i in p:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression 
    poly_reg = PolynomialFeatures(degree=i)              # polynomial regression with particular degree
    X_poly = poly_reg.fit_transform(np.array(X_train).reshape(-1, 1))  # building regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly,np.array(X_label_train).reshape(-1, 1))  # fitting into a polynomial equation 
    y_train_pred = lin_reg.predict(X_poly)        # predicting the values of a polynomial
    ans = findPredictionAcc(y_train_pred,np.array(X_label_train))      # calcuating accuray of training data set
    print("Prediction Accuracy on train dataset with p value ",i,"is ",ans)
    X1.append(i)
    Y1.append(ans)
print("Plot Prediction Accuracy on train dataset with  different p values ")    
plt.bar(X1,Y1)                    # bar plot of Accuracy with different values of p
plt.ylabel("RMSE values")
plt.xlabel("p-values")
plt.show()
print("")
print("........................2b.........................")
Y2 = []
for i in p:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    poly_reg = PolynomialFeatures(degree=i)               # polynommial regression with particular degree
    X_poly = poly_reg.fit_transform(np.array(X_train).reshape(-1, 1)) # # building regression model
    X_poly1 = poly_reg.fit_transform(np.array(X_test).reshape(-1, 1))
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly,np.array(X_label_train).reshape(-1, 1))    # fitting into a polynomial equation
    y_test_pred = lin_reg.predict(X_poly1)                # predicting the values of test dataset
    ans = findPredictionAcc(y_test_pred,np.array(X_label_test))  # calculating accuracy of test dataset
    print("Prediction Accuracy on test dataset with p value ",i,"is ",ans)
    Y2.append(ans)
    
print("Plot Prediction Accuracy on train dataset with  different p values ")    
plt.bar(X1,Y2)    # bar plot of Accuracy with different values of p
plt.ylabel("RMSE values")
plt.xlabel("p-values")
plt.show()

print("")
print("........................2c.........................")
print(" polynomial curve fitting having minimum RMSE error is with p=5")  # curve fitting having minimum error that is maximum accuracy
X = np.asarray(X_train)
Y  = y_train_pred    
Y = Y.T
Y = Y[0].tolist();
plt.scatter(X_train, X_label_train);   # scatter plot of actuall values
x1 = np.polyfit(X,Y,5)  # fit into a polynomial of degree 5
poly = np.poly1d(x1)         # coverting into id polynomial
Y = [poly(x) for x in range(30,1100)]   # calculating values using equation
X = [x for x in range(30,1100)]
plt.plot(X,Y,color = "r")  # ploting curve
plt.xlabel("pressure")  
plt.ylabel("temperature")
plt.title("best fitting curve with low RMSE for a p")
plt.show()


print("")
print("........................2d.........................")
print("plot of predicted temperature v/s actual having minimum RMSE error is with p=5")
plt.scatter(X_label_test,y_test_pred) #ploting predicted temperature vs actual temperature 
plt.xlabel("Actual Temperature")
plt.ylabel("predicted temperature")
plt.show()
  


    
    
