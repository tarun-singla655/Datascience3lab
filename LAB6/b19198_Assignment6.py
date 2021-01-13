import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from math import sqrt
from sklearn.metrics import mean_squared_error


FileData = pd.read_csv("datasetA6_HP.csv")

print("..........QUESTION_1............................")
#1a
x = FileData["Date"]
y = FileData["HP"]
print("................1a.............................")
plt.plot(x,y)
plt.xlabel("Days")
plt.ylabel("Power Consumed in MegaWatt")
plt.show()


#1b
print("................1b.............................")
x1 = y.loc[1:]
x2 = y.iloc[0:x.size-1]
x1 = list(x1)
x2 = list(x2)
Corelation = np.corrcoef(x1,x2)
print("Auto Correlation with 1 time lag ",end = "")
print(Corelation[0][1])


# 1c
print(".................1c...........................")
print("Plot between x(t) and x(t-1) ")
plt.scatter(x1,x2)
plt.xlabel("x(t)")
plt.ylabel("x(t-1)")
plt.show()

# 1d
print("..................1d...........................")
y3 = []
x3 = []
for i in range(1,8):
    x1 = y.loc[i:]
    x1 = list(x1)
    x2 = y.iloc[0:x.size-i]
    x2 = list(x2)
    Correlation = np.corrcoef(x1,x2)
    y3.append(Correlation[0][1])
    x3.append(i);
plt.stem(x3,y3,use_line_collection= True)
plt.xlabel("different time Lags")
plt.ylabel("AutoCorrelation for different timelags")
plt.show()

# # 1e
print(".................1e...............................")
sm.graphics.tsa.plot_acf(y,lags = [1,2,3,4,5,6,7]);
plt.xlabel("different time lags")
plt.ylabel("AutoCorrelation fuction plot")
plt.show()



#2
print("..........QUESTION_2............................")
y = FileData["HP"]
X = y.values
test = X[len(X)-250:]
Actual = test[:test.size-1]
Predicted = test[1:]
rmse = sqrt(mean_squared_error(Actual,Predicted))
print("RMSE using presistent algorithm ")
print("rmse ",rmse);

# 3a
print("..........QUESTION_3............................")
print("..........3a............................")
X = y.values
train, test = X[1:len(X)-250], X[len(X)-250:]
model = AutoReg(train, lags=5)
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
rmse = sqrt(mean_squared_error(test, predictions))
print("RMSE error for the test data ");
print('Test RMSE: %.3f' % rmse)
x1 = [x for x in range(0,250)];
plt.plot(x1,test,label = "Actual")
plt.plot(x1,predictions , label = "predicted")
plt.xlabel("days")
plt.ylabel("Power Consumed")
plt.legend()
plt.show()

lag = 0
value = 43
#3b
print("..........3b............................")
l = [1,5,10,15,25]
RMSE = []
for i in l:
    model = AutoReg(train, lags=i)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    rmse = sqrt(mean_squared_error(test, predictions))
    if rmse < value:
        value = rmse
        lag = i
    RMSE.append(rmse)
    
    print('Test RMSE for',i, 'is : %.3f' % rmse)


print("..........3c......................")
lag_val=1
T=len(test)-1
while True:
    xx=test[:-lag_val]
    yy=test[lag_val:]
    cof=abs(np.corrcoef(xx,yy)[0][1])
    if(cof>2/T**(1/2)):
        T-=1
        lag_val+=1
    else:
        lag_val-=1
        break
print("Optimal value of Time lag ")
print(lag_val)

model = AutoReg(train, lags=lag_val)
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
rmse = sqrt(mean_squared_error(test, predictions))
print("RMSE error with Optimal time Lag value")
print(rmse)

print("...........3d....................")
print("The optimal number of lags without using heuristics for calculating optimal lag is ",lag ," and RMSE is ",value)
print("The optimal number of lags using heuristics for calculating optimal lag is ",lag_val," and value is ",rmse);

