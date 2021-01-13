import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy
FileData = pd.read_csv("landslide_data3.csv")  # reading file
Columns = FileData.columns;       # extract all columns
Columns = list(Columns)      
Columns = Columns[2:]   # removing dates and stationid
FileData = FileData.drop(['stationid','dates'],axis=1)
for i in Columns:
    q1 = FileData[i].quantile(q=0.25)
    q3 = FileData[i].quantile(q=0.75)
    # identifying outliers
    outliers = FileData[i][(FileData[i] <= q1 - 1.5 * (q3 - q1)) | (FileData[i] >= q3 + 1.5 * (q3 - q1))]
    # using replace() in pandas.DataFrame to replace outliers by NaN
    FileData[i].replace(to_replace=outliers, value=np.nan, inplace=True)
    # using median() in pandas.DataFrame to calculate median without outliers
    median = FileData[i].median()
    # using replace() in pandas.DataFrame to replace outliers by its respective median
    FileData[i].replace(to_replace=outliers, value=median, inplace=True)
    # using fillna() in pandas.DataFrame to fill missing values by its respective median
    FileData[i].fillna(value=median, inplace=True)

print("")
print("Question 1a...................\n")
print("Minimum value before min-max normalization\n")
Minimum = FileData.min(axis = 0);    # minimum valu
print(Minimum)
print("Maximum value before min-max normalization\n")
Maximum = FileData.max(axis = 0);   # maximum value 
print(Maximum)
FileData1 = FileData.apply(lambda x : ((x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)))*3 + 6)
Minimum = FileData1.min(axis = 0);  # minimum after min max normalization
print("Minimum value after min-max normalization\n")
print(Minimum)
Maximum = FileData1.max(axis = 0);    # maximum after min max normalization
print("Minimum value after min-max normalization\n")
print(Maximum)

print("")
print("Question 1b...................\n")
Mean = FileData.mean(axis=0)    # mean 
print("Mean before Standardizing\n")
print(Mean)
Std = FileData.std(axis=0)   # standard deviation
print("Standard deviation before Standardizing\n")
print(Std)
FileData1 = FileData.apply(lambda x : (x-x.mean(axis=0))/x.std(axis=0))
Std = FileData1.std(axis = 0);   # standard deviation after standaradization
Mean = FileData1.mean(axis = 0);   # mean after standardization
print("Mean after Standardizing\n")
print(Mean)
print("Standard deviation after Standardizing\n")
print(Std)


print("")
print("Question 2a...................\n")
mean=[0,0]   # mean vector
cov=[[6.84806467,7.63444163],[7.63444163,13.02074623]]  # covariance matrix
D_mat=[[],[]]
no_samp=1000
for i in range(no_samp):
    xx=np.random.multivariate_normal(mean,cov)  # generating multivariate sample 
    D_mat[0].append(xx[0])   
    D_mat[1].append(xx[1])
plt.scatter(D_mat[0],D_mat[1])   # scatter plot
plt.title('plot of multivariate data')

plt.show()
print("Question 2b...................\n")
eige=np.linalg.eig(cov)  # calculating eigenvalue , eigenvector
plt.scatter(D_mat[0],D_mat[1])   
plt.title('plot of multivariate data with direction of vectors')
plt.quiver(0,0,eige[1][0][0],eige[1][1][0],scale=3,color='red')  #quiver plot in direction of first eigen vector
plt.quiver(0,0,eige[1][0][1],eige[1][1][1],scale=3,color='red')  # quiver plot in the direction of second eigen vector 
plt.axis('equal')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
print("Question 2c...................\n")
xx1=[]
yy1=[]
print("for EigenVector 1")
for i in range(1000):
    dot=eige[1][0][0]*D_mat[0][i]+eige[1][1][0]*D_mat[1][i]  # calculating component
    xx1.append(dot*eige[1][0][0])   # component with direction towards eigenvector
    yy1.append(dot*eige[1][1][0])   
plt.scatter(D_mat[0],D_mat[1])  # scatter plot
plt.quiver(0,0,eige[1][0][0],eige[1][1][0],scale=3,color='red')
plt.quiver(0,0,eige[1][0][1],eige[1][1][1],scale=3,color='red')
plt.scatter(xx1,yy1,marker='x')   # component along eigenvector
plt.title('plot of multivariate data with projection of eigenvector 1')
plt.axis('equal')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
xx=[]
yy=[]

print("for EigenVector 2 ")
for i in range(1000):
    dot=eige[1][0][1]*D_mat[0][i]+eige[1][1][1]*D_mat[1][i]  #component of second eigen vector
    xx.append(dot*eige[1][0][1])  # component in the direction of vector     
    yy.append(dot*eige[1][1][1])
plt.scatter(D_mat[0],D_mat[1])  
plt.quiver(0,0,eige[1][0][0],eige[1][1][0],scale=3,color='red')  # direction to the eigen vector
plt.quiver(0,0,eige[1][0][1],eige[1][1][1],scale=3,color='red')  # 
plt.scatter(xx,yy,marker='x')   # component along vector
plt.title('plot of multivariate data with projection of eigenvector 2')
plt.axis('equal')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

print("Question 2d...................\n")
for i in range(1000):
    xx[i] += xx1[i];      # component in direction of both
    yy[i] += yy1[i];     
MSE = 0;
for i in range(1000):
    error = (D_mat[0][i]-xx[i])**2 + (D_mat[1][i]-yy[i])**2;  #MSE Error
    MSE += error;   
MSE/=1000    # dividing by total number of samples
print("MSE error after regeneration data \n")
print(MSE)






standardized_data = FileData1
sample_data = standardized_data #we have already standardized data
coav_matrix = np.dot(sample_data.T,sample_data)  # calculating covariance matrix



print("Question 3a...................\n")
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)   # value of l = 2 project on top two eigen vectors
PC = pca.fit_transform(FileData1) # projecting components
PCDf = pd.DataFrame(data = PC, columns=('principal component1','principal component2')) #converting into data frame
x = PCDf['principal component1']   #component1
y = PCDf['principal component2']   #component2 
print("Variance in the direction of projection\n")
print(pca.explained_variance_)
from scipy.linalg import eigh
coav_matrix/=944  # dividing by total number of samples
values,vectors = eigh(coav_matrix,eigvals=(5,6))  # calculating top 2 eigen values
values = list(values)  
values.reverse();  # just printing in decreasing order
print("printing Eigenvalues \n",values);
plt.scatter(x,y)    # scatter plot of projection
plt.title("plot after dimension Reduction")
plt.xlabel("principle component along eigenvector 1")
plt.ylabel("principle component along eigenvector 2")
plt.show();


print("Question 3b...................\n")
values,vectors = eigh(coav_matrix/944,eigvals=(0,6))  # all seven eigen vectors in increasing order
values = list(values)   
values.reverse();  # to make it in decreasing order
print("decreasing order of eigen values \n")
print(values);
plt.plot(values);
plt.xlabel("indics")
plt.ylabel("EigenValues");
plt.show();



print("Question 3c...................\n")
lx  = []
ly = []
val = 0
for i in range(1,8):
    val = 0;    
    pca = PCA(n_components = i)  # value of l = i
    pca.fit(FileData1)
    PC = pca.fit_transform(FileData1)  # project on i components
    vectors = pca.components_ # calculating eigen vector
    ans = np.dot(PC,vectors)    # reconstructing error
    PCdf2 = pd.DataFrame(data = ans,columns = ['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture'])  # converting into dataframe
    for j in Columns:
        for k in FileData[j].index:    
            val += (FileData1[j][k]-PCdf2[j][k])**2;  # calculating error
    lx.append(i)         # appending value of L
    ly.append((val/FileData1.shape[0])**0.5)   # appending RMSE
plt.bar(lx,ly)     #ploting bar graph
plt.xlabel("Value of l" , weight = "bold")
plt.ylabel("error ",weight = "bold")
plt.title("RMSE error\n")
plt.show();













