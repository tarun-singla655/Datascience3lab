import pandas as pd
import matplotlib.pyplot as plt
import statistics
import math
OriginalData  = pd.read_csv("pima_indians_diabetes_original.csv")
MissingData  = pd.read_csv("pima_indians_diabetes_miss.csv")



#problem 1
Data = MissingData.isnull()   # Checking for Na values having Na values is True 
Data3 = MissingData.dtypes;  # extracting Attributes for ploting graphs 
print("PROBLEM 1....................")
print("")
Attributes = []
Missing_Values = [];
for x,y in Data3.iteritems():
    Attributes.append(x)      # pushing attributes in  List
for i in Attributes:
    seriesObj = Data[i]     # extracting particular attribute  
    numOfRows = len(seriesObj[seriesObj == True].index)  #num of rows with particular Attributes missing
    Missing_Values.append(numOfRows)  
fig = plt.figure()   # ploting graph
ax = fig.add_axes([0,0,1,1])
ax.bar(Attributes,Missing_Values)
plt.title("Total missing Values of each attributes")
plt.show()
print("")


def FindMode(data):
    Mode = []
    for i in Attributes:
        Mode.append(statistics.mode(data[i]))
    return Mode;
        



# # problem 2a
print("PROBLEM 2a.......................")
print(".....................................................")
print("")
MissingValueRow = MissingData.isnull().sum(axis = 1) # calculating Number missingValues in each row
droprow = []
for i in MissingValueRow.index:  
    if MissingValueRow[i]>=3:    # checking missing values greater than or equal to 3
        droprow.append(i)
MissingData.drop(droprow,inplace = True)   # droping rows
print("Total number of tuples deleted ", len(droprow) ) # printing length of rows
print("Row number of deleted tuples is",droprow)  # printing rows


print("")
MissingData2 = MissingData.isnull();  # checking to boolen values with missing to true
DropRow = []







print("PROBLEM 2b...............")
print(".....................................................")
print("")
#problem 2b
for i in MissingData2.index:
    if(MissingData2['class'][i] == True):   # checking with class attribute missing
        DropRow.append(i);
MissingData.drop(DropRow,inplace = True)
print("Total number of deleted tuples", len(DropRow) )  #total Deleted tuples
print("Rw number of deleted rows",DropRow)   # index number










print("")
# #problem 3
print("PROBLEM 3..........")
print(".....................................................")
print("")
TotalMissingValue = MissingData.isnull().sum(axis = 1).sum() # total number of missing values
print( "Total Missing values are ",TotalMissingValue)
MissingRows =  MissingData.isnull().sum(axis = 0);  # total number of missing values in each row
print( "Missing Values of each Attribute\n",MissingRows)   #MissingRows
print("")








#4..
MissingValues = MissingData.isnull();  # conv
RowWithValMissing = []
for i in Attributes:  
    Rows = [];
    for j in MissingData[i].index:
        if(MissingValues[i][j] == True):  # storing row index of missing values of each attributes
            Rows.append(j);
    RowWithValMissing.append(Rows);  # appending rows
print("PROBLEM 4a....................")
print("")
Mean = OriginalData.mean();              #mean
StandardDeviation = OriginalData.std();   #standardDeviation
Median = OriginalData.median();            #Median
Mode = OriginalData.mode();            #Mode
print("MEAN MEDIAN MODE STANDARAD DEVIATION OF ORIGINAL DATA")
print(".....................................................")
print("")
print("Mean of Orignal data is\n", Mean)
print("Median of Orignal data is\n", Median)
print("Mode of Orignal data is\n" );
Mode = FindMode(OriginalData);
for i in range(len(Attributes)):
    print(Attributes[i] , "    ", Mode[i]);
print("StandardDeviation of Orignal data is\n", StandardDeviation)
print("")
MissingData2 = MissingData.fillna(MissingData.mean());  #Filling MissingData with mean
l1 = MissingData.index.size;
RMSValues = [];
Mean = MissingData2.mean();
StandardDeviation = MissingData2.std();
Median = MissingData2.median();
Mode = MissingData2.mode();
print(".......................................")
print("Mean after filling with mean is\n", Mean)
print("Median after filling with mean is\n", Median)
Mode = FindMode(MissingData2);
for i in range(len(Attributes)):
    print(Attributes[i] , "    ", Mode[i]);
print("StandardDeviation after filling with mean is\n", StandardDeviation)
print("")
for i in range(len(Attributes)):
    Total = 0;
    Na = 0
    for j in MissingData[Attributes[i]].index:  # Looping
        if j not in RowWithValMissing[i]:
            continue;
        Total += (OriginalData[Attributes[i]][j] - MissingData2[Attributes[i]][j])**2; # cakculating total
        Na += 1
    if Total == 0:
        RMSValues.append(0);
        continue;
    Total /= Na
    Total = math.sqrt(Total);  #RMSE square Error
    RMSValues.append(Total);   #appending RMS value
print("RMS values After replacing missing values with mean")

for i in range(len(Attributes)):
    print(Attributes[i], "   ",RMSValues[i]);
plt.bar(Attributes,RMSValues);  # plotting values   
plt.title("RMSE Value of Attributes after filing with mean")
plt.xlabel("attributes")
plt.ylabel("RMSE Values")
plt.show();







RMSValues = []
print("PROBLEM 4b.......................")
print(".....................................................")
print("")

MissingData3 = MissingData.fillna(MissingData.interpolate()); #filling values using interpolation
Mean = MissingData3.mean();
StandardDeviation = MissingData3.std();
Median = MissingData3.median();
Mode = MissingData3.mode();
print("")
print("Mean after filling with interpolation is\n", Mean)   #mean
print("Median after filling with interpolation is\n", Median)  #medain
Mode = FindMode(MissingData3);
print("Median after filling with interpolation is ")
for i in range(len(Attributes)):                    #mode
    print(Attributes[i] , "    ", Mode[i]);
print("StandardDeviation after filling with interpolation is\n", StandardDeviation) # standard deviation 
print("")
for i in range(len(Attributes)):   
    Total = 0;
    Na = 0;
    for j in MissingData[Attributes[i]].index:
        if j not in RowWithValMissing[i]:  # checking if value is missing or not
            continue
        Total += ( OriginalData[Attributes[i]][j] - MissingData3[Attributes[i]][j] )**2 #calculating RMS
        Na += 1
    if Total == 0:
        RMSValues.append(0);
        continue
    Total /= Na
    Total = math.sqrt(Total);
    RMSValues.append(Total);  # appending RMS value
print("RMS values After replacing missing values with interpolation")
for i in range(len(Attributes)):  #printing RMS
    print(Attributes[i], "   ",RMSValues[i]);
plt.bar(Attributes,RMSValues);
plt.title("RMSE Value of Attributes after filing with Interpolation")
plt.xlabel("attributes")
plt.ylabel("RMSE Values")
plt.show();    # ploting graph









print("")
# # # 5...
print("PROBLEM 5......................")
print(".....................................................")
print("")
Outlier = []
Array = ['BMI' , 'Age']   # for BMI and Age
valueOutlier = []
for i in Array:
    Values = MissingData3[i];   #particular value attributes
    q2 = Values.quantile(.5);   #Q1
    q1 = Values.quantile(.25);  #Q2
    q3 = Values.quantile(.75);  #Q3
    Outliers = []
    for j in MissingData3[i].index:
        if( (1.5*abs(q1-q3) + q3) <= MissingData3[i][j] or (q1 - 1.5*abs(q1-q3)) >= MissingData3[i][j]):  # checking for outliers
            Outliers.append(j);
            valueOutlier.append(MissingData3[i][j]);
    plt.boxplot(MissingData3[i]);  # boxplot
    s1 = "Boxplot of "+i
    plt.title(s1)
    plt.show();
    Outlier.append(Outliers);   # appending outliers
    print("outliers for ",i," is ",valueOutlier);  
for i in Array:
    Median = MissingData3[i].median();   # median after interpolation
    index = 0
    if(i=='Age'): 
        index = 1
    for j in MissingData3[i].index:  
        if j in Outlier[index]:       # checking if its outlier
            MissingData3[i][j] = Median  # replacing outlier with median
    plt.boxplot(MissingData3[i])
    s1 = "Boxplot of "+i+" after replacing outliers with median "
    plt.title(s1)  #title
    plt.show();  #ploting 
            
