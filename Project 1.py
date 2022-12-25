#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd

df=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - With missing values.csv')


# In[20]:


#select rows and columns from the dataset and print them
data=df.iloc[:7,:7]
print("The first 7 rows and 7 columns are:")
print(data)


# In[21]:


#description of the data
df.info()


# In[22]:


#summary of the data
df.describe()


# In[23]:


import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()


# In[24]:


#check if any missing values
df.isna()
   


# In[25]:


#number of missing values
df.isna().sum()


# In[26]:


#drop the row having any missing value
dataAfterDrop=df.dropna(axis=0,how='any')
#save the data with no missing values
dataAfterDrop.to_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - WithOut missing values.csv')


# In[27]:


dataAfterDrop.info()


# In[33]:


#detect the outliers using the z-score for each column

from sklearn.preprocessing import LabelBinarizer
#encoding gender column
df=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - With missing values.csv')
lb=LabelBinarizer()
df['gender']=lb.fit_transform(df[['gender']])
df.to_csv(r'diabetes - With missing values final.csv', index=False)
meanValue=df['hip'].mean()
df['hip'].fillna(value=meanValue,inplace=True)
#detect outliers
import numpy as np
from scipy import stats
zscores=np.abs(stats.zscore(df))
absZscores=np.abs(zscores) 
filteredEnteries=(absZscores>3).any(axis=1)
df=pd.read_csv(r'diabetes - With missing values final.csv')
dataOutliers=df[filteredEnteries]
withoutOutliers = df[~filteredEnteries]
dataOutliers.to_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - dataOutliers.csv', index=False)
withoutOutliers.to_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - correctData-noOutliers.csv', index=False)


# In[34]:


df=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - correctData-noOutliers.csv')
#calculate the mean of the hip column having the missing values 
meanValue=df['hip'].mean()
#replace with the mean value the missing values
df['hip'].fillna(value=meanValue,inplace=True)
#save the new data
df.to_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - afterReplaceWithMean.csv')


# In[35]:


#calculate the median
df=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - correctData-noOutliers.csv')
medianValue=df['hip'].median()
#replace with the median
df['hip'].fillna(value=medianValue,inplace=True)
df.to_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - afterReplaceWithMedian.csv')


# In[36]:


#replace with interpolation
df=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - correctData-noOutliers.csv')
df['hip'].interpolate(inplace=True)
df.to_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - afterReplaceWithInterpolate.csv')


# In[37]:


#replace with constant
df=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - correctData-noOutliers.csv')
df['hip'].fillna(50,inplace=True)
df.to_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - afterReplaceWithConstant.csv')


# In[42]:


df=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - Complete dataset.csv')
df=df[~filteredEnteries]
df.to_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - Complete dataset-NoOutliers.csv')


# In[43]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#save the initial values
df=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - Complete dataset-NoOutliers.csv')
initialValues=df['hip']


# In[44]:


#checking predictions for mean value
#values after replacing with mean
df1=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - afterReplaceWithMean.csv')
valuesAfterReplacingwithMean=df1['hip']
meanRMSE=mean_squared_error(initialValues, valuesAfterReplacingwithMean, squared=False)
meanMAE=mean_absolute_error(initialValues, valuesAfterReplacingwithMean)
print(meanRMSE)
print(meanMAE)


# In[45]:


#checking predictions for median value
#values after replacing the median
df1=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - afterReplaceWithMedian.csv')
valuesAfterReplacingwithMedian=df1['hip']
medianRMSE=mean_squared_error(initialValues, valuesAfterReplacingwithMedian, squared=False)
medianMAE=mean_absolute_error(initialValues, valuesAfterReplacingwithMedian)
print(medianRMSE)
print(medianMAE)


# In[46]:


#checking predictions for constant value

#values after replacing the constant
df1=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - afterReplaceWithConstant.csv')
valuesAfterReplacingwithConstant=df1['hip']
cnstRMSE=mean_squared_error(initialValues, valuesAfterReplacingwithConstant,squared=False)
cnstMAE=mean_absolute_error(initialValues, valuesAfterReplacingwithConstant)
print(cnstRMSE)
print(cnstMAE)


# In[47]:


#checking predictions for interpolate value

#values after replacing the interpolate
df1=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - afterReplaceWithInterpolate.csv')
valuesAfterReplacingwithInterpolate=df1['hip']
interRMSE=mean_squared_error(initialValues, valuesAfterReplacingwithInterpolate,squared=False)
interMAE=mean_absolute_error(initialValues, valuesAfterReplacingwithInterpolate)
print(interRMSE)
print(interMAE)


# In[51]:


#applying sample random sampling
df=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - afterReplaceWithMean.csv')

#sample
for x in np.arange(0.1,1,0.1):
    sample=df.sample(frac=x)
    sample.to_csv('F:\\Al Maaref Uni\\Second Year 2021-2022\\Spring 21-22\\CSC 458 AI\\Project 1\\diabetes - sample%s.csv' % (int(x*100)))


# In[52]:


#finding the optimal knn
bestK=np.ones(9)
j=0
for x in np.arange(0.1,1,0.1):
    sample=pd.read_csv("F:\\Al Maaref Uni\\Second Year 2021-2022\\Spring 21-22\\CSC 458 AI\\Project 1\\diabetes - sample%s.csv" % (int(x*100)))
    #two datsets: x without the label and y with the labeli
    X=sample.drop('result',axis=1).values
    Y=sample['result'].values    
    #split the data
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=42,stratify=Y)
    #generate an array from 1-10 and an empty array
    neighbors=np.arange(1,11)
    test_accuracy=np.empty(len(neighbors))
    #check the accuarcy for different values of k: setup the k and apply ot on the training set, compute the accuarcy
    from sklearn.neighbors import KNeighborsClassifier     
    i=0
    for k in neighbors:
        
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,Y_train)
        test_accuracy[i]=knn.score(X_test,Y_test)
        if test_accuracy[int(bestK[j])-1]<test_accuracy[i]:
            bestK[j]=k
        i+=1
    j+=1
print(bestK)


# In[53]:


#select the optimal k
predictionScore=np.empty(9)
YPrediction=[]
j=0
for x in np.arange(0.1,1,0.1):
    sample=pd.read_csv(r'F:\Al Maaref Uni\Second Year 2021-2022\Spring 21-22\CSC 458 AI\Project 1\diabetes - sample%s.csv' % int(x*100))
    X=sample.drop('result',axis=1).values
    Y=sample['result'].values   
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=42,stratify=Y)
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    YPrediction.append(knn.predict(X_test))
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    c_matrix=confusion_matrix(Y_test,YPrediction[j])
    print('Sampling ratio: %s' %x)
    print(c_matrix)        
    print(classification_report(Y_test,YPrediction[j]))
    j+=1
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




