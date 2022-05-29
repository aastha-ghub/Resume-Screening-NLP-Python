#!/usr/bin/env python
# coding: utf-8

# <table align="left" width=100%>
#     <tr>
#         <td width="10%">
#             <img src="title.png">
#         </td>
#         <td>
#             <div align="left">
#                 <font color="#21618C" size=8px>
#                   <b>RESUME SCREENING
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# ## PROBLEM STATEMENT:
# 
# Companies often receive thousands of resumes for each job posting and employ dedicated screening officers to screen qualified candidates.
# Typically, large companies do not have enough time to open each CV, so they use machine learning algorithms for the Resume Screening task.
# 
# Here,we will go through a machine learning project on Resume Screening with Python programming language.

# ## STEPS IMPLEMENTED
# 
# 1. **[Import Packages](#import_packages)**
# 2. **[Read Data](#Read_Data)**
# 3. **[Understand and Prepare the Data](#data_preparation)**
#     - 3.1 - [Data Types and Dimensions](#Data_Types)
#     - 3.2 - [Missing Data Treatment](#Missing_Data_Treatment)
#     - 3.3 - [Statistical Summary](#Statistical_Summary)
# 4. **[EDA](#EDA)**    
# 5. **[Label Encoding for categorical Variable](#Label_Encoding_for_categorical_Variable)**
# 6. **[ML Models](#ML_Models)** 
# 7. **[Make Predictions](#Prediction_on_Models)** 

# <a id='import_packages'></a>
# ## 1. Import Packages

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier


# <a id='Read_Data'></a>
# ## 2. Read the Data

# In[54]:


# importing and reading the .csv file
resumeData = pd.read_csv('/Users/Admin/Desktop/CAPSTONE_1/UpdatedResumeDataSet.csv' ,encoding='utf-8')
#resumeData['cleaned_resume'] = ''
print("The number of rows are", resumeData.shape[0],"and the number of columns are", resumeData.shape[1])


# In[55]:


# print the first five rows of the data
resumeData.head()


# <a id='data_preparation'></a>
# ## 3. Understand and Prepare the Data

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>The process of data preparation entails cleansing, structuring and integrating data to make it ready for analysis. <br><br>
#                         Here we will analyze and prepare data :<br>
#                         1. Check dimensions and data types of the dataframe <br>
#                         2. Check for missing values<br>
#                         3. Study summary statistics<br> 
#                                        </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Data_Types'></a>
# ## 3.1 Data Types and Dimensions

# In[56]:


# get the shape
print(resumeData.shape)


# In[57]:


# Checking the information of the dataframe(i.e the dataset)
resumeData.info()


# <a id='Missing_Data_Treatment'></a>
# ## 3.3. Missing Data Treatment

# In[58]:


# get the count of missing values
missing_values = resumeData.isnull().sum()

# print the count of missing values
print(missing_values)


# There are no missing values present.

# <a id='Statistical_Summary'></a>
# ## 3.4. Statistical Summary
# Here we take a look at the summary of each attribute.

# In[59]:


# data frame with categorical features
resumeData.describe(include='object')


# In[60]:


# Checking all the different unique values

resumeData.nunique()


# Categories of resumes present in the dataset

# In[61]:


print ("Displaying the distinct categories of resume -")
print (resumeData['Category'].unique())

print("total unique category: {}". format(len(resumeData['Category'].unique())))


# Distinct categories of resume and the number of records belonging to each category

# In[62]:


print ("Displaying the distinct categories of resume and the number of records belonging to each category -")
print (resumeData['Category'].value_counts())


# <a id='EDA'></a>
# ## 4. EDA

# Let’s visualize the number of categories in the dataset

# In[63]:


# Plotting the distribution of Categories as a Count Plot

plt.figure(figsize=(15,15))

sns.countplot(y="Category", data=resumeData)


# Visualizing the distribution of categories

# In[64]:


# Plotting the distribution of Categories as a Pie Plot

plt.figure(figsize = (18,18))
count = resumeData['Category'].value_counts()             #storing the count 
Labels = resumeData['Category'].value_counts().keys()     #storing the value of labels

plt.title("Categorywise Distribution", fontsize=20)
plt.pie(count, labels = Labels, autopct = '%1.2f%%')

resumeData["Category"].value_counts()*100/resumeData.shape[0]


# # Defining a function to remove the URLs, hashtags, mentions, special letters, and punctuations

# In[65]:


# Function to clean the resumeData

#re.sub() function stands for a substring and returns a string where all matching occurrences of the 
#specified pattern are replaced by the replace string.
#Multiple elements can be replaced using a list when we use this function.


import re
#re.compile('<title>(.*)</title>')

def clean(data):
    data = re.sub('httpS+s*', ' ', data)                                                # Removing the links
    data = re.sub('RT|cc', ' ', data)                                                   # Removing the RT and cc
    data = re.sub('#S+', ' ', data)                                                     # Removing the hashtags
    data = re.sub('@S+', ' ', data)                                                     # Removing the mentions
    data = data.lower()                                                                 # Changing text to lowercase
    data = ''.join([i if 32 < ord(i) < 128 else ' ' for i in data])                     # Removing the special characters
    data = re.sub('s+', 's', data)                                                      # Removing extra whitespaces
    data = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', data) # Removing punctuations
    return data

resumeData['cleaned_resume'] = resumeData.Resume.apply(lambda x: clean(x))


# In[66]:


resumeData.head()


# In[67]:


resumeData_1=resumeData.copy()


# Now as we have cleared the dataset, the next task is to have a look at the Wordcloud. 

# In[68]:


#Word Cloud is a data visualization technique used for representing text data in which 
#the size of each word indicates its frequency or importance.
#A Wordcloud represents the most numbers of words larger and vice versa

import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])

totalWords =[]                               #making an empty list totalwords to store our words
Sentences = resumeData['Resume'].values
cleanedSentences = ""
for i in range(0,160):
    cleanedText = clean(Sentences[i])        #using the user-defined 'clean' fn to clean sentences
    cleanedSentences += cleanedText          #updating 'cleanedSentences'using 'cleanedText'
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:  #String of ASCIIchar considered punctuation char
            totalWords.append(word)
            
#Now we have to find the word freq distribution            
wordfreqdist = nltk.FreqDist(totalWords)            
mostcommon = wordfreqdist.most_common(50)           #find most common words
print(mostcommon)

wc = WordCloud().generate(cleanedSentences)         #generating cleaned sent that we made above
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')             #used to display smoother image.
plt.axis("off")
plt.show()


# <a id='Label_Encoding_for_categorical_Variable'></a>
# ## 5. Label Encoding for categorical Variable

# Converting these words into categorical values

# In[69]:


# Encoding the Category column using LabelEncoder

from sklearn.preprocessing import LabelEncoder

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeData[i] = le.fit_transform(resumeData[i])


# In[70]:


resumeData.head()


# In[71]:


resumeData.Category.value_counts()


# In[72]:


resumeData_1.Category.value_counts() #understanding decode LabelEncoder


# In[73]:


del resumeData_1     #clearing the space occupied 


# In[75]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

requiredText = resumeData['cleaned_resume'].values
requiredTarget = resumeData['Category'].values

word_vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english')   #cleaned_resume into vector format using tfidf
word_vectorizer.fit(requiredText)                                            #fitting vector on text
WordFeatures = word_vectorizer.transform(requiredText)

print ("Feature completed .....")


# <a id="ML_Models"> </a>
# ## 6. ML Models

# Training Machine Learning Model for Resume Screening

# In[77]:


# Splitting the data into train, test, printing the shape of each
 
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=2, test_size = 0.2)
print('The shape of the training data', X_train.shape)
print('The shape of the test data',X_test.shape)


# In[81]:


#Now let’s train the model and print the classification report:
#running KNeighborsClassifier with OneVsRest method

model = OneVsRestClassifier(KNeighborsClassifier())
model.fit(X_train, y_train)


# In[82]:


# Predicting the values using the model built with train data and checking the appropriate metrics

prediction = model.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(model.score(X_test, y_test)))


# In[83]:


print("Classification report for classifier %s:\n%s\n" % (model, metrics.classification_report(y_test, prediction)))


# 
# 
# 
# 
# 
# 

# In[88]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))


# evaluate each model in turn
#using k-fold method of cross validation
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring) 
    results.append(cv_results)
    names.append(name) 
    msg = "%s: %f " % (name, cv_results.mean()) 
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[89]:


#Make predictions on validation dataset
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict (X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report (y_test, predictions))


# In[ ]:




