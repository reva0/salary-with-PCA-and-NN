
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model

from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, auc, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
# from sklearn.cross_validation import KFold # old version

from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

cur_dir = os.getcwd()


# # Step 1 - Data Cleaning and Data Preparation

# In[4]:


# Using a small subset of original data
# full data can be found on Kaggle : https://www.kaggle.com/c/job-salary-prediction
Salaries = pd.read_csv("Salary_Train_sample.csv", encoding = "ISO-8859-1") 
Salaries_Train,Salaries_Test = train_test_split(Salaries,test_size=0.33, random_state=13)


# In[5]:


Salaries.head()


# Data prep function - explanations for each step provided below

# In[6]:


# Function to Clean Data - Explanations Provided in Cells Below

def clean_null(Salaries):
    Salaries.dropna(subset=['Title'],inplace=True)
    Salaries['ContractType'].fillna(Salaries['ContractType'].mode()[0],inplace=True)
    Salaries.loc[Salaries['ContractTime'].isnull(), 'ContractTime'] = 'Unknown'
    Salaries.loc[Salaries['Company'].isnull(), 'Company'] = 'Unknown'
    return Salaries


# In[7]:


def featurize(Salaries_Train, Salaries_Test):

    vectorizer = CountVectorizer(analyzer = "word", 
                                 tokenizer = None, 
                                 preprocessor = None, 
                                 stop_words = 'english', 
                                 max_features = 200,
                                 ngram_range = (1,2))\
                                .fit(Salaries_Train['FullDescription']) 
    
    train_words = vectorizer.transform(Salaries_Train['FullDescription'])
    test_words = vectorizer.transform(Salaries_Test['FullDescription'])
    
    title_vectorizer = vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, 
                                                preprocessor = None, 
                                                stop_words = 'english', 
                                                max_features = 200,
                                                ngram_range = (2,3))\
                                                .fit(Salaries_Train['Title'])

    train_title_words = title_vectorizer.transform(Salaries_Train['Title'])
    test_title_words = title_vectorizer.transform(Salaries_Test['Title'])
    
    location_counts = Salaries_Train.LocationNormalized.value_counts()
    value_mask = Salaries_Train.LocationNormalized.isin(location_counts.index[location_counts < 100])
    Salaries_Train.loc[value_mask,'LocationNormalized'] = "Other"
    Salaries_Test.loc[Salaries_Test.LocationNormalized.isin(list(location_counts.index[location_counts > 100])) == False,                      "LocationNormalized"] = "Other"
    
    Salaries_Train = pd.get_dummies(data=Salaries_Train, columns=['LocationNormalized', 'ContractType','Category','ContractTime'])
    Salaries_Test = pd.get_dummies(data=Salaries_Test, columns=['LocationNormalized', 'ContractType','Category','ContractTime'])
    
    
    # In case there are missing columns in Test
    missing_cols = set( Salaries_Train.columns ) - set(Salaries_Test.columns )
    for column in missing_cols:
        Salaries_Test[column] = 0
    Salaries_Test = Salaries_Test[Salaries_Train.columns]
 
    #Combine all features into sparse dataframe
    
    #TRAIN -------------------------------------------------------

    features_train = Salaries_Train.drop(['FullDescription',
                         'Title','Id','LocationRaw','Company',
                         'SalaryRaw','SourceName'], axis=1)
    title_train =  pd.DataFrame(data = train_title_words.toarray(), columns = title_vectorizer.get_feature_names())
    description_train = pd.DataFrame(data = train_words.toarray(), columns = vectorizer.get_feature_names())

    features_train.reset_index(drop=True, inplace=True)
    title_train.reset_index(drop=True, inplace=True)
    description_train.reset_index(drop=True, inplace=True)

    Salaries_Train = pd.concat([features_train,title_train,description_train], axis = 1)
    
    Salaries_Y = Salaries_Train['SalaryNormalized']
    Salaries_X = Salaries_Train.drop(['SalaryNormalized'], axis=1)
    
     #TEST -------------------------------------------------------

    features_test = Salaries_Test.drop(['FullDescription',
                         'Title','Id','LocationRaw','Company',
                         'SalaryRaw','SourceName'], axis=1)
    title_test =  pd.DataFrame(data = test_title_words.toarray(), columns = title_vectorizer.get_feature_names())
    description_test = pd.DataFrame(data = test_words.toarray(), columns = vectorizer.get_feature_names())

    features_test.reset_index(drop=True, inplace=True)
    title_test.reset_index(drop=True, inplace=True)
    description_test.reset_index(drop=True, inplace=True)

    Salaries_Test = pd.concat([features_test,title_test,description_test], axis = 1)
    
    Salaries_Y_Test = Salaries_Test['SalaryNormalized']
    Salaries_X_Test = Salaries_Test.drop(['SalaryNormalized'], axis=1)

    return Salaries_X, Salaries_Y, Salaries_X_Test, Salaries_Y_Test


# In[8]:


Salaries_Train = clean_null(Salaries_Train)
Salaries_Test = clean_null(Salaries_Test)


# In[9]:


Salaries_X, Salaries_Y, Salaries_X_Test, Salaries_Y_Test = featurize(Salaries_Train, Salaries_Test)


# In[10]:


Salaries_X_Test.shape


# ## Logic for Data Prep 

# 1 a) - Cleaning NaN 

# In[20]:


Salaries = pd.read_csv("Salary_Train_sample.csv", encoding = "ISO-8859-1") 


# In[21]:


# See how many null values are in each column
Salaries.isnull().sum(axis=0)


# In[22]:


Salaries[Salaries['Title'].isna()]


# In[23]:


# There are very few cases where the title is not provided, therefore these will be simply removed
Salaries = Salaries.dropna(subset=['Title'])


# In[24]:


# could drop rows (or columns) above a certain number of nulls 
# thresh = Require that many non-NA values.
Salaries.dropna(axis=0, how='any', thresh=9, subset=None, inplace=True)


# In[25]:


# There are a lot of null ContractType positions
Salaries['ContractType'].unique()


# In[26]:


# Can see that most Contract Types are full-time
Salaries['ContractType'].value_counts(normalize=True)


# In[27]:


# There are several methods to determine how to handle NaN values 
#- including dropping them, filling with mode or mean, or replacing with an "Other" value
# start with data exploration - can check if there a pattern in where the NaN values appear or if the rows with NaN are randomly distributed

# For Example - is the a pattern in job categories where NaN contract type appears or is it similar to the full data set?

a = Salaries['Category'].value_counts(normalize=True)
b = Salaries[Salaries['ContractType'].isnull()]['Category'].value_counts(normalize=True)
print (pd.DataFrame({'Alldata': a, 'NaN ContractType':b}))


# In[28]:


# fill contracttype with mode
Salaries['ContractType'].fillna(Salaries['ContractType'].mode()[0],inplace=True)


# In[29]:


#Contract time


# In[30]:


Salaries['ContractTime'].value_counts(normalize=True)


# There is a higher proportion of jobs in healthcare, hospitality and catering and 'other' where the contract time is NaN. Although this may require some futher exploration, we will now fill them with a filler 'Unknown' variable.

# In[11]:


Salaries.loc[Salaries['ContractTime'].isnull(), 'ContractTime'] = 'Unknown'


# In[33]:


# Next, missing Company Names are dealt with 
Salaries[Salaries['Company'].isnull()]['FullDescription'][0:10] 


# Some data exploration shows that missing companies names are caused by a variety of factors - Some posings may have the company name in the full description, while others are third party recruiters, recruiting for a client. NaNs will be filled with a 'Unknown' value. This value might capture recruitment by a third party. This could be further refined - for example filling with 'Third Party' IF Description contains 'Client'.

# In[34]:


Salaries.loc[Salaries['Company'].isnull(), 'Company'] = 'Unknown'


# 1 b) Cleaning Text Data

# The dataset contains a significant amount of uncleaned and unstructured text in the 'Title' and 'Full Description' columns.
# A simple approach would be to clean up the text, remove stop words, and do one-hot encoding using countvectorizer. 

# In[35]:


vectorizer = CountVectorizer(analyzer = "word", 
                             tokenizer = None, 
                             preprocessor = None, 
                             stop_words = 'english', 
                             max_features = 200,
                             ngram_range = (1,2)) 


# In[36]:


Words = vectorizer.fit_transform(Salaries['FullDescription'])


# In[37]:


print(vectorizer.get_feature_names()[0:25])


# The most common words seem to be relevant to salary information, although more stopwords could possibly be added to list. A better approach may be to set a high # of max_features and run a linear regression for feature selection. 

# Several of the most frequent words such 'car,'linux','worker','digital' look like they belong to a sequence of words, so the vectorizer could benefit from using n - grams.

# In[38]:


title_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, 
                                                preprocessor = None, 
                                                stop_words = 'english', 
                                                max_features = 200,
                                                ngram_range = (2,3))

Title_Words = title_vectorizer.fit_transform(Salaries['Title'])
print(title_vectorizer.get_feature_names()[0:10])


# The features from n-grams seem more appropriate.

# In[39]:


#Salaries.Company.value_counts()


# In[40]:


Salaries.LocationNormalized.nunique()


# Because there are a lot of unique company names, it may be excessive to get dummies for each one, and there may not be enough examples of each company name to be significant in model training. An option would be to get dummy variables for the top 100 most popular companies or companies above a threshold count.

# There is also a significant amount of unique Company locations (2000+). To allow the models to run in reasonable time and eliminate locations that only appear once, we will only consider locations that appear above a certain frequency. Everything else we can replace with "Other".

# In[41]:


location_counts = Salaries.LocationNormalized.value_counts()
value_mask = Salaries.LocationNormalized.isin(location_counts.index[location_counts < 100])
Salaries.loc[value_mask,'LocationNormalized'] = "Other"
Salaries.LocationNormalized.nunique()


# In[42]:


# Since the other features are categorical and not ordinal they can be encoded using dummy variables.
Salaries = pd.get_dummies(data=Salaries, columns=['LocationNormalized', 'ContractType','Category','ContractTime'])


# 1c) Combining Features

# In[43]:


# Combine all feautures - Option One, Combine Into Matrix
features = Salaries.drop(['FullDescription',
                         'Title','Id','LocationRaw','Company',
                         'SalaryRaw','SalaryNormalized','SourceName'], axis=1).values

Title_Features = Title_Words.toarray()
Description_Features = Words.toarray()

all_features = np.hstack([features, Title_Features, Description_Features])


# In[44]:


#Combine all features - Option Two, Combine Into Sparse Dataframe

features = Salaries.drop(['FullDescription',
                         'Title','Id','LocationRaw','Company',
                         'SalaryRaw','SalaryNormalized','SourceName'], axis=1)
title =  pd.DataFrame(data = Title_Words.toarray(), columns = title_vectorizer.get_feature_names())
description = pd.DataFrame(data = Words.toarray(), columns = vectorizer.get_feature_names())

features.reset_index(drop=True, inplace=True)
title.reset_index(drop=True, inplace=True)
description.reset_index(drop=True, inplace=True)

Salaries_Feature = pd.concat([features,title,description], axis = 1)


# # 2 Exploratory Data Analysis

# An example of visualization

# In[45]:


Salaries_Plot = pd.read_csv("Salary_Train_sample.csv", encoding = "ISO-8859-1")


# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.hist(Salaries_Plot['SalaryNormalized'], bins=10, edgecolor='black')
plt.title("Salary Histogram 10 bins")
plt.xlabel("Salary Bin")
plt.ylabel("Number of Samples")
plt.show()


# # 3 Model Implementation

# In[55]:


#set up cross validation

def run_kfold(model):
    
    X = Salaries_X
    Y = Salaries_Y
    
    kf = KFold(n_splits=10) #n_splits previously n_folds
    
    outcomes = []
    fold = 0
    
    for train_index, test_index in kf.split(X):
        fold += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        Y_train, Y_test = Y.values[train_index], Y.values[test_index]
        
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        
        accuracy = r2_score(Y_test, predictions) # can try mean absolute error instead
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))   
        
    mean_outcome = np.mean(outcomes)
    std_outcome=np.std(outcomes)
    print("Mean r2: {0}".format(mean_outcome)) 
    print("Standard Deviation: {0}".format(std_outcome)) 


# ## 3 A Linear Regression

# In[56]:


model_lr = linear_model.LinearRegression()
run_kfold (model_lr)

predictions = model_lr.predict(Salaries_X_Test)
Test_Score = r2_score(Salaries_Y_Test, predictions)

print ("------------------\n Test Score: " + str(Test_Score))


# In[57]:


mean_absolute_error(Salaries_Y_Test,predictions)


# Look at features and coefficients

# In[58]:


print(sorted(list(zip(model_lr.coef_, Salaries_X)))[0:10])


# In[45]:


print(sorted(list(zip(model_lr.coef_, Salaries_X)),reverse=True)[0:10])


# Coefficients seem very high, try regulatization and feature elimination. 

# # Feature Selection

# ### Recursive Feature Elimination

# In[59]:


rfe = RFE(model_lr)
fit = rfe.fit(Salaries_X[0:10000], Salaries_Y[0:10000]) #Sampling because of slow run time


# In[60]:


sorted(list(zip(fit.ranking_,Salaries_X))[0:10])


# ### Lasso Regularization

# In[61]:


reg = linear_model.Lasso(alpha = 0.5,max_iter=10000)
reg.fit(Salaries_X, Salaries_Y)
reg.score(Salaries_X, Salaries_Y)


# Example of Grid Search

# In[74]:


from sklearn.metrics import make_scorer, r2_score, confusion_matrix

reg_gridsearch = linear_model.Lasso(random_state=42)
#Parameters to test
parameters = {'alpha':[0.5,1,3], # Constant that multiplies the L1 term. Defaults to 1.0.
             'normalize':[True,False]} #

# Compare parameters by score of model 
acc_scorer_lm = make_scorer(r2_score)

# Run the grid search
grid_obj_lm = GridSearchCV(reg_gridsearch, parameters, scoring=acc_scorer_lm)
grid_obj_lm = grid_obj_lm.fit(Salaries_X, Salaries_Y)

reg_gridsearch = grid_obj_lm.best_estimator_  #Select best parameter combination


# In[78]:


reg_gridsearch # print out the optimal params so grid search does not need to be rerun


# In[80]:


reg_gridsearch.fit(Salaries_X, Salaries_Y)
reg_gridsearch.score(Salaries_X, Salaries_Y)


# Print optimal parameters 

# In[81]:


print('alpha (Constant that multiplies the L1 term):',grid_obj_lm.best_estimator_.alpha) 
print('normalize:',grid_obj_lm.best_estimator_.normalize)


# In[96]:


predictions_lasso=reg_gridsearch.predict(Salaries_X_Test)


# In[97]:


mean_absolute_error(Salaries_Y_Test,predictions_lasso)


# In[64]:


run_kfold (reg)


# In[65]:


# can see more consistent results with regularization than regular linear regression


# In[82]:


sorted(list(zip(reg.coef_, Salaries_X)),reverse=True)[0:10]
#Can see that regularized coefficeints are more reasonable 


# In[ ]:


rfe_lasso = RFE(reg)
fit_lasso = rfe_lasso.fit(Salaries_X[0:10000], Salaries_Y[0:10000]) #Sampling because of slow run time


# In[84]:


sorted(list(zip(model_lr.coef_, Salaries_X))[0:10])


# ### PCA - Example

# In[236]:


# does not show any improvements - higher score with more components


# In[85]:


from sklearn.decomposition import PCA
pca = PCA(n_components=150)
pca.fit(Salaries_X)
PCA_X = pca.transform(Salaries_X)
PCA_X_Test = pca.transform(Salaries_X_Test)
model_lr_pca = linear_model.LinearRegression()
model_lr_pca.fit(PCA_X,Salaries_Y)

predictions_pca_train = model_lr_pca.predict(PCA_X)
predictions_pca_test = model_lr_pca.predict(PCA_X_Test)

print ("Train r2 Score")

print (r2_score(Salaries_Y, predictions_pca_train))  

print ("Test r2 Score")

print (r2_score(Salaries_Y_Test,predictions_pca_test))


# In[87]:


#------------------------------------------------------------------------------
#Extra for Assignment 2
#------------------------------------------------------------------------------
'''
# thresh = require that many non-NA values

Salaries = Salaries.dropna(thresh=len(Salaries) - 10000, axis=1) #drop columns with too many NAs

# Drop or explore columns which have too many unique values are will be difficult to encode

Salaries_X.drop([col for col, val in Salaries_X.nunique().iteritems() if val > 500], axis=1, inplace = True) '''


# # Neural Network

# In[88]:


clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,8,4), random_state=1, max_iter=1000)
clf.fit(Salaries_X, Salaries_Y)


# In[92]:


clf.score(Salaries_X, Salaries_Y)


# In[91]:


clf.score(Salaries_X_Test, Salaries_Y_Test) #overfitting..


# In[221]:


run_kfold(clf)

