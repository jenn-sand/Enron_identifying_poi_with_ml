
# coding: utf-8

# In[59]:

import warnings
warnings.filterwarnings('ignore')


# In[60]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd
import numpy as np
import plotly.plotly as py
import cufflinks as cf
import plotly.graph_objs as go
from plotly.graph_objs import *

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from time import time
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit

from tester import test_classifier


# In[3]:

### Load the dictionary containing the dataset
data_dict = pickle.load(
    open("../final_project/final_project_dataset.pkl", "r"))


# In[4]:

#Select what features to use:

#all in USD, features ordered like financial pdf:
financial_features = [
    'salary', 'bonus', 'long_term_incentive', 'deferred_income',
    'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees',
    'total_payments', 'exercised_stock_options', 'restricted_stock',
    'restricted_stock_deferred', 'total_stock_value'
]
#(units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)
email_features = [
    'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
    'from_this_person_to_poi', 'shared_receipt_with_poi'
]

#for now, use all.  Let's trim later.
features_list = financial_features + email_features + ['poi']


# In[5]:

#Move data into pandas dataframe
#Cribbed heavily from Miles: https://discussions.udacity.com/t/pickling-pandas-df/174753
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
df.columns.values
# __, cols = df.shape
# print cols


# In[6]:

#The data set is using the string 'NaN' instead of nan, which doesn't play well with describe
df = df.replace('NaN', np.nan)

df.describe()


# In[7]:

persons_of_interest = [k for k in data_dict if data_dict[k]['poi']]

#known issue in this data -- email list is restricted to persons of interest who worked for Enron
print 'persons of interest: {0}'.format(len(persons_of_interest))
poi_names = pd.read_csv("../final_project/poi_names.txt", "\s+")

print(
    'There are {0} persons in the data, {1} or {2:.2f}% are persons of interest.'
).format(len(df), len(poi_names), 100. * len(poi_names) / len(df))


# In[8]:

print 'Where is data missing?  What values have NaN, and what percentage of those is missing data?'
for col in df.columns.values:
    isnan = df[col].isnull().sum()
    print '{0} NaN values: {1} people = {2:.2f}%'.format(col, isnan, 100. *
                                                         isnan / len(df))


# In[9]:

#Let's see how compensation 'balances' horizontally, so we know which stock options have the richest information.
for col in df.columns.values:
    if 'stock' in col:
        print '{0} total is: {1}'.format(col, df[col].sum())
print ''
print 'total_stock_value should be the sum of the other stock options, as it\'s labeled total. Let\'s see if that holds true.'
print ''
print 'The balanced stock data is: {0}, we would expect zero.'.format(
    (df['total_stock_value'].sum() - df['exercised_stock_options'].sum() -
     (df['restricted_stock'].sum() + df['restricted_stock_deferred'].sum())))


# In[10]:

#Who in the data has a line that isn't balancing?
df.loc[df['total_stock_value'].fillna(0) -
       df['exercised_stock_options'].fillna(0) -
       (df['restricted_stock'].fillna(0) +
        df['restricted_stock_deferred'].fillna(0)) != 0]


# In[11]:

#And their names?
print employees[24]
print employees[118]


# In[12]:

#After a visual check, these two people are NOT reflecting what's in the FindLaw PDF of payments to insiders.
print 'Semi-manually adjusting financials for {0}'.format(employees[24])
#24 is BELFER ROBERT, his payments are all shifted to the right one starting at deferred_income.
for col in financial_features[3:-1]:
    col_loc = df.columns.get_loc(col)
    next_col_loc = df.columns.get_loc(financial_features[
        financial_features.index(col) + 1])
    #     print col_loc, df.iloc[24,col_loc], df.iloc[24,next_col_loc]
    df.iloc[24, col_loc] = df.iloc[24, next_col_loc]

df.iloc[24, df.columns.get_loc('total_stock_value')] = np.nan


# In[13]:

print 'Semi-manually adjusting financials for {0}'.format(employees[118])

#Sanjay's values all need to slide one to the right after 'Other', which was missing a - in the pdf.
#Loan Advances is already NaN, so we can use that value to fill other.
for col in reversed(financial_features[6:]):
    col_loc = df.columns.get_loc(col)
    prev_col_loc = df.columns.get_loc(financial_features[
        financial_features.index(col) - 1])
    #     print df.columns[col_loc], df.iloc[118,col_loc], df.iloc[118,prev_col_loc]
    df.iloc[118, col_loc] = df.iloc[118, prev_col_loc]

# print df.iloc[118]


# In[14]:

#Check the stock balancing now that we've adjusted some figures:
for col in df.columns.values:
    if 'stock' in col:
        print '{0} total is: {1}'.format(col, df[col].sum())
print ''
print 'total_stock_value should be the sum of the other stock options, as it\'s labeled total. Let\'s see if that holds true.'
print ''
print 'The balanced stock data is now: {0}, we would expect zero.'.format(
    (df['total_stock_value'].sum() - df['exercised_stock_options'].sum() -
     (df['restricted_stock'].sum() + df['restricted_stock_deferred'].sum())))


# Clean up outliers

# In[15]:

#Let's do the same check for total payments:
print 'The balanced payments data is: {0}, we would expect zero.'.format(
    (df['total_payments'].sum() - df['salary'].sum() - df['bonus'].sum() -
     df['long_term_incentive'].sum() - df['deferred_income'].sum() -
     df['deferral_payments'].sum() - df['loan_advances'].sum() -
     df['other'].sum() - df['expenses'].sum() - df['director_fees'].sum()))


# In[16]:

#Quick visual check for outliers
df[['salary', 'bonus']].iplot(
    kind='scatter',
    mode='markers',
    x='salary',
    y='bonus',
    filename='cufflinks/simple-scatter')


# In[17]:

outlier_salary_bonus = {
    k: v
    for k, v in data_dict.items()
    if v['bonus'] >= 90000000 and v['bonus'] <> 'NaN' and v['salary'] >=
    26000000
}

outlier_salary_bonus


# In[18]:

#get the total line out of _everything_.
data_dict.pop('TOTAL', 0)

total_idx = employees[employees == 'TOTAL'].index[0]
df = df.drop(df.index[total_idx])

employees = employees.drop(employees[employees == 'TOTAL'].index[0])


# In[19]:

#Re-run the visual check for outliers
df[['salary', 'bonus']].iplot(
    kind='scatter',
    mode='markers',
    x='salary',
    y='bonus',
    filename='cufflinks/simple-scatter')


# In[20]:

#possible TO-DO -- imputation -- backfill all financial columns with zeroes when they're NaN


# In[21]:

#Add a new column, total_compensation, to summarize the overall compensation someone's getting from the company.
#this is not scaled! Stock is a smaller proportion of compensation and I do not believe it should be weighed as 'more' of the package.
df['total_compensation'] = df['total_payments'].fillna(0) + df[
    'total_stock_value'].fillna(0)


# In[22]:

#Add the employee names into the pandas dataframe
df = pd.concat([employees, df], axis=1)
df = df.rename(index=str, columns={0: "employee_name"})


# Let's follow the money -- were people who were significantly higher funded more likely to be persons of interest?

# In[23]:

#Fun with Graphs: total_payments+total_stock_value with Employee Names included
# df[['total_payments','total_stock_value', 'employee_name', 'poi']].iplot(kind='scatter', mode='markers', x='total_payments', y='total_stock_value',
#                                                                   text= 'poi', filename='cufflinks/simple-scatter')

trace = go.Scatter(
    x=df.total_payments,
    y=df.total_stock_value,
    mode='markers',
    marker=dict(
        size='16',
        color=map(lambda x: 1 if x else 0, df.poi),
        colorscale='Viridis',
        showscale=True),
    text=df.employee_name, )
# layout = Layout(
#     xaxis = dict(title='total payments'),
#     yaxis = dict(title='total stock value'))
py.iplot([trace], filename='scatter-plot-with-colorscale')


# In[24]:

#What if it's just salary and bonus?

trace = go.Scatter(
    x=df.salary,
    y=df.bonus,
    mode='markers',
    marker=dict(
        size='12',
        color=map(lambda x: 1 if x else 0, df.poi),
        colorscale='Viridis',
        showscale=True),
    text=df.employee_name, )
py.iplot([trace], filename='scatter-plot-with-colorscale')


# In[25]:

#simple/beautiful outlier detection function from https://github.com/joferkington/oost_paper_code/blob/master/utilities.py
#input is a list and an optional z-score, output is a True/False for whether that specific value would be above the z-score.
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


# In[26]:

#Let's mark the people who are extremely well paid
df['total_comp_outlier'] = is_outlier(df.total_compensation)


# In[27]:

#visual check of the z-score method
trace = go.Scatter(
    x=df.total_payments,
    y=df.total_stock_value,
    mode='markers',
    marker=dict(
        size='12',
        color=map(lambda x: 1 if x else 0, is_outlier(df.total_compensation)),
        colorscale='Viridis',
        showscale=True),
    text=df.employee_name, )
py.iplot([trace], filename='scatter-plot-with-colorscale')


# In[28]:

#Quick check of how many persons of interest were and weren't in our exceptionally well-paid category
# df.groupby(['poi', 'total_comp_outlier']).size()
df[['poi', 'total_comp_outlier', 'employee_name']].groupby(
    ['poi', 'total_comp_outlier']).agg(['count'])


# In[29]:

#Because the data was gathered as part of a lawsuit, we can safely assume that any NaN values in the financial fields had no related payments.
for col in financial_features:
    df[col] = df[col].fillna(0)


# In[30]:

#convert the dataframe back into a dictionary
my_dataset = df.to_dict('index')


# In[31]:

# #First, we'll do a basic check to see how much overall compensation could predict whether someone's a poi
# #based on the outlier correlation we saw above, I'm assuming it's low
features_list = ['poi', 'total_payments', 'total_stock_value']


def create_test_split(features_list):
    data = featureFormat(my_dataset, features_list)
    global labels, features
    labels, features = targetFeatureSplit(data)
    global features_train, features_test, labels_train, labels_test
    features_train, features_test, labels_train, labels_test =         train_test_split(features, labels, random_state=42, test_size=0.25)


create_test_split(features_list)


def classify_SVC(f_train, l_train, f_test, l_test):
    clf = svm.SVC(kernel='rbf', C=10000)
    t0 = time()

    # features_train = features_train[:len(features_train)/100] 
    # labels_train = labels_train[:len(labels_train)/100] 

    clf.fit(f_train, l_train)
    print "training time:", round(time() - t0, 3), "s"

    t0 = time()
    pred = clf.predict(f_test)
    print pred
    print "prediction time:", round(time() - t0, 3), "s"

    t0 = time()
    print accuracy_score(l_test, pred)
    print "accuracy time:", round(time() - t0, 3), "s"

    t0 = time()
    print recall_score(l_test, pred)
    print "recall time:", round(time() - t0, 3), "s"    
    
    cm = confusion_matrix(labels_test, pred)
    print(
        'There are {0} True Negatives, {1} False Positives, {2} False Negatives, and {3} True Positives'
    ).format(cm[0][0], cm[0][1], cm[1][0], cm[1][1])


classify_SVC(features_train, labels_train, features_test, labels_test)


# In[32]:

# #TODO: Consider making graph of actual y/n poi on left, predicted y/n poi on right


# Based on total payments and total stock value alone, we can determine the chance of someone being a person of interest with a reasonable accuracy.  While the prediction numbers look good at face value, this will not do as this model is predicting _everyone_ to not be a person of interest.  This result has far too many Type II errors to be a good indicator of who to investigate.

# In[33]:

#How often were persons of interest emailing each other?
trace = go.Scatter(
    x=df.from_poi_to_this_person,
    y=df.from_this_person_to_poi,
    mode='markers',
    marker=dict(
        size='12',
        color=map(lambda x: 1 if x else 0, df.poi),
        colorscale='Viridis',
        showscale=True),
    text=df.employee_name, )
py.iplot([trace], filename='scatter-plot-with-colorscale')


# In[34]:

scaler = MinMaxScaler()
#Assuming if we have no emails to a person in the records, there are none, as this data was obtained through legal discovery.
df['from_poi_to_this_person'] = df['from_poi_to_this_person'].fillna(0)
df['from_this_person_to_poi'] = df['from_this_person_to_poi'].fillna(0)
df['from_poi_scaled'] = df['from_poi_to_this_person']
df['to_poi_scaled'] = df['from_this_person_to_poi']

df[['from_poi_scaled', 'to_poi_scaled']] = scaler.fit_transform(df[
    ['from_poi_to_this_person', 'from_this_person_to_poi']])


# In[35]:

#How often were persons of interest emailing each other?
trace = go.Scatter(
    x=df.from_poi_scaled,
    y=df.to_poi_scaled,
    mode='markers',
    marker=dict(
        size='12',
        color=map(lambda x: 1 if x else 0, df.poi),
        colorscale='Viridis',
        showscale=True),
    text=df.employee_name, )
py.iplot([trace], filename='scatter-plot-with-colorscale')


# In[36]:

#That was interesting to see, but shouldn't be used for testing -- we don't want to scale based on all data points
#then train PCA on some data points.
df = df.drop(['from_poi_scaled'], axis=1)
df = df.drop(['to_poi_scaled'], axis=1)


# In[37]:

#convert the dataframe back into a dictionary
my_dataset = df.to_dict('index')


# In[38]:

features_list = ['poi', 'from_poi_to_this_person', 'from_this_person_to_poi']
create_test_split(features_list)


# In[39]:

features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)


# In[40]:

#create & fit PCA
from sklearn.decomposition import PCA
t0 = time()
pca = PCA(n_components=2).fit(features_train_scaled)
print "done in %0.3fs" % (time() - t0)


# In[41]:

features_train_transformed = pca.transform(features_train_scaled)
features_test_transformed = pca.transform(features_test_scaled)


# In[42]:

print('Predict without scaling or PCA')
classify_SVC(features_train, labels_train, features_test, labels_test)


# In[43]:

print('Predict with scaling only')
classify_SVC(features_train_scaled, labels_train, features_test_scaled,
             labels_test)


# In[44]:

print('Predict with scaling and PCA')
classify_SVC(features_train_transformed, labels_train,
             features_test_transformed, labels_test)


# Again, based on the number of emails going between persons and known persons of interest, we cannot predict whether someone would be a person of interest with enough certainty.  I'd still like more false positives than false negatives, as we'd like to know who else to investigate when someone is suspect.

# Which brings us to my fleshed-out question -- were persons of interest better connected than non-POIs?  
# I'm specifically interested in how often persons of interest emailed the higher-paid staff at Enron vs non-POIs.

# In[45]:

features_list = [
    'poi', 'total_payments', 'total_stock_value', 'from_poi_to_this_person',
    'from_this_person_to_poi'
]
create_test_split(features_list)


# In[46]:

print('Predict using total payments, total stock, from poi, to poi')
classify_SVC(features_train, labels_train, features_test, labels_test)


# That's even worse.  Clearly we cannot use just these values to predict who is a POI.

# In[47]:

#Let's try selecting features based on the financial features and connectivity / how often the person sent poi emails
features_list = ['poi'] + financial_features + [
    'from_poi_to_this_person', 'from_this_person_to_poi'
]
create_test_split(features_list)

kbest = SelectKBest(f_classif, k=10)
selected_features = kbest.fit_transform(features_train, labels_train)


# In[48]:

features_selected = [
    features_list[i + 1] for i in kbest.get_support(indices=True)
]

print 'Features selected by SelectKBest:'
print features_selected


# SelectKBest is suggesting we disregard email frequency to POIs in our predictions.

# In[49]:

print 'Looking at the f score for each to validate that our selected scores had significantly higher scores than non-selected'
print [i for i in kbest.get_support(indices=True)]
kbest.scores_


# In[64]:

create_test_split(features_list)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

# svc = svm.SVC()
pipe = make_pipeline(MinMaxScaler(), PCA(), LinearSVC(), GradientBoostingClassifier())

parameters = {
    'pca__n_components': range(1,16),
    'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
    'linearsvc__C': range(1,100)
}

gs = GridSearchCV(pipe, parameters, cv=cv, scoring='f1')

gs.fit(features, labels)
print("The best parameters are: {0}").format(gs.best_params_)

clf = gs.best_estimator_


# In[70]:

print test_classifier(clf, my_dataset, features_list)


# In[78]:

create_test_split(features_list)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

# svc = svm.SVC()
pipe = make_pipeline(MinMaxScaler(), PCA(), LinearSVC(), GradientBoostingClassifier())

parameters = {
    'pca__n_components': range(1,16),
    'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
    'linearsvc__C': range(1,100)
}

gs = GridSearchCV(pipe, parameters, cv=cv, scoring='recall')

gs.fit(features, labels)
print("The best parameters are: {0}").format(gs.best_params_)

clf = gs.best_estimator_


# In[79]:

print test_classifier(clf, my_dataset, features_list)


# In[80]:

from sklearn.ensemble import AdaBoostClassifier

create_test_split(features_list)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

pipe = make_pipeline(SelectKBest(), AdaBoostClassifier())

parameters = {
    'selectkbest__k': range(1,15),
    'adaboostclassifier__n_estimators': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'adaboostclassifier__learning_rate': np.arange(0.1,1,0.1)
}

gs = GridSearchCV(pipe, parameters, cv=cv, scoring='recall')

gs.fit(features, labels)
print("The best parameters are: {0}").format(gs.best_params_)

clf = gs.best_estimator_


# In[81]:

print test_classifier(clf, my_dataset, features_list)


# In[83]:

#Add scaling
create_test_split(features_list)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

pipe = make_pipeline(MinMaxScaler(), SelectKBest(), AdaBoostClassifier())

parameters = {
    'selectkbest__k': range(1,15),
    'adaboostclassifier__n_estimators': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'adaboostclassifier__learning_rate': np.arange(0.1,1,0.1)
}

gs = GridSearchCV(pipe, parameters, cv=cv, scoring='recall')

gs.fit(features, labels)
print("The best parameters are: {0}").format(gs.best_params_)

clf = gs.best_estimator_


# In[84]:

print test_classifier(clf, my_dataset, features_list)


# In[85]:

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:

#Future possibility, elephant is too large for scope right now: Consider checking for -- is there a group of words where the poi to poi emails [no non-pois in email?] had a higher usage
#than those words in gen pop

