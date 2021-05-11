# Loading data and libraries
import numpy as np
import pandas as pd
import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("ted_main.csv")
print(df.columns)



df = df[['name', 'title', 'description', 'main_speaker',
         'speaker_occupation', 'num_speaker', 'duration',
         'event', 'film_date', 'published_date', 'comments',
         'tags', 'languages', 'ratings', 'related_talks', 'url', 'views']]



import datetime
df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))


month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

print(df[['event']])

df['month'] = df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])

month_df = pd.DataFrame(df['month'].value_counts()).reset_index()
month_df.columns = ['month', 'talks']


ted = pd.read_csv('ted_main.csv')

# Categorize events into TED and TEDx; exclude those that are non-TED events
ted = ted[ted['event'].str[0:3]=='TED'].reset_index()


ted.loc[:,'event_cat'] = ted['event'].apply(lambda x: 'TEDx' if x[0:4]=='TEDx' else 'TED')
print ("No. of talks remain: ", len(ted))


# Here, we change the Unix timstamp to human readable date format.
# Then we extract month and day of week from film date and published date.


ted['film_date'] = ted['film_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
ted['published_date'] = ted['published_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)))
ted['film_month'] = ted['film_date'].apply(lambda x: x.month)
ted['pub_month'] = ted['published_date'].apply(lambda x: x.month)
ted['film_weekday'] = ted['film_date'].apply(lambda x: x.weekday()) # Monday: 0, Sunday: 6
ted['pub_weekday'] = ted['published_date'].apply(lambda x: x.weekday())
ted[['film_date','published_date']].head()


# TED users can give ratings to each talk. There are 14 possible ratings and they will be categorized as positive,
# negative and neutral:
#
# Positive: 'Beautiful', 'Courageous', 'Fascinating', 'Funny', 'Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping',
# 'Persuasive' Negative: 'Confusing', 'Longwinded', 'Obnoxious', 'Unconvincing' Neutral: 'OK'
#
# Here, we define a "popular" TED talk by its ratio of positive to negative ratings
# (which we call it "popularity ratio" here). If the popularity ratio is above 5, it is defined as "Popular",
# otherwise it is "Not Popular". Transformation is made to avoid "divided by zero" error. The following code is adopted
# from this kernel to convert 'ratings' column (a JSON object) into columns of each rating


ted['ratings']=ted['ratings'].str.replace("'",'"')
ted=ted.merge(ted.ratings.apply(lambda x: pd.Series(pd.read_json(x)['count'].values,index=pd.read_json(x)['name'])),
            left_index=True, right_index=True)


Positive = ['Beautiful', 'Courageous', 'Fascinating', 'Funny', 'Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', 'Persuasive']
Negative = ['Confusing', 'Longwinded', 'Obnoxious', 'Unconvincing']
ted['positive']=ted.loc[:,Positive].sum(axis=1)+1
ted['negative']=ted.loc[:,Negative].sum(axis=1)+1
ted['pop_ratio']=ted['positive']/ted['negative']
ted.loc[:,'Popular'] = ted['pop_ratio'].apply (lambda x: 1 if x >5 else 0)

print ("No. of Not Popular talks: ", len(ted[ted['Popular']==0]))
# print ("Ratio of Popular talks: {:.4f}".format(len(ted[ted['Popular']==1])/ float(len(ted))))
overall_mean_popular = np.mean(ted.Popular)
print ("Ratio of Popular talks: {:.4f}".format(overall_mean_popular))


nums = ['comments', 'duration', 'languages', 'num_speaker', 'views']
sns.pairplot(ted, vars=nums, hue='Popular', hue_order = [1,0], diag_kind='kde', height=3);




ratings = ['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Longwinded', 'Confusing', 'Informative', 'Fascinating', 'Unconvincing',
           'Persuasive', 'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring', 'Popular']
plt.figure(figsize=(10,8))
sns.heatmap(ted[ratings].corr(), annot=True, cmap='RdBu');


# Then we do count vectorizer on 'speaker_occupation'. Before that, some data cleaning is needed.


ted.loc[:,'occ'] = ted.speaker_occupation.copy()
ted.occ = ted.occ.fillna('Unknown')
ted.occ = ted.occ.str.replace('singer/songwriter', 'singer, songwriter')
ted.occ = ted.occ.str.replace('singer-songwriter', 'singer, songwriter')
count_vector2 = CountVectorizer(stop_words='english', min_df=20/len(ted))
occ_array = count_vector2.fit_transform(ted.occ).toarray()
occ_matrix = pd.DataFrame(occ_array, columns = count_vector2.get_feature_names())
all_occ = occ_matrix.columns
occ_matrix = pd.concat([occ_matrix, ted.Popular], axis=1)
by_occ = dict()
for col in all_occ:
    by_occ[col]=occ_matrix.groupby(col)['Popular'].mean()[1] - overall_mean_popular
occ_rank = pd.DataFrame.from_dict(by_occ, orient='index')
occ_rank.columns = ['pop_rate_diff']

plt.figure(figsize=(16,7))
plt.subplot(121)
bar_2 = occ_rank.sort_values(by='pop_rate_diff', ascending=False)[:10]
sns.barplot(x=bar_2.pop_rate_diff, y=bar_2.index, color='blue')
plt.title('10 Most Popular Occupation Keywords', fontsize=14)
plt.xlabel('Ratio of Popular Talk (Net of Mean)')
plt.yticks(fontsize=12)
plt.subplot(122)
bar_1 = occ_rank.sort_values(by='pop_rate_diff')[:10]
sns.barplot(x=bar_1.pop_rate_diff, y=bar_1.index, color='red')
plt.title('10 Most Unpopular Occupation Keywords', fontsize=14)
plt.xlabel('Ratio of Popular Talk (Net of Mean)')
plt.yticks(fontsize=12)
plt.show()

print("Number of non-Ted talks that we removed were:", len(df) - len(ted))

# Structure of the data
from scipy.stats import pearsonr

corr, _ = pearsonr(df['comments'], df['views']); corr
# As can be seen, not much of a correlation between the two.

# if we include all the talks will the correlation between comments and views increase or decrease?

corr, _ = pearsonr(ted['comments'], ted['views']); corr # it decreases!



## Building a neural network to predict views

# Modeling univariable Deep Learning Neural Network

# TRAINING AND TEST SETS.
df_comments = df['comments']; df_views = df['views']
print(df_comments)
print(df_views)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_comments, df_views, test_size=0.2, random_state=1)

for set in [X_train, X_test, y_train, y_test]:
    print( len(set))

#pip install --upgrade keras
#pip install --upgrade tensorflow



#Import required packages

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredLogarithmicError, CosineSimilarity
from keras import metrics

# check if tensorflow works
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
print(hello)


### Simple neural network: # Model 1: comments as input and views as output.

# Specify the modelmodel = Sequential()
model_1_neuron = Sequential()
model_1_neuron.add(Dense(1, input_dim = 1, activation = "relu"))
model_1_neuron.compile(loss='mse', optimizer='adam',
          #metrics = ["mse", "RMSE", "mape"])
            metrics = [ MeanSquaredError(), RootMeanSquaredError(),MeanAbsoluteError(), MeanAbsolutePercentageError(), MeanSquaredLogarithmicError()])


# Fit the model, or in other words, train the model.

#Train the model and make predictions
history = model_1_neuron.fit(X_train, y_train, epochs=100 , batch_size=32, verbose = 0, validation_split=0.2)
score = model_1_neuron.evaluate(X_test, y_test, verbose = 0)

#Make predictions from the trained model
predictions = model_1_neuron.predict(X_test)

#print performance and loss
print("Performance on Test set:", zip(model_1_neuron.metrics_names, score))

(np.mean((np.array(y_test) - predictions)**2))**.5 # we got 2545996.0559998746 04 April 2021

# "Plot Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()



# "Plot Loss"
#plt.plot(history.history['MeanSquaredError'])
#plt.plot(history.history['val_loss'])
#plt.title('model Metrics')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper right')
#plt.show()

# plot metrics
plt.plot(history.history['mean_squared_error'])
#plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['mean_absolute_percentage_error'])
#pyplot.plot(history.history['cosine_proximity'])
plt.show()



# plot metrics
plt.plot(history.history['mean_absolute_error'])
plt.show()


# plot metrics
plt.plot(history.history['mean_squared_logarithmic_error'])
plt.show()


# plot metrics
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['mean_squared_logarithmic_error'])
plt.show()



# Model 2

# Specify the modelmodel = Sequential()
model_5_neuron = Sequential()
model_5_neuron.add(Dense(5, input_dim = 1, activation = "relu")) #Input layer, Why is number of neurons and input_dim different things
model_5_neuron.add(Dense(3,activation = "relu")) # Hidden layer
model_5_neuron.add(Dense(1,activation = "relu")) # Output layer
model_5_neuron.compile(loss='mse', optimizer='adam',
metrics=['mse'])

# Fit the model, or in other words, train the model.

#Train the model and make predictions
model_5_neuron.fit(X_train, y_train, epochs=100 ,batch_size=32)
#Make predictions from the trained model
predictions = model_5_neuron.predict(X_test)

(np.mean((np.array(y_test) - predictions)**2))**.5



# Model 3

# Specify the modelmodel = Sequential()
model_1_neuron = Sequential()
model_1_neuron.add(Dense(1, input_dim = 1, activation = "relu"))
#model_1_neuron.add(Dropout(rate = 0.1,seed=100))
model_1_neuron.add(Dense(1,activation = "relu"))
model_1_neuron.compile(loss='mse', optimizer='adam',
metrics=['mse'])

# Fit the model, or in other words, train the model.

#Train the model and make predictions
model_1_neuron.fit(X_train, y_train, epochs=10, batch_size=32)
#Make predictions from the trained model
predictions = model_1_neuron.predict(X_test)

(np.mean((np.array(y_test) - predictions)**2))**.5


# Model 4

model = Sequential()
model.add(Dense(5, input_dim = 1, activation = "relu"))
model.add(Dropout(rate = 0.1,seed=100))
model.add(Dense(1,activation = "relu"))
model.compile(loss='mse', optimizer='adam',
metrics=['mse'])

#Train the model and make predictions
model.fit(X_train, y_train, epochs=100, batch_size=100)
#Make predictions from the trained model
predictions = model.predict(X_test)

(np.mean((np.array(y_test) - predictions)**2))**.5


# Write a function that gives the NN results

def train_given_optimiser(optimiser):
    model = Sequential()
    model.add(Dense(2, input_dim=1))
    model.add(Activation(activation='relu'))

    model.compile(optimizer=optimiser, loss='msle', metrics=['mse', 'mape', RootMeanSquaredError(), 'msle'])

    score = model.evaluate(X_train, y_train, verbose=2)

    print("Optimiser: ", optimiser)
    print("Before Training:", list(zip(model.metrics_names, score)))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)
    print("After Training:", list(zip(model.metrics_names, score)))
    print(score)
    print(model.metrics_names)
    print(" \n ")


# Running function for different optimizers
train_given_optimiser("sgd")
train_given_optimiser("rmsprop")
train_given_optimiser("adagrad")
train_given_optimiser("adadelta")
train_given_optimiser("adam")
train_given_optimiser("adamax")
train_given_optimiser("nadam")


