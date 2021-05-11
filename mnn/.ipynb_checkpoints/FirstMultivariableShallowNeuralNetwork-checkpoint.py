# Data Prep

# Loading data and libraries
import numpy as np
import pandas as pd
import datetime
# import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("ted_main.csv")
print(df.columns)

df = df[['name', 'title', 'description', 'main_speaker',
         'speaker_occupation', 'num_speaker', 'duration',
         'event', 'film_date', 'published_date', 'comments',
         'tags', 'languages', 'ratings', 'related_talks', 'url', 'views']]

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

# Modeling

# Once the data is split into train and test. We could further divide the training data into train and validation sets. So our split of the data will be 60:20:20. What we can then do is while training the model using the 60% of the data also indicate in val_per = .2 so that cross validatio is also performed during the training phase. This three phases allow us to first train all the models, then on the validation set compare all the models and choosing the best performing one, and finally using the test set to find the accuracy of the best model in production. Now one last word. If the data set is very very large people at Dessa implementing deep learning for corporate clients have told me that cross validation can be left out, because the data is so large. My reasoning, if this is indeed the case, is because there is only redundant information learned by performing cross validation. The data is so large the model learns all the information in the data with regular training. Cross validation therefore, I surmise, can also be thought as a data augmentation method, nor a data trend augmentation, but data trend augmentation. Data trend augmentation is creating and replicating patterns in the data that already exist and making them stronger or more detectable.


# Again look at the data columns and the type of variables they are.

print(ted.columns)
print(len(ted.columns))

# We will create at the end a cleaned data set called ted_clean.
ted1 = ted.drop(['index', 'url', 'main_speaker', 'speaker_occupation', 'occ', 'film_date',
                 'event', 'description', 'ratings', 'name', 'title', 'published_date' , 'related_talks', 'tags'], axis =1)
print(ted1.columns)
print(len(ted1.columns))


## We will now create dummy variables.
#ted2 = ted1.drop([''])
ted2 = pd.get_dummies(ted1)


print(ted2.columns)
print(len(ted2.columns)) # WOW, Too many variable were created.

# First the proper non-trivial, but still basic model with proper variables and a deep neural network. Statistically speaking we will now do multivariable analysis, instead of univariable analysis, though still using shallow networks.

# Seperate the data first into training and testing.
X_train, X_test, y_train, y_test = train_test_split(ted2.drop('views', axis=1), ted['views'], test_size=0.2,
                                                    random_state=1)

for set in [X_train, X_test, y_train, y_test]:
    print(len(set))

print(X_train.shape)

print(X_train.shape)
X_train.shape[1]


# This is getting very long. Need to covert this into a script.
def model_results(model_num,

                  #X_train = X_train,
                  #y_train = y_train,
                  #X_test = X_test,
                  #y_test = y_test,

                  optimizer='adam',
                  loss_fn='mse',
                  activation_fn='relu',
                  output_activation_fn='relu',

                  epochs=100,
                  batch_size=32,
                  validation_split=0.2,

                  neurons_in_inputlayer=50,
                  num_hiddenlayer=1,
                  neurons_in_hiddenlayer=40,
                  num_dropout_layers=0,

                  verbose=0):
    # The Model

    input_dimensions = X_train.shape[1]

    model = Sequential()
    # Input Layer
    model.add(Dense(neurons_in_inputlayer, input_dim=input_dimensions, activation=activation_fn))
    # Dropout Layer
    #model.add(Dropout(rate = 0.1,seed=100))
    # Hidden Layers
    model.add(Dense(neurons_in_hiddenlayer, activation=activation_fn))
    # Output Layer
    model.add(Dense(1, activation=output_activation_fn))

    # Compiling the model
    model.compile(loss=loss_fn, optimizer=optimizer,
                  #metrics=['mse', 'mae', 'mape', CosineSimilarity(), RootMeanSquaredError() , MeanSquaredLogarithmicError() ])
                  metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()
                      , CosineSimilarity(), RootMeanSquaredError(), MeanSquaredLogarithmicError()])

    # Train the model and make predictions
    model_fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                          validation_split=validation_split)
    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=verbose)

    # Make predictions from the trained model
    #predictions = model.predict(X_test)

    # Store results
    dict = {'model_num': [model_num],
            'optimizer': [optimizer],
            'loss_fn': [loss_fn],
            'activation_fn': [activation_fn],
            'output_activation_fn': [output_activation_fn],
            'epochs': [epochs],
            'batch_size': [batch_size],
            'validation_split': [validation_split],
            'input_dimensions': [input_dimensions],
            'neurons_in_inputlayer': [neurons_in_inputlayer],
            'num_hiddenlayer': [num_hiddenlayer],
            'neurons_in_hiddenlayer': [neurons_in_hiddenlayer],
            'num_dropout_layers': [num_dropout_layers],
            'loss': [score[0]],
            'mse_test': [score[1]],
            'mae_test': [score[2]],
            'mape_test': [score[3]],
            'cosine_similarity_test': [score[4]],
            'rmse_test': [score[5]],
            'msle_test': [score[6]]
            }

    # "Plot Loss"
    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # Plot metrics
    for metric in ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error',
                   'mean_absolute_percentage_error', 'cosine_similarity', 'mean_squared_logarithmic_error']:
        plt.plot(model_fit.history[metric])
        plt.title('Model Metric: ' + metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        #plt.plot(model_fit.history['mean_squared_error'])
        #plt.plot(model_fit.history['mean_absolute_error'])
        #plt.plot(model_fit.history['mean_absolute_percentage_error'])
        #pyplot.plot(model_fit.history['cosine_proximity'])
        plt.show()

    results_df = pd.DataFrame(dict)
    #print ("After Training:", list(zip(model.metrics_names, score)))
    return (results_df)


model_0 = model_results(0)
model_0

print(804724.75/2035016.625)
print(100*(2035016.625 - 804724.75) / 2035016.625)
print((804724.75/2035016.625)**-1)

ax = sns.barplot(x=["univariable neural network", "multivariable neural network"], y=[2035016.625,804724.75])


# We can see there is a clear improvement using the multivariable model over the univariable model. A 60.46% improvement over the univariable model.























