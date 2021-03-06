

# Data Prep


# Loading data and libraries
import numpy as np
import pandas as pd
import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("ted_main.csv")
df.columns



df = df[['name', 'title', 'description', 'main_speaker',
         'speaker_occupation', 'num_speaker', 'duration',
         'event', 'film_date', 'published_date', 'comments',
         'tags', 'languages', 'ratings', 'related_talks', 'url', 'views']]



import datetime
df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))


month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df[['event']]

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



###############################################################################


# Modeling

def model_results(model_num,
                  optimizer='adam',
                  loss_fn='mse',
                  activation_fn='relu',
                  output_activation_fn='relu',

                  epochs=100,
                  batch_size=32,

                  neurons_in_inputlayer=5,
                  num_hiddenlayer=1,
                  neurons_in_hiddenlayer=2,
                  num_dropout_layers=0,

                  verbose=0):
    # The Model

    model = Sequential()
    # Input Layer
    model.add(Dense(neurons_in_inputlayer, input_dim=1, activation=activation_fn))
    # Dropout Layer
    # model.add(Dropout(rate = 0.1,seed=100))
    # Hidden Layers
    model.add(Dense(neurons_in_hiddenlayer, activation=activation_fn))
    # Output Layer
    model.add(Dense(1, activation=output_activation_fn))

    # Compiling the model
    model.compile(loss=loss_fn, optimizer=optimizer,
                  metrics=['mse', 'mae', 'mape', CosineSimilarity(), RootMeanSquaredError(),
                           MeanSquaredLogarithmicError()])
    # Train the model and make predictions
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=verbose)

    # Make predictions from the trained model
    # predictions = model.predict(X_test)

    # Store results
    dict = {'model_num': [model_num],
            'optimizer': [optimizer],
            'loss_fn': [loss_fn],
            'activation_fn': [activation_fn],
            'output_activation_fn': [output_activation_fn],
            'epochs': [epochs],
            'batch_size': [batch_size],
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

    results_df = pd.DataFrame(dict)
    # print ("After Training:", list(zip(model.metrics_names, score)))
    return (results_df)


model_results(0)



dict = {'model_num':[],
        'optimizer':[ ],
        'loss_fn':[],
        'activation_fn':[],
        'output_activation_fn':[],
        'epochs':[],
        'batch_size':[],
        'neurons_in_inputlayer':[],
        'num_hiddenlayer':[],
        'neurons_in_hiddenlayer':[],
        'num_dropout_layers':[],
        'loss':[],
        'mse_test':[],
        'mae_test':[],
        'mape_test':[],
        'cosine_similarity_test':[],
        'rmse_test':[],
        'msle_test':[]
}

results_df = pd.DataFrame(dict)
results_df

# HyperParameter Search Options.

optimizer_list = ["nadam", "adam"]
loss_fn_list = ["mse", "mae", "msle"]
activation_fn_list = ["relu", "selu"]
epochs_list = [50, 100]
batch_size_list = [1, 10, 32, 100]
num_hiddenlayer_list = [1]
neurons_in_hiddenlayer_list = [1, 5, 10]
neurons_in_inputlayer_list = [1, 5, 10]
num_dropout_layers_list = [0]

number_parameters = len(optimizer_list) * len(loss_fn_list) * len(activation_fn_list) * len(epochs_list) * len(batch_size_list) * len(neurons_in_inputlayer_list) * len(neurons_in_hiddenlayer_list) * len(num_dropout_layers_list)
print("number_parameters :", number_parameters)

print(2*3*2*2*4*3*3)

from datetime import datetime

start = datetime.now()

number_parameters = len(optimizer_list) * len(loss_fn_list) * len(activation_fn_list) * len(epochs_list) * len(
    batch_size_list) * len(neurons_in_inputlayer_list) * len(neurons_in_hiddenlayer_list) * len(num_dropout_layers_list)
print("number_parameters :", number_parameters)
print(number_parameters)

i = 0
for opt in optimizer_list:
    for loss in loss_fn_list:
        for activation in activation_fn_list:
            for epochs in epochs_list:
                for batch in batch_size_list:
                    for neurons_in_inputlayer in neurons_in_inputlayer_list:
                        for neurons_in_hiddenlayer in neurons_in_hiddenlayer_list:
                            results = model_results(model_num=i,
                                                    optimizer=opt,
                                                    loss_fn=loss,
                                                    activation_fn=activation,
                                                    output_activation_fn='relu',
                                                    epochs=epochs,
                                                    batch_size=batch,
                                                    neurons_in_inputlayer=neurons_in_inputlayer,
                                                    num_hiddenlayer=1,
                                                    neurons_in_hiddenlayer=neurons_in_hiddenlayer,
                                                    num_dropout_layers=0,
                                                    verbose=0
                                                    )

                            results_df = results_df.append(results, ignore_index=True)
                            print('i:', i)
                            print("Percent Complete", 100 * i / number_parameters, '%')
                            print(' \n ')
                            print(results)
                            print(' \n ')
                            i += 1

stop = datetime.now()

print('Time: ', stop - start)


# Add information about the data structure to the model results.
results_df['corr'] = corr


runtime = stop - start
print('Time: ', runtime)

results_df['total_runtime'] = runtime

#results_df.to_csv('Experiment4_results_views_vs_comments.csv')

results_df.shape
results_df.describe()


results_df = pd.read_csv("Experimental_results/Experiment4_results_views_vs_comments.csv")
results_df.shape

results_df.columns

results_df_metrics = results_df.loc[:, results_df.columns.str.contains('test')]
print(results_df_metrics.shape)
print(results_df_metrics.columns)
type(results_df_metrics.columns)


results_df_metrics.max()


maxValueIndex = results_df_metrics.idxmax()
maxValueIndex



print(results_df_metrics.min())

print()
minValueIndex = results_df_metrics.idxmin()
print(minValueIndex)

print(minValueIndex.values)
results_df.iloc[minValueIndex.values]


# Convert the data from wide format to long format. use pd.melt()

results_df_long0 = pd.melt(results_df, id_vars='model_num', value_vars=results_df_metrics.columns)
results_df_long0

#results_df_long = pd.wide_to_long(results_df, i = "model_num", j="metric", stubnames= results_df_metrics.columns)
#results_df_long


# Summarize the results as boxplots

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='variable', y='value',
            data= results_df_long0[results_df_long0.variable!= 'mse_test'],
            palette="muted", ax = ax)
plt.show()

for variable in results_df_metrics:
    fig, ax = plt.subplots(nrows=1, ncols=1)  # ,figsize=(15, 8))
    sns.boxplot(x='variable', y='value',
                data=results_df_long0[results_df_long0.variable == variable],
                palette="muted", ax=ax)
    plt.show()


# histogram using sns distribution

#sns.histplot(results_df.rmse_test)
#sns.displot(results_df.rmse_test)

#fig, ax = plt.subplots(nrows=3, ncols=2)
for metric in results_df_metrics.columns:

    #print(metric)
    sns.histplot(results_df[metric])
    plt.show()


# This code is amazingly beautiful and I give full credit from where I copied it: Stackoverflow: https://stackoverflow.com/questions/34439/finding-what-methods-a-python-object-has

import pandas as pd df = pd.DataFrame([[10, 20, 30], [100, 200, 300]], columns=['foo', 'bar', 'baz']) def get_methods(object, spacing=20): methodList = [] for method_name in dir(object): try: if callable(getattr(object, method_name)): methodList.append(str(method_name)) except: methodList.append(str(method_name)) processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s) for method in methodList: try: print(str(method.ljust(spacing)) + ' ' + processFunc(str(getattr(object, method).doc)[0:90])) except: print(method.ljust(spacing) + ' ' + ' getattr() failed')

get_methods(df['foo'])

# Summarize the results

#plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot( data=results_df['mse_test', 'mae_test', 'rmse_test', 'msle_test']
plt.show()























