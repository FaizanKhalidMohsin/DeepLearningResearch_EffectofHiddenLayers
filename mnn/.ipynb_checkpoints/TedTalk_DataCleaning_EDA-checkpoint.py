


############################################################################
# Loading Packages
import numpy as np
import pandas as pd
import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import datetime


#############################################################################
# Loading the data
df = pd.read_csv("ted_main.csv")
print(df.columns)

##############################################################################
# Data Cleaning and Preparation

df = df[['name', 'title', 'description', 'main_speaker', 'speaker_occupation', 'num_speaker', 'duration', 'event', 'film_date', 'published_date', 'comments', 'tags', 'languages', 'ratings', 'related_talks', 'url', 'views']]

df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
print(df.head())

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df[['event']]

################################################################################
# EDA

pop_talks = df[['title', 'main_speaker', 'views', 'film_date']].sort_values('views', ascending=False)[:15]
pop_talks

pop_talks['abbr'] = pop_talks['main_speaker'].apply(lambda x: x[:3])
sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
sns.barplot(x='abbr', y='views', data=pop_talks)

df[['views', 'comments']].describe()

#The average number of views on TED Talks in 1.6 million. The median number of views is 1.12 million. 


## Scatter Plot
sns.jointplot(x='views', y='comments', data=df)

## Correlation between X and Y
df[['views', 'comments']].corr()

## 10 most commented TED Talks of all time.
df[['title', 'main_speaker','views', 'comments']].sort_values('comments', ascending=False).head(10)

#Ken Robinson has second most comment video. The title of the topics are more contravertial in general.


# Ted Talk Popular Months and Days

df['month'] = df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])

month_df = pd.DataFrame(df['month'].value_counts()).reset_index()
month_df.columns = ['month', 'talks']

sns.barplot(x='month', y='talks', data=month_df, order=month_order)

#February is clearly the most popular month for TED Conferences as the official TED Conferences are held in February.

## TED Speakers

speaker_df = df.groupby('main_speaker').count().reset_index()[['main_speaker', 'comments']]
speaker_df.columns = ['main_speaker', 'appearances']
speaker_df = speaker_df.sort_values('appearances', ascending=False)
speaker_df.head(10)

#Hans Rosling has the highest number of appearences: 9 !


## Which Occupation is invited to Speak the most at TED Talks?

occupation_df = df.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]
occupation_df.columns = ['occupation', 'appearances']
occupation_df = occupation_df.sort_values('appearances', ascending=False)

plt.figure(figsize=(15,5))
sns.barplot(x='occupation', y='appearances', data=occupation_df.head(10))
plt.show()

#Writer is the most invited Occupation with 45 writers appearing.Followed by Artist and Designer with 33 appearances. 


## Which Occupation draws more views?

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='speaker_occupation', y='views', data=df[df['speaker_occupation'].isin(occupation_df.head(10)['occupation'])], palette="muted", ax =ax)
ax.set_ylim([0, 0.4e7])
plt.show()

#The Occupation Writer and Psychologist draws the most views! With hightest Medians.


## Look at the data a little. Look at events.

event_df.columns = ['event', 'talks']
event_df


# Among 2550 talks in the dataset, some are in fact not TED or TEDx events (for example, there is a video filmed in 1972, even before TED is established). They will be removed in this study.

ted = pd.read_csv('ted_main.csv')

# Categorize events into TED and TEDx; exclude those that are non-TED events
ted = ted[ted['event'].str[0:3]=='TED'].reset_index()

ted.loc[:,'event_cat'] = ted['event'].apply(lambda x: 'TEDx' if x[0:4]=='TEDx' else 'TED')

print ("No. of talks remain: ", len(ted))

df.shape

print("Number of non-Ted talks that we removed were:", len(df) - len(ted))

































