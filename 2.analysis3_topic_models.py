#Topic modeling

print('Lets start!')

# Packages
import pandas as pd
import re
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics

import gensim
from gensim.matutils import kullback_leibler
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns 

## Helper functions
import util_analysis1_word_count as util

def tf_idfVectorizer(df, input_text, stp_wrds=[]):
    '''
    Creates a TfidfVectorizer class using skit learn and fits it to an input text
    from a pandas df
    
    Input:
        df(pandas df): dataframe
        input_text(pandas serias): pandas columns with tokenized text
        stp_wrds(list): list of stop words
    
    Output:
        dfTFVectorizer(class): class that sarisfy conditions within function 
        dfTFVects(fit model): matrix of vectorized text usinf tf idf
    '''
    dfTFVectorizer = \
            sklearn.feature_extraction.text.TfidfVectorizer(max_df=.5, 
                                                            max_features=1000, 
                                                            min_df=3, 
                                                            norm='l2',
                                                            stop_words=stp_wrds)
    dfTFVects = dfTFVectorizer.fit_transform(df[input_text])
    
    return dfTFVectorizer, dfTFVects
    

def dropMissing(wordLst, vocab):
    return [w for w in wordLst if w in vocab]
    

def convert_to_prob(bow):
    '''
    Input: 
        bow: topics distribution
    
    '''
    ps = []
    for topic_no, topic_prob in bow:
        ps.append(topic_prob)
    return ps
    
    
print('functions loaded!')


## Data
data = pd.read_csv('data/2021_03_data.csv')
corpus = data

## Text preprocesing:
## Delete duplicates
corpus = corpus.drop_duplicates(subset=['Text'])


# Lematize 
# corpus['lemmatized_text'] = \
#    corpus['normalized_text'].apply(lambda x: lematize_list(x))


# Create a dataset in which each politician is mentioned and 
# the source comes from the politician‘s country
santos = corpus[(corpus["Text"].str.contains("Santos")) & (corpus["Country"]=="CO")]
uribe = corpus[(corpus["Text"].str.contains("Uribe")) & (corpus["Country"]=="CO")]
pena_nieto = corpus[(corpus["Text"].str.contains("Peña Nieto")) & (corpus["Country"]=="MX")]
correa = corpus[(corpus["Text"].str.contains("Correa")) & (corpus["Country"]=="EC")]
morales = corpus[corpus["Text"].str.contains("Evo") & (corpus["Country"]=="BO")]
chavez = corpus[corpus["Text"].str.contains("Chávez") & (corpus["Country"]=="VE")]
maduro = corpus[corpus["Text"].str.contains("Maduro") & (corpus["Country"]=="VE")]
kirchner = corpus[corpus["Text"].str.contains("Kirchner") & (corpus["Country"]=="AR")]
ortega = corpus[corpus["Text"].str.contains("Ortega") & (corpus["Country"]=="NI")]
bachelet = corpus[corpus["Text"].str.contains("Bachelet") & (corpus["Country"]=="CL")]
mujica = corpus[corpus["Text"].str.contains("Mujica") & (corpus["Country"]=="UY")]
print('data frames for each president created')

list_df = [santos, uribe, pena_nieto, correa,\
            morales, chavez, maduro, kirchner, ortega, bachelet, mujica]

list_names = ['santos', 'uribe', 'pena_nieto', 'correa',\
              'morales', 'chavez', 'maduro', 'kirchner', 'ortega', 
              'bachelet', 'mujica']

i = 0
for df in list_df:
    df.name = list_names[i]
    i += 1

for df in list_df:
    print(df.name, 'df:', df.shape)

for df in list_df:
    df['text2'] = df['Text'].apply(lambda x: re.sub('[¡!@#$:).;,¿?&]', '', x.lower()))
    df['text2'] = \
            df['text2'].apply(lambda x: re.sub("\d+", "", x)) #substracts digits

####################################
# THIS IS COMPUTATIONALLY EXPENSIVE
for df in list_df:
    df['normalized_tokens'] = df['text2'].apply(lambda x: util.normalize(x))
###################################

stp_wrds = ['me', 
 'mi', 
 'yo', 
 'era', 
 'había', 
 'muy', 
 'estaba',
 'qué', 
 'he', 
 'día', 
 'tnn', 
 'me',
 'qué',
 'ni', 
 'gente', #I don't think you want to take this word out. 
 'muy', 
 'yo', 
 'bien', #I don't think you want to take this word out.
 'decir',  
 'puede', 
 'esa', 
 'te', 
 'usted']

santosTFVectorizer, santosTFVects = tf_idfVectorizer(santos, 'text2', stp_wrds)
uribeTFVectorizer, uribeTFVects = tf_idfVectorizer(uribe, 'text2', stp_wrds)
pena_nietoTFVectorizer, pena_nietoTFVects = tf_idfVectorizer(pena_nieto, 'text2', stp_wrds)
correaTFVectorizer, correaTFVects = tf_idfVectorizer(correa, 'text2', stp_wrds)
moralesTFVectorizer, moralesTFVects = tf_idfVectorizer(morales, 'text2', stp_wrds)
chavezTFVectorizer, chavezTFVects = tf_idfVectorizer(chavez, 'text2', stp_wrds)
maduroTFVectorizer, maduroTFVects = tf_idfVectorizer(maduro, 'text2', stp_wrds)
kirchnerTFVectorizer, kirchnerTFVects = tf_idfVectorizer(kirchner, 'text2', stp_wrds)
ortegaTFVectorizer, ortegaTFVects = tf_idfVectorizer(ortega, 'text2', stp_wrds)
bacheletTFVectorizer, bacheletTFVects = tf_idfVectorizer(bachelet, 'text2', stp_wrds)
mujicaTFVectorizer, mujicaTFVects = tf_idfVectorizer(mujica, 'text2', stp_wrds)

# List of TFVectors
list_tfvect_obj = [santosTFVectorizer, uribeTFVectorizer, pena_nietoTFVectorizer,
                correaTFVectorizer, moralesTFVectorizer, chavezTFVectorizer,
                maduroTFVectorizer, kirchnerTFVectorizer, ortegaTFVectorizer, 
                bacheletTFVectorizer, mujicaTFVectorizer] 

print('loop and dropMissing!')
i = 0
for df in list_df:
    df['reduced_tokens'] = df['normalized_tokens'].apply(lambda x: dropMissing(x, list_tfvect_obj[i].vocabulary_.keys()))
    i+=1

###############

# Get 25 topics for all candidates 
for df in list_df:
    dictionary = gensim.corpora.Dictionary(df['reduced_tokens'])
    corpus = [dictionary.doc2bow(text) for text in df['reduced_tokens']]
    gensim.corpora.MmCorpus.serialize('df.mm', corpus)
    dfmm = gensim.corpora.MmCorpus('df.mm')
    # Here the model is being created
        # Notice that the number of topics is defined here
        # **** As I increase the number of topics each topic seems to be better differentiated ****
    num_topics = 25
    dflda = gensim.models.ldamodel.LdaModel(corpus=dfmm, id2word=dictionary, num_topics=num_topics, alpha='auto', eta='auto')

        # Show df of topics
    topicsDict = {}
    for topicNum in range(dflda.num_topics):
        topicWords = [w for w, p in dflda.show_topic(topicNum)]
        topicsDict['Topic_{}'.format(topicNum)] = topicWords

    wordRanksDF = pd.DataFrame(topicsDict)
    wordRanksDF.to_csv('results/topic_models/{}_{}topics.csv'.format(df.name, num_topics))

columns = santos.columns
df_all = pd.DataFrame(columns=columns)
for df in list_df:
    df['president_name'] = df.name
    df_all = df_all.append(df)

dictionary = gensim.corpora.Dictionary(df_all['reduced_tokens'])
corpus = [dictionary.doc2bow(text) for text in df_all['reduced_tokens']]
gensim.corpora.MmCorpus.serialize('df_all.mm', corpus)
df_allcorpus = gensim.corpora.MmCorpus('df_all.mm')
num_topics = 25
df_all_lda = gensim.models.ldamodel.LdaModel(corpus=df_allcorpus, id2word=dictionary, num_topics=num_topics, alpha='auto', eta='auto')


'''

for name in list_names:
    president_all_words = []
    for df in list_df:
        sent = df
        for word in sent:
            actor_all_words += word
    df[name]['topic_distribution'] = df_all_lda[dictionary.doc2bow(util.normalize(president_all_words))]
## KL divergence


for 

'''