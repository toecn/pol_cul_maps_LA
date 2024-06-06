
## Helper functions

import spacy

nlp = spacy.load("es_core_news_sm")

def normalize(text, stopwords=[]):
    '''
    Normalizes: takes out punctuation and stopwords (defined by SpaCy) 
    Input:
        text (str): text made of string
        stopwords(list): list 
        
    Output:
        lexical_tokens(list): list of strings. Text has been clean of puntuation, numbers,
                        stop words, and made lower case
    '''
    test = nlp(text)
    words = [t.orth_ for t in test if not t.is_punct | t.is_stop] #here is filtering
    lexical_tokens = [t.lower() for t in words if t.isalpha()] #make lowercase and leave nonnumerical
    
    for word in list(lexical_tokens):
        if word in stopwords:
            lexical_tokens.remove(word)
    
    return lexical_tokens


def lemmatizer(text):
    '''
    Lemmatizes a text using the SpaCy lemmatizer, which makes errors 
    '''
    test = nlp(text)
    lemmas = [tok.lemma_.lower() for tok in test]
    
    return lemmas


def lematize_list(word_list):
    '''
    Takes a list of words and applies the lematizer function
    
    Input:
        word_list(list)
    
    Output:
        lem_list(list): list of lemmatized words
        
    '''
    lem_list=[]
    for word in word_list:
        lem_word = lemmatizer(word)
        lem_list.append(lem_word[0])
    
    return lem_list