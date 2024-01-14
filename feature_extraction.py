import math
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import nltk
from nltk import bigrams, trigrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from spellchecker import SpellChecker
from sklearn.preprocessing import StandardScaler


# Read data
def read_data(path):
    return pd.read_csv(path)

# Text length
def text_len(text):
    tokens = word_tokenize(text)
    return len(tokens)


# Type-to-text ratio
def ttr(text):
    text = text.lower()
    tokens = word_tokenize(text)
    types = nltk.Counter(tokens)
    return (len(types)/len(tokens))


# Ratio of stopwords to text length
def stops_ratio(text):
    tokens = word_tokenize(text)
    c = 0
    for i in tokens:
        if i in stops:
            c += 1
    return round((c/len(tokens)),2)


# Cosine similarity between the student summary and the prompt text
def get_cosine(t1, t2):
    word = re.compile(r"\w+")
    word1 = word.findall(t1)
    word2 = word.findall(t2)
    vec1 = Counter(word1)
    vec2 = Counter(word2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
    

# Total contraction words in the student summary text
def tot_conts(text):
    c = 0
    tokens = word_tokenize(text)
    for i in tokens:
        if i in contract_keys:
            c += 1
    return c


# Proper nouns
def ppn_extract(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word not in set(stopwords.words('english'))]
    poses = nltk.pos_tag(words)
    return [word for (word,tag) in poses if tag != 'NNP']

# No.of misspelled words. Note that the proper nouns and punctuations have been removed from the text.
def misspellings(text):
    # remove punctuations.
    st = re.sub(r'[^\w\s]','',text)
    # remove proper nouns using pos tagging
    words = ppn_extract(st)
    ms = sp.unknown(words)
    return len(ms)


# The quality of student text depends on a lot of factors. What better way to say about the quality of the text than
# examining the parts of speech.
# The pos will be represented as 7 features namely nouns, adjectives, verbs, pronouns, numerics, conjunctions
def pos_count(text):
    nouns = 0
    adjs = 0
    vbs = 0
    prons = 0
    nums = 0
    conjns = 0

    poses = nltk.pos_tag(text.split())
    
    for pos in poses:
        if pos[1] in ['NN','NNP','NNS']:
            nouns += 1
        if pos[1] in ['JJ','JJR','JJS']:
            adjs += 1
        if pos[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            vbs += 1
        if pos[1] in ['PRP','PRP$']:
            prons += 1
        if pos[1] in ['CD']:
            nums += 1
        if pos[1] in ['CC','IN']:
            conjns += 1
            
    return pd.Series([nouns, adjs, vbs, prons, nums, conjns])


# drop unnecessary cols
def drops(df):
    df = df.drop(drop_cols, axis = 1)
    return df

# feature scaling
def scales(df):
    df_sc_cols = df[scale_cols]
    sc_df = scaler.fit_transform(df_sc_cols)
    df[scale_cols] = sc_df
    return df

# save csv files
def save_data(dfs):
    for i in range(len(dfs)):
        dfs[i].to_csv(f"data/training_data/df_{i+1}.csv", index = False)



if __name__ == "__main__":
    
    stops = stopwords.words('english')
    contract_url = "https://gist.githubusercontent.com/Lewy09-Tm25/2ca6392c5741b5522e5abccf267a2cf0/raw/c8e7f7ccd3aad74d7b2e0135cc1f3e9e5e0f72f9/contractions.json"
    contract_dict = pd.read_json(contract_url, typ = 'series')
    contract_keys = list(contract_dict.keys())
    sp = SpellChecker()

    df_1 = read_data("data/split_data/39c16e.csv")
    df_2 = read_data("data/split_data/3b9047.csv")
    df_3 = read_data("data/split_data/ebad26.csv")
    df_4 = read_data("data/split_data/814d6b.csv")

    dfs = [df_1, df_2, df_3, df_4]
    for df in dfs:
        df['len'] = df['text'].apply(text_len)
        df['ttr'] = df['text'].apply(ttr)
        df['stopwords_ratio'] = df['text'].apply(stops_ratio)
        df['cossim'] = df.apply(lambda x:get_cosine(x['text'],x['prompt_text']), axis = 1)
        df['conts'] = df['text'].apply(tot_conts)
        df['misspelled'] = df['text'].apply(misspellings)
        df[["nouns", "adjs", "vbs", "prons", "nums", "conjns"]] = df['text'].apply(pos_count)
        print(f"feature extraction done --------------\n")
    
    # dropping columns and feature scaling
    drop_cols = ['prompt_id','prompt_question','prompt_title','prompt_text','student_id','text']
    scale_cols = ['len','conts','misspelled','nouns','adjs','vbs','prons','nums','conjns']
    scaler = StandardScaler()
    for df in dfs:
        df = drops(df)
        df = scales(df)
        print(f"prepared for training --------------\n")

    save_data(dfs)