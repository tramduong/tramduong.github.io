
# Latent Semantic Analysis

#### Libraries used 


```python
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
import nltk, re
from nltk.stem.wordnet import WordNetLemmatizer
import time
import gensim, operator
from scipy import spatial
import numpy as np
from gensim.models import KeyedVectors
```

#### Data Exploratory


```python
DATA_FILE = r'C:/Users/tramh/github/Data-Science-Portfolio/Airlines Covid-19/data/Airlines_dedup.json'
```


```python
# Parsing json file
def parse_json_file(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    parsed_json = [json.loads(line) for line in lines]
    return parsed_json
```


```python
# Load in deduped dataset - clean
mydata = parse_json_file(DATA_FILE)
```


```python
# Extract stories
stories = [feed['text'] for feed in mydata]
```


```python
# Develop my own tokenizer
def tokenize_stories_lemma(story):
    tokens = nltk.word_tokenize(story)
    lmtzr = WordNetLemmatizer()
    filtered_tokens = []
    for token in tokens:
        token = token.replace("'s", " ").replace("n't", " not").replace("'ve", " have")
        token = re.sub(r'[^a-zA-Z0-9 ]', '', token)
        if token not in stopwords.words('english'):
            filtered_tokens.append(token.lower())
    lemmas = [lmtzr.lemmatize(t, 'v') for t in filtered_tokens]
    return lemmas
```

#### Latent Semantic Analysis

##### Testing to be done to identify best set of parameters for min_df, max_df and the best number of topics. Best model is then applied to articles in each week


```python
# Tokenize using tfidf
### Create tokenizer based on parameters
def create_tfidf_tokenizer(min_df, max_df, ngram, max_features):
    tf_vectorizer = TfidfVectorizer(min_df=min_df, 
                                    max_df=max_df, 
                                    ngram_range=(ngram, ngram), 
                                    tokenizer=tokenize_stories_lemma, 
                                    max_features=max_features)
    return tf_vectorizer


### Create tokenizer based on variable ngrams
def create_flexible_tfidf_tokenizer(min_df, max_df, ngram, max_features):
    tf_vectorizer = TfidfVectorizer(min_df=min_df, 
                                    max_df=max_df, 
                                    ngram_range=ngram, 
                                    tokenizer=tokenize_stories_lemma, 
                                    max_features=max_features)
    return tf_vectorizer

```


```python
### Apply LSA to tfidf-vectorized text
def apply_LSA(n_components, tfidf_docs):
    svd = TruncatedSVD(n_components=n_components, n_iter=10)
    svd_topic_vectors = svd.fit_transform(tfidf_docs)
    return [svd, svd_topic_vectors]


```


```python
### Function to build LSA model 
def build_LSA_model(min_df, max_df, ngram, n_components, max_features, text_data, top_words=3):
    print('Building tokenizer...')
    
    if isinstance(ngram, int):
        tf_vectorizer = create_tfidf_tokenizer(min_df=min_df, max_df=max_df, ngram=ngram, max_features=max_features)
    else:
        tf_vectorizer = create_flexible_tfidf_tokenizer(min_df=min_df, max_df=max_df, ngram=ngram, max_features=max_features)
    tfidf_docs = tf_vectorizer.fit_transform(text_data)
    print('==================================================================')
    
    print('Generating LSA model outputs...')
    svd, svd_topic_vectors = apply_LSA(n_components=n_components, tfidf_docs=tfidf_docs)
    print('==================================================================')
    
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    print('Generating topics...')
    topics = dict()
    for topic_idx, topic in enumerate(svd.components_):
        topics[topic_idx] = [tf_feature_names[i] for i in topic.argsort()[:-top_words-1:-1]]
        print('Topic ' + str(topic_idx))
        print(" | ".join(tf_feature_names[i] for i in topic.argsort()[:-top_words-1:-1]))
    
    print('==================================================================')
    return topics
```


```python
### TESTING LSA MODEL

# Optimize for topics and number of features
n_topics = [2, 3]
features = [20, 50, 100]
for topic in n_topics:
    for feature in features:
        try:
            tic = time.perf_counter()
            print('Applying LSA model for: {0} topics and max_features: {1}'.format(topic, feature))
            result = build_LSA_model(min_df=100, max_df=1000, ngram=2, n_components=topic, max_features=feature, text_data=stories)
            toc = time.perf_counter()
            print('Process completed in {} mins'.format((toc-tic)/60))
            print('==================================================================')
            print()
        except:
            continue

# 3 topics, 50 features seems good. Optimize for min and max df:
min_df = [100, 500]
max_df = [1000, 5000]
for mini in min_df:
    for maxi in max_df:
        try:
            tic = time.perf_counter()
            print('Applying LSA model for min_df: {0} and max_df: {1}'.format(mini, maxi))
            result = build_LSA_model(min_df=mini, max_df=maxi, ngram=2, n_components=3, max_features=50, text_data=stories)
            toc = time.perf_counter()
            print('Process completed in {} mins'.format((toc-tic)/60))
            print('==================================================================')
            print()
        except:
            continue
```

    Applying LSA model for: 2 topics and max_features: 20
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
     airlines |  american |   
    Topic 1
       | southwest airlines |  get
    ==================================================================
    Process completed in 7.8181107950000035 mins
    ==================================================================
    
    Applying LSA model for: 2 topics and max_features: 50
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
       |  american |  airlines
    Topic 1
       | credit card |  stock
    ==================================================================
    Process completed in 7.881262741666664 mins
    ==================================================================
    
    Applying LSA model for: 2 topics and max_features: 100
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
       |  airlines |  american
    Topic 1
       |   not | right 
    ==================================================================
    Process completed in 8.059177126666661 mins
    ==================================================================
    
    Applying LSA model for: 3 topics and max_features: 20
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
     airlines |  american |   
    Topic 1
       | southwest airlines |  get
    Topic 2
     get | time  |  one
    ==================================================================
    Process completed in 7.613548364999997 mins
    ==================================================================
    
    Applying LSA model for: 3 topics and max_features: 50
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
       |  american |  airlines
    Topic 1
       | credit card |  stock
    Topic 2
     get |  go |  one
    ==================================================================
    Process completed in 7.364663015000004 mins
    ==================================================================
    
    Applying LSA model for: 3 topics and max_features: 100
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
       |  airlines |  american
    Topic 1
       |   not | right 
    Topic 2
      not | fly  |  people
    ==================================================================
    Process completed in 7.345298148333328 mins
    ==================================================================
    
    Applying LSA model for min_df: 100 and max_df: 1000
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
       |  american |  airlines
    Topic 1
       | credit card |  stock
    Topic 2
     get |  go |  one
    ==================================================================
    Process completed in 7.311624175000005 mins
    ==================================================================
    
    Applying LSA model for min_df: 100 and max_df: 5000
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
      | airlines  |  say
    Topic 1
    wear mask | mask  |   
    Topic 2
    airlines  | american airlines | flight 
    ==================================================================
    Process completed in 7.368646959999993 mins
    ==================================================================
    
    Applying LSA model for min_df: 500 and max_df: 1000
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
     american |  airlines |   
    Topic 1
       |   not |  take
    Topic 2
      not | fly  |  get
    ==================================================================
    Process completed in 7.296270904999998 mins
    ==================================================================
    
    Applying LSA model for min_df: 500 and max_df: 5000
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
      | airlines  |  say
    Topic 1
    wear mask | mask  |   
    Topic 2
    airlines  | american airlines | flight 
    ==================================================================
    Process completed in 7.29425408666666 mins
    ==================================================================
    
    


```python
### min_df 100 is too low - use 500
### max df not bad, maybe can test a bit higher

max_df = [6000, 7000]
for maxi in max_df:
    tic = time.perf_counter()
    print('Applying LSA model for max_df: {}'.format(maxi))
    result = build_LSA_model(min_df=500, max_df=maxi, ngram=2, n_components=3, max_features=50, text_data=stories)
    toc = time.perf_counter()
    print('Process completed in {} mins'.format((toc-tic)/60))
    print('==================================================================')
    print()


### Looks better - let's test some more for top words and number of topics
tic = time.perf_counter()
result = build_LSA_model(min_df=100, max_df=8000, ngram=3, n_components=2, max_features=300, text_data=stories, top_words=4)
toc = time.perf_counter()
print('Process completed in {} mins'.format((toc-tic)/60))
print('==================================================================')

```

    Applying LSA model for max_df: 6000
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
      | airlines  |  say
    Topic 1
    wear mask | mask  |   
    Topic 2
    airlines  | american airlines | flight 
    ==================================================================
    Process completed in 7.330493481666675 mins
    ==================================================================
    
    Applying LSA model for max_df: 7000
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
      | airlines  |  say
    Topic 1
    wear mask | mask  |   
    Topic 2
    airlines  | american airlines | flight 
    ==================================================================
    Process completed in 7.304339103333344 mins
    ==================================================================
    
    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
       |   say | say   | wear mask 
    Topic 1
    wear mask  |  wear mask | refuse wear mask | mask  
    ==================================================================
    Process completed in 7.336434568333334 mins
    ==================================================================
    


```python
### Let's try a mix of grams '''
tic = time.perf_counter()
result = build_LSA_model(min_df=10, max_df=8000, ngram=(1, 3), n_components=2, max_features=3000, text_data=stories, top_words=5)
toc = time.perf_counter()
print('Process completed in {} mins'.format((toc-tic)/60))
print('==================================================================')
```

    Building tokenizer...
    ==================================================================
    Generating LSA model outputs...
    ==================================================================
    Generating topics...
    Topic 0
      | airlines |    | mask | flight
    Topic 1
    mask | wear | wear mask | mask  | wear mask 
    ==================================================================
    Process completed in 7.374587050000006 mins
    ==================================================================
    


```python
### Retrieving the top 5 articles titles for each topic 

### Generate a list of titles
titles = [feed['title'] for feed in mydata]

### Build a taxonomy to return top article titles from LSA model
topic_taxonomy = {
    'COVID': {
        'COVID prevention protocol': 'wear mask'
    },
    'Airline companies': {
        'Companies': 'american airlines united airlines southwest airlines',
    }
}
```


```python
### Saving lists of keywords, labels and topics
keyword_list = []
label_list = []
topic_list = []
for key, value in topic_taxonomy.items():
    topic_list.append(key)
    for label, keywords in value.items():
        keyword_list.append(keywords.lower())
        label_list.append(label)

```

#### Word2Vec Model


```python
### Use word2vec to calculate similarities
def load_wordvec_model(modelName, modelFile, flagBin):
    print('Loading ' + modelName + ' model...')
    model = KeyedVectors.load_word2vec_format(modelFile, binary=flagBin)
    print('Finished loading ' + modelName + ' model...')
    return model

model_word2vec = load_wordvec_model('Word2Vec', r'C:/Users/tramh/github/Data-Science-Portfolio/Airlines Covid-19/data/GoogleNews-vectors-negative300.bin.gz', True)
```

    Loading Word2Vec model...
    Finished loading Word2Vec model...
    


```python
# calculate vector similarity between two inputs
def vec_similarity(input1, input2, vectors):
    term_vectors = [np.zeros(300), np.zeros(300)]
    terms = [input1, input2]
        
    for index, term in enumerate(terms):
        for i, t in enumerate(term.split(' ')):
            try:
                term_vectors[index] += vectors[t]
            except:
                term_vectors[index] += 0
        
    result = (1 - spatial.distance.cosine(term_vectors[0], term_vectors[1]))
    if result is 'nan':
        result = 0
        
    return result
```


```python
# function checks whether the input words are present in the vocabulary for the model
def vocab_check(vectors, words):
    
    output = list()
    for word in words:
        if word in vectors.vocab:
            output.append(word.strip())
            
    return output
```


```python
# function calculates similarity between two strings using a particular word vector model
def calc_similarity(input1, input2, vectors):
    s1words = set(vocab_check(vectors, input1.split()))
    s2words = set(vocab_check(vectors, input2.split()))
    
    output = vectors.n_similarity(s1words, s2words)
    return output
```


```python
# Sort and return top 3 articles for each item in topic_taxonomy
def classify_topics(input, vectors):
    feed_score = dict()
    for key, value in topic_taxonomy.items():
        max_value_score = dict()
        for label, keywords in value.items():
            max_value_score[label] = 0
            topic = (key + ' ' + keywords).strip()
            max_value_score[label] += float(calc_similarity(input, topic, vectors))
            
        sorted_max_score = sorted(max_value_score.items(), key=operator.itemgetter(1), reverse=True)[0]
        feed_score[sorted_max_score[0]] = sorted_max_score[1]
    return sorted(feed_score.items(), key=operator.itemgetter(1), reverse=True)[:3]

```


```python
# Collate topic classifications into a dictionary - key is title index, value is the classification topics and their similarity scores
results = {}
for i in range(len(titles)):
    try:
        results[i] = classify_topics(titles[i], model_word2vec)
    except:
        continue

''' Mapping topics to Top 5 Articles '''
# For given topic, return dictionary of article index (key) mapped to similarity score (value)
def extract_top_articles_from_topic(topic, label, classified_topics):
    topic_articles = {}
    for feed_idx, labels in classified_topics.items():
        for label_value_tuple in labels:
            if label_value_tuple[0] == label:
                topic_articles[feed_idx] = label_value_tuple[1]
    # Return just top 10 articles
    sorted_topic_articles = {k: v for k, v in sorted(topic_articles.items(), key=lambda item: item[1], reverse=True)}
    # Return just top 5 articles
    return list(sorted_topic_articles.items())[:5]
```


```python
# Map topic articles to actual article titles
def get_article_titles(top_topics, titles):
    articles_to_similarity = {}
    for feed in top_topics:
        articles_to_similarity[feed[0]] = [titles[feed[0]], feed[1]]
    return articles_to_similarity
```


```python
# Reverse the taxonomy
reverse_taxonomy = {}
for topic, label_dict in topic_taxonomy.items():
    for label, keywords in label_dict.items():
        reverse_taxonomy[label] = topic


for label, topic in reverse_taxonomy.items():
    print('Topic: {} \n '.format(topic))
    print('Label: {} \n '.format(label))
    print('Top 5 articles and similarity scores \n')
    top_articles = extract_top_articles_from_topic(topic=topic, label=label, classified_topics=results)
    final_results = get_article_titles(top_articles, titles)
    for result in final_results:
        print(final_results[result][0], round(final_results[result][1], 2))
    print()
    print('=========================================================================================')
```

    Topic: COVID 
     
    Label: COVID prevention protocol 
     
    Top 5 articles and similarity scores 
    
    Alaska Airlines: Wear mask or else! 0.68
    Airline mask policy: Do you have to wear a mask on a plane? 0.66
    Coronavirus: United Airlines says you need to wear face mask in airport 0.6
    Airlines are actually banning flyers who won’t wear masks 0.58
    Alaska Airlines could suspend passengers who refuse to wear a mask 0.57
    
    =========================================================================================
    Topic: Airline companies 
     
    Label: Companies 
     
    Top 5 articles and similarity scores 
    
    southwest philly airlines 0.75
    united airlines to require masks at all airports 0.7
    Information for american airlines cancellation policy 0.68
    TC-JND A330-200 Turkish airlines 17-08-16 ebbr maarten-sr 0.68
    eight airlines pinned, they “violate passenger rights” 0.67
    
    =========================================================================================
    
