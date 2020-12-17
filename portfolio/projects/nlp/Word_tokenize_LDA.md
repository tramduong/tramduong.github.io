
# Library Imports


```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk, re
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(nltk.corpus.stopwords.words('english'))
```

# Load feeds into DF


```python
import json
google_json=open("/Github/google_deduplicated.json").readlines()
```


```python
feed_text = []

for feed in google_json:
    a = json.loads(feed)
    feed_text.append(a['text'])
```


```python
print("Total number of text: " + str(len(feed_text)))
```

    Total number of text: 18116
    

# Word Tokenizer


```python
def tokenize_stories(text):
    tokens = nltk.word_tokenize(text)
    lmtzr = WordNetLemmatizer()
    filtered_tokens = []
    
    for token in tokens:
        token = token.replace("'s", " ").replace("n’t", " not").replace("’ve", " have")
        token = re.sub(r'[^a-zA-Z0-9 ]', '', token)
        if token not in stopwords:
            filtered_tokens.append(token.lower())
    
    lemmas = [lmtzr.lemmatize(t,'v') for t in filtered_tokens]

    return lemmas
```

# Training LDA Model


```python
# Through multiple testings, the best results for topic modeling are the below parameters
#max_df = 0.15
#min_df = 0.01
#max_features = 1000
#max_iter = 500
```


```python
def test_lda_model(tf, tf_vectorizer, num_topics, max_iter, n_top_words):
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=max_iter, learning_method='batch', learning_offset=10, random_state=1)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()

    topics = dict()
    for topic_idx, topic in enumerate(lda.components_):
        topics[topic_idx] = [tf_feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]

    return topics
```


```python
tf_vectorizer = CountVectorizer(max_df=0.15, min_df=0.01, max_features=1000, tokenizer=tokenize_stories, ngram_range=(1, 1))
tf = tf_vectorizer.fit_transform(feed_text)
```


```python
lda = LatentDirichletAllocation(n_components=8, max_iter=500, learning_method='batch', learning_offset=10, random_state=1)
lda_model = lda.fit(tf)
```


```python
topics = test_lda_model(tf, tf_vectorizer, 8, 500, 10)
print(topics)
```

    {0: ['cloud', 'technology', 'team', 'digital', 'health', 'network', 'platform', 'design', 'tool', 'develop'], 1: ['page', 'https', 'website', 'web', 'site', 'chrome', 'browser', 'link', 'file', 'user'], 2: ['trump', 'podcast', 'president', 'tech', 'law', 'privacy', 'government', 'order', 'tweet', 'claim'], 3: ['police', 'black', 'coronavirus', 'city', 'health', 'case', 'officer', 'floyd', 'protest', 'pm'], 4: ['million', 'india', 'per', 'pay', 'digital', 'increase', 'billion', 'stock', 'businesses', 'revenue'], 5: ['game', 'good', 'really', 'lot', 'school', 'nt', 'things', 'students', 'something', 'every'], 6: ['android', 'phone', 'apps', 'apple', 'de', 'game', 'device', 'pixel', 'store', 'devices'], 7: ['log', 'smart', 'tv', 'amazon', 'music', 'voice', 'stream', 'assistant', 'youtube', 'never']}
    

# LDA on 10 random articles


```python
import random
sample = random.sample(range(1, len(feed_text)), 10)

random_10_text = [feed_text[i] for i in sample]
```


```python
lda_results = lda.fit_transform(tf)
sample_text_results = lda_results[sample,]
```


```python
import pandas as pd
df = pd.DataFrame(sample_text_results, index=sample)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3387</th>
      <td>0.001625</td>
      <td>0.001625</td>
      <td>0.001626</td>
      <td>0.001626</td>
      <td>0.040522</td>
      <td>0.001626</td>
      <td>0.949724</td>
      <td>0.001626</td>
    </tr>
    <tr>
      <th>7547</th>
      <td>0.002720</td>
      <td>0.002722</td>
      <td>0.002723</td>
      <td>0.520908</td>
      <td>0.002720</td>
      <td>0.407885</td>
      <td>0.002725</td>
      <td>0.057595</td>
    </tr>
    <tr>
      <th>16954</th>
      <td>0.005012</td>
      <td>0.005002</td>
      <td>0.005006</td>
      <td>0.225241</td>
      <td>0.005014</td>
      <td>0.610612</td>
      <td>0.139092</td>
      <td>0.005020</td>
    </tr>
    <tr>
      <th>15396</th>
      <td>0.381700</td>
      <td>0.136717</td>
      <td>0.001509</td>
      <td>0.001507</td>
      <td>0.001507</td>
      <td>0.201019</td>
      <td>0.274533</td>
      <td>0.001508</td>
    </tr>
    <tr>
      <th>4546</th>
      <td>0.208270</td>
      <td>0.000513</td>
      <td>0.196529</td>
      <td>0.000513</td>
      <td>0.254618</td>
      <td>0.058234</td>
      <td>0.280810</td>
      <td>0.000513</td>
    </tr>
    <tr>
      <th>10721</th>
      <td>0.001150</td>
      <td>0.001148</td>
      <td>0.200276</td>
      <td>0.588801</td>
      <td>0.145510</td>
      <td>0.060818</td>
      <td>0.001148</td>
      <td>0.001148</td>
    </tr>
    <tr>
      <th>6700</th>
      <td>0.256704</td>
      <td>0.000921</td>
      <td>0.033379</td>
      <td>0.000921</td>
      <td>0.492184</td>
      <td>0.188476</td>
      <td>0.000920</td>
      <td>0.026496</td>
    </tr>
    <tr>
      <th>12455</th>
      <td>0.013889</td>
      <td>0.013892</td>
      <td>0.013891</td>
      <td>0.013889</td>
      <td>0.013889</td>
      <td>0.013891</td>
      <td>0.013889</td>
      <td>0.902770</td>
    </tr>
    <tr>
      <th>13220</th>
      <td>0.621430</td>
      <td>0.345943</td>
      <td>0.005442</td>
      <td>0.005436</td>
      <td>0.005438</td>
      <td>0.005437</td>
      <td>0.005438</td>
      <td>0.005436</td>
    </tr>
    <tr>
      <th>15689</th>
      <td>0.008415</td>
      <td>0.000057</td>
      <td>0.458526</td>
      <td>0.076257</td>
      <td>0.066695</td>
      <td>0.389936</td>
      <td>0.000057</td>
      <td>0.000057</td>
    </tr>
  </tbody>
</table>
</div>




```python
for x in range(len(sample)):
    print("For index " + str(sample[x])+
         ", the max value comes from topics " + str(int(df.iloc[[x]].idxmax(1)))+
         ", with a max value of " + str(round(float(max(sample_text_results[x])),4)))
```

    For index 3387, the max value comes from topics 6, with a max value of 0.9497
    For index 7547, the max value comes from topics 3, with a max value of 0.5209
    For index 16954, the max value comes from topics 5, with a max value of 0.6106
    For index 15396, the max value comes from topics 0, with a max value of 0.3817
    For index 4546, the max value comes from topics 6, with a max value of 0.2808
    For index 10721, the max value comes from topics 3, with a max value of 0.5888
    For index 6700, the max value comes from topics 4, with a max value of 0.4922
    For index 12455, the max value comes from topics 7, with a max value of 0.9028
    For index 13220, the max value comes from topics 0, with a max value of 0.6214
    For index 15689, the max value comes from topics 2, with a max value of 0.4585
    
