
# Similarity Score Project 

Use Word2Vec pre-trained model and your Webhose dataset, to identify 100 most similar titles to any one chosen title.

###  Google Word2Vec Model 

### Loads the downloaded pre-trained Google Word2Vec model from your computer


```python
#!pip install gensim --user
```


```python
import gensim, operator
from scipy import spatial
import numpy as np
from gensim.models import KeyedVectors

model_path = 'C:/Users/tramh/github/data'
```


```python
def load_wordvec_model(modelName, modelFile, flagBin):
    print('Loading ' + modelName + ' model...')
    model = KeyedVectors.load_word2vec_format(model_path + modelFile, binary=flagBin)
    print('Finished loading ' + modelName + ' model...')
    return model
```


```python
model_word2vec = load_wordvec_model('Word2Vec', 'GoogleNews-vectors-negative300.bin.gz', True)
```

    Loading Word2Vec model...
    Finished loading Word2Vec model...
    

### Import webhose data


```python
#Reads JSON objects of newsfeeds into list of dictionaries
import json
json_data=open("C:/data/webhose_apple.json").readlines()
```


```python
# Prints the number of newsfeeds (JSON objects) in the collection
newsfeeds_read = []
for line in json_data:
    newsfeeds_read.append(json.loads(line))
len(newsfeeds_read)
```




    10800




```python
import random

title_list = [x['title'] for x in newsfeeds_read]
article_title = random.choice(title_list)
```

### Checking Similarity


```python
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
sim_list=[]

for i in title_list:
    try:
        sim = calc_similarity(article_title, i, model_word2vec)
        sim_list.append(sim)
    except:
        #sim_list.append(0, 'ERROR ZERO DIV '+i)
        sim_list.append(0)
```

### 100 most similar titles in a descending order of similarity scores


```python
import pandas as pd
df = pd.DataFrame(list(zip(title_list, sim_list)),columns =['Title', 'Similarity'])
most_similar = df.sort_values(['Similarity'], ascending=0)
most_similar[:100]
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
      <th>Title</th>
      <th>Similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5255</th>
      <td>New iMac, iPad Pro and 16 inch MacBook Pro com...</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>6077</th>
      <td>Rumor: iPad Pro, 16-inch MacBook Pro, iMac wit...</td>
      <td>0.855856</td>
    </tr>
    <tr>
      <th>6851</th>
      <td>New 13 inch MacBook Pro gets unboxed again (Vi...</td>
      <td>0.855090</td>
    </tr>
    <tr>
      <th>159</th>
      <td>Apple’s 16-inch MacBook Pro, 2018’s 12.9-inch ...</td>
      <td>0.815411</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Deals: iPads back for $249, Apple Watch 5 $299...</td>
      <td>0.802628</td>
    </tr>
    <tr>
      <th>2033</th>
      <td>May 2020 in review: 'iPhone 12' leaked, MacBoo...</td>
      <td>0.788718</td>
    </tr>
    <tr>
      <th>1266</th>
      <td>Latest 13-inch MacBook Pro hits new low, 10.2-...</td>
      <td>0.783056</td>
    </tr>
    <tr>
      <th>7286</th>
      <td>Apple iPad Pro (2020) review: Is it a notebook...</td>
      <td>0.779337</td>
    </tr>
    <tr>
      <th>10665</th>
      <td>MacBook Pro $300 off, Apple Watch $179, iMac P...</td>
      <td>0.778373</td>
    </tr>
    <tr>
      <th>2147</th>
      <td>May 2020 in Review: at last the MacBook Pro 13...</td>
      <td>0.776781</td>
    </tr>
    <tr>
      <th>1852</th>
      <td>Apple 13in MacBook Pro review 2020: going out ...</td>
      <td>0.776526</td>
    </tr>
    <tr>
      <th>3529</th>
      <td>Apple iPad Pro (2020) review: Still the best t...</td>
      <td>0.768484</td>
    </tr>
    <tr>
      <th>1693</th>
      <td>Apple 13in MacBook Pro review 2020: going out ...</td>
      <td>0.766282</td>
    </tr>
    <tr>
      <th>6079</th>
      <td>Analyst Jeff Pu: 16-Inch MacBook Pro, iPad Pro...</td>
      <td>0.760718</td>
    </tr>
    <tr>
      <th>6096</th>
      <td>16-Inch MacBook Pro, iPad Pro, and iMac Pro Wi...</td>
      <td>0.758488</td>
    </tr>
    <tr>
      <th>5091</th>
      <td>MacBook 12-inch: Apple’s first ARM-based Mac e...</td>
      <td>0.756616</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>Apple just made the MacBook Pro 2020 a worse v...</td>
      <td>0.753645</td>
    </tr>
    <tr>
      <th>7563</th>
      <td>Apple Rumor- A New 13-inch MacBook Pro to Laun...</td>
      <td>0.753357</td>
    </tr>
    <tr>
      <th>7652</th>
      <td>Today only: $630 to $700 off Apple's 2019 13-i...</td>
      <td>0.753085</td>
    </tr>
    <tr>
      <th>7746</th>
      <td>Today only: $630 to $700 off Apple's 2019 13-i...</td>
      <td>0.753085</td>
    </tr>
    <tr>
      <th>6667</th>
      <td>Best Memorial Day Deals on AirPods, Apple Watc...</td>
      <td>0.749021</td>
    </tr>
    <tr>
      <th>5764</th>
      <td>Mini-LED 16-Inch MacBook Pro, iPad Pro, and iM...</td>
      <td>0.743874</td>
    </tr>
    <tr>
      <th>4829</th>
      <td>Apple's new 13-inch MacBook Pro is $100 off in...</td>
      <td>0.743729</td>
    </tr>
    <tr>
      <th>3460</th>
      <td>new ipad professional: Apple iPad Pro now on s...</td>
      <td>0.743277</td>
    </tr>
    <tr>
      <th>8103</th>
      <td>New 23-inch iMac rumoured to launch this year</td>
      <td>0.741977</td>
    </tr>
    <tr>
      <th>790</th>
      <td>This Dell laptop is the best Windows alternati...</td>
      <td>0.741813</td>
    </tr>
    <tr>
      <th>9234</th>
      <td>Apple's 16-inch MacBook Pro powerhouse laptop ...</td>
      <td>0.737959</td>
    </tr>
    <tr>
      <th>791</th>
      <td>This Dell laptop is the best Windows alternati...</td>
      <td>0.737039</td>
    </tr>
    <tr>
      <th>6644</th>
      <td>Apple Memorial Day sale- 13-inch MacBook Pro g...</td>
      <td>0.734248</td>
    </tr>
    <tr>
      <th>4803</th>
      <td>Apple iPad Pro 12.9" 256GB WiFi Tablet (Late 2...</td>
      <td>0.731697</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>Apple to release four iPhone 12 models this year</td>
      <td>0.706253</td>
    </tr>
    <tr>
      <th>1695</th>
      <td>Apple doubles RAM upgrade charges for 13-inch ...</td>
      <td>0.706240</td>
    </tr>
    <tr>
      <th>1820</th>
      <td>Save $199 on Apple's latest 13-inch MacBook Pr...</td>
      <td>0.706043</td>
    </tr>
    <tr>
      <th>6714</th>
      <td>Apple is expected to release a larger iPhone 1...</td>
      <td>0.705738</td>
    </tr>
    <tr>
      <th>5378</th>
      <td>Apple plans to change Magic Keyboard for iPad ...</td>
      <td>0.704676</td>
    </tr>
    <tr>
      <th>3013</th>
      <td>New 12.9-inch iPad Pro now $963, lowest price ...</td>
      <td>0.704365</td>
    </tr>
    <tr>
      <th>7753</th>
      <td>Apple 13" Intel i5 MacBook Pro (2019)</td>
      <td>0.704308</td>
    </tr>
    <tr>
      <th>834</th>
      <td>Apple doubles the price of adding more RAM to ...</td>
      <td>0.702685</td>
    </tr>
    <tr>
      <th>2994</th>
      <td>Apple's new 13-inch MacBook Pro is the best wo...</td>
      <td>0.702585</td>
    </tr>
    <tr>
      <th>352</th>
      <td>Apple 16 and quot; MacBook Pro (Late 2019, Sil...</td>
      <td>0.701802</td>
    </tr>
    <tr>
      <th>4071</th>
      <td>Apple MacBook Pro (13-Inch, 2020) Review: Port...</td>
      <td>0.701299</td>
    </tr>
    <tr>
      <th>8326</th>
      <td>Apple MacBook Air, MacBook Pro, and Dell XPS 1...</td>
      <td>0.700124</td>
    </tr>
    <tr>
      <th>5872</th>
      <td>Anker’s iPhone accessory Gold Box starts at $1...</td>
      <td>0.699247</td>
    </tr>
    <tr>
      <th>1550</th>
      <td>Apple doubles cost of upgrading RAM on MacBook...</td>
      <td>0.699143</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>Apple increases RAM upgrade prices on entry-le...</td>
      <td>0.698734</td>
    </tr>
    <tr>
      <th>5844</th>
      <td>Apple MacBook Pro Intel i5-7360U 13" Retina La...</td>
      <td>0.698125</td>
    </tr>
    <tr>
      <th>8252</th>
      <td>Apple updates 13-inch MacBook Pro with Magic K...</td>
      <td>0.696214</td>
    </tr>
    <tr>
      <th>6849</th>
      <td>Apple updates 13-inch MacBook Pro with Magic K...</td>
      <td>0.696214</td>
    </tr>
    <tr>
      <th>6717</th>
      <td>Apple is expected to release a larger iPhone 1...</td>
      <td>0.696129</td>
    </tr>
    <tr>
      <th>6754</th>
      <td>Apple is expected to release a larger iPhone 1...</td>
      <td>0.696129</td>
    </tr>
    <tr>
      <th>4260</th>
      <td>Apple 12.9-inch iPad Pros are up to $450 off t...</td>
      <td>0.696043</td>
    </tr>
    <tr>
      <th>6596</th>
      <td>With a quality keyboard case, your iPad Pro is...</td>
      <td>0.695787</td>
    </tr>
    <tr>
      <th>5762</th>
      <td>Samsung Galaxy Tab S7 will beat iPad Pro with ...</td>
      <td>0.695344</td>
    </tr>
    <tr>
      <th>2073</th>
      <td>Handson: Magic Keyboard for iPad Pro impressions</td>
      <td>0.695217</td>
    </tr>
    <tr>
      <th>4515</th>
      <td>New iPad Pro (2021) release date, price &amp; spec...</td>
      <td>0.694778</td>
    </tr>
    <tr>
      <th>7033</th>
      <td>iPhone eleven Pro and 11 Pro Max evaluation: T...</td>
      <td>0.692998</td>
    </tr>
    <tr>
      <th>10115</th>
      <td>Microsoft’s Surface Book 3 is a powerhouse riv...</td>
      <td>0.692949</td>
    </tr>
    <tr>
      <th>10397</th>
      <td>Microsoft’s Surface Book 3 is a powerhouse riv...</td>
      <td>0.692949</td>
    </tr>
    <tr>
      <th>10399</th>
      <td>Microsoft’s Surface Book 3 is a powerhouse riv...</td>
      <td>0.692949</td>
    </tr>
    <tr>
      <th>6051</th>
      <td>Analyst reiterates 2021 launch for Mini-LED iP...</td>
      <td>0.692441</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>


