

```python
# Load deduplicated json file
import json
airlines=open("C:/Users/tramh/github/Data-Science-Portfolio/Airlines Covid-19/data/Airlines_dedup.json").readlines()
```


```python
# Extract text from json data
feed_text = []

for feed in airlines:
    a = json.loads(feed)
    feed_text.append(a['text'])
```


```python
import spacy as sp
# Load pre-existing spacy model
nlp = sp.load("en_core_web_sm")
```


```python
article = []
for i in feed_text:
    b = nlp(i)
    article.append(b)
```


```python
#Create the list of Company or Organization entities in the picked article
org_list = []
for line in article:    
    for x in line.ents:
        if x.label_ == 'ORG':
            org_list.append(x.text)
len(org_list)
```




    53628




```python
# Create the list of USA airlines. 
airlines_list = ['Alaska','Allegiant','American Air','Boeing','Delta','Frontier','Hawaiian',
                'JetBlue','Southwest','Spirit','United Air']
```


```python
# Count the US Airlines mentioned in articles
import pandas as pd

orgs = pd.DataFrame(org_list)

orgcount = orgs.stack().value_counts()
```


```python
orgcount[:20]
```




    United                2108
    American Airlines     2087
    United Airlines       1118
    Southwest Airlines     938
    Delta                  754
    Boeing                 709
    Alaska Airlines        683
    Spirit Airlines        608
    COVID-19               544
    JetBlue                350
    Airbus                 336
    American               324
    NYSE                   304
    Delta Air Lines        304
    Reuters                291
    COVID                  261
    IATA                   253
    Hawaiian Airlines      244
    Air Canada             238
    LUV                    236
    dtype: int64




```python
# America Airlines mentioned in articles within the last 30 days 
import re
y=[]
z=[]
air = []
cnt = []
for x in airlines_list:
    count = sum(orgcount.filter(regex=re.compile(x, re.IGNORECASE)))
    if x[-3:]=='Air':
        y='lines'
    else:
        y= ' Airlines'
    z = x+y
    air.append(z)
    cnt.append(count)
    
    print('For ' +z + ", the number of mentions is " + str(count))
```

    For Alaska Airlines, the number of mentions is 965
    For Allegiant Airlines, the number of mentions is 70
    For American Airlines, the number of mentions is 2733
    For Boeing Airlines, the number of mentions is 781
    For Delta Airlines, the number of mentions is 1331
    For Frontier Airlines, the number of mentions is 291
    For Hawaiian Airlines, the number of mentions is 320
    For JetBlue Airlines, the number of mentions is 506
    For Southwest Airlines, the number of mentions is 1353
    For Spirit Airlines, the number of mentions is 913
    For United Airlines, the number of mentions is 1409
    


```python
prop_list = []
for i in cnt:
    prop = i/len(feed_text)*100
    prop_list.append(prop)
```


```python
us_air_df = pd.DataFrame(list(zip(air, cnt, prop_list)),columns =['Airlines', 'Count', 'Percentage'])

us_air_df = us_air_df.sort_values(by = 'Count', ascending = False) 
us_air_df
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
      <th>Airlines</th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>American Airlines</td>
      <td>2733</td>
      <td>20.485721</td>
    </tr>
    <tr>
      <th>10</th>
      <td>United Airlines</td>
      <td>1409</td>
      <td>10.561427</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Southwest Airlines</td>
      <td>1353</td>
      <td>10.141669</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Delta Airlines</td>
      <td>1331</td>
      <td>9.976763</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alaska Airlines</td>
      <td>965</td>
      <td>7.233341</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Spirit Airlines</td>
      <td>913</td>
      <td>6.843565</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Boeing Airlines</td>
      <td>781</td>
      <td>5.854134</td>
    </tr>
    <tr>
      <th>7</th>
      <td>JetBlue Airlines</td>
      <td>506</td>
      <td>3.792819</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hawaiian Airlines</td>
      <td>320</td>
      <td>2.398621</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Frontier Airlines</td>
      <td>291</td>
      <td>2.181246</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Allegiant Airlines</td>
      <td>70</td>
      <td>0.524698</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
us_air_df.plot(kind= 'bar',x='Airlines', y= 'Percentage', title = 'Proportion of US Airlines Media Mentions')
plt.show()
```


    <Figure size 640x480 with 1 Axes>



```python
# Proportion of Covid keyword mentions in the dataset 
sum(orgcount.filter(regex=re.compile('COVID', re.IGNORECASE)))/len(feed_text)
```




    0.06423806311370962


