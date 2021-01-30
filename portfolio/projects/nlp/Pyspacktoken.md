
# Word2Vec and Pyspark Similarity

Process and tokenize Webhose article bodies and train a Word2Vec model  using Spark MLLib library. Demonstrate a search query implementation and retrieved article titles.

**Word2Vec Pyspark** 


```python
#!pip install pyspark
```

    Collecting pyspark
    [?25l  Downloading https://files.pythonhosted.org/packages/8e/b0/bf9020b56492281b9c9d8aae8f44ff51e1bc91b3ef5a884385cb4e389a40/pyspark-3.0.0.tar.gz (204.7MB)
    [K     |████████████████████████████████| 204.7MB 64kB/s 
    [?25hCollecting py4j==0.10.9
    [?25l  Downloading https://files.pythonhosted.org/packages/9e/b6/6a4fb90cd235dc8e265a6a2067f2a2c99f0d91787f06aca4bcf7c23f3f80/py4j-0.10.9-py2.py3-none-any.whl (198kB)
    [K     |████████████████████████████████| 204kB 43.0MB/s 
    [?25hBuilding wheels for collected packages: pyspark
      Building wheel for pyspark (setup.py) ... [?25l[?25hdone
      Created wheel for pyspark: filename=pyspark-3.0.0-py2.py3-none-any.whl size=205044182 sha256=5321e96b41dad0c1710dfd6d8b284c2ea3b9ed9070e442b86f6e000fde174478
      Stored in directory: /root/.cache/pip/wheels/57/27/4d/ddacf7143f8d5b76c45c61ee2e43d9f8492fc5a8e78ebd7d37
    Successfully built pyspark
    Installing collected packages: py4j, pyspark
    Successfully installed py4j-0.10.9 pyspark-3.0.0
    

**Install Libraries**


```python
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
sc = SparkContext() 
sqlContext = SQLContext(sc)
spark = SparkSession(sc)
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
from nltk.stem.wordnet import WordNetLemmatizer
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec

print("Using Apache Spark Version", sc.version)
```

    Using Apache Spark Version 3.0.0
    

**Load obtained dataset of Webhose news articles into a Spark dataframe**


```python
crunchbase_df = sqlContext.read.option("header", "true").option("delimiter", ",") \
                    .option("inferSchema", "true") \
                    .json("/content/sample_data/webhose_apple.json")

```


```python
crunchbase_df.show()
```

    +--------------------+--------------------+--------------------+---------------+--------------------+-------------+--------------------+--------------+--------+-------------+----------+-------------------+------+--------------------+--------------------+--------------------+--------------------+--------------------+
    |              author|             crawled|            entities|external_images|      external_links|highlightText|highlightThreadTitle|highlightTitle|language|ord_in_thread|parent_url|          published|rating|                text|              thread|               title|                 url|                uuid|
    +--------------------+--------------------+--------------------+---------------+--------------------+-------------+--------------------+--------------+--------+-------------+----------+-------------------+------+--------------------+--------------------+--------------------+--------------------+--------------------+
    |   Roland Hutchinson| 2020-06-03 07:10:13|[[[lightning port...|             []|                  []|             |                    |              | english|            0|      null|2020-06-03 07:00:00|  null|Apple is expected...|[US, 59479, https...|New iPad Air may ...|http://omgili.com...|958670c1717dd8f1e...|
    |                    |2020-06-03 07:07:...|[[[cupertino, non...|             []|[https://www.thev...|             |                    |              | english|            0|      null|2020-06-03 06:58:00|  null|iOS 14 Will Suppo...|[US, 847, https:/...|iOS 14 Will Repor...|http://omgili.com...|4aa124a2c78843f84...|
    |                  Ki|2020-06-03 06:55:...|[[[united states,...|             []|[https://www.twit...|             |                    |              | english|            0|      null|2020-06-03 06:52:00|  null|in: News iPhone L...|[PH, 80996, https...|iPhone Looters Be...|http://omgili.com...|26ebb8ab008ed759d...|
    |       Bradley Prior|2020-06-03 06:45:...|[[], [[apple, neg...|             []|[https://bhavukja...|             |                    |              | english|            0|      null|2020-06-03 06:41:00|  null|Bradley Prior 3 J...|[ZA, 5035, https:...|Apple bug exposed...|http://omgili.com...|cb43510b88a39af75...|
    |                    |2020-06-03 06:30:...|[[], [[google, no...|             []|                  []|             |                    |              | english|            0|      null|2020-06-03 06:25:00|  null|French govt's Sto...|[NL,, https://www...|French govt's Sto...|http://omgili.com...|cfe464ff046a7ad47...|
    |     OfficeChai Team|2020-06-03 07:08:...|[[[america, none]...|             []|[https://twitter....|             |                    |              | english|            0|      null|2020-06-03 06:22:00|  null|— Marvel Studios ...|[US, 52557, https...|American Companie...|http://omgili.com...|f96cdd7df78fdcbe1...|
    |   Roland Hutchinson|2020-06-03 06:10:...|[[], [[apple, non...|             []|[http://www.youtu...|             |                    |              | english|            0|      null|2020-06-03 06:01:00|  null|Yesterday we saw ...|[US, 59479, https...|iOS 13.5.1 vs iOS...|http://omgili.com...|67dadbbd72117060c...|
    |      stockwatch.com|2020-06-03 05:56:...|[[[south korea, n...|             []|                  []|             |                    |              | english|            0|      null|2020-06-03 05:45:00|  null|Mr. Ranjeet Sundh...|[CA, 39897, , 1, ...|Mr. Ranjeet Sundh...|http://omgili.com...|d81d04e2538487a10...|
    |                    |2020-06-03 06:00:...|[[[uk, none], [us...|             []|                  []|             |                    |              | english|            0|      null|2020-06-03 05:22:00|  null|YouTube Kids is a...|[US, 70391, http:...|Apple TV Users Ca...|http://omgili.com...|8c3c8567e9b1ed83b...|
    |                    | 2020-06-03 05:19:44|[[[canada, none],...|             []|[https://www.forb...|             |                    |              | english|            0|      null|2020-06-03 05:12:00|  null|Tech giants conde...|[ZA,, https://www...|Tech giants conde...|http://omgili.com...|017660f92bfbef23c...|
    |  BGR — Yoni Heisler|2020-06-03 05:37:...|[[[new york city,...|             []|[https://www.bgr....|             |                    |              | english|            0|      null|2020-06-03 05:09:00|  null|Looters over the ...|[US,, https://vam...|Looters find that...|http://omgili.com...|bd1ccaa4de64e60ca...|
    |      stockwatch.com|2020-06-03 05:56:...|[[[indonesia, non...|             []|[https://globenew...|             |                    |              | english|            0|      null|2020-06-03 05:05:00|  null|Indonesia’s cobal...|[CA, 39897, , 1, ...|Indonesia’s cobal...|http://omgili.com...|131cb15bd740b475a...|
    |        Jaime Toplin| 2020-06-03 05:18:00|[[[us, none]], [[...|             []|                  []|             |                    |              | english|            0|      null|2020-06-03 05:01:00|  null|Two crossed lines...|[US, 221, https:/...|Why are Apple Pay...|http://omgili.com...|ed020e6bb34a3c933...|
    |     Samuel Martinez|2020-06-03 05:48:...|[[[united states,...|             []|                  []|             |                    |              | english|            0|      null|2020-06-03 04:49:00|  null|It seems that the...|[US, 80738, https...|Canalys expects d...|http://omgili.com...|e4ff2b4a02fc20a99...|
    |                    |2020-06-03 05:38:...|[[[new york city,...|             []|[http://www.youtu...|             |                    |              | english|            0|      null|2020-06-03 04:47:00|  null|Home Opinion A Bi...|[US,, https://chr...|Jha'asryel-Akquil...|http://omgili.com...|501d5062e889ee86b...|
    |           IIC Deals|2020-06-03 04:44:...|[[], [[apple, neg...|             []|[https://apple.sj...|             |                    |              | english|            0|      null|2020-06-03 04:42:00|  null|Help support iPho...|[CA, 94202, https...|Apple Canada Refu...|http://omgili.com...|5881df5d587cb529d...|
    |               admin|2020-06-03 05:15:...|[[[us, none]], [[...|             []|[https://cnet.com...|             |                    |              | english|            0|      null|2020-06-03 04:11:00|  null|Michael B. Jordan...|[US,, https://llo...|Michael B. Jordan...|http://omgili.com...|42e26dc0f165a54c9...|
    |                    |2020-06-03 04:19:...|[[[indonesia, non...|             []|[https://boltmeta...|             |                    |              | english|            0|      null|2020-06-03 04:05:00|  null|June 03, 2020 00:...|[US, 34729, http:...|60% of Global Cob...|http://omgili.com...|4f2eba7cf6bb107ff...|
    |Editor - Entertai...| 2020-06-03 04:17:52|[[[gwyneth paltro...|             []|                  []|             |                    |              | english|            0|      null|2020-06-03 04:04:00|  null|Posted by Editor ...|[US,, https://www...|Gwyneth Paltrow's...|http://omgili.com...|3bdb1715bedef5bf1...|
    |                 AFP|2020-06-03 04:00:...|[[[france, none],...|             []|[https://nhsx.nhs...|             |                    |              | english|            0|      null|2020-06-03 03:45:00|  null|French Virus Trac...|[US,, , 1, 0, 202...|French Virus Trac...|http://omgili.com...|df16e4f695e565001...|
    +--------------------+--------------------+--------------------+---------------+--------------------+-------------+--------------------+--------------+--------+-------------+----------+-------------------+------+--------------------+--------------------+--------------------+--------------------+--------------------+
    only showing top 20 rows
    
    

**Cleans up and tokenizes article bodies using the RegexTokenizer and Stopword remover functions**


```python
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load( disable=['parser', 'tagger','ner'] )

def cleanup_pretokenize(text):
    #text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'http\S+', '', text)
    text = text.replace("'s", " ")
    text = text.replace("n't", " not ")
    text = text.replace("'ve", " have ")
    text = text.replace("'re", " are ")
    text = text.replace("I'm"," I am ")
    text = text.replace("you're"," you are ")
    text = text.replace("You're"," You are ")
    text = text.replace("-"," ")
    text = text.replace("/"," ")
    text = text.replace("("," ")
    text = text.replace(")"," ")
    text = text.replace("%"," percent ")
    return text

lmtzr = WordNetLemmatizer()
def text_cleanup(row):
    desc = row[2].strip().lower()
    tokens = [w.lemma_ for w in nlp(cleanup_pretokenize(desc))]
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if len(token) > 3]
    #tokens = [lmtzr.lemmatize(token,'v') for token in tokens]
    row[2] = ' '.join(tokens)
    return row

regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'text', outputCol = 'tokens')
swr = StopWordsRemover(inputCol = 'tokens', outputCol = 'tokens_sw_removed')
```


```python
crunchbase_data = crunchbase_df['uuid','title','text']
```


```python
df_tokens = regexTokenizer.transform(crunchbase_data)
desc_swr = swr.transform(df_tokens)
desc_swr.show(3)
```

    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |                uuid|               title|                text|              tokens|   tokens_sw_removed|
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |958670c1717dd8f1e...|New iPad Air may ...|Apple is expected...|[apple, is, expec...|[apple, expected,...|
    |4aa124a2c78843f84...|iOS 14 Will Repor...|iOS 14 Will Suppo...|[ios, 14, will, s...|[ios, 14, support...|
    |26ebb8ab008ed759d...|iPhone Looters Be...|in: News iPhone L...|[in, news, iphone...|[news, iphone, lo...|
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    only showing top 3 rows
    
    

**Trains a Word2Vec model based on the text column**


```python
def cossim(v1, v2): 
    return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1)) / (np.sqrt(np.dot(v2, v2))+.1)
```


```python
word2vec = Word2Vec(vectorSize = 300, minCount = 5, inputCol = 'tokens_sw_removed', outputCol = 'wordvectors')
model = word2vec.fit(desc_swr)
wordvectors = model.transform(desc_swr)
#wordvectors.select('wordvectors').show(1, truncate = True)
crunchbase_desc = wordvectors.select('uuid','title','wordvectors').rdd.toDF()
crunchbase_desc.show(10)
```

    +--------------------+--------------------+--------------------+
    |                uuid|               title|         wordvectors|
    +--------------------+--------------------+--------------------+
    |958670c1717dd8f1e...|New iPad Air may ...|[-0.0327451773791...|
    |4aa124a2c78843f84...|iOS 14 Will Repor...|[0.02522358425096...|
    |26ebb8ab008ed759d...|iPhone Looters Be...|[-0.0403018059497...|
    |cb43510b88a39af75...|Apple bug exposed...|[-0.0362135766530...|
    |cfe464ff046a7ad47...|French govt's Sto...|[-0.0102957411447...|
    |f96cdd7df78fdcbe1...|American Companie...|[-0.0279899754667...|
    |67dadbbd72117060c...|iOS 13.5.1 vs iOS...|[0.06957130055058...|
    |d81d04e2538487a10...|Mr. Ranjeet Sundh...|[0.02859138866347...|
    |8c3c8567e9b1ed83b...|Apple TV Users Ca...|[-0.0809914982685...|
    |017660f92bfbef23c...|Tech giants conde...|[-0.0061011842118...|
    +--------------------+--------------------+--------------------+
    only showing top 10 rows
    
    


```python
synonyms = model.findSynonyms("tiktok", 10)   
synonyms.show()
```

    +---------------+-------------------+
    |           word|         similarity|
    +---------------+-------------------+
    |       telegram| 0.5649581551551819|
    |      flipboard| 0.5230107307434082|
    |        tiktook|   0.51828932762146|
    |      instagram| 0.5145618915557861|
    |      smartnews|0.49309733510017395|
    |       kuaishou| 0.4913390278816223|
    |davidphelan2009| 0.4827902317047119|
    |         douyin| 0.4825259745121002|
    |davidphelantech|   0.47369584441185|
    |      pinterest| 0.4644016921520233|
    +---------------+-------------------+
    
    

**Implements any sample search query**


```python
#chunk = crunchbase_desc.filter(lambda r: r[1]>=0 and r[1]<1000).collect()
chunk = crunchbase_desc.take(50000)
#chunk = crunchbase_desc.collect()
```


```python
SEARCH_QUERY = "I love bacon cheeseburger"
```


```python
query_df  = sc.parallelize([(1,SEARCH_QUERY)]).toDF(['index','text'])
query_tok = regexTokenizer.transform(query_df)
query_swr = swr.transform(query_tok)
query_swr.show()
query_vec = model.transform(query_swr)
query_vec = query_vec.select('wordvectors').collect()[0][0]
query_vec
```

    +-----+--------------------+--------------------+--------------------+
    |index|                text|              tokens|   tokens_sw_removed|
    +-----+--------------------+--------------------+--------------------+
    |    1|I love bacon chee...|[i, love, bacon, ...|[love, bacon, che...|
    +-----+--------------------+--------------------+--------------------+
    
    




    DenseVector([-0.055, 0.0312, 0.0163, 0.0527, 0.0418, 0.0508, -0.1263, 0.0107, 0.0522, -0.1525, 0.0123, -0.0097, 0.0569, 0.0047, 0.0283, -0.0125, 0.0006, 0.001, 0.0009, 0.0716, -0.0026, 0.071, -0.0728, 0.0201, -0.0875, -0.0735, 0.0095, 0.0253, 0.0196, 0.0719, -0.1274, -0.0668, -0.0557, 0.0356, 0.033, -0.123, 0.0218, -0.0357, -0.0067, -0.0225, 0.0221, 0.0257, 0.0527, -0.0662, -0.0231, -0.0444, 0.0524, -0.0056, 0.0372, 0.0236, -0.0226, -0.0675, 0.0672, -0.0502, 0.0527, 0.0642, 0.0426, 0.0082, 0.0087, -0.0067, -0.0047, -0.1092, -0.0528, -0.0891, 0.0583, 0.0541, -0.1014, -0.0297, 0.0105, 0.0242, 0.0326, 0.0618, 0.0171, -0.0027, -0.0869, 0.0105, -0.0812, -0.0343, 0.0341, -0.0492, -0.0291, 0.0042, 0.0531, -0.029, -0.0143, 0.0987, -0.0951, 0.0298, -0.0423, 0.0441, -0.03, -0.0296, -0.0522, 0.0748, -0.0488, -0.013, 0.0139, 0.0012, 0.0048, 0.0937, -0.0365, -0.0621, 0.0461, 0.0474, -0.0746, -0.0675, 0.1117, 0.1574, -0.004, -0.0032, -0.0663, 0.0301, -0.0092, -0.0226, 0.0318, -0.0071, 0.0472, 0.0253, -0.0195, 0.0124, 0.0296, 0.0812, -0.0581, 0.0697, 0.0065, 0.0147, -0.0203, 0.0391, 0.0681, 0.0388, 0.0885, -0.0669, 0.1094, -0.0257, -0.013, -0.0136, -0.0856, -0.023, 0.0381, 0.0263, 0.0193, -0.035, -0.0601, 0.0139, 0.099, -0.0082, -0.0068, 0.0743, -0.0242, -0.0035, 0.0102, -0.0074, -0.0147, 0.0166, -0.0458, -0.0605, -0.0748, -0.0493, 0.0284, -0.1023, -0.089, -0.0057, -0.0807, -0.0024, 0.0214, -0.1085, 0.0529, -0.0277, -0.0576, -0.0489, -0.0187, 0.0218, 0.0036, -0.0207, -0.0239, -0.1174, 0.0312, 0.0559, -0.0212, -0.0157, 0.0267, -0.0171, 0.1107, -0.022, -0.0403, -0.0918, -0.0321, -0.0374, 0.0202, 0.1119, 0.0071, -0.0516, -0.0036, 0.0713, -0.0249, 0.0415, 0.0272, -0.0391, 0.0698, -0.063, 0.029, 0.066, -0.0246, -0.0156, -0.0932, 0.0174, 0.0525, 0.0109, -0.0905, 0.0668, 0.0232, 0.0342, -0.0391, 0.0294, 0.0709, 0.0004, 0.0063, 0.0446, -0.0764, -0.0833, 0.018, -0.036, 0.1279, 0.0033, -0.0393, 0.0149, -0.1127, 0.0451, -0.006, -0.0069, -0.0356, 0.0602, -0.062, 0.0623, 0.0153, -0.007, -0.0054, -0.028, -0.0283, 0.0314, 0.0254, -0.1691, -0.0286, -0.0474, -0.046, 0.0456, -0.0174, -0.1011, -0.0151, 0.0098, 0.032, -0.0615, -0.0019, 0.038, 0.006, -0.0289, -0.113, 0.0373, 0.0646, 0.0061, -0.0295, -0.0496, 0.0563, 0.0328, -0.0078, 0.0312, 0.0446, 0.094, -0.0057, 0.0022, -0.0086, -0.0161, 0.0138, -0.0571, 0.0551, -0.0713, -0.0724, 0.0019, 0.0405, -0.051, 0.0004, 0.1092, -0.0187, 0.006, -0.026, -0.0446, 0.0055, -0.0369, 0.0508, -0.0417, 0.0828, -0.0221, -0.0353, 0.0158, -0.0099, -0.053, -0.0796, 0.095, -0.0102, 0.053])



**Produces matching article titles**


```python
import numpy as np
sim_rdd = sc.parallelize((i[0], i[1], float(cossim(query_vec, i[2]))) for i in chunk)
sim_df  = sqlContext.createDataFrame(sim_rdd).\
                   withColumnRenamed('_1', 'crunchbase_uuid').\
                   withColumnRenamed('_2', 'title').\
                   withColumnRenamed('_3', 'similarity').\
                   orderBy("similarity", ascending = False)
sim_df.show(5, truncate = False)
```

    +----------------------------------------+--------------------------------------------------------------------+------------------+
    |crunchbase_uuid                         |title                                                               |similarity        |
    +----------------------------------------+--------------------------------------------------------------------+------------------+
    |a4be8c58a900b7afb1ba4a2434484d5e0d503b78|SpongeBob: Patty Pursuit Launches on Apple Arcade                   |0.4610071773652834|
    |b31aca513fbe1448f5dabe4d73a91caf7a1ccf2f|Apple TV+’s Central Park Finds the Musical Joy in Being Outside     |0.4521597186787441|
    |5926befef806aa08a1bbb0dca8707259a5d0b628|Music and Family Combine in Apple TV's Wonderful Central Park       |0.4511184847199503|
    |61f5ad11fa2647c4fa4c05df22e9aed3d0ec9421|Review: ‘Central Park’ is a Refreshing Shot of Family-Friendly Funny|0.4462476544760373|
    |ba678715246f4a0d2b0ede51ebdae96a7466b5a0|Stars say 'Central Park' celebrates family, musicals, being outside |0.4443306144002278|
    +----------------------------------------+--------------------------------------------------------------------+------------------+
    only showing top 5 rows
    
    
