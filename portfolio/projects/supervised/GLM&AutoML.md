
# Supervised Machine Learning with GLM and AutoML

Tram Duong
<br>November 23, 2020
<br>APAN-5420

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Supervised-Machine-Learning-with-GLM-and-AutoML" data-toc-modified-id="Supervised-Machine-Learning-with-GLM-and-AutoML-1">Supervised Machine Learning with GLM and AutoML</a></span><ul class="toc-item"><li><span><a href="#Table-of-Contents:" data-toc-modified-id="Table-of-Contents:-1.1">Table of Contents:</a></span></li><li><span><a href="#Part-1:-EDA-and-FE-" data-toc-modified-id="Part-1:-EDA-and-FE--1.2">Part 1: EDA and FE <a class="anchor" id="Part_1"></a></a></span><ul class="toc-item"><li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-1.2.1">Data Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Exploring-the-data" data-toc-modified-id="Exploring-the-data-1.2.1.1">Exploring the data</a></span></li><li><span><a href="#The-two-columns--unnamed-0-and-unnamed-0.1-do--not-contain-any-important-information-for-our-target,-leading-me-to-drop-them." data-toc-modified-id="The-two-columns--unnamed-0-and-unnamed-0.1-do--not-contain-any-important-information-for-our-target,-leading-me-to-drop-them.-1.2.1.2">The two columns  unnamed 0 and unnamed 0.1 do  not contain any important information for our target, leading me to drop them.</a></span></li><li><span><a href="#-99-and--999:" data-toc-modified-id="-99-and--999:-1.2.1.3">-99 and -999:</a></span></li></ul></li></ul></li><li><span><a href="#Part-2:-Data-Preparation-" data-toc-modified-id="Part-2:-Data-Preparation--1.3">Part 2: Data Preparation <a class="anchor" id="Part_2"></a></a></span><ul class="toc-item"><li><span><a href="#Feature-Correlation" data-toc-modified-id="Feature-Correlation-1.3.1">Feature Correlation</a></span></li></ul></li><li><span><a href="#Part-3:-Supervised-Learning-with-Generalized-Linear-Models-(GLM)-" data-toc-modified-id="Part-3:-Supervised-Learning-with-Generalized-Linear-Models-(GLM)--1.4">Part 3: Supervised Learning with Generalized Linear Models (GLM) <a class="anchor" id="Part_3"></a></a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Used-functions-throughout-the-modeling-approach" data-toc-modified-id="Used-functions-throughout-the-modeling-approach-1.4.0.1">Used functions throughout the modeling approach</a></span></li><li><span><a href="#Installing:-H2O" data-toc-modified-id="Installing:-H2O-1.4.0.2">Installing: H2O</a></span></li><li><span><a href="#Split-train-test-data" data-toc-modified-id="Split-train-test-data-1.4.0.3">Split train-test data</a></span></li><li><span><a href="#Model-1:" data-toc-modified-id="Model-1:-1.4.0.4">Model 1:</a></span></li></ul></li><li><span><a href="#Model-2:" data-toc-modified-id="Model-2:-1.4.1">Model 2:</a></span></li><li><span><a href="#Model-3:-Hyperparameter-Tuning" data-toc-modified-id="Model-3:-Hyperparameter-Tuning-1.4.2">Model 3: Hyperparameter Tuning</a></span></li><li><span><a href="#AutoML" data-toc-modified-id="AutoML-1.4.3">AutoML</a></span><ul class="toc-item"><li><span><a href="#Model-1" data-toc-modified-id="Model-1-1.4.3.1">Model 1</a></span></li><li><span><a href="#Model-2" data-toc-modified-id="Model-2-1.4.3.2">Model 2</a></span></li><li><span><a href="#Ensemble-Exploration" data-toc-modified-id="Ensemble-Exploration-1.4.3.3">Ensemble Exploration</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-1.4.4">Conclusion</a></span></li></ul></li></ul></li></ul></div>

## Part 1: EDA and FE <a class="anchor" id="Part_1"></a>
- Data Exploration
- Data Cleaning
- Feature Engineerings


```python
#!pip install h2o
```


```python
import numpy as np
import datetime
import h2o 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import roc_curve,auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.automl import H2OAutoML

import warnings
warnings.filterwarnings('ignore')
```

    C:\Users\tramh\Anaconda3\lib\site-packages\statsmodels\tools\_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    


```python
# Read data
data = pd.read_csv("/Github/Data/XYZloan_default_selected_vars.csv")
```

### Data Preprocessing

#### Exploring the data
*Following cell output was not included to save space in report*


```python
#data.info()
#data.describe()
#data.head()
```

#### The two columns  unnamed 0 and unnamed 0.1 do  not contain any important information for our target, leading me to drop them.


```python
data = data.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
```

#### -99 and -999:

By looking through the dataset, there are numerious amount for "-99" and "-999" in the data. These values are most likely NAs that are handled differently by separate systems and seem to be hold no actual value. Thus, I will replace them with **na** for futher analysis. 


```python
data = data.replace(-99, np.nan)
data = data.replace(-999, np.nan)
```


```python
#data.head()
```

**Null Values**


```python
# Proportuib of null value for the entire data
null_prop = round(sum(data.isnull().sum()) /(data.shape[1] * data.shape[0]),2) * 100
print("There is " + str(null_prop) + " % of the data is missing" )
```

    There is 11.0 % of the data is missing
    


```python
# Proportion of null values in each columns
null_cols = data[data.columns[data.isnull().any()]].isnull().sum() * 100 / data.shape[0]
null_cols = null_cols.sort_values(ascending=False)
```


```python
null_cols[:10]
```




    TD048    99.99875
    TD055    99.99875
    TD062    99.99875
    TD044    99.99625
    TD051    99.99375
    TD061    99.98750
    TD054    99.98500
    TD022    24.49750
    TD023     9.57875
    TD024     4.61750
    dtype: float64



From this information, we can see that some features won't be relevant in our analysis as there are too many missing values (over 99% of the data is null). Therefore, I removed those variables as they do not provide useful information to work with.


```python
# Create a list of dropped columns 
# The count function do not count null value.
data_prep = data[[column for column in data if data[column].count() / len(data) >= 0.1]]
# The list of droping features
null_list = []
for c in data.columns:
    if c not in data_prep.columns:
        null_list.append(c)
```


```python
# Create the new dataframe we will work on
all_col = data.columns.to_list()
selected_columns = [column for column in all_col if column not in set(null_list)]
data = data[selected_columns]
```


```python
# The data now has dropped 9 columns 
data.head()
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
      <th>id</th>
      <th>loan_default</th>
      <th>AP001</th>
      <th>AP002</th>
      <th>AP003</th>
      <th>AP004</th>
      <th>AP005</th>
      <th>AP006</th>
      <th>AP007</th>
      <th>AP008</th>
      <th>...</th>
      <th>CD162</th>
      <th>CD164</th>
      <th>CD166</th>
      <th>CD167</th>
      <th>CD169</th>
      <th>CD170</th>
      <th>CD172</th>
      <th>CD173</th>
      <th>MB005</th>
      <th>MB007</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>31</td>
      <td>2</td>
      <td>1</td>
      <td>12</td>
      <td>2017/7/6 10:21</td>
      <td>ios</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1449.0</td>
      <td>1449.0</td>
      <td>2249.0</td>
      <td>2249.0</td>
      <td>7.0</td>
      <td>IPHONE7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>2017/4/6 12:51</td>
      <td>h5</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WEB</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>33</td>
      <td>1</td>
      <td>4</td>
      <td>12</td>
      <td>2017/7/1 14:11</td>
      <td>h5</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>143.0</td>
      <td>110.0</td>
      <td>8.0</td>
      <td>WEB</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>34</td>
      <td>2</td>
      <td>4</td>
      <td>12</td>
      <td>2017/7/7 10:10</td>
      <td>android</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>OPPO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>47</td>
      <td>2</td>
      <td>1</td>
      <td>12</td>
      <td>2017/7/6 14:37</td>
      <td>h5</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WEB</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



*Now I look at the others null values columns to see how to deal with them*


```python
null_cols = data[data.columns[data.isnull().any()]].isnull().sum() * 100 / data.shape[0]
null_cols = null_cols.sort_values(ascending=False)
#null_cols
```

After running some basic functions and checking the data dictionary, the columns that have missing values are all numeric type and are mostly related to the phone info, call details and credit center. Due to the type and the category of these variables, I assume that the missing values were either never provided to the company or were never recorded. 

For this dataset, instead of removing these missing valua, I will impute 0 to all of them in order to not exclude or mispresent any essential data. Thus, with 0 value, I can assume that there is no info for phone, call, bank, or loan to these users while still keeping the information available to me.


```python
data = data.fillna(0)
```

***Categorical Data and Time Data***

From the dictionary and data exploration, there are 3 variables that not numeric datatypes which are AP005, AP006, and MB007. 
  - AP005: DATETIME
  - AP006: OS_TYPE
  - MB007: MOBILE_BRAND


```python
data.select_dtypes(exclude=['int64', 'float64']).columns
```




    Index(['AP005', 'AP006', 'MB007'], dtype='object')




```python
AP006_df = pd.DataFrame(data.AP006.value_counts())
AP006_df
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
      <th>AP006</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>h5</th>
      <td>44246</td>
    </tr>
    <tr>
      <th>ios</th>
      <td>17159</td>
    </tr>
    <tr>
      <th>android</th>
      <td>17140</td>
    </tr>
    <tr>
      <th>api</th>
      <td>1455</td>
    </tr>
  </tbody>
</table>
</div>



AP006 is good for a categorical column as the entire column only consists of 4 different values.


```python
MB007_df = pd.DataFrame(data.MB007.value_counts())
MB007_df['Cum_Percentage'] = MB007_df.cumsum()/ len(data) *100
#len(MB007_df)
#MB007_df[1:11]
```

*The column MB007 contains 112 categorical values, but there are 11 values contibute to approximately 95% of the data, including the "Noinfo" value. Thus, I renamed the other 5% of categorical values as "Other"*


```python
data['MB007'] = data['MB007'].apply(lambda i: i if i in MB007_df[:11].index else 'Other')
```


```python
from sklearn.preprocessing import LabelEncoder
def create_dummies(df):
    for i in df.columns:
        if df[i].dtypes=='object':
            print(i,'encoded')
            mask = ~df[i].isnull()
            label_encoder = LabelEncoder() 
            try:
                df[i][mask] = label_encoder.fit_transform(df[i][mask])
                df[i] = df[i].astype(int)
            except Exception as e:
                print(e)
    return df
```


```python
data = create_dummies(data)
```

    AP005 encoded
    AP006 encoded
    MB007 encoded
    


```python
# AP004 is for loan term appication which only contains value of: 3,6,9,12. 
AP004_df = pd.DataFrame(data.AP004.value_counts())
#AP004_df
data = data.drop(columns = "AP004")
```

###### AP004

As the majority of users choose 12 month term, this column does would skew the analysis due to its large proportion value. Therefore, I dropped this column. 

***DateTime values***

The column AP005 represents the data and time recording. Instead of leaving it as be, I broke the information into two separate columns, date and time, to allow for easier formatting before removing the original column.


```python
data['AP005'] = pd.to_datetime(data['AP005'])
data['Date'] = data['AP005'].dt.strftime('%d/%m/%Y')
data['Time'] = data['AP005'].dt.strftime('%H:%M')
data = data.drop(columns = "AP005")
```

**Duplicated values**


```python
# Check duplciation values
duplicate = sum(data.id.duplicated())
print("There is " + str(duplicate) + " duplicated value" )
```

    There is 0 duplicated value
    

## Part 2: Data Preparation <a class="anchor" id="Part_2"></a>


### Feature Correlation 


```python
x = data.drop(columns = "loan_default", axis = 1)
y = data.loan_default.values
```


```python
corr = x.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f28008b940>




![png](/assets/img/glm_automl/output_45_1.png)


In general, feautures that are overly correlated do not improve model efficiency and also affect the performance of linear regression and random forest models, making the learning algorithms slower to create and train. Therefore, I removed highly correlated features to prevent multicollinearity throughout the following function:


```python
# Function to remove collum with high correlation value
def correlation(dataset, threshold):
    """
    Remove columns that do exceeed correlation threshold
    """
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return(dataset)
```


```python
data = correlation(data, 0.80)
data_new = data.drop(columns = 'id')
```

Since the models I am planning only works with numerical features, I will convert whichever strings the data may contain to numeric values. Also, the date, time, object data do not work in these model. Therefore, I will drop these columns before working on the models. 


```python
data_clean = data_new.drop(columns = ["Date", "Time"])
```


```python
data_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 80000 entries, 0 to 79999
    Data columns (total 47 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   loan_default  80000 non-null  int64  
     1   AP001         80000 non-null  int64  
     2   AP002         80000 non-null  int64  
     3   AP003         80000 non-null  int64  
     4   AP006         80000 non-null  int32  
     5   AP007         80000 non-null  int64  
     6   AP008         80000 non-null  int64  
     7   AP009         80000 non-null  int64  
     8   TD001         80000 non-null  int64  
     9   TD002         80000 non-null  int64  
     10  TD005         80000 non-null  int64  
     11  TD006         80000 non-null  int64  
     12  TD013         80000 non-null  int64  
     13  TD014         80000 non-null  int64  
     14  TD015         80000 non-null  int64  
     15  TD023         80000 non-null  float64
     16  TD024         80000 non-null  float64
     17  TD025         80000 non-null  float64
     18  TD026         80000 non-null  float64
     19  TD027         80000 non-null  float64
     20  TD028         80000 non-null  float64
     21  TD029         80000 non-null  float64
     22  CR004         80000 non-null  int64  
     23  CR005         80000 non-null  int64  
     24  CR009         80000 non-null  int64  
     25  CR012         80000 non-null  int64  
     26  CR015         80000 non-null  int64  
     27  CR017         80000 non-null  int64  
     28  PA022         80000 non-null  float64
     29  PA028         80000 non-null  float64
     30  PA030         80000 non-null  float64
     31  CD008         80000 non-null  float64
     32  CD018         80000 non-null  float64
     33  CD071         80000 non-null  float64
     34  CD072         80000 non-null  float64
     35  CD088         80000 non-null  float64
     36  CD100         80000 non-null  float64
     37  CD113         80000 non-null  float64
     38  CD115         80000 non-null  float64
     39  CD130         80000 non-null  float64
     40  CD131         80000 non-null  float64
     41  CD152         80000 non-null  float64
     42  CD153         80000 non-null  float64
     43  CD160         80000 non-null  float64
     44  CD166         80000 non-null  float64
     45  MB005         80000 non-null  float64
     46  MB007         80000 non-null  int32  
    dtypes: float64(25), int32(2), int64(20)
    memory usage: 28.1 MB
    

**Distribution Plots**


```python
def plot_X_and_Y(var):

    z= data_clean.groupby(var)['loan_default'].agg(['count','mean']).reset_index() 
    z['count_pcnt'] = z['count']/z['count'].sum()
    x = z[var]
    y_mean = z['mean']
    count_pcnt = z['count_pcnt']
    ind = np.arange(0, len(x))
    width = .5

    fig = plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.bar(ind, count_pcnt, width, color='r')
    # plt.ylabel('X')
    plt.title(var + ' Distribution')
    plt.xticks(ind,x.tolist(), rotation=45)

    plt.subplot(122)
    plt.bar(ind, y_mean, width, color='b')
    #plt.ylabel('Y by X')
    plt.xticks(ind,x.tolist(), rotation=45)
    plt.tight_layout()
    plt.title('Response mean by ' + var)
    plt.show()
```

The clean data has 59 features, which is still a very large number to work with. After further exploration and looking at the variable dictionary, I saw that multiple variables contain a large range of values and thus are not supporting any insight to distribution plot. I removed these variables out of the list of features for distribution plotting.


```python
features_dis = data_clean.drop(columns=['TD025','TD026','TD027','TD028','CR009','CR012',
                                      'PA022','PA028','PA030','CD008','CD018','CD071',
                                      'CD072','CD088','CD100','CD113','CD115','CD130',
                                      'CD131','CD152','CD153','CD160','CD166'])
features_dis = features_dis.columns
```


```python
for i in features_dis:
    plot_X_and_Y(i)
```


![png](/assets/img/glm_automl/output_56_0.png)



![png](/assets/img/glm_automl/output_56_1.png)



![png](/assets/img/glm_automl/output_56_2.png)



![png](/assets/img/glm_automl/output_56_3.png)



![png](/assets/img/glm_automl/output_56_4.png)



![png](/assets/img/glm_automl/output_56_5.png)



![png](/assets/img/glm_automl/output_56_6.png)



![png](/assets/img/glm_automl/output_56_7.png)



![png](/assets/img/glm_automl/output_56_8.png)



![png](/assets/img/glm_automl/output_56_9.png)



![png](/assets/img/glm_automl/output_56_10.png)



![png](/assets/img/glm_automl/output_56_11.png)



![png](/assets/img/glm_automl/output_56_12.png)



![png](/assets/img/glm_automl/output_56_13.png)



![png](/assets/img/glm_automl/output_56_14.png)



![png](/assets/img/glm_automl/output_56_15.png)



![png](/assets/img/glm_automl/output_56_16.png)



![png](/assets/img/glm_automl/output_56_17.png)



![png](/assets/img/glm_automl/output_56_18.png)



![png](/assets/img/glm_automl/output_56_19.png)



![png](/assets/img/glm_automl/output_56_20.png)



![png](/assets/img/glm_automl/output_56_21.png)



![png](/assets/img/glm_automl/output_56_22.png)



![png](/assets/img/glm_automl/output_56_23.png)


## Part 3: Supervised Learning with Generalized Linear Models (GLM) <a class="anchor" id="Part_3"></a>

Generalized Linear Models (GLM) estimate regression models for outcomes following exponential distributions. In addition to the Gaussian (i.e. normal) distribution, these include Poisson, binomial, and gamma distributions. Each serves a different purpose, and depending on distribution and link function choice, can be used either for prediction or classification. 
<br><br>
[Reference](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html)


#### Used functions throughout the modeling approach


```python
def VarImp(model_name):
    
    from sklearn.metrics import roc_curve,auc
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    
    # plot the variable importance
    plt.rcdefaults()
    variables = model_name._model_json['output']['coefficients_table']['names']
    y_pos = np.arange(len(variables))
    fig, ax = plt.subplots(figsize = (6,len(variables)/4))
    scaled_importance = model_name._model_json['output']['coefficients_table']['standardized_coefficients']
    ax.barh(y_pos,scaled_importance,align='center',color='lightblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.set_xlabel('standardized_coefficients')
    ax.set_title('Variable Importance')
    plt.show()
    
def createGains(model):
    predictions = model.predict(test_hex)['p1']
    test_scores = test_hex['loan_default'].cbind(predictions).as_data_frame()

    #sort on prediction (descending), add id, and decile for groups containing 1/10 of datapoints
    test_scores = test_scores.sort_values(by='p1',ascending=False)
    test_scores['row_id'] = range(0,0+len(test_scores))
    test_scores['decile'] = ( test_scores['row_id'] / (len(test_scores)/10) ).astype(int)
    #see count by decile
    test_scores.loc[test_scores['decile'] == 10]=9
    test_scores['decile'].value_counts()

    #create gains table
    gains = test_scores.groupby('decile')['loan_default'].agg(['count','sum'])
    gains.columns = ['count','actual']
    gains

    #add features to gains table
    gains['non_actual'] = gains['count'] - gains['actual']
    gains['cum_count'] = gains['count'].cumsum()
    gains['cum_actual'] = gains['actual'].cumsum()
    gains['cum_non_actual'] = gains['non_actual'].cumsum()
    gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
    gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
    gains['if_random'] = np.max(gains['cum_actual']) /10 
    gains['if_random'] = gains['if_random'].cumsum()
    gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
    gains['K_S'] = np.abs( gains['percent_cum_actual'] -  gains['percent_cum_non_actual'] ) * 100
    gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
    gains = pd.DataFrame(gains)
    return(gains)

def ROC_AUC(my_result,df,target):
    from sklearn.metrics import roc_curve,auc
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    # ROC
    y_actual = df[target].as_data_frame()
    y_pred = my_result.predict(df)['p1'].as_data_frame()
    fpr = list()
    tpr = list()
    roc_auc = list()
    fpr,tpr,_ = roc_curve(y_actual,y_pred)
    roc_auc = auc(fpr,tpr)
    
    # Precision-Recall
    average_precision = average_precision_score(y_actual,y_pred)

    print('')
    print('   * ROC curve: The ROC curve plots the true positive rate vs. the false positive rate')
    print('')
    print('	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy')
    print('')
    print('   * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)')
    print('')
    
    # plotting
    plt.figure(figsize=(10,4))

    # ROC
    plt.subplot(1,2,1)
    plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC curve (aare=%0.2f)' % roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=3,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: AUC={0:0.4f}'.format(roc_auc))
    plt.legend(loc='lower right')

    # Precision-Recall
    plt.subplot(1,2,2)
    precision,recall,_ = precision_recall_curve(y_actual,y_pred)
    plt.step(recall,precision,color='b',alpha=0.2,where='post')
    plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.title('Precision-Recall curve: PR={0:0.4f}'.format(average_precision))
    plt.show()
    
```

#### Installing: H2O


```python
h2o.init()
```

    Checking whether there is an H2O instance running at http://localhost:54321 ..... not found.
    Attempting to start a local H2O server...
    ; Java HotSpot(TM) 64-Bit Server VM (build 14.0.1+7, mixed mode, sharing)
      Starting server from C:\Users\tramh\Anaconda3\lib\site-packages\h2o\backend\bin\h2o.jar
      Ice root: C:\Users\tramh\AppData\Local\Temp\tmpd5jsdjvp
      JVM stdout: C:\Users\tramh\AppData\Local\Temp\tmpd5jsdjvp\h2o_tramh_started_from_python.out
      JVM stderr: C:\Users\tramh\AppData\Local\Temp\tmpd5jsdjvp\h2o_tramh_started_from_python.err
      Server is running at http://127.0.0.1:54321
    Connecting to H2O server at http://127.0.0.1:54321 ... successful.
    


<div style="overflow:auto"><table style="width:50%"><tr><td>H2O_cluster_uptime:</td>
<td>01 secs</td></tr>
<tr><td>H2O_cluster_timezone:</td>
<td>America/New_York</td></tr>
<tr><td>H2O_data_parsing_timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O_cluster_version:</td>
<td>3.30.1.3</td></tr>
<tr><td>H2O_cluster_version_age:</td>
<td>1 month and 24 days </td></tr>
<tr><td>H2O_cluster_name:</td>
<td>H2O_from_python_tramh_91nxxh</td></tr>
<tr><td>H2O_cluster_total_nodes:</td>
<td>1</td></tr>
<tr><td>H2O_cluster_free_memory:</td>
<td>3.967 Gb</td></tr>
<tr><td>H2O_cluster_total_cores:</td>
<td>8</td></tr>
<tr><td>H2O_cluster_allowed_cores:</td>
<td>8</td></tr>
<tr><td>H2O_cluster_status:</td>
<td>accepting new members, healthy</td></tr>
<tr><td>H2O_connection_url:</td>
<td>http://127.0.0.1:54321</td></tr>
<tr><td>H2O_connection_proxy:</td>
<td>{"http": null, "https": null, "no": "192.168.99.100"}</td></tr>
<tr><td>H2O_internal_security:</td>
<td>False</td></tr>
<tr><td>H2O_API_Extensions:</td>
<td>Amazon S3, Algos, AutoML, Core V3, TargetEncoder, Core V4</td></tr>
<tr><td>Python_version:</td>
<td>3.7.3 final</td></tr></table></div>


#### Split train-test data 


```python
train, test = train_test_split(
    data_clean, test_size=0.30, random_state=23)

target = 'loan_default'
predictors = train.columns[1:]
```


```python
# full data
train_full = h2o.H2OFrame(train)
test_full = h2o.H2OFrame(test)

# sample data to test 
train_smpl = train.sample(frac=0.1, random_state=1)
test_smpl = test.sample(frac=0.1, random_state=1)
train_hex = h2o.H2OFrame(train_smpl)
test_hex = h2o.H2OFrame(test_smpl)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    

#### Model 1: 

I started by modeling using the train and test sample dataset. This approach helps to define the code performance before applying them into the entire dataset.

For defining parameters, I used the following below:
 + family= "binomial": target variable has binary values. The "binomial" value require  the response must be categorical 2 levels/classes or binary
 + alpha=.05 = Distribution of regularization between the L1 (Lasso) and L2 (Ridge) penalties
 + balance_classes=True =Balance training data class counts via over/under-sampling (for imbalanced data).
 + early_stopping=True = Stop early when there is no more relative improvement on train or validation 

Given that we have an highly unbalanced dataset, I'm using the H2O balance_classes. The balance_classes option can be used to balance the class distribution. When enabled, H2O will either undersample the majority classes or oversample the minority classes. Additionally, other key parameters were achieved through my research using the references link below. Additionally, after some running tests, with the data we have, the lower the alpha is, the better the lift score is. 

[Reference](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2ogeneralizedlinearestimator)


```python
glm_v1 = H2OGeneralizedLinearEstimator(family= "binomial",
                                       alpha=.05, #Distribution of regularization between the L1 (Lasso) and L2 (Ridge) penalties
                                       balance_classes=True, # Balance training data class counts via over/under-sampling (for imbalanced data).
                                       seed=1234,
                                       early_stopping=True) #Stop early when there is no more relative improvement on train or validation 
```


```python
glm_v1.train(list(predictors),target,training_frame=train_hex,validation_frame = test_hex)
```

    glm Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
VarImp(glm_v1)
```


![png](/assets/img/glm_automl/output_68_0.png)


The standardized coefficients are the predictor weights of the standardized data and are included only for informational purposes (e.g. to compare relative variable importance). Standardized coefficients are useful for comparing the relative contribution of different predictors to the model. 


```python
# Print the Coefficients table
coefs = glm_v1._model_json['output']['coefficients_table'].as_data_frame()
coefs = pd.DataFrame(coefs)
coefs.sort_values(by='standardized_coefficients',ascending=False)
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
      <th>names</th>
      <th>coefficients</th>
      <th>standardized_coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>CD113</td>
      <td>4.826001e-02</td>
      <td>0.318767</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TD005</td>
      <td>9.932365e-02</td>
      <td>0.275861</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AP008</td>
      <td>1.336872e-01</td>
      <td>0.175250</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PA022</td>
      <td>4.154934e-03</td>
      <td>0.149584</td>
    </tr>
    <tr>
      <th>16</th>
      <td>TD024</td>
      <td>2.234966e-02</td>
      <td>0.137984</td>
    </tr>
    <tr>
      <th>17</th>
      <td>TD029</td>
      <td>1.734215e-01</td>
      <td>0.136451</td>
    </tr>
    <tr>
      <th>39</th>
      <td>CD166</td>
      <td>6.652242e-05</td>
      <td>0.095174</td>
    </tr>
    <tr>
      <th>13</th>
      <td>TD014</td>
      <td>3.389002e-02</td>
      <td>0.084246</td>
    </tr>
    <tr>
      <th>35</th>
      <td>CD131</td>
      <td>9.406307e-04</td>
      <td>0.078624</td>
    </tr>
    <tr>
      <th>27</th>
      <td>CD018</td>
      <td>1.864265e-04</td>
      <td>0.070055</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AP001</td>
      <td>9.577424e-03</td>
      <td>0.067588</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TD002</td>
      <td>7.182254e-02</td>
      <td>0.066058</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PA028</td>
      <td>4.789024e-04</td>
      <td>0.047568</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TD013</td>
      <td>8.891037e-03</td>
      <td>0.045523</td>
    </tr>
    <tr>
      <th>28</th>
      <td>CD071</td>
      <td>5.515946e-04</td>
      <td>0.033731</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CR017</td>
      <td>1.273239e-02</td>
      <td>0.030680</td>
    </tr>
    <tr>
      <th>34</th>
      <td>CD130</td>
      <td>1.726869e-04</td>
      <td>0.014643</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CR004</td>
      <td>8.553618e-03</td>
      <td>0.009473</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TD006</td>
      <td>5.167950e-03</td>
      <td>0.007228</td>
    </tr>
    <tr>
      <th>37</th>
      <td>CD153</td>
      <td>3.959519e-07</td>
      <td>0.005867</td>
    </tr>
    <tr>
      <th>38</th>
      <td>CD160</td>
      <td>1.943266e-04</td>
      <td>0.003436</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TD001</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>CD088</td>
      <td>-1.909189e-04</td>
      <td>-0.014176</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CR005</td>
      <td>-1.888491e-02</td>
      <td>-0.020752</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PA030</td>
      <td>-4.461517e-04</td>
      <td>-0.034699</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TD015</td>
      <td>-4.699530e-02</td>
      <td>-0.041611</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CR009</td>
      <td>-1.088527e-06</td>
      <td>-0.064896</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AP007</td>
      <td>-5.048598e-02</td>
      <td>-0.067758</td>
    </tr>
    <tr>
      <th>41</th>
      <td>MB007</td>
      <td>-2.190158e-02</td>
      <td>-0.070522</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TD023</td>
      <td>-9.858185e-03</td>
      <td>-0.072158</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CD008</td>
      <td>-6.645931e-06</td>
      <td>-0.083144</td>
    </tr>
    <tr>
      <th>33</th>
      <td>CD115</td>
      <td>-2.110097e-03</td>
      <td>-0.110122</td>
    </tr>
    <tr>
      <th>29</th>
      <td>CD072</td>
      <td>-2.027435e-03</td>
      <td>-0.115914</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CR015</td>
      <td>-1.234391e-01</td>
      <td>-0.136403</td>
    </tr>
    <tr>
      <th>36</th>
      <td>CD152</td>
      <td>-8.234768e-06</td>
      <td>-0.139390</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AP002</td>
      <td>-3.004534e-01</td>
      <td>-0.139917</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AP009</td>
      <td>-3.465978e-01</td>
      <td>-0.153405</td>
    </tr>
    <tr>
      <th>40</th>
      <td>MB005</td>
      <td>-5.368420e-02</td>
      <td>-0.198963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AP006</td>
      <td>-2.000011e-01</td>
      <td>-0.201458</td>
    </tr>
    <tr>
      <th>31</th>
      <td>CD100</td>
      <td>-9.348230e-03</td>
      <td>-0.283871</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AP003</td>
      <td>-3.039011e-01</td>
      <td>-0.366838</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>-1.091871e+00</td>
      <td>-1.592336</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions = glm_v1.predict(test_hex)
predictions.head()
test_scores = test_hex['loan_default'].cbind(predictions).as_data_frame()
test_scores.head()
```

    glm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <th>loan_default</th>
      <th>predict</th>
      <th>p0</th>
      <th>p1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.852239</td>
      <td>0.147761</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0.889551</td>
      <td>0.110449</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0.932246</td>
      <td>0.067754</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0.808106</td>
      <td>0.191894</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0.884863</td>
      <td>0.115137</td>
    </tr>
  </tbody>
</table>
</div>




```python
createGains(glm_v1)
```

    glm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <th>count</th>
      <th>actual</th>
      <th>non_actual</th>
      <th>cum_count</th>
      <th>cum_actual</th>
      <th>cum_non_actual</th>
      <th>percent_cum_actual</th>
      <th>percent_cum_non_actual</th>
      <th>if_random</th>
      <th>lift</th>
      <th>K_S</th>
      <th>gain</th>
    </tr>
    <tr>
      <th>decile</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>240</td>
      <td>96</td>
      <td>144</td>
      <td>240</td>
      <td>96</td>
      <td>144</td>
      <td>0.21</td>
      <td>0.07</td>
      <td>46.8</td>
      <td>2.05</td>
      <td>14.0</td>
      <td>40.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>74</td>
      <td>166</td>
      <td>480</td>
      <td>170</td>
      <td>310</td>
      <td>0.36</td>
      <td>0.16</td>
      <td>93.6</td>
      <td>1.82</td>
      <td>20.0</td>
      <td>35.42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>52</td>
      <td>188</td>
      <td>720</td>
      <td>222</td>
      <td>498</td>
      <td>0.47</td>
      <td>0.26</td>
      <td>140.4</td>
      <td>1.58</td>
      <td>21.0</td>
      <td>30.83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>58</td>
      <td>182</td>
      <td>960</td>
      <td>280</td>
      <td>680</td>
      <td>0.60</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.50</td>
      <td>25.0</td>
      <td>29.17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>42</td>
      <td>198</td>
      <td>1200</td>
      <td>322</td>
      <td>878</td>
      <td>0.69</td>
      <td>0.45</td>
      <td>234.0</td>
      <td>1.38</td>
      <td>24.0</td>
      <td>26.83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>40</td>
      <td>200</td>
      <td>1440</td>
      <td>362</td>
      <td>1078</td>
      <td>0.77</td>
      <td>0.56</td>
      <td>280.8</td>
      <td>1.29</td>
      <td>21.0</td>
      <td>25.14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>37</td>
      <td>203</td>
      <td>1680</td>
      <td>399</td>
      <td>1281</td>
      <td>0.85</td>
      <td>0.66</td>
      <td>327.6</td>
      <td>1.22</td>
      <td>19.0</td>
      <td>23.75</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>28</td>
      <td>212</td>
      <td>1920</td>
      <td>427</td>
      <td>1493</td>
      <td>0.91</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.14</td>
      <td>14.0</td>
      <td>22.24</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>22</td>
      <td>218</td>
      <td>2160</td>
      <td>449</td>
      <td>1711</td>
      <td>0.96</td>
      <td>0.89</td>
      <td>421.2</td>
      <td>1.07</td>
      <td>7.0</td>
      <td>20.79</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>19</td>
      <td>221</td>
      <td>2400</td>
      <td>468</td>
      <td>1932</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>468.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>19.50</td>
    </tr>
  </tbody>
</table>
</div>



The gain table shows that the lift, K_S, and gain scores are fairly good. For example, let's look at decile 0.

- Lift = decile 0 of model 1 has 2.05 times greater lift than random selection.
- K_S= abs(cumulative % of total good loan applicants— cumulative % of total bad loan applicants) -> The higher the value, the better the model is at separating the positive cases from negative ones. 


```python
ROC_AUC(glm_v1,test_hex,'loan_default')
```

    glm prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false positive rate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/glm_automl/output_74_1.png)


The AUC and PR scores also have positive results (0.6676 and 0.3383). Overall, the parameters used in the first model are more likely to work well.  

### Model 2: 

After testing on small datasets and the results are promising, I apply the same code to the entire dataset. 


```python
glm_v2 = H2OGeneralizedLinearEstimator(family= "binomial",
                                       alpha=.05, #Distribution of regularization between the L1 (Lasso) and L2 (Ridge) penalties
                                       balance_classes=True, # Balance training data class counts via over/under-sampling (for imbalanced data).
                                       seed=1234,
                                       early_stopping=True) #Stop early when there is no more relative improvement on train or validation 
```


```python
glm_v2.train(list(predictors),target,training_frame=train_full,validation_frame = test_full)
```

    glm Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
predictions_2 = glm_v2.predict(test_full)
predictions_2.head()
test_scores_2 = test_full['loan_default'].cbind(predictions_2).as_data_frame()
test_scores_2.head()
```

    glm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <th>loan_default</th>
      <th>predict</th>
      <th>p0</th>
      <th>p1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0.668360</td>
      <td>0.331640</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0.856900</td>
      <td>0.143100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0.834750</td>
      <td>0.165250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0.786682</td>
      <td>0.213318</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0.600507</td>
      <td>0.399493</td>
    </tr>
  </tbody>
</table>
</div>




```python
createGains(glm_v2)
```

    glm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <th>count</th>
      <th>actual</th>
      <th>non_actual</th>
      <th>cum_count</th>
      <th>cum_actual</th>
      <th>cum_non_actual</th>
      <th>percent_cum_actual</th>
      <th>percent_cum_non_actual</th>
      <th>if_random</th>
      <th>lift</th>
      <th>K_S</th>
      <th>gain</th>
    </tr>
    <tr>
      <th>decile</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>240</td>
      <td>95</td>
      <td>145</td>
      <td>240</td>
      <td>95</td>
      <td>145</td>
      <td>0.20</td>
      <td>0.08</td>
      <td>46.8</td>
      <td>2.03</td>
      <td>12.0</td>
      <td>39.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>79</td>
      <td>161</td>
      <td>480</td>
      <td>174</td>
      <td>306</td>
      <td>0.37</td>
      <td>0.16</td>
      <td>93.6</td>
      <td>1.86</td>
      <td>21.0</td>
      <td>36.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>53</td>
      <td>187</td>
      <td>720</td>
      <td>227</td>
      <td>493</td>
      <td>0.49</td>
      <td>0.26</td>
      <td>140.4</td>
      <td>1.62</td>
      <td>23.0</td>
      <td>31.53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>51</td>
      <td>189</td>
      <td>960</td>
      <td>278</td>
      <td>682</td>
      <td>0.59</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.49</td>
      <td>24.0</td>
      <td>28.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>53</td>
      <td>187</td>
      <td>1200</td>
      <td>331</td>
      <td>869</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>234.0</td>
      <td>1.41</td>
      <td>26.0</td>
      <td>27.58</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>30</td>
      <td>210</td>
      <td>1440</td>
      <td>361</td>
      <td>1079</td>
      <td>0.77</td>
      <td>0.56</td>
      <td>280.8</td>
      <td>1.29</td>
      <td>21.0</td>
      <td>25.07</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>35</td>
      <td>205</td>
      <td>1680</td>
      <td>396</td>
      <td>1284</td>
      <td>0.85</td>
      <td>0.66</td>
      <td>327.6</td>
      <td>1.21</td>
      <td>19.0</td>
      <td>23.57</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>33</td>
      <td>207</td>
      <td>1920</td>
      <td>429</td>
      <td>1491</td>
      <td>0.92</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.15</td>
      <td>15.0</td>
      <td>22.34</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>23</td>
      <td>217</td>
      <td>2160</td>
      <td>452</td>
      <td>1708</td>
      <td>0.97</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.07</td>
      <td>9.0</td>
      <td>20.93</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>16</td>
      <td>224</td>
      <td>2400</td>
      <td>468</td>
      <td>1932</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>468.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>19.50</td>
    </tr>
  </tbody>
</table>
</div>



The gain table of model 2 results in similar scores compare to model 1, as well as the AUC and PR scores. 


```python
ROC_AUC(glm_v2,test_full,'loan_default')
```

    glm prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false positive rate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/glm_automl/output_82_1.png)


The model 2 (full data) use the same set of parameters like model 1 (sample data). The results of both models are positive. However, I would like to define the model with Hyperparameter Tuning approach using H2O Grid-search, so that we might be able to optimize the predictive power.

### Model 3: Hyperparameter Tuning 
In this section, I used H2O Grid-search to find the optimal hyper-parameters for the model.

In order to create less complex (parsimonious) model as our data has large number of features, I used some of the regularization techniques to address over-fitting and feature selection.

+ L1 Regularization = Lasso Regression: Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function.
+ L2 Regularization = Ridge Regression: Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function.

The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.

While traditional methods like cross-validation, stepwise regression to handle overfitting, and perform feature selection work well with a small set of features but these techniques are a great alternative when we are dealing with a broad set of features.

The alpha parameter controls the distribution between the 1 (Lasso) and 2 (Ridge regression) penalties. A value of 1.0 for alpha represents Lasso, and an alpha value of 0.0 produces ridge regression. The lambda parameter controls the amount of regularization applied. If lambda is 0.0, no regularization is applied, and the alpha parameter is ignored.

I'm testing out a variety of parameters for L1 and L2 Regularization.

[Reference](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)


```python
hyper_params_tune = {'alpha': [x/100. for x in range(0,100,5)],
                'lambda': [x/100. for x in range(0,100,5)],}


glm_v3 = H2OGeneralizedLinearEstimator(
        family= "binomial",
        balance_classes=True,
        early_stopping=True,
        seed = 1234) 

#Build grid search with previously made GBM and hyper parameters
glm_grid = H2OGridSearch(glm_v3, hyper_params = hyper_params_tune,
                         search_criteria = {'strategy': "Cartesian"})
```


```python
glm_grid.train(list(predictors),target,training_frame=train_hex, validation_frame = test_hex)
```

    glm Grid Build progress: |████████████████████████████████████████████████| 100%
    


```python
sorted_glm_grid = glm_grid.get_grid(sort_by='auc',decreasing=False)
print(sorted_glm_grid)
```

             alpha  lambda  \
    0       [0.95]   [0.2]   
    1       [0.25]  [0.25]   
    2        [0.3]  [0.25]   
    3       [0.35]  [0.25]   
    4        [0.4]  [0.25]   
    ..  ..     ...     ...   
    395      [0.3]   [0.0]   
    396     [0.35]   [0.0]   
    397      [0.4]   [0.0]   
    398     [0.05]  [0.05]   
    399      [0.0]  [0.05]   
    
                                                                     model_ids  \
    0    Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    1    Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    2    Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    3    Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    4    Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    ..                                                                     ...   
    395  Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    396  Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    397  Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    398  Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    399  Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_mo...   
    
                        auc  
    0                   0.5  
    1                   0.5  
    2                   0.5  
    3                   0.5  
    4                   0.5  
    ..                  ...  
    395  0.6674054608837217  
    396  0.6674054608837217  
    397  0.6674054608837217  
    398  0.6681879412857673  
    399  0.6684174320043885  
    
    [400 rows x 5 columns]
    
    

The AUC result in the same set of optimal hyperparameters as in the model summary above.The model metrics also show positive insights for the other statistics.

From the above, I will move forward to train the model with the full data.


```python
best_glm = sorted_glm_grid.models[0]
print(best_glm)
```

    Model Details
    =============
    H2OGeneralizedLinearEstimator :  Generalized Linear Modeling
    Model Key:  Grid_GLM_Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex_model_python_1606060975622_7_model_100
    
    
    GLM Model: summary
    


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
      <th></th>
      <th>family</th>
      <th>link</th>
      <th>regularization</th>
      <th>number_of_predictors_total</th>
      <th>number_of_active_predictors</th>
      <th>number_of_iterations</th>
      <th>training_frame</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>binomial</td>
      <td>logit</td>
      <td>Elastic Net (alpha = 0.95, lambda = 0.2 )</td>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>Key_Frame__upload_a97e7b95be3b0554405055ca3abd40a3.hex</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    ModelMetricsBinomialGLM: glm
    ** Reported on train data. **
    
    MSE: 0.15566326530611185
    RMSE: 0.39454184227545736
    LogLoss: 0.49033943049059775
    Null degrees of freedom: 5599
    Residual degrees of freedom: 5599
    Null deviance: 5491.801621495794
    Residual deviance: 5491.801621494695
    AIC: 5493.801621494695
    AUC: 0.5
    AUCPR: 0.19285714285714287
    Gini: 0.0
    
    Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.19285714285714067: 
    


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
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Error</th>
      <th>Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>4520.0</td>
      <td>1.0</td>
      <td>(4520.0/4520.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0</td>
      <td>1080.0</td>
      <td>0.0</td>
      <td>(0.0/1080.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Total</td>
      <td>0.0</td>
      <td>5600.0</td>
      <td>0.8071</td>
      <td>(4520.0/5600.0)</td>
    </tr>
  </tbody>
</table>
</div>


    
    Maximum Metrics: Maximum metrics at their respective thresholds
    


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
      <th>metric</th>
      <th>threshold</th>
      <th>value</th>
      <th>idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>max f1</td>
      <td>0.192857</td>
      <td>0.323353</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>max f2</td>
      <td>0.192857</td>
      <td>0.544355</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>max f0point5</td>
      <td>0.192857</td>
      <td>0.229983</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>max accuracy</td>
      <td>0.192857</td>
      <td>0.192857</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max precision</td>
      <td>0.192857</td>
      <td>0.192857</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>max recall</td>
      <td>0.192857</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>max specificity</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>max absolute_mcc</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>max min_per_class_accuracy</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>max mean_per_class_accuracy</td>
      <td>0.192857</td>
      <td>0.500000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>max tns</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>max fns</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>max fps</td>
      <td>0.192857</td>
      <td>4520.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>max tps</td>
      <td>0.192857</td>
      <td>1080.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>max tnr</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>max fnr</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>max fpr</td>
      <td>0.192857</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>max tpr</td>
      <td>0.192857</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Gains/Lift Table: Avg response rate: 19.29 %, avg score: 19.29 %
    


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
      <th>group</th>
      <th>cumulative_data_fraction</th>
      <th>lower_threshold</th>
      <th>lift</th>
      <th>cumulative_lift</th>
      <th>response_rate</th>
      <th>score</th>
      <th>cumulative_response_rate</th>
      <th>cumulative_score</th>
      <th>capture_rate</th>
      <th>cumulative_capture_rate</th>
      <th>gain</th>
      <th>cumulative_gain</th>
      <th>kolmogorov_smirnov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.0</td>
      <td>0.192857</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.192857</td>
      <td>0.192857</td>
      <td>0.192857</td>
      <td>0.192857</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    ModelMetricsBinomialGLM: glm
    ** Reported on validation data. **
    
    MSE: 0.15697959183674048
    RMSE: 0.396206501507409
    LogLoss: 0.4934070396750237
    Null degrees of freedom: 2399
    Residual degrees of freedom: 2399
    Null deviance: 2368.3537904401014
    Residual deviance: 2368.3537904401137
    AIC: 2370.3537904401137
    AUC: 0.5
    AUCPR: 0.19499999999999998
    Gini: 0.0
    
    Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.19285714285714067: 
    


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
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>Error</th>
      <th>Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>1932.0</td>
      <td>1.0</td>
      <td>(1932.0/1932.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0</td>
      <td>468.0</td>
      <td>0.0</td>
      <td>(0.0/468.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Total</td>
      <td>0.0</td>
      <td>2400.0</td>
      <td>0.805</td>
      <td>(1932.0/2400.0)</td>
    </tr>
  </tbody>
</table>
</div>


    
    Maximum Metrics: Maximum metrics at their respective thresholds
    


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
      <th>metric</th>
      <th>threshold</th>
      <th>value</th>
      <th>idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>max f1</td>
      <td>0.192857</td>
      <td>0.326360</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>max f2</td>
      <td>0.192857</td>
      <td>0.547753</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>max f0point5</td>
      <td>0.192857</td>
      <td>0.232420</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>max accuracy</td>
      <td>0.192857</td>
      <td>0.195000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max precision</td>
      <td>0.192857</td>
      <td>0.195000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>max recall</td>
      <td>0.192857</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>max specificity</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>max absolute_mcc</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>max min_per_class_accuracy</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>max mean_per_class_accuracy</td>
      <td>0.192857</td>
      <td>0.500000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>max tns</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>max fns</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>max fps</td>
      <td>0.192857</td>
      <td>1932.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>max tps</td>
      <td>0.192857</td>
      <td>468.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>max tnr</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>max fnr</td>
      <td>0.192857</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>max fpr</td>
      <td>0.192857</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>max tpr</td>
      <td>0.192857</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Gains/Lift Table: Avg response rate: 19.50 %, avg score: 19.29 %
    


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
      <th>group</th>
      <th>cumulative_data_fraction</th>
      <th>lower_threshold</th>
      <th>lift</th>
      <th>cumulative_lift</th>
      <th>response_rate</th>
      <th>score</th>
      <th>cumulative_response_rate</th>
      <th>cumulative_score</th>
      <th>capture_rate</th>
      <th>cumulative_capture_rate</th>
      <th>gain</th>
      <th>cumulative_gain</th>
      <th>kolmogorov_smirnov</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.0</td>
      <td>0.192857</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.195</td>
      <td>0.192857</td>
      <td>0.195</td>
      <td>0.192857</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Scoring History: 
    


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
      <th></th>
      <th>timestamp</th>
      <th>duration</th>
      <th>iterations</th>
      <th>negative_log_likelihood</th>
      <th>objective</th>
      <th>training_rmse</th>
      <th>training_logloss</th>
      <th>training_r2</th>
      <th>training_auc</th>
      <th>training_pr_auc</th>
      <th>training_lift</th>
      <th>training_classification_error</th>
      <th>validation_rmse</th>
      <th>validation_logloss</th>
      <th>validation_r2</th>
      <th>validation_auc</th>
      <th>validation_pr_auc</th>
      <th>validation_lift</th>
      <th>validation_classification_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>2020-11-22 11:03:21</td>
      <td>0.000 sec</td>
      <td>0</td>
      <td>2745.900811</td>
      <td>0.490339</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>2020-11-22 11:03:21</td>
      <td>0.000 sec</td>
      <td>1</td>
      <td>2745.900811</td>
      <td>0.490339</td>
      <td>0.394542</td>
      <td>0.490339</td>
      <td>6.81677e-14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>0.807143</td>
      <td>0.396207</td>
      <td>0.493407</td>
      <td>-2.9252e-05</td>
      <td>0.5</td>
      <td>0.195</td>
      <td>1</td>
      <td>0.805</td>
    </tr>
  </tbody>
</table>
</div>


    
    


```python
best_glm.train(list(predictors),target,training_frame=train_full, validation_frame = test_full)
```

    glm Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
# Prediction with the best model
predictions_3 = best_glm.predict(test_full)
predictions_3.head()
test_scores_3 = test_full['loan_default'].cbind(predictions_3).as_data_frame()
test_scores_3.head()
```

    glm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <th>loan_default</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.318183</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.162836</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.169249</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.221212</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.376131</td>
    </tr>
  </tbody>
</table>
</div>




```python
VarImp(best_glm)
```


![png](/assets/img/glm_automl/output_93_0.png)


In the best glm model, the standardized coefficients show that the variables importance levels are very different from the previous ones. Additionally, there are less negative sizes and values and more positive sizes and values among all variables. 


```python
def createGains_1(model):
    predictions = model.predict(test_hex)
    test_scores = test_hex['loan_default'].cbind(predictions).as_data_frame()

    #sort on prediction (descending), add id, and decile for groups containing 1/10 of datapoints
    test_scores = test_scores.sort_values(by='predict',ascending=False)
    test_scores['row_id'] = range(0,0+len(test_scores))
    test_scores['decile'] = ( test_scores['row_id'] / (len(test_scores)/10) ).astype(int)
    #see count by decile
    test_scores.loc[test_scores['decile'] == 10]=9
    test_scores['decile'].value_counts()

    #create gains table
    gains = test_scores.groupby('decile')['loan_default'].agg(['count','sum'])
    gains.columns = ['count','actual']
    gains

    #add features to gains table
    gains['non_actual'] = gains['count'] - gains['actual']
    gains['cum_count'] = gains['count'].cumsum()
    gains['cum_actual'] = gains['actual'].cumsum()
    gains['cum_non_actual'] = gains['non_actual'].cumsum()
    gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
    gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
    gains['if_random'] = np.max(gains['cum_actual']) /10 
    gains['if_random'] = gains['if_random'].cumsum()
    gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
    gains['K_S'] = np.abs( gains['percent_cum_actual'] -  gains['percent_cum_non_actual'] ) * 100
    gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
    gains = pd.DataFrame(gains)
    return(gains)
createGains_1(best_glm)
```

    glm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <th>count</th>
      <th>actual</th>
      <th>non_actual</th>
      <th>cum_count</th>
      <th>cum_actual</th>
      <th>cum_non_actual</th>
      <th>percent_cum_actual</th>
      <th>percent_cum_non_actual</th>
      <th>if_random</th>
      <th>lift</th>
      <th>K_S</th>
      <th>gain</th>
    </tr>
    <tr>
      <th>decile</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>240</td>
      <td>99</td>
      <td>141</td>
      <td>240</td>
      <td>99</td>
      <td>141</td>
      <td>0.21</td>
      <td>0.07</td>
      <td>46.8</td>
      <td>2.12</td>
      <td>14.0</td>
      <td>41.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>71</td>
      <td>169</td>
      <td>480</td>
      <td>170</td>
      <td>310</td>
      <td>0.36</td>
      <td>0.16</td>
      <td>93.6</td>
      <td>1.82</td>
      <td>20.0</td>
      <td>35.42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>50</td>
      <td>190</td>
      <td>720</td>
      <td>220</td>
      <td>500</td>
      <td>0.47</td>
      <td>0.26</td>
      <td>140.4</td>
      <td>1.57</td>
      <td>21.0</td>
      <td>30.56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>58</td>
      <td>182</td>
      <td>960</td>
      <td>278</td>
      <td>682</td>
      <td>0.59</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.49</td>
      <td>24.0</td>
      <td>28.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>54</td>
      <td>186</td>
      <td>1200</td>
      <td>332</td>
      <td>868</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>234.0</td>
      <td>1.42</td>
      <td>26.0</td>
      <td>27.67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>32</td>
      <td>208</td>
      <td>1440</td>
      <td>364</td>
      <td>1076</td>
      <td>0.78</td>
      <td>0.56</td>
      <td>280.8</td>
      <td>1.30</td>
      <td>22.0</td>
      <td>25.28</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>34</td>
      <td>206</td>
      <td>1680</td>
      <td>398</td>
      <td>1282</td>
      <td>0.85</td>
      <td>0.66</td>
      <td>327.6</td>
      <td>1.21</td>
      <td>19.0</td>
      <td>23.69</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>31</td>
      <td>209</td>
      <td>1920</td>
      <td>429</td>
      <td>1491</td>
      <td>0.92</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.15</td>
      <td>15.0</td>
      <td>22.34</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>25</td>
      <td>215</td>
      <td>2160</td>
      <td>454</td>
      <td>1706</td>
      <td>0.97</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.08</td>
      <td>9.0</td>
      <td>21.02</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>14</td>
      <td>226</td>
      <td>2400</td>
      <td>468</td>
      <td>1932</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>468.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>19.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
def ROC_AUC(my_result,df,target):
    from sklearn.metrics import roc_curve,auc
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    # ROC
    y_actual = df[target].as_data_frame()
    y_pred = my_result.predict(df).as_data_frame()
    fpr = list()
    tpr = list()
    roc_auc = list()
    fpr,tpr,_ = roc_curve(y_actual,y_pred)
    roc_auc = auc(fpr,tpr)
    
    # Precision-Recall
    average_precision = average_precision_score(y_actual,y_pred)

    print('')
    print('   * ROC curve: The ROC curve plots the true positive rate vs. the false positive rate')
    print('')
    print('	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy')
    print('')
    print('   * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)')
    print('')
    
    # plotting
    plt.figure(figsize=(10,4))

    # ROC
    plt.subplot(1,2,1)
    plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC curve (aare=%0.2f)' % roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=3,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: AUC={0:0.4f}'.format(roc_auc))
    plt.legend(loc='lower right')

    # Precision-Recall
    plt.subplot(1,2,2)
    precision,recall,_ = precision_recall_curve(y_actual,y_pred)
    plt.step(recall,precision,color='b',alpha=0.2,where='post')
    plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.title('Precision-Recall curve: PR={0:0.4f}'.format(average_precision))
    plt.show()
ROC_AUC(best_glm,test_full,'loan_default')
```

    glm prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false positive rate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/glm_automl/output_96_1.png)


The best model has a better result in the lift score and AUC score compared to other previous models, which is 2.12 for lift score in gain table and 0.66 for AUC score.

### AutoML

Automated machine learning (AutoML) is the process of automating applying machine learning. AutoML tends to automate the maximum number of steps in an ML pipeline - with a minimun amount of human effort - without compromising the model's performance. 

AutoML very broadly includes:
+ Automating certain parts of data preparation, e.g. imputation, standardization, feature selection, etc.
+ Being able to generate various models automatically, e.g. random grid search, Bayesian Hyperparameter Optimization, etc.
+ Getting the best model out of all the generated models, which most of the time is an Ensemble, e.g. ensemble selection, stacking, etc.

H2O AutoML can be used for automating the machine learning workflow, which includes automatic training and tuning of many models within a user-specified time-limit.<br><br>
[References](https://towardsdatascience.com/a-deep-dive-into-h2os-automl-4b1fe51d3f3e)

#### Model 1


```python
aml_v1 = H2OAutoML(max_models=15, seed=1234)#,balance_classes= True)
aml_v1.train(list(predictors),target,training_frame=train_hex, validation_frame = test_hex)
```

    AutoML progress: |
    11:03:57.281: User specified a validation frame with cross-validation still enabled. Please note that the models will still be validated using cross-validation only, the validation frame will be used to provide purely informative validation metrics on the trained models.
    11:03:57.291: AutoML: XGBoost is not available; skipping it.
    
    ████████████████████████████████████████████████████████| 100%
    

The AutoML object includes a “leaderboard” of models that were trained in the process, including the 5-fold cross-validated model performance (by default)


```python
aml_v1.leaderboard.head()
```


<table>
<thead>
<tr><th>model_id                                           </th><th style="text-align: right;">  mean_residual_deviance</th><th style="text-align: right;">    rmse</th><th style="text-align: right;">     mse</th><th style="text-align: right;">     mae</th><th style="text-align: right;">   rmsle</th></tr>
</thead>
<tbody>
<tr><td>StackedEnsemble_AllModels_AutoML_20201122_110357   </td><td style="text-align: right;">                0.145458</td><td style="text-align: right;">0.381389</td><td style="text-align: right;">0.145458</td><td style="text-align: right;">0.291212</td><td style="text-align: right;">0.267751</td></tr>
<tr><td>StackedEnsemble_BestOfFamily_AutoML_20201122_110357</td><td style="text-align: right;">                0.145467</td><td style="text-align: right;">0.381402</td><td style="text-align: right;">0.145467</td><td style="text-align: right;">0.291773</td><td style="text-align: right;">0.267773</td></tr>
<tr><td>GLM_1_AutoML_20201122_110357                       </td><td style="text-align: right;">                0.145611</td><td style="text-align: right;">0.38159 </td><td style="text-align: right;">0.145611</td><td style="text-align: right;">0.292631</td><td style="text-align: right;">0.268104</td></tr>
<tr><td>GBM_5_AutoML_20201122_110357                       </td><td style="text-align: right;">                0.146927</td><td style="text-align: right;">0.383311</td><td style="text-align: right;">0.146927</td><td style="text-align: right;">0.291945</td><td style="text-align: right;">0.269298</td></tr>
<tr><td>GBM_grid__1_AutoML_20201122_110357_model_3         </td><td style="text-align: right;">                0.147296</td><td style="text-align: right;">0.383791</td><td style="text-align: right;">0.147296</td><td style="text-align: right;">0.292889</td><td style="text-align: right;">0.269634</td></tr>
<tr><td>GBM_grid__1_AutoML_20201122_110357_model_2         </td><td style="text-align: right;">                0.148224</td><td style="text-align: right;">0.384999</td><td style="text-align: right;">0.148224</td><td style="text-align: right;">0.294471</td><td style="text-align: right;">0.270336</td></tr>
<tr><td>GBM_1_AutoML_20201122_110357                       </td><td style="text-align: right;">                0.150274</td><td style="text-align: right;">0.387652</td><td style="text-align: right;">0.150274</td><td style="text-align: right;">0.296103</td><td style="text-align: right;">0.273468</td></tr>
<tr><td>GBM_2_AutoML_20201122_110357                       </td><td style="text-align: right;">                0.150334</td><td style="text-align: right;">0.387729</td><td style="text-align: right;">0.150334</td><td style="text-align: right;">0.294454</td><td style="text-align: right;">0.273334</td></tr>
<tr><td>DeepLearning_1_AutoML_20201122_110357              </td><td style="text-align: right;">                0.151614</td><td style="text-align: right;">0.389376</td><td style="text-align: right;">0.151614</td><td style="text-align: right;">0.293723</td><td style="text-align: right;">0.274793</td></tr>
<tr><td>GBM_3_AutoML_20201122_110357                       </td><td style="text-align: right;">                0.151983</td><td style="text-align: right;">0.38985 </td><td style="text-align: right;">0.151983</td><td style="text-align: right;">0.293853</td><td style="text-align: right;">0.274906</td></tr>
</tbody>
</table>





    




```python
createGains_1(aml_v1)
```

    stackedensemble prediction progress: |████████████████████████████████████| 100%
    




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
      <th>count</th>
      <th>actual</th>
      <th>non_actual</th>
      <th>cum_count</th>
      <th>cum_actual</th>
      <th>cum_non_actual</th>
      <th>percent_cum_actual</th>
      <th>percent_cum_non_actual</th>
      <th>if_random</th>
      <th>lift</th>
      <th>K_S</th>
      <th>gain</th>
    </tr>
    <tr>
      <th>decile</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>240</td>
      <td>97</td>
      <td>143</td>
      <td>240</td>
      <td>97</td>
      <td>143</td>
      <td>0.21</td>
      <td>0.07</td>
      <td>46.8</td>
      <td>2.07</td>
      <td>14.0</td>
      <td>40.42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>62</td>
      <td>178</td>
      <td>480</td>
      <td>159</td>
      <td>321</td>
      <td>0.34</td>
      <td>0.17</td>
      <td>93.6</td>
      <td>1.70</td>
      <td>17.0</td>
      <td>33.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>71</td>
      <td>169</td>
      <td>720</td>
      <td>230</td>
      <td>490</td>
      <td>0.49</td>
      <td>0.25</td>
      <td>140.4</td>
      <td>1.64</td>
      <td>24.0</td>
      <td>31.94</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>54</td>
      <td>186</td>
      <td>960</td>
      <td>284</td>
      <td>676</td>
      <td>0.61</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.52</td>
      <td>26.0</td>
      <td>29.58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>46</td>
      <td>194</td>
      <td>1200</td>
      <td>330</td>
      <td>870</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>234.0</td>
      <td>1.41</td>
      <td>26.0</td>
      <td>27.50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>38</td>
      <td>202</td>
      <td>1440</td>
      <td>368</td>
      <td>1072</td>
      <td>0.79</td>
      <td>0.55</td>
      <td>280.8</td>
      <td>1.31</td>
      <td>24.0</td>
      <td>25.56</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>32</td>
      <td>208</td>
      <td>1680</td>
      <td>400</td>
      <td>1280</td>
      <td>0.85</td>
      <td>0.66</td>
      <td>327.6</td>
      <td>1.22</td>
      <td>19.0</td>
      <td>23.81</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>37</td>
      <td>203</td>
      <td>1920</td>
      <td>437</td>
      <td>1483</td>
      <td>0.93</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.17</td>
      <td>16.0</td>
      <td>22.76</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>17</td>
      <td>223</td>
      <td>2160</td>
      <td>454</td>
      <td>1706</td>
      <td>0.97</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.08</td>
      <td>9.0</td>
      <td>21.02</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>14</td>
      <td>226</td>
      <td>2400</td>
      <td>468</td>
      <td>1932</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>468.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>19.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
ROC_AUC(aml_v1,test_hex,'loan_default')
```

    stackedensemble prediction progress: |████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false positive rate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/glm_automl/output_104_1.png)


The first automl model has a positive result in the lift score and AUC score, which is 2.07 for lift score in gain table and 0.67 for AUC. Thus, I am going to use it for the entire dataset.

#### Model 2


```python
aml_v2 = H2OAutoML(max_models=15, seed=1234)
aml_v2.train(list(predictors),target,training_frame=train_full, validation_frame = test_full)
```

    AutoML progress: |
    11:22:19.319: User specified a validation frame with cross-validation still enabled. Please note that the models will still be validated using cross-validation only, the validation frame will be used to provide purely informative validation metrics on the trained models.
    11:22:19.319: AutoML: XGBoost is not available; skipping it.
    
    ████████████████████████████████████████████████████████| 100%
    


```python
aml_v2.leaderboard.head()
```


<table>
<thead>
<tr><th>model_id                                           </th><th style="text-align: right;">  mean_residual_deviance</th><th style="text-align: right;">    rmse</th><th style="text-align: right;">     mse</th><th style="text-align: right;">     mae</th><th style="text-align: right;">   rmsle</th></tr>
</thead>
<tbody>
<tr><td>StackedEnsemble_AllModels_AutoML_20201122_112219   </td><td style="text-align: right;">                0.146072</td><td style="text-align: right;">0.382194</td><td style="text-align: right;">0.146072</td><td style="text-align: right;">0.292318</td><td style="text-align: right;">0.268171</td></tr>
<tr><td>StackedEnsemble_BestOfFamily_AutoML_20201122_112219</td><td style="text-align: right;">                0.146261</td><td style="text-align: right;">0.382441</td><td style="text-align: right;">0.146261</td><td style="text-align: right;">0.292759</td><td style="text-align: right;">0.26838 </td></tr>
<tr><td>GBM_grid__1_AutoML_20201122_112219_model_2         </td><td style="text-align: right;">                0.146756</td><td style="text-align: right;">0.383088</td><td style="text-align: right;">0.146756</td><td style="text-align: right;">0.294665</td><td style="text-align: right;">0.268574</td></tr>
<tr><td>GBM_1_AutoML_20201122_112219                       </td><td style="text-align: right;">                0.146937</td><td style="text-align: right;">0.383324</td><td style="text-align: right;">0.146937</td><td style="text-align: right;">0.294009</td><td style="text-align: right;">0.268962</td></tr>
<tr><td>GBM_grid__1_AutoML_20201122_112219_model_3         </td><td style="text-align: right;">                0.147001</td><td style="text-align: right;">0.383407</td><td style="text-align: right;">0.147001</td><td style="text-align: right;">0.294523</td><td style="text-align: right;">0.268942</td></tr>
<tr><td>GBM_2_AutoML_20201122_112219                       </td><td style="text-align: right;">                0.14759 </td><td style="text-align: right;">0.384175</td><td style="text-align: right;">0.14759 </td><td style="text-align: right;">0.294314</td><td style="text-align: right;">0.269653</td></tr>
<tr><td>GLM_1_AutoML_20201122_112219                       </td><td style="text-align: right;">                0.147644</td><td style="text-align: right;">0.384244</td><td style="text-align: right;">0.147644</td><td style="text-align: right;">0.296478</td><td style="text-align: right;">0.269806</td></tr>
<tr><td>GBM_5_AutoML_20201122_112219                       </td><td style="text-align: right;">                0.1477  </td><td style="text-align: right;">0.384317</td><td style="text-align: right;">0.1477  </td><td style="text-align: right;">0.29299 </td><td style="text-align: right;">0.270075</td></tr>
<tr><td>GBM_3_AutoML_20201122_112219                       </td><td style="text-align: right;">                0.147863</td><td style="text-align: right;">0.384529</td><td style="text-align: right;">0.147863</td><td style="text-align: right;">0.293963</td><td style="text-align: right;">0.270106</td></tr>
<tr><td>DeepLearning_1_AutoML_20201122_112219              </td><td style="text-align: right;">                0.14812 </td><td style="text-align: right;">0.384863</td><td style="text-align: right;">0.14812 </td><td style="text-align: right;">0.294655</td><td style="text-align: right;">0.270597</td></tr>
</tbody>
</table>





    



The leaderboard displays the top 10 models built by AutoML with their parameters. The best model is a Stacked Ensemble(placed on the top) and is stored as aml.leader.

The standard model_performance() method can be applied to the AutoML leader model and a test set to generate an H2O model performance object.


```python
perf = aml_v2.leader.model_performance(test_hex)
perf
```

    
    ModelMetricsRegressionGLM: stackedensemble
    ** Reported on test data. **
    
    MSE: 0.14586616904884925
    RMSE: 0.38192429753663126
    MAE: 0.2867923860387727
    RMSLE: 0.26684395979703573
    R^2: 0.07076815385348445
    Mean Residual Deviance: 0.14586616904884925
    Null degrees of freedom: 2399
    Residual degrees of freedom: 2387
    Null deviance: 376.74231505102404
    Residual deviance: 350.0788057172382
    AIC: 2218.747212170441
    




    




```python
createGains_1(aml_v2)
```

    stackedensemble prediction progress: |████████████████████████████████████| 100%
    




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
      <th>count</th>
      <th>actual</th>
      <th>non_actual</th>
      <th>cum_count</th>
      <th>cum_actual</th>
      <th>cum_non_actual</th>
      <th>percent_cum_actual</th>
      <th>percent_cum_non_actual</th>
      <th>if_random</th>
      <th>lift</th>
      <th>K_S</th>
      <th>gain</th>
    </tr>
    <tr>
      <th>decile</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>240</td>
      <td>102</td>
      <td>138</td>
      <td>240</td>
      <td>102</td>
      <td>138</td>
      <td>0.22</td>
      <td>0.07</td>
      <td>46.8</td>
      <td>2.18</td>
      <td>15.0</td>
      <td>42.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>74</td>
      <td>166</td>
      <td>480</td>
      <td>176</td>
      <td>304</td>
      <td>0.38</td>
      <td>0.16</td>
      <td>93.6</td>
      <td>1.88</td>
      <td>22.0</td>
      <td>36.67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>55</td>
      <td>185</td>
      <td>720</td>
      <td>231</td>
      <td>489</td>
      <td>0.49</td>
      <td>0.25</td>
      <td>140.4</td>
      <td>1.65</td>
      <td>24.0</td>
      <td>32.08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>55</td>
      <td>185</td>
      <td>960</td>
      <td>286</td>
      <td>674</td>
      <td>0.61</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.53</td>
      <td>26.0</td>
      <td>29.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>40</td>
      <td>200</td>
      <td>1200</td>
      <td>326</td>
      <td>874</td>
      <td>0.70</td>
      <td>0.45</td>
      <td>234.0</td>
      <td>1.39</td>
      <td>25.0</td>
      <td>27.17</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>42</td>
      <td>198</td>
      <td>1440</td>
      <td>368</td>
      <td>1072</td>
      <td>0.79</td>
      <td>0.55</td>
      <td>280.8</td>
      <td>1.31</td>
      <td>24.0</td>
      <td>25.56</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>37</td>
      <td>203</td>
      <td>1680</td>
      <td>405</td>
      <td>1275</td>
      <td>0.87</td>
      <td>0.66</td>
      <td>327.6</td>
      <td>1.24</td>
      <td>21.0</td>
      <td>24.11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>32</td>
      <td>208</td>
      <td>1920</td>
      <td>437</td>
      <td>1483</td>
      <td>0.93</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.17</td>
      <td>16.0</td>
      <td>22.76</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>18</td>
      <td>222</td>
      <td>2160</td>
      <td>455</td>
      <td>1705</td>
      <td>0.97</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.08</td>
      <td>9.0</td>
      <td>21.06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>13</td>
      <td>227</td>
      <td>2400</td>
      <td>468</td>
      <td>1932</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>468.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>19.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
ROC_AUC(aml_v2,test_full,'loan_default')
```

    stackedensemble prediction progress: |████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false positive rate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/glm_automl/output_113_1.png)


With AutoML in the entire dataset, the result is more promising than the sample dataset, when the lift is 2.18 and AUC is 0.67.  

#### Ensemble Exploration

The “All Models” ensemble is an ensemble of all of the individual models in the AutoML run. This method is often the top-performing model on the leaderboard.


```python
# Get model ids for all models in the AutoML Leaderboard
model_ids = list(aml_v2.leaderboard['model_id'].as_data_frame().iloc[:,0])
# Get the "All Models" Stacked Ensemble model
se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
# Get the Stacked Ensemble metalearner model
metalearner = h2o.get_model(se.metalearner()['name'])
```

Examine the variable importance of the metalearner (combiner) algorithm in the ensemble. This shows us how much each base learner is contributing to the ensemble.


```python
metalearner.std_coef_plot()
```


![png](/assets/img/glm_automl/output_118_0.png)


### Conclusion

For Generalized Linear Models (GLM) models, the predictive power increases when we apply optimal hyperparementers (lift: 2.12 and AUC: 0.66). From my observations, regularization significantly improves model performance and reduces model variances. Additionally, the GLM models performed much better than RF and GBM in previous assignments with the given dataset. This outcome provides an assumption of a strong linear relationship between the variables. 

For AutoML model, the predictive power increased much more when I applied to the entire dataset (lift:2.18 and AUC: 0.67), instead of only the sample dataset. Additionally, the purpose of AutoML is to automate the repetitive tasks like pipeline creation and hyperparameter tuning so that we can dedicate more time and resources for the business problem on hand. AutoML helps to accelerate the ML process so that the real effectiveness of machine learning can be utilized.
