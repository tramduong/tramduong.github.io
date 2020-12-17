
# Supervised Machine Learning with Random Forest and Sampling

Tram Duong
<br>November 2, 2020

## Table of Contents:
* [Part 1: EDA and FE](#Part_1)
* [Part 2: Data Preparation](#Part_2)
* [Part 3: Supervised Learning](#Part_3)

## Part 1: EDA and FE <a class="anchor" id="Part_1"></a>
- Data Exploration
- Data Cleaning
- Feature Engineerings


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

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch

import warnings
warnings.filterwarnings('ignore')
```


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



#### From this information, we can see that some features won't be relevant in our analysis as there are too many missing values (over 99% of the data is null). Therefore, I removed those variables as they do not provide useful information to work with.


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




    <AxesSubplot:>




![png](/assets/img/sampling_rf/output_44_1.png)


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


![png](/assets/img/sampling_rf/output_55_0.png)



![png](/assets/img/sampling_rf/output_55_1.png)



![png](/assets/img/sampling_rf/output_55_2.png)



![png](/assets/img/sampling_rf/output_55_3.png)



![png](/assets/img/sampling_rf/output_55_4.png)



![png](/assets/img/sampling_rf/output_55_5.png)



![png](/assets/img/sampling_rf/output_55_6.png)



![png](/assets/img/sampling_rf/output_55_7.png)



![png](/assets/img/sampling_rf/output_55_8.png)



![png](/assets/img/sampling_rf/output_55_9.png)



![png](/assets/img/sampling_rf/output_55_10.png)



![png](/assets/img/sampling_rf/output_55_11.png)



![png](/assets/img/sampling_rf/output_55_12.png)



![png](/assets/img/sampling_rf/output_55_13.png)



![png](/assets/img/sampling_rf/output_55_14.png)



![png](/assets/img/sampling_rf/output_55_15.png)



![png](/assets/img/sampling_rf/output_55_16.png)



![png](/assets/img/sampling_rf/output_55_17.png)



![png](/assets/img/sampling_rf/output_55_18.png)



![png](/assets/img/sampling_rf/output_55_19.png)



![png](/assets/img/sampling_rf/output_55_20.png)



![png](/assets/img/sampling_rf/output_55_21.png)



![png](/assets/img/sampling_rf/output_55_22.png)



![png](/assets/img/sampling_rf/output_55_23.png)


#### Split train-test data 


```python
train, test = train_test_split(
    data_clean, test_size=0.30, random_state=23)

target = 'loan_default'
predictors = train.columns[1:]
```


```python
train['loan_default'].value_counts(dropna=False)
```




    0    45135
    1    10865
    Name: loan_default, dtype: int64



## Part 3: Supervised Learning with Random Forest <a class="anchor" id="Part_3"></a>

#### Used functions throughout the modeling approach


```python
def VarImp(model_name):
    
    from sklearn.metrics import roc_curve,auc
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    
    # plot the variable importance
    plt.rcdefaults()
    variables = model_name._model_json['output']['variable_importances']['variable']
    y_pos = np.arange(len(variables))
    fig, ax = plt.subplots(figsize = (6,len(variables)/2))
    scaled_importance = model_name._model_json['output']['variable_importances']['scaled_importance']
    ax.barh(y_pos,scaled_importance,align='center',color='lightblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.set_xlabel('Scaled Importance')
    ax.set_title('Variable Importance')
    plt.show()
    
def createGains(model):
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
    print('   * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate')
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
    ; OpenJDK 64-Bit Server VM (build 11.0.6+8-b765.1, mixed mode)
      Starting server from C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\h2o\backend\bin\h2o.jar
      Ice root: C:\Users\tramh\AppData\Local\Temp\tmpnphmwdzb
      JVM stdout: C:\Users\tramh\AppData\Local\Temp\tmpnphmwdzb\h2o_tramh_started_from_python.out
      JVM stderr: C:\Users\tramh\AppData\Local\Temp\tmpnphmwdzb\h2o_tramh_started_from_python.err
      Server is running at http://127.0.0.1:54321
    Connecting to H2O server at http://127.0.0.1:54321 ... successful.
    


<div style="overflow:auto"><table style="width:50%"><tr><td>H2O_cluster_uptime:</td>
<td>02 secs</td></tr>
<tr><td>H2O_cluster_timezone:</td>
<td>America/New_York</td></tr>
<tr><td>H2O_data_parsing_timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O_cluster_version:</td>
<td>3.30.1.3</td></tr>
<tr><td>H2O_cluster_version_age:</td>
<td>1 month and 5 days </td></tr>
<tr><td>H2O_cluster_name:</td>
<td>H2O_from_python_tramh_7q0c05</td></tr>
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
<td>3.7.1 final</td></tr></table></div>



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

The main difference between AUC and AUCPR is that AUC calculates the area under the ROC curve and AUCPR calculates the area under the Precision Recall curve. The Precision Recall curve does not care about True Negatives. For imbalanced data, a large quantity of True Negatives usually overshadows the effects of changes in other metrics like False Positives. The AUCPR will be much more sensitive to True Positives, False Positives, and False Negatives than AUC. As such, AUCPR is recommended over AUC for highly imbalanced data and should provide more meaningful results.

[Reference](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance-and-prediction.html?highlight=AUC)


```python
rf_v1 = H2ORandomForestEstimator(
        model_id = 'rf_v1',
        ntrees = 600,
        stopping_metric = "AUCPR",
        nfolds=10,
        min_rows=100,
        seed=1234)
```


```python
rf_v1.train(list(predictors),target,training_frame=train_hex)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
VarImp(rf_v1)
```


![png](/assets/img/sampling_rf/output_68_0.png)



```python
predictions = rf_v1.predict(test_hex)
predictions.head()
test_scores = test_hex['loan_default'].cbind(predictions).as_data_frame()
test_scores.head()
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>0.190111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.145541</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.167426</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.179221</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.150215</td>
    </tr>
  </tbody>
</table>
</div>




```python
createGains(rf_v1)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>89</td>
      <td>151</td>
      <td>240</td>
      <td>89</td>
      <td>151</td>
      <td>0.19</td>
      <td>0.08</td>
      <td>46.8</td>
      <td>1.90</td>
      <td>11.0</td>
      <td>37.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>66</td>
      <td>174</td>
      <td>480</td>
      <td>155</td>
      <td>325</td>
      <td>0.33</td>
      <td>0.17</td>
      <td>93.6</td>
      <td>1.66</td>
      <td>16.0</td>
      <td>32.29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>65</td>
      <td>175</td>
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
      <td>46</td>
      <td>194</td>
      <td>960</td>
      <td>266</td>
      <td>694</td>
      <td>0.57</td>
      <td>0.36</td>
      <td>187.2</td>
      <td>1.42</td>
      <td>21.0</td>
      <td>27.71</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>42</td>
      <td>198</td>
      <td>1200</td>
      <td>308</td>
      <td>892</td>
      <td>0.66</td>
      <td>0.46</td>
      <td>234.0</td>
      <td>1.32</td>
      <td>20.0</td>
      <td>25.67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>52</td>
      <td>188</td>
      <td>1440</td>
      <td>360</td>
      <td>1080</td>
      <td>0.77</td>
      <td>0.56</td>
      <td>280.8</td>
      <td>1.28</td>
      <td>21.0</td>
      <td>25.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>46</td>
      <td>194</td>
      <td>1680</td>
      <td>406</td>
      <td>1274</td>
      <td>0.87</td>
      <td>0.66</td>
      <td>327.6</td>
      <td>1.24</td>
      <td>21.0</td>
      <td>24.17</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>24</td>
      <td>216</td>
      <td>1920</td>
      <td>430</td>
      <td>1490</td>
      <td>0.92</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.15</td>
      <td>15.0</td>
      <td>22.40</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>21</td>
      <td>219</td>
      <td>2160</td>
      <td>451</td>
      <td>1709</td>
      <td>0.96</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.07</td>
      <td>8.0</td>
      <td>20.88</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>17</td>
      <td>223</td>
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

- Lift = decile 0 of model 1 has 1.9 times greater lift than random selection.
- K_S= abs(cumulative % of total good loan applicants— cumulative % of total bad loan applicants) -> The higher the value, the better the model is at separating the positive cases from negative ones.


```python
ROC_AUC(rf_v1,test_hex,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_72_1.png)


### Model 2: 

After testing on small datasets and the results are promising, I apply the same code to the entire dataset. 


```python
rf_v2 = H2ORandomForestEstimator(
        model_id = 'rf_v2',
        ntrees = 600,
        stopping_metric = "AUCPR",
        nfolds=10,
        min_rows=100,
        seed=1234)
```


```python
rf_v2.train(list(predictors),target,training_frame=train_full)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
predictions_2 = rf_v2.predict(test_hex)
predictions_2.head()
test_scores_2 = test_hex['loan_default'].cbind(predictions_2).as_data_frame()
test_scores_2.head()
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>0.172478</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.109992</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.138127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.184747</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.103186</td>
    </tr>
  </tbody>
</table>
</div>




```python
createGains(rf_v2)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>93</td>
      <td>147</td>
      <td>240</td>
      <td>93</td>
      <td>147</td>
      <td>0.20</td>
      <td>0.08</td>
      <td>46.8</td>
      <td>1.99</td>
      <td>12.0</td>
      <td>38.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>69</td>
      <td>171</td>
      <td>480</td>
      <td>162</td>
      <td>318</td>
      <td>0.35</td>
      <td>0.16</td>
      <td>93.6</td>
      <td>1.73</td>
      <td>19.0</td>
      <td>33.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>53</td>
      <td>187</td>
      <td>720</td>
      <td>215</td>
      <td>505</td>
      <td>0.46</td>
      <td>0.26</td>
      <td>140.4</td>
      <td>1.53</td>
      <td>20.0</td>
      <td>29.86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>55</td>
      <td>185</td>
      <td>960</td>
      <td>270</td>
      <td>690</td>
      <td>0.58</td>
      <td>0.36</td>
      <td>187.2</td>
      <td>1.44</td>
      <td>22.0</td>
      <td>28.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>43</td>
      <td>197</td>
      <td>1200</td>
      <td>313</td>
      <td>887</td>
      <td>0.67</td>
      <td>0.46</td>
      <td>234.0</td>
      <td>1.34</td>
      <td>21.0</td>
      <td>26.08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>44</td>
      <td>196</td>
      <td>1440</td>
      <td>357</td>
      <td>1083</td>
      <td>0.76</td>
      <td>0.56</td>
      <td>280.8</td>
      <td>1.27</td>
      <td>20.0</td>
      <td>24.79</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>34</td>
      <td>206</td>
      <td>1680</td>
      <td>391</td>
      <td>1289</td>
      <td>0.84</td>
      <td>0.67</td>
      <td>327.6</td>
      <td>1.19</td>
      <td>17.0</td>
      <td>23.27</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>41</td>
      <td>199</td>
      <td>1920</td>
      <td>432</td>
      <td>1488</td>
      <td>0.92</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.15</td>
      <td>15.0</td>
      <td>22.50</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>22</td>
      <td>218</td>
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



The gain table of model 2 results in better scores compare to model 1, as well as the AUC score.


```python
ROC_AUC(rf_v2,test_full,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_79_1.png)


### Model 3:  Use H2O's "balance_classes"

Given that we have an highly unbalanced dataset, I'm using the H2O balance_classes. The balance_classes option can be used to balance the class distribution. When enabled, H2O will either undersample the majority classes or oversample the minority classes.


```python
rf_v3 = H2ORandomForestEstimator(
        model_id = 'rf_v3',
        ntrees = 300,
        stopping_metric = "AUCPR", 
        nfolds=10,
        min_rows=100,
        balance_classes = True,
        seed=1234)
rf_v3.train(list(predictors),target,training_frame=train_full)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
predictions_3 = rf_v2.predict(test_hex)
predictions_3.head()
test_scores_3 = test_hex['loan_default'].cbind(predictions_3).as_data_frame()
test_scores_3.head()
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>0.172478</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.109992</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.138127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.184747</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.103186</td>
    </tr>
  </tbody>
</table>
</div>




```python
createGains(rf_v3)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>92</td>
      <td>148</td>
      <td>240</td>
      <td>92</td>
      <td>148</td>
      <td>0.20</td>
      <td>0.08</td>
      <td>46.8</td>
      <td>1.97</td>
      <td>12.0</td>
      <td>38.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>70</td>
      <td>170</td>
      <td>480</td>
      <td>162</td>
      <td>318</td>
      <td>0.35</td>
      <td>0.16</td>
      <td>93.6</td>
      <td>1.73</td>
      <td>19.0</td>
      <td>33.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>55</td>
      <td>185</td>
      <td>720</td>
      <td>217</td>
      <td>503</td>
      <td>0.46</td>
      <td>0.26</td>
      <td>140.4</td>
      <td>1.55</td>
      <td>20.0</td>
      <td>30.14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>53</td>
      <td>187</td>
      <td>960</td>
      <td>270</td>
      <td>690</td>
      <td>0.58</td>
      <td>0.36</td>
      <td>187.2</td>
      <td>1.44</td>
      <td>22.0</td>
      <td>28.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>42</td>
      <td>198</td>
      <td>1200</td>
      <td>312</td>
      <td>888</td>
      <td>0.67</td>
      <td>0.46</td>
      <td>234.0</td>
      <td>1.33</td>
      <td>21.0</td>
      <td>26.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>43</td>
      <td>197</td>
      <td>1440</td>
      <td>355</td>
      <td>1085</td>
      <td>0.76</td>
      <td>0.56</td>
      <td>280.8</td>
      <td>1.26</td>
      <td>20.0</td>
      <td>24.65</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>34</td>
      <td>206</td>
      <td>1680</td>
      <td>389</td>
      <td>1291</td>
      <td>0.83</td>
      <td>0.67</td>
      <td>327.6</td>
      <td>1.19</td>
      <td>16.0</td>
      <td>23.15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>42</td>
      <td>198</td>
      <td>1920</td>
      <td>431</td>
      <td>1489</td>
      <td>0.92</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.15</td>
      <td>15.0</td>
      <td>22.45</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>22</td>
      <td>218</td>
      <td>2160</td>
      <td>453</td>
      <td>1707</td>
      <td>0.97</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.08</td>
      <td>9.0</td>
      <td>20.97</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>15</td>
      <td>225</td>
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



Model 3 has lower results in gain table compare to model 2, but still better than model 1. For AUC, this model also ranked as the second place amoung 3 models.


```python
ROC_AUC(rf_v3,test_full,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_85_1.png)


### Model 4: Hyperparameter Tuning 
In this section, I used H2O Grid-search to find the optimal hyper-parameters for the model.

As 'loan_default' contains binary values represented by 0's and 1's, it is assumed to be numeric in H20 and is applied as such. Thus, it would be trained as a regression model instead of a binary classification model, and AUC is not allowed. I will use as.factor() on that column, just after uploading it into H2O to ensure that column is interpreted correctly.


```python
# sample data to test 
train_smpl_1 = train.sample(frac=0.1, random_state=1)
test_smpl_1 = test.sample(frac=0.1, random_state=1)
train_hex_1 = h2o.H2OFrame(train_smpl_1)
test_hex_1 = h2o.H2OFrame(test_smpl_1)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    


```python
# For the full data 
#train_full[target] = train_full [target].asfactor()
#test_full[target] = test_full[target].asfactor()
# For sample data to test
train_hex_1[target] = train_hex_1[target].asfactor()
#test_hex[target] = test_hex[target].asfactor()
```


```python
rf_v4 = H2ORandomForestEstimator(nfolds = 5,seed = 1234,balance_classes = True)

hyper_params = {'max_depth': [1, 5,10,20],
                 'ntrees': [200,400,600, 800]}

rf_grid = H2OGridSearch(model = rf_v4, hyper_params = hyper_params,
                     search_criteria = {'strategy': "RandomDiscrete",'max_models': 30, 
                                        'seed': 1,"stopping_metric": "AUCPR",
                                        "stopping_tolerance": 1e-4, 
                                        "stopping_rounds": 5})
```


```python
rf_grid.train(x = list(predictors), y = target, training_frame=train_hex_1)
```

    drf Grid Build progress: |████████████████████████████████████████████████| 100%
    


```python
# Get the grid results, sorted by validation AUC
sorted_rf_grid = rf_grid.get_grid(sort_by='auc',decreasing=True)
best_rf = sorted_rf_grid.models[0]
```


```python
print(best_rf)
```

    Model Details
    =============
    H2ORandomForestEstimator :  Distributed Random Forest
    Model Key:  Grid_DRF_py_13_sid_9c55_model_python_1604352498129_1_model_3
    
    
    Model Summary: 
    


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
      <th>number_of_trees</th>
      <th>number_of_internal_trees</th>
      <th>model_size_in_bytes</th>
      <th>min_depth</th>
      <th>max_depth</th>
      <th>mean_depth</th>
      <th>min_leaves</th>
      <th>max_leaves</th>
      <th>mean_leaves</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>800.0</td>
      <td>800.0</td>
      <td>3051870.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>161.0</td>
      <td>403.0</td>
      <td>298.16</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    ModelMetricsBinomial: drf
    ** Reported on train data. **
    
    MSE: 0.23676255438870758
    RMSE: 0.48658252577410505
    LogLoss: 0.645491325489853
    Mean Per-Class Error: 0.12036748445533774
    AUC: 0.9545834413505689
    AUCPR: 0.9565482862869644
    Gini: 0.9091668827011379
    
    Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.206981036479151: 
    


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
      <td>3911.0</td>
      <td>609.0</td>
      <td>0.1347</td>
      <td>(609.0/4520.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>496.0</td>
      <td>4010.0</td>
      <td>0.1101</td>
      <td>(496.0/4506.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Total</td>
      <td>4407.0</td>
      <td>4619.0</td>
      <td>0.1224</td>
      <td>(1105.0/9026.0)</td>
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
      <td>0.206981</td>
      <td>0.878904</td>
      <td>256.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>max f2</td>
      <td>0.158301</td>
      <td>0.917593</td>
      <td>298.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>max f0point5</td>
      <td>0.258163</td>
      <td>0.904306</td>
      <td>217.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>max accuracy</td>
      <td>0.225276</td>
      <td>0.879681</td>
      <td>241.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max precision</td>
      <td>0.661631</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>max recall</td>
      <td>0.095665</td>
      <td>1.000000</td>
      <td>356.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>max specificity</td>
      <td>0.661631</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>max absolute_mcc</td>
      <td>0.225276</td>
      <td>0.760804</td>
      <td>241.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>max min_per_class_accuracy</td>
      <td>0.211429</td>
      <td>0.878982</td>
      <td>252.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>max mean_per_class_accuracy</td>
      <td>0.225276</td>
      <td>0.879633</td>
      <td>241.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>max tns</td>
      <td>0.661631</td>
      <td>4520.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>max fns</td>
      <td>0.661631</td>
      <td>4504.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>max fps</td>
      <td>0.036625</td>
      <td>4520.000000</td>
      <td>399.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>max tps</td>
      <td>0.095665</td>
      <td>4506.000000</td>
      <td>356.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>max tnr</td>
      <td>0.661631</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>max fnr</td>
      <td>0.661631</td>
      <td>0.999556</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>max fpr</td>
      <td>0.036625</td>
      <td>1.000000</td>
      <td>399.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>max tpr</td>
      <td>0.095665</td>
      <td>1.000000</td>
      <td>356.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Gains/Lift Table: Avg response rate: 49.92 %, avg score: 23.66 %
    


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
      <td>0.010082</td>
      <td>0.557453</td>
      <td>2.003107</td>
      <td>2.003107</td>
      <td>1.000000</td>
      <td>0.589794</td>
      <td>1.000000</td>
      <td>0.589794</td>
      <td>0.020195</td>
      <td>0.020195</td>
      <td>100.310697</td>
      <td>100.310697</td>
      <td>0.020195</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.020053</td>
      <td>0.529356</td>
      <td>2.003107</td>
      <td>2.003107</td>
      <td>1.000000</td>
      <td>0.542790</td>
      <td>1.000000</td>
      <td>0.566422</td>
      <td>0.019973</td>
      <td>0.040169</td>
      <td>100.310697</td>
      <td>100.310697</td>
      <td>0.040169</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.030024</td>
      <td>0.511138</td>
      <td>2.003107</td>
      <td>2.003107</td>
      <td>1.000000</td>
      <td>0.519249</td>
      <td>1.000000</td>
      <td>0.550756</td>
      <td>0.019973</td>
      <td>0.060142</td>
      <td>100.310697</td>
      <td>100.310697</td>
      <td>0.060142</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.040106</td>
      <td>0.496710</td>
      <td>2.003107</td>
      <td>2.003107</td>
      <td>1.000000</td>
      <td>0.503813</td>
      <td>1.000000</td>
      <td>0.538955</td>
      <td>0.020195</td>
      <td>0.080337</td>
      <td>100.310697</td>
      <td>100.310697</td>
      <td>0.080337</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.050078</td>
      <td>0.480812</td>
      <td>2.003107</td>
      <td>2.003107</td>
      <td>1.000000</td>
      <td>0.488699</td>
      <td>1.000000</td>
      <td>0.528948</td>
      <td>0.019973</td>
      <td>0.100311</td>
      <td>100.310697</td>
      <td>100.310697</td>
      <td>0.100311</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.100044</td>
      <td>0.424116</td>
      <td>2.003107</td>
      <td>2.003107</td>
      <td>1.000000</td>
      <td>0.449688</td>
      <td>1.000000</td>
      <td>0.489362</td>
      <td>0.100089</td>
      <td>0.200399</td>
      <td>100.310697</td>
      <td>100.310697</td>
      <td>0.200399</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.150011</td>
      <td>0.387749</td>
      <td>1.980900</td>
      <td>1.995710</td>
      <td>0.988914</td>
      <td>0.404084</td>
      <td>0.996307</td>
      <td>0.460957</td>
      <td>0.098979</td>
      <td>0.299379</td>
      <td>98.089957</td>
      <td>99.570997</td>
      <td>0.298272</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.200089</td>
      <td>0.357374</td>
      <td>1.963222</td>
      <td>1.987579</td>
      <td>0.980088</td>
      <td>0.372606</td>
      <td>0.992248</td>
      <td>0.438845</td>
      <td>0.098313</td>
      <td>0.397692</td>
      <td>96.322210</td>
      <td>98.757901</td>
      <td>0.394595</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.300022</td>
      <td>0.307329</td>
      <td>1.927602</td>
      <td>1.967601</td>
      <td>0.962306</td>
      <td>0.331998</td>
      <td>0.982275</td>
      <td>0.403256</td>
      <td>0.192632</td>
      <td>0.590324</td>
      <td>92.760183</td>
      <td>96.760138</td>
      <td>0.579705</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.400066</td>
      <td>0.257892</td>
      <td>1.705857</td>
      <td>1.902147</td>
      <td>0.851606</td>
      <td>0.282954</td>
      <td>0.949598</td>
      <td>0.373172</td>
      <td>0.170661</td>
      <td>0.760985</td>
      <td>70.585743</td>
      <td>90.214727</td>
      <td>0.720720</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.500000</td>
      <td>0.210691</td>
      <td>1.188096</td>
      <td>1.759432</td>
      <td>0.593126</td>
      <td>0.233365</td>
      <td>0.878351</td>
      <td>0.345229</td>
      <td>0.118731</td>
      <td>0.879716</td>
      <td>18.809560</td>
      <td>75.943187</td>
      <td>0.758256</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.600044</td>
      <td>0.173720</td>
      <td>0.700977</td>
      <td>1.582958</td>
      <td>0.349945</td>
      <td>0.191907</td>
      <td>0.790251</td>
      <td>0.319666</td>
      <td>0.070129</td>
      <td>0.949845</td>
      <td>-29.902348</td>
      <td>58.295750</td>
      <td>0.698517</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.699978</td>
      <td>0.140115</td>
      <td>0.375305</td>
      <td>1.410545</td>
      <td>0.187361</td>
      <td>0.156360</td>
      <td>0.704179</td>
      <td>0.296351</td>
      <td>0.037506</td>
      <td>0.987350</td>
      <td>-62.469504</td>
      <td>41.054494</td>
      <td>0.573855</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.800022</td>
      <td>0.112151</td>
      <td>0.104259</td>
      <td>1.247191</td>
      <td>0.052049</td>
      <td>0.125940</td>
      <td>0.622628</td>
      <td>0.275041</td>
      <td>0.010431</td>
      <td>0.997781</td>
      <td>-89.574083</td>
      <td>24.719138</td>
      <td>0.394905</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.899956</td>
      <td>0.084371</td>
      <td>0.022207</td>
      <td>1.111166</td>
      <td>0.011086</td>
      <td>0.098342</td>
      <td>0.554721</td>
      <td>0.255420</td>
      <td>0.002219</td>
      <td>1.000000</td>
      <td>-97.779261</td>
      <td>11.116583</td>
      <td>0.199779</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>1.000000</td>
      <td>0.034346</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.067333</td>
      <td>0.499224</td>
      <td>0.236603</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-100.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    ModelMetricsBinomial: drf
    ** Reported on cross-validation data. **
    
    MSE: 0.15102772513540358
    RMSE: 0.3886228571962843
    LogLoss: 0.47577934005630707
    Mean Per-Class Error: 0.37366027531956736
    AUC: 0.6641486397902326
    AUCPR: 0.31305383378431545
    Gini: 0.3282972795804653
    
    Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.14684065158575366: 
    


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
      <td>2925.0</td>
      <td>1595.0</td>
      <td>0.3529</td>
      <td>(1595.0/4520.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>426.0</td>
      <td>654.0</td>
      <td>0.3944</td>
      <td>(426.0/1080.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Total</td>
      <td>3351.0</td>
      <td>2249.0</td>
      <td>0.3609</td>
      <td>(2021.0/5600.0)</td>
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
      <td>0.146841</td>
      <td>0.392911</td>
      <td>216.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>max f2</td>
      <td>0.093896</td>
      <td>0.561699</td>
      <td>310.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>max f0point5</td>
      <td>0.171283</td>
      <td>0.333470</td>
      <td>176.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>max accuracy</td>
      <td>0.346744</td>
      <td>0.807857</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max precision</td>
      <td>0.388885</td>
      <td>0.600000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>max recall</td>
      <td>0.034370</td>
      <td>1.000000</td>
      <td>398.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>max specificity</td>
      <td>0.461535</td>
      <td>0.999779</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>max absolute_mcc</td>
      <td>0.156135</td>
      <td>0.203516</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>max min_per_class_accuracy</td>
      <td>0.142852</td>
      <td>0.623148</td>
      <td>222.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>max mean_per_class_accuracy</td>
      <td>0.146841</td>
      <td>0.626340</td>
      <td>216.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>max tns</td>
      <td>0.461535</td>
      <td>4519.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>max fns</td>
      <td>0.461535</td>
      <td>1080.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>max fps</td>
      <td>0.033139</td>
      <td>4520.000000</td>
      <td>399.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>max tps</td>
      <td>0.034370</td>
      <td>1080.000000</td>
      <td>398.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>max tnr</td>
      <td>0.461535</td>
      <td>0.999779</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>max fnr</td>
      <td>0.461535</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>max fpr</td>
      <td>0.033139</td>
      <td>1.000000</td>
      <td>399.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>max tpr</td>
      <td>0.034370</td>
      <td>1.000000</td>
      <td>398.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Gains/Lift Table: Avg response rate: 19.29 %, avg score: 14.02 %
    


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
      <td>0.01</td>
      <td>0.314313</td>
      <td>2.592593</td>
      <td>2.592593</td>
      <td>0.500000</td>
      <td>0.355131</td>
      <td>0.500000</td>
      <td>0.355131</td>
      <td>0.025926</td>
      <td>0.025926</td>
      <td>159.259259</td>
      <td>159.259259</td>
      <td>0.019731</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.02</td>
      <td>0.287325</td>
      <td>2.314815</td>
      <td>2.453704</td>
      <td>0.446429</td>
      <td>0.297988</td>
      <td>0.473214</td>
      <td>0.326559</td>
      <td>0.023148</td>
      <td>0.049074</td>
      <td>131.481481</td>
      <td>145.370370</td>
      <td>0.036021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.03</td>
      <td>0.273643</td>
      <td>1.481481</td>
      <td>2.129630</td>
      <td>0.285714</td>
      <td>0.279466</td>
      <td>0.410714</td>
      <td>0.310862</td>
      <td>0.014815</td>
      <td>0.063889</td>
      <td>48.148148</td>
      <td>112.962963</td>
      <td>0.041986</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.04</td>
      <td>0.261543</td>
      <td>2.592593</td>
      <td>2.245370</td>
      <td>0.500000</td>
      <td>0.267873</td>
      <td>0.433036</td>
      <td>0.300114</td>
      <td>0.025926</td>
      <td>0.089815</td>
      <td>159.259259</td>
      <td>124.537037</td>
      <td>0.061717</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.05</td>
      <td>0.253599</td>
      <td>1.481481</td>
      <td>2.092593</td>
      <td>0.285714</td>
      <td>0.257472</td>
      <td>0.403571</td>
      <td>0.291586</td>
      <td>0.014815</td>
      <td>0.104630</td>
      <td>48.148148</td>
      <td>109.259259</td>
      <td>0.067683</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.10</td>
      <td>0.222775</td>
      <td>1.888889</td>
      <td>1.990741</td>
      <td>0.364286</td>
      <td>0.237512</td>
      <td>0.383929</td>
      <td>0.264549</td>
      <td>0.094444</td>
      <td>0.199074</td>
      <td>88.888889</td>
      <td>99.074074</td>
      <td>0.122747</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.15</td>
      <td>0.203773</td>
      <td>1.462963</td>
      <td>1.814815</td>
      <td>0.282143</td>
      <td>0.212230</td>
      <td>0.350000</td>
      <td>0.247109</td>
      <td>0.073148</td>
      <td>0.272222</td>
      <td>46.296296</td>
      <td>81.481481</td>
      <td>0.151426</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.20</td>
      <td>0.189758</td>
      <td>1.314815</td>
      <td>1.689815</td>
      <td>0.253571</td>
      <td>0.196758</td>
      <td>0.325893</td>
      <td>0.234522</td>
      <td>0.065741</td>
      <td>0.337963</td>
      <td>31.481481</td>
      <td>68.981481</td>
      <td>0.170928</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.30</td>
      <td>0.165934</td>
      <td>1.407407</td>
      <td>1.595679</td>
      <td>0.271429</td>
      <td>0.177817</td>
      <td>0.307738</td>
      <td>0.215620</td>
      <td>0.140741</td>
      <td>0.478704</td>
      <td>40.740741</td>
      <td>59.567901</td>
      <td>0.221403</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.40</td>
      <td>0.146845</td>
      <td>1.240741</td>
      <td>1.506944</td>
      <td>0.239286</td>
      <td>0.156459</td>
      <td>0.290625</td>
      <td>0.200830</td>
      <td>0.124074</td>
      <td>0.602778</td>
      <td>24.074074</td>
      <td>50.694444</td>
      <td>0.251229</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.50</td>
      <td>0.130780</td>
      <td>0.944444</td>
      <td>1.394444</td>
      <td>0.182143</td>
      <td>0.138442</td>
      <td>0.268929</td>
      <td>0.188352</td>
      <td>0.094444</td>
      <td>0.697222</td>
      <td>-5.555556</td>
      <td>39.444444</td>
      <td>0.244346</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.60</td>
      <td>0.115727</td>
      <td>0.888889</td>
      <td>1.310185</td>
      <td>0.171429</td>
      <td>0.123101</td>
      <td>0.252679</td>
      <td>0.177477</td>
      <td>0.088889</td>
      <td>0.786111</td>
      <td>-11.111111</td>
      <td>31.018519</td>
      <td>0.230580</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.70</td>
      <td>0.100380</td>
      <td>0.638889</td>
      <td>1.214286</td>
      <td>0.123214</td>
      <td>0.108067</td>
      <td>0.234184</td>
      <td>0.167561</td>
      <td>0.063889</td>
      <td>0.850000</td>
      <td>-36.111111</td>
      <td>21.428571</td>
      <td>0.185841</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.80</td>
      <td>0.085917</td>
      <td>0.564815</td>
      <td>1.133102</td>
      <td>0.108929</td>
      <td>0.093096</td>
      <td>0.218527</td>
      <td>0.158253</td>
      <td>0.056481</td>
      <td>0.906481</td>
      <td>-43.518519</td>
      <td>13.310185</td>
      <td>0.131924</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.90</td>
      <td>0.070200</td>
      <td>0.472222</td>
      <td>1.059671</td>
      <td>0.091071</td>
      <td>0.077834</td>
      <td>0.204365</td>
      <td>0.149318</td>
      <td>0.047222</td>
      <td>0.953704</td>
      <td>-52.777778</td>
      <td>5.967078</td>
      <td>0.066536</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>1.00</td>
      <td>0.032906</td>
      <td>0.462963</td>
      <td>1.000000</td>
      <td>0.089286</td>
      <td>0.058321</td>
      <td>0.192857</td>
      <td>0.140218</td>
      <td>0.046296</td>
      <td>1.000000</td>
      <td>-53.703704</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Cross-Validation Metrics Summary: 
    


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
      <th>mean</th>
      <th>sd</th>
      <th>cv_1_valid</th>
      <th>cv_2_valid</th>
      <th>cv_3_valid</th>
      <th>cv_4_valid</th>
      <th>cv_5_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>accuracy</td>
      <td>0.63339216</td>
      <td>0.060881127</td>
      <td>0.64499575</td>
      <td>0.5351027</td>
      <td>0.6223214</td>
      <td>0.6903283</td>
      <td>0.6742126</td>
    </tr>
    <tr>
      <th>1</th>
      <td>auc</td>
      <td>0.6654752</td>
      <td>0.008363225</td>
      <td>0.65197897</td>
      <td>0.66795117</td>
      <td>0.66469514</td>
      <td>0.674636</td>
      <td>0.6681146</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aucpr</td>
      <td>0.3177904</td>
      <td>0.017030368</td>
      <td>0.34251213</td>
      <td>0.31493726</td>
      <td>0.32268253</td>
      <td>0.3133214</td>
      <td>0.2954986</td>
    </tr>
    <tr>
      <th>3</th>
      <td>err</td>
      <td>0.36660784</td>
      <td>0.060881127</td>
      <td>0.35500428</td>
      <td>0.46489727</td>
      <td>0.37767857</td>
      <td>0.3096717</td>
      <td>0.3257874</td>
    </tr>
    <tr>
      <th>4</th>
      <td>err_count</td>
      <td>412.2</td>
      <td>83.39784</td>
      <td>415.0</td>
      <td>543.0</td>
      <td>423.0</td>
      <td>349.0</td>
      <td>331.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>f0point5</td>
      <td>0.33257973</td>
      <td>0.02614182</td>
      <td>0.33761293</td>
      <td>0.30431825</td>
      <td>0.32768103</td>
      <td>0.37393767</td>
      <td>0.31934878</td>
    </tr>
    <tr>
      <th>6</th>
      <td>f1</td>
      <td>0.40379867</td>
      <td>0.01786044</td>
      <td>0.4062947</td>
      <td>0.3973363</td>
      <td>0.40338504</td>
      <td>0.43066883</td>
      <td>0.3813084</td>
    </tr>
    <tr>
      <th>7</th>
      <td>f2</td>
      <td>0.5175354</td>
      <td>0.035954196</td>
      <td>0.51005745</td>
      <td>0.57225066</td>
      <td>0.52457815</td>
      <td>0.50769234</td>
      <td>0.47309834</td>
    </tr>
    <tr>
      <th>8</th>
      <td>lift_top_group</td>
      <td>2.5401468</td>
      <td>0.8786933</td>
      <td>3.7954545</td>
      <td>3.0829563</td>
      <td>2.1406727</td>
      <td>1.6404657</td>
      <td>2.0411854</td>
    </tr>
    <tr>
      <th>9</th>
      <td>logloss</td>
      <td>0.4752418</td>
      <td>0.017334543</td>
      <td>0.4854248</td>
      <td>0.46821377</td>
      <td>0.47960383</td>
      <td>0.49386168</td>
      <td>0.449105</td>
    </tr>
    <tr>
      <th>10</th>
      <td>max_per_class_error</td>
      <td>0.43203494</td>
      <td>0.05878689</td>
      <td>0.38528138</td>
      <td>0.5290391</td>
      <td>0.3858093</td>
      <td>0.4235808</td>
      <td>0.4364641</td>
    </tr>
    <tr>
      <th>11</th>
      <td>mcc</td>
      <td>0.22342017</td>
      <td>0.01615416</td>
      <td>0.21712719</td>
      <td>0.22309135</td>
      <td>0.2155643</td>
      <td>0.25112364</td>
      <td>0.21019442</td>
    </tr>
    <tr>
      <th>12</th>
      <td>mean_per_class_accuracy</td>
      <td>0.63757753</td>
      <td>0.0067451634</td>
      <td>0.63358533</td>
      <td>0.64045787</td>
      <td>0.635077</td>
      <td>0.6478978</td>
      <td>0.63086975</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mean_per_class_error</td>
      <td>0.36242247</td>
      <td>0.0067451634</td>
      <td>0.36641467</td>
      <td>0.35954216</td>
      <td>0.364923</td>
      <td>0.3521022</td>
      <td>0.36913025</td>
    </tr>
    <tr>
      <th>14</th>
      <td>mse</td>
      <td>0.15081765</td>
      <td>0.0067610065</td>
      <td>0.15423578</td>
      <td>0.14853068</td>
      <td>0.15224142</td>
      <td>0.15850234</td>
      <td>0.14057805</td>
    </tr>
    <tr>
      <th>15</th>
      <td>pr_auc</td>
      <td>0.3177904</td>
      <td>0.017030368</td>
      <td>0.34251213</td>
      <td>0.31493726</td>
      <td>0.32268253</td>
      <td>0.3133214</td>
      <td>0.2954986</td>
    </tr>
    <tr>
      <th>16</th>
      <td>precision</td>
      <td>0.2979564</td>
      <td>0.02947021</td>
      <td>0.30341882</td>
      <td>0.2632353</td>
      <td>0.29124236</td>
      <td>0.34375</td>
      <td>0.2881356</td>
    </tr>
    <tr>
      <th>17</th>
      <td>r2</td>
      <td>0.0297492</td>
      <td>0.0068825097</td>
      <td>0.027254298</td>
      <td>0.031812236</td>
      <td>0.028806357</td>
      <td>0.021024825</td>
      <td>0.039848283</td>
    </tr>
    <tr>
      <th>18</th>
      <td>recall</td>
      <td>0.64411837</td>
      <td>0.09947746</td>
      <td>0.6147186</td>
      <td>0.80995476</td>
      <td>0.6559633</td>
      <td>0.57641923</td>
      <td>0.5635359</td>
    </tr>
    <tr>
      <th>19</th>
      <td>rmse</td>
      <td>0.3882735</td>
      <td>0.0087555405</td>
      <td>0.39272863</td>
      <td>0.38539678</td>
      <td>0.39018127</td>
      <td>0.39812353</td>
      <td>0.3749374</td>
    </tr>
  </tbody>
</table>
</div>


    
    See the whole table with table.as_data_frame()
    
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
      <th>number_of_trees</th>
      <th>training_rmse</th>
      <th>training_logloss</th>
      <th>training_auc</th>
      <th>training_pr_auc</th>
      <th>training_lift</th>
      <th>training_classification_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  6.834 sec</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  6.861 sec</td>
      <td>1.0</td>
      <td>0.513185</td>
      <td>1.199601</td>
      <td>0.742498</td>
      <td>0.712600</td>
      <td>1.655536</td>
      <td>0.370245</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  6.873 sec</td>
      <td>2.0</td>
      <td>0.507847</td>
      <td>1.314510</td>
      <td>0.760105</td>
      <td>0.714712</td>
      <td>1.561004</td>
      <td>0.321422</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  6.886 sec</td>
      <td>3.0</td>
      <td>0.507421</td>
      <td>1.268220</td>
      <td>0.770557</td>
      <td>0.724050</td>
      <td>1.527369</td>
      <td>0.308975</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  6.904 sec</td>
      <td>4.0</td>
      <td>0.501674</td>
      <td>1.136477</td>
      <td>0.790257</td>
      <td>0.749886</td>
      <td>1.564070</td>
      <td>0.294607</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  6.923 sec</td>
      <td>5.0</td>
      <td>0.500463</td>
      <td>1.016178</td>
      <td>0.802394</td>
      <td>0.761718</td>
      <td>1.573870</td>
      <td>0.282210</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  6.943 sec</td>
      <td>6.0</td>
      <td>0.499454</td>
      <td>0.913544</td>
      <td>0.811844</td>
      <td>0.772524</td>
      <td>1.597414</td>
      <td>0.276883</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  6.965 sec</td>
      <td>7.0</td>
      <td>0.497051</td>
      <td>0.819577</td>
      <td>0.823319</td>
      <td>0.788646</td>
      <td>1.646999</td>
      <td>0.256804</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  6.987 sec</td>
      <td>8.0</td>
      <td>0.497002</td>
      <td>0.794552</td>
      <td>0.832858</td>
      <td>0.801345</td>
      <td>1.614444</td>
      <td>0.248806</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.010 sec</td>
      <td>9.0</td>
      <td>0.496142</td>
      <td>0.765382</td>
      <td>0.842627</td>
      <td>0.815644</td>
      <td>1.606452</td>
      <td>0.249691</td>
    </tr>
    <tr>
      <th>10</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.032 sec</td>
      <td>10.0</td>
      <td>0.494834</td>
      <td>0.735571</td>
      <td>0.850547</td>
      <td>0.826975</td>
      <td>1.694937</td>
      <td>0.245110</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.054 sec</td>
      <td>11.0</td>
      <td>0.493788</td>
      <td>0.714430</td>
      <td>0.859332</td>
      <td>0.839180</td>
      <td>1.760973</td>
      <td>0.236716</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.079 sec</td>
      <td>12.0</td>
      <td>0.492059</td>
      <td>0.695008</td>
      <td>0.866866</td>
      <td>0.846939</td>
      <td>1.804997</td>
      <td>0.209227</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.102 sec</td>
      <td>13.0</td>
      <td>0.490382</td>
      <td>0.686892</td>
      <td>0.872760</td>
      <td>0.854269</td>
      <td>1.760973</td>
      <td>0.205771</td>
    </tr>
    <tr>
      <th>14</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.124 sec</td>
      <td>14.0</td>
      <td>0.489647</td>
      <td>0.670048</td>
      <td>0.878496</td>
      <td>0.863987</td>
      <td>1.893046</td>
      <td>0.200754</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.151 sec</td>
      <td>15.0</td>
      <td>0.489144</td>
      <td>0.664593</td>
      <td>0.883616</td>
      <td>0.870602</td>
      <td>1.893046</td>
      <td>0.201464</td>
    </tr>
    <tr>
      <th>16</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.174 sec</td>
      <td>16.0</td>
      <td>0.488834</td>
      <td>0.659744</td>
      <td>0.887353</td>
      <td>0.876341</td>
      <td>1.915058</td>
      <td>0.198182</td>
    </tr>
    <tr>
      <th>17</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.199 sec</td>
      <td>17.0</td>
      <td>0.488624</td>
      <td>0.659091</td>
      <td>0.890248</td>
      <td>0.879676</td>
      <td>1.915058</td>
      <td>0.192154</td>
    </tr>
    <tr>
      <th>18</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.221 sec</td>
      <td>18.0</td>
      <td>0.487914</td>
      <td>0.656868</td>
      <td>0.894869</td>
      <td>0.884886</td>
      <td>1.894242</td>
      <td>0.190139</td>
    </tr>
    <tr>
      <th>19</th>
      <td></td>
      <td>2020-11-02 16:50:11</td>
      <td>2 min  7.246 sec</td>
      <td>19.0</td>
      <td>0.487468</td>
      <td>0.654930</td>
      <td>0.899978</td>
      <td>0.889370</td>
      <td>1.915058</td>
      <td>0.190671</td>
    </tr>
  </tbody>
</table>
</div>


    
    See the whole table with table.as_data_frame()
    
    Variable Importances: 
    


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
      <th>variable</th>
      <th>relative_importance</th>
      <th>scaled_importance</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TD013</td>
      <td>28246.951172</td>
      <td>1.000000</td>
      <td>0.060145</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TD005</td>
      <td>21326.464844</td>
      <td>0.755001</td>
      <td>0.045409</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MB005</td>
      <td>19927.867188</td>
      <td>0.705487</td>
      <td>0.042431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AP003</td>
      <td>18405.435547</td>
      <td>0.651590</td>
      <td>0.039190</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TD014</td>
      <td>18105.884766</td>
      <td>0.640985</td>
      <td>0.038552</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AP001</td>
      <td>16733.414062</td>
      <td>0.592397</td>
      <td>0.035630</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TD024</td>
      <td>16200.927734</td>
      <td>0.573546</td>
      <td>0.034496</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CD008</td>
      <td>14725.038086</td>
      <td>0.521297</td>
      <td>0.031353</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PA028</td>
      <td>14204.986328</td>
      <td>0.502886</td>
      <td>0.030246</td>
    </tr>
    <tr>
      <th>9</th>
      <td>TD001</td>
      <td>13787.530273</td>
      <td>0.488107</td>
      <td>0.029357</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CR015</td>
      <td>13233.312500</td>
      <td>0.468486</td>
      <td>0.028177</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CD131</td>
      <td>13196.512695</td>
      <td>0.467184</td>
      <td>0.028099</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CD113</td>
      <td>12717.770508</td>
      <td>0.450235</td>
      <td>0.027079</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CD115</td>
      <td>12695.989258</td>
      <td>0.449464</td>
      <td>0.027033</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CD152</td>
      <td>12597.620117</td>
      <td>0.445982</td>
      <td>0.026824</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CD018</td>
      <td>12542.010742</td>
      <td>0.444013</td>
      <td>0.026705</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CD130</td>
      <td>12369.836914</td>
      <td>0.437918</td>
      <td>0.026339</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PA022</td>
      <td>12206.709961</td>
      <td>0.432143</td>
      <td>0.025991</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CR017</td>
      <td>11895.869141</td>
      <td>0.421138</td>
      <td>0.025329</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TD023</td>
      <td>11443.859375</td>
      <td>0.405136</td>
      <td>0.024367</td>
    </tr>
  </tbody>
</table>
</div>


    
    See the whole table with table.as_data_frame()
    
    

The best rf model contains the hyperparameters as in the model summary above. The model metrics also show high scores on AUC and AUCPR (Both are approximately 0.95). Additionaly the other statistics also provide positive insights.  

From all above, I am going to train the model with the full data. 


```python
best_rf.train(list(predictors),target,training_frame=train_full)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
# Prediction with the best model
predictions_4 = best_rf.predict(test_full)
predictions_4.head()
test_scores_4 = test_full['loan_default'].cbind(predictions_4).as_data_frame()
test_scores_4.head()
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>0.416000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.174541</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.219001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.187464</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.353922</td>
    </tr>
  </tbody>
</table>
</div>




```python
VarImp(best_rf)
```


![png](/assets/img/sampling_rf/output_97_0.png)



```python
createGains(best_rf)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>92</td>
      <td>148</td>
      <td>240</td>
      <td>92</td>
      <td>148</td>
      <td>0.20</td>
      <td>0.08</td>
      <td>46.8</td>
      <td>1.97</td>
      <td>12.0</td>
      <td>38.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>62</td>
      <td>178</td>
      <td>480</td>
      <td>154</td>
      <td>326</td>
      <td>0.33</td>
      <td>0.17</td>
      <td>93.6</td>
      <td>1.65</td>
      <td>16.0</td>
      <td>32.08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>60</td>
      <td>180</td>
      <td>720</td>
      <td>214</td>
      <td>506</td>
      <td>0.46</td>
      <td>0.26</td>
      <td>140.4</td>
      <td>1.52</td>
      <td>20.0</td>
      <td>29.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>56</td>
      <td>184</td>
      <td>960</td>
      <td>270</td>
      <td>690</td>
      <td>0.58</td>
      <td>0.36</td>
      <td>187.2</td>
      <td>1.44</td>
      <td>22.0</td>
      <td>28.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>47</td>
      <td>193</td>
      <td>1200</td>
      <td>317</td>
      <td>883</td>
      <td>0.68</td>
      <td>0.46</td>
      <td>234.0</td>
      <td>1.35</td>
      <td>22.0</td>
      <td>26.42</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>43</td>
      <td>197</td>
      <td>1440</td>
      <td>360</td>
      <td>1080</td>
      <td>0.77</td>
      <td>0.56</td>
      <td>280.8</td>
      <td>1.28</td>
      <td>21.0</td>
      <td>25.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>36</td>
      <td>204</td>
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
      <td>34</td>
      <td>206</td>
      <td>1920</td>
      <td>430</td>
      <td>1490</td>
      <td>0.92</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.15</td>
      <td>15.0</td>
      <td>22.40</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>18</td>
      <td>222</td>
      <td>2160</td>
      <td>448</td>
      <td>1712</td>
      <td>0.96</td>
      <td>0.89</td>
      <td>421.2</td>
      <td>1.06</td>
      <td>7.0</td>
      <td>20.74</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>20</td>
      <td>220</td>
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
ROC_AUC(best_rf,test_hex,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_99_1.png)


### Undersampling

Undersampling refers to a group of techniques designed to balance the class distribution for a classification dataset that has a skewed class distribution. 

Undersampling techniques remove examples from the training dataset that belong to the majority class in order to better balance the class distribution, such as reducing the skew from a 1:100 ratio to a 1:10.

There are three major approaches to handle imbalanced data: data sampling, algorithm modifications, and cost-sensitive learning. In this part, I will focus on **undersampling technique** for my approach. There are 7 common undersampling methods, as following: 
- Random under-sampling for the majority class
- NearMiss
- Condensed Nearest Neighbor Rule (CNN)
- TomekLinks
- Edited Nearest Neighbor Rule (ENN)
- NeighbourhoodCleaningRule
- ClusterCentroids

[Reference](https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/)

####  Random under-sampling for the majority class
Random undersampling involves randomly selecting examples from the majority class to delete from the training dataset.This has the effect of reducing the number of examples in the majority class in the transformed version of the training dataset.

In random under-sampling (potentially), vast quantities of data are discarded.This can be highly problematic, as the loss of such data can make the decision boundary between minority and majority instances harder to learn, resulting in a loss in classification performance.



```python
from collections import Counter
from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)
```


```python
y = data_clean[target]
X = data_clean.drop(target,axis=1)
```


```python
# RandomUnderSampler
sampler = RandomUnderSampler(sampling_strategy='majority', random_state = 42)
X_rs, y_rs = sampler.fit_sample(X, y)
print('Random undersampling {}'.format(Counter(y_rs)))
df = pd.concat([X_rs,y_rs],axis=1, sort=False)
```

    Random undersampling Counter({0: 15488, 1: 15488})
    


```python
from sklearn.model_selection import train_test_split
train,test = train_test_split(df,test_size=0.4,random_state=1234)
df_hex = h2o.H2OFrame(df)
train_hex = h2o.H2OFrame(train)
test_hex = h2o.H2OFrame(test)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    


```python
rf_v5 = H2ORandomForestEstimator(
        model_id = 'rf_v5',
        ntrees = 300,
        nfolds=10,
        min_rows=100,
        seed=1234)
rf_v5.train(list(predictors),target,training_frame=train_hex)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
createGains(rf_v5)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>1240</td>
      <td>904</td>
      <td>336</td>
      <td>1240</td>
      <td>904</td>
      <td>336</td>
      <td>0.14</td>
      <td>0.05</td>
      <td>623.9</td>
      <td>1.45</td>
      <td>9.0</td>
      <td>72.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1239</td>
      <td>827</td>
      <td>412</td>
      <td>2479</td>
      <td>1731</td>
      <td>748</td>
      <td>0.28</td>
      <td>0.12</td>
      <td>1247.8</td>
      <td>1.39</td>
      <td>16.0</td>
      <td>69.83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1239</td>
      <td>765</td>
      <td>474</td>
      <td>3718</td>
      <td>2496</td>
      <td>1222</td>
      <td>0.40</td>
      <td>0.20</td>
      <td>1871.7</td>
      <td>1.33</td>
      <td>20.0</td>
      <td>67.13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1239</td>
      <td>708</td>
      <td>531</td>
      <td>4957</td>
      <td>3204</td>
      <td>1753</td>
      <td>0.51</td>
      <td>0.28</td>
      <td>2495.6</td>
      <td>1.28</td>
      <td>23.0</td>
      <td>64.64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1239</td>
      <td>647</td>
      <td>592</td>
      <td>6196</td>
      <td>3851</td>
      <td>2345</td>
      <td>0.62</td>
      <td>0.38</td>
      <td>3119.5</td>
      <td>1.23</td>
      <td>24.0</td>
      <td>62.15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1239</td>
      <td>607</td>
      <td>632</td>
      <td>7435</td>
      <td>4458</td>
      <td>2977</td>
      <td>0.71</td>
      <td>0.48</td>
      <td>3743.4</td>
      <td>1.19</td>
      <td>23.0</td>
      <td>59.96</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1239</td>
      <td>567</td>
      <td>672</td>
      <td>8674</td>
      <td>5025</td>
      <td>3649</td>
      <td>0.81</td>
      <td>0.59</td>
      <td>4367.3</td>
      <td>1.15</td>
      <td>22.0</td>
      <td>57.93</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1239</td>
      <td>512</td>
      <td>727</td>
      <td>9913</td>
      <td>5537</td>
      <td>4376</td>
      <td>0.89</td>
      <td>0.71</td>
      <td>4991.2</td>
      <td>1.11</td>
      <td>18.0</td>
      <td>55.86</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1239</td>
      <td>402</td>
      <td>837</td>
      <td>11152</td>
      <td>5939</td>
      <td>5213</td>
      <td>0.95</td>
      <td>0.85</td>
      <td>5615.1</td>
      <td>1.06</td>
      <td>10.0</td>
      <td>53.26</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1239</td>
      <td>300</td>
      <td>939</td>
      <td>12391</td>
      <td>6239</td>
      <td>6152</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>6239.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>50.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
ROC_AUC(rf_v5,test_hex,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_108_1.png)


#### Nearest Neighbours Clearning Rule 
The Neighborhood Cleaning Rule, or NCR for short, is an undersampling technique that combines both the Condensed Nearest Neighbor (CNN) Rule to remove redundant examples and the Edited Nearest Neighbors (ENN) Rule to remove noisy or ambiguous examples.

The number of neighbors used in the ENN and CNN steps can be specified via the n_neighbors argument that defaults to three. The threshold_cleaning controls whether or not the CNN is applied to a given class, which might be useful if there are multiple minority classes with similar sizes. This is kept at 0.5.


```python
# NeighbourhoodCleaningRule
sampler = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
X_rs, y_rs = sampler.fit_sample(X, y)
print('NearestNeighbours Clearning Rule undersampling {}'.format(Counter(y_rs)))
df = pd.concat([X_rs,y_rs],axis=1, sort=False)
```

    NearestNeighbours Clearning Rule undersampling Counter({0: 34903, 1: 15488})
    


```python
train,test = train_test_split(df,test_size=0.4,random_state=1234)
df_hex = h2o.H2OFrame(df)
train_hex = h2o.H2OFrame(train)
test_hex = h2o.H2OFrame(test)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    


```python
rf_v6 = H2ORandomForestEstimator(
        model_id = 'rf_v6',
        ntrees = 300,
        nfolds=10,
        min_rows=100,
        seed=1234)
rf_v6.train(list(predictors),target,training_frame=train_hex)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
createGains(rf_v6)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>2016</td>
      <td>1097</td>
      <td>919</td>
      <td>2016</td>
      <td>1097</td>
      <td>919</td>
      <td>0.18</td>
      <td>0.07</td>
      <td>623.8</td>
      <td>1.76</td>
      <td>11.0</td>
      <td>54.41</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016</td>
      <td>946</td>
      <td>1070</td>
      <td>4032</td>
      <td>2043</td>
      <td>1989</td>
      <td>0.33</td>
      <td>0.14</td>
      <td>1247.6</td>
      <td>1.64</td>
      <td>19.0</td>
      <td>50.67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>796</td>
      <td>1220</td>
      <td>6048</td>
      <td>2839</td>
      <td>3209</td>
      <td>0.46</td>
      <td>0.23</td>
      <td>1871.4</td>
      <td>1.52</td>
      <td>23.0</td>
      <td>46.94</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>698</td>
      <td>1317</td>
      <td>8063</td>
      <td>3537</td>
      <td>4526</td>
      <td>0.57</td>
      <td>0.33</td>
      <td>2495.2</td>
      <td>1.42</td>
      <td>24.0</td>
      <td>43.87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016</td>
      <td>628</td>
      <td>1388</td>
      <td>10079</td>
      <td>4165</td>
      <td>5914</td>
      <td>0.67</td>
      <td>0.42</td>
      <td>3119.0</td>
      <td>1.34</td>
      <td>25.0</td>
      <td>41.32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016</td>
      <td>559</td>
      <td>1457</td>
      <td>12095</td>
      <td>4724</td>
      <td>7371</td>
      <td>0.76</td>
      <td>0.53</td>
      <td>3742.8</td>
      <td>1.26</td>
      <td>23.0</td>
      <td>39.06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015</td>
      <td>517</td>
      <td>1498</td>
      <td>14110</td>
      <td>5241</td>
      <td>8869</td>
      <td>0.84</td>
      <td>0.64</td>
      <td>4366.6</td>
      <td>1.20</td>
      <td>20.0</td>
      <td>37.14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016</td>
      <td>421</td>
      <td>1595</td>
      <td>16126</td>
      <td>5662</td>
      <td>10464</td>
      <td>0.91</td>
      <td>0.75</td>
      <td>4990.4</td>
      <td>1.13</td>
      <td>16.0</td>
      <td>35.11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016</td>
      <td>364</td>
      <td>1652</td>
      <td>18142</td>
      <td>6026</td>
      <td>12116</td>
      <td>0.97</td>
      <td>0.87</td>
      <td>5614.2</td>
      <td>1.07</td>
      <td>10.0</td>
      <td>33.22</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015</td>
      <td>212</td>
      <td>1803</td>
      <td>20157</td>
      <td>6238</td>
      <td>13919</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>6238.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>30.95</td>
    </tr>
  </tbody>
</table>
</div>




```python
ROC_AUC(rf_v6,test_hex,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_114_1.png)


#### Near Miss
Near Miss refers to a collection of undersampling methods that select examples based on the distance of majority class examples to minority class examples.

Here, distance is determined in feature space using Euclidean distance or similar.

- NearMiss-1: Majority class examples with minimum average distance to three closest minority class examples.
- NearMiss-2: Majority class examples with minimum average distance to three furthest minority class examples.
- NearMiss-3: Majority class examples with minimum distance to each minority class example.


```python
# NearMiss-3 that selects the closest examples from the majority class for each minority class.
sampler = NearMiss(version=3, n_neighbors=3)
X_rs, y_rs = sampler.fit_sample(X, y)
print('NearMiss{}'.format(Counter(y_rs)))
df = pd.concat([X_rs,y_rs],axis=1, sort=False)
```

    NearMissCounter({0: 15488, 1: 15488})
    


```python
train,test = train_test_split(df,test_size=0.4,random_state=1234)
df_hex = h2o.H2OFrame(df)
train_hex = h2o.H2OFrame(train)
test_hex = h2o.H2OFrame(test)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    


```python
rf_v7 = H2ORandomForestEstimator(
        model_id = 'rf_v7',
        ntrees = 300,
        nfolds=10,
        min_rows=100,
        seed=1234)
rf_v7.train(list(predictors),target,training_frame=train_hex)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
createGains(rf_v7)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>1240</td>
      <td>1238</td>
      <td>2</td>
      <td>1240</td>
      <td>1238</td>
      <td>2</td>
      <td>0.20</td>
      <td>0.00</td>
      <td>623.9</td>
      <td>1.98</td>
      <td>20.0</td>
      <td>99.84</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1239</td>
      <td>1131</td>
      <td>108</td>
      <td>2479</td>
      <td>2369</td>
      <td>110</td>
      <td>0.38</td>
      <td>0.02</td>
      <td>1247.8</td>
      <td>1.90</td>
      <td>36.0</td>
      <td>95.56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1239</td>
      <td>854</td>
      <td>385</td>
      <td>3718</td>
      <td>3223</td>
      <td>495</td>
      <td>0.52</td>
      <td>0.08</td>
      <td>1871.7</td>
      <td>1.72</td>
      <td>44.0</td>
      <td>86.69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1239</td>
      <td>640</td>
      <td>599</td>
      <td>4957</td>
      <td>3863</td>
      <td>1094</td>
      <td>0.62</td>
      <td>0.18</td>
      <td>2495.6</td>
      <td>1.55</td>
      <td>44.0</td>
      <td>77.93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1239</td>
      <td>589</td>
      <td>650</td>
      <td>6196</td>
      <td>4452</td>
      <td>1744</td>
      <td>0.71</td>
      <td>0.28</td>
      <td>3119.5</td>
      <td>1.43</td>
      <td>43.0</td>
      <td>71.85</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1239</td>
      <td>492</td>
      <td>747</td>
      <td>7435</td>
      <td>4944</td>
      <td>2491</td>
      <td>0.79</td>
      <td>0.40</td>
      <td>3743.4</td>
      <td>1.32</td>
      <td>39.0</td>
      <td>66.50</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1239</td>
      <td>439</td>
      <td>800</td>
      <td>8674</td>
      <td>5383</td>
      <td>3291</td>
      <td>0.86</td>
      <td>0.53</td>
      <td>4367.3</td>
      <td>1.23</td>
      <td>33.0</td>
      <td>62.06</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1239</td>
      <td>347</td>
      <td>892</td>
      <td>9913</td>
      <td>5730</td>
      <td>4183</td>
      <td>0.92</td>
      <td>0.68</td>
      <td>4991.2</td>
      <td>1.15</td>
      <td>24.0</td>
      <td>57.80</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1239</td>
      <td>302</td>
      <td>937</td>
      <td>11152</td>
      <td>6032</td>
      <td>5120</td>
      <td>0.97</td>
      <td>0.83</td>
      <td>5615.1</td>
      <td>1.07</td>
      <td>14.0</td>
      <td>54.09</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1239</td>
      <td>207</td>
      <td>1032</td>
      <td>12391</td>
      <td>6239</td>
      <td>6152</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>6239.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>50.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
ROC_AUC(rf_v7,test_hex,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_120_1.png)


**Undersampling**: Overall, I used 3 out of 7 undersampling methods to evaluate the random forest performance which are Random under-sampling for the majority class, Nearest Neighbours Cleaning Rule, and Near Miss. From these approaches, the NearMiss method had the best results compared to the other two in both the gain table and the scores of AUC and AUCPR. 

However, with undersampling methods, there is a possibility of a loss in classification performance, thus the scores might not be reliable to realistic problems. However, I would recommend apply the models to the real world use case to track and evaulate performance in a live setting.


### Oversampling

OVersampling is another technique to deal with extremely imbalanced data. Oversampling increases the weight of the minority class by replicating the minority class examples. Although it does not increase information, it raises the over-fitting issue, which causes the model to be too specific. 

There are 3 common oversampling techniques, as following:
- Random oversampling for the minority class
- Synthetic Minority Oversampling Technique (SMOTE)
- ADASYN: Adaptive Synthetic Sampling.

[Reference](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/)


```python
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)
```

#### Random oversampling for the minority class
Random oversampling involves randomly duplicating examples from the minority class and adding them to the training dataset.This means that examples from the minority class can be chosen and added to the new “more balanced” training dataset multiple times. They are selected from the original training dataset, added to the new training dataset, and then returned or “replaced” in the original dataset, allowing them to be selected again.

The random oversampling may increase the likelihood of overfitting, since it makes exact copies of the minority class examples.


```python
sampler = RandomOverSampler(sampling_strategy='minority')
X_rs, y_rs = sampler.fit_sample(X, y)
print('RandomOverSampler {}'.format(Counter(y_rs)))
df = pd.concat([X_rs,y_rs],axis=1, sort=False)
```

    RandomOverSampler Counter({1: 64512, 0: 64512})
    


```python
train,test = train_test_split(df,test_size=0.4,random_state=1234)
df_hex = h2o.H2OFrame(df)
train_hex = h2o.H2OFrame(train)
test_hex = h2o.H2OFrame(test)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    


```python
rf_v8 = H2ORandomForestEstimator(
        model_id = 'rf_v8',
        ntrees = 300,
        nfolds=10,
        min_rows=100,
        seed=1234)
rf_v8.train(list(predictors),target,training_frame=train_hex)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
createGains(rf_v8)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>5161</td>
      <td>4076</td>
      <td>1085</td>
      <td>5161</td>
      <td>4076</td>
      <td>1085</td>
      <td>0.16</td>
      <td>0.04</td>
      <td>2578.0</td>
      <td>1.58</td>
      <td>12.0</td>
      <td>78.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5161</td>
      <td>3546</td>
      <td>1615</td>
      <td>10322</td>
      <td>7622</td>
      <td>2700</td>
      <td>0.30</td>
      <td>0.10</td>
      <td>5156.0</td>
      <td>1.48</td>
      <td>20.0</td>
      <td>73.84</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5161</td>
      <td>3275</td>
      <td>1886</td>
      <td>15483</td>
      <td>10897</td>
      <td>4586</td>
      <td>0.42</td>
      <td>0.18</td>
      <td>7734.0</td>
      <td>1.41</td>
      <td>24.0</td>
      <td>70.38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5161</td>
      <td>3127</td>
      <td>2034</td>
      <td>20644</td>
      <td>14024</td>
      <td>6620</td>
      <td>0.54</td>
      <td>0.26</td>
      <td>10312.0</td>
      <td>1.36</td>
      <td>28.0</td>
      <td>67.93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5161</td>
      <td>2868</td>
      <td>2293</td>
      <td>25805</td>
      <td>16892</td>
      <td>8913</td>
      <td>0.66</td>
      <td>0.35</td>
      <td>12890.0</td>
      <td>1.31</td>
      <td>31.0</td>
      <td>65.46</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5161</td>
      <td>2528</td>
      <td>2633</td>
      <td>30966</td>
      <td>19420</td>
      <td>11546</td>
      <td>0.75</td>
      <td>0.45</td>
      <td>15468.0</td>
      <td>1.26</td>
      <td>30.0</td>
      <td>62.71</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5161</td>
      <td>2164</td>
      <td>2997</td>
      <td>36127</td>
      <td>21584</td>
      <td>14543</td>
      <td>0.84</td>
      <td>0.56</td>
      <td>18046.0</td>
      <td>1.20</td>
      <td>28.0</td>
      <td>59.74</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5161</td>
      <td>1962</td>
      <td>3199</td>
      <td>41288</td>
      <td>23546</td>
      <td>17742</td>
      <td>0.91</td>
      <td>0.69</td>
      <td>20624.0</td>
      <td>1.14</td>
      <td>22.0</td>
      <td>57.03</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5161</td>
      <td>1410</td>
      <td>3751</td>
      <td>46449</td>
      <td>24956</td>
      <td>21493</td>
      <td>0.97</td>
      <td>0.83</td>
      <td>23202.0</td>
      <td>1.08</td>
      <td>14.0</td>
      <td>53.73</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5161</td>
      <td>824</td>
      <td>4337</td>
      <td>51610</td>
      <td>25780</td>
      <td>25830</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>25780.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>49.95</td>
    </tr>
  </tbody>
</table>
</div>




```python
ROC_AUC(rf_v8,test_hex,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_129_1.png)


####  Synthetic Minority Oversampling Technique (SMOTE)
SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.

A general downside of the approach is that synthetic examples are created without considering the majority class, possibly resulting in ambiguous examples if there is a strong overlap for the classes.


```python
sampler = SMOTE(sampling_strategy='minority',random_state=42)
X_rs, y_rs = sampler.fit_sample(X, y)
print('SMOTE {}'.format(Counter(y_rs)))
df = pd.concat([X_rs,y_rs],axis=1, sort=False)
```

    SMOTE Counter({1: 64512, 0: 64512})
    

The dataset is transformed using the SMOTE and the new class distribution is summarized, showing a balanced distribution now with 64,512 examples in the minority class


```python
train,test = train_test_split(df,test_size=0.4,random_state=1234)
df_hex = h2o.H2OFrame(df)
train_hex = h2o.H2OFrame(train)
test_hex = h2o.H2OFrame(test)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    


```python
rf_v9 = H2ORandomForestEstimator(
        model_id = 'rf_v9',
        ntrees = 300,
        nfolds=10,
        min_rows=100,
        seed=1234)
rf_v9.train(list(predictors),target,training_frame=train_hex)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
createGains(rf_v9)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>5161</td>
      <td>5161</td>
      <td>0</td>
      <td>5161</td>
      <td>5161</td>
      <td>0</td>
      <td>0.20</td>
      <td>0.00</td>
      <td>2578.0</td>
      <td>2.00</td>
      <td>20.0</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5161</td>
      <td>5116</td>
      <td>45</td>
      <td>10322</td>
      <td>10277</td>
      <td>45</td>
      <td>0.40</td>
      <td>0.00</td>
      <td>5156.0</td>
      <td>1.99</td>
      <td>40.0</td>
      <td>99.56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5161</td>
      <td>4678</td>
      <td>483</td>
      <td>15483</td>
      <td>14955</td>
      <td>528</td>
      <td>0.58</td>
      <td>0.02</td>
      <td>7734.0</td>
      <td>1.93</td>
      <td>56.0</td>
      <td>96.59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5161</td>
      <td>3455</td>
      <td>1706</td>
      <td>20644</td>
      <td>18410</td>
      <td>2234</td>
      <td>0.71</td>
      <td>0.09</td>
      <td>10312.0</td>
      <td>1.79</td>
      <td>62.0</td>
      <td>89.18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5161</td>
      <td>2238</td>
      <td>2923</td>
      <td>25805</td>
      <td>20648</td>
      <td>5157</td>
      <td>0.80</td>
      <td>0.20</td>
      <td>12890.0</td>
      <td>1.60</td>
      <td>60.0</td>
      <td>80.02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5161</td>
      <td>1669</td>
      <td>3492</td>
      <td>30966</td>
      <td>22317</td>
      <td>8649</td>
      <td>0.87</td>
      <td>0.33</td>
      <td>15468.0</td>
      <td>1.44</td>
      <td>54.0</td>
      <td>72.07</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5161</td>
      <td>1300</td>
      <td>3861</td>
      <td>36127</td>
      <td>23617</td>
      <td>12510</td>
      <td>0.92</td>
      <td>0.48</td>
      <td>18046.0</td>
      <td>1.31</td>
      <td>44.0</td>
      <td>65.37</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5161</td>
      <td>1017</td>
      <td>4144</td>
      <td>41288</td>
      <td>24634</td>
      <td>16654</td>
      <td>0.96</td>
      <td>0.64</td>
      <td>20624.0</td>
      <td>1.19</td>
      <td>32.0</td>
      <td>59.66</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5161</td>
      <td>744</td>
      <td>4417</td>
      <td>46449</td>
      <td>25378</td>
      <td>21071</td>
      <td>0.98</td>
      <td>0.82</td>
      <td>23202.0</td>
      <td>1.09</td>
      <td>16.0</td>
      <td>54.64</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5161</td>
      <td>402</td>
      <td>4759</td>
      <td>51610</td>
      <td>25780</td>
      <td>25830</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>25780.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>49.95</td>
    </tr>
  </tbody>
</table>
</div>




```python
ROC_AUC(rf_v9,test_hex,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_136_1.png)


#### ADASYN: Adaptive Synthetic Sampling

The key idea of ADASYN algorithm is to use a density distribution as a criterion to automatically decide the number of synthetic samples that need to be generated for each minority data example.


```python
sampler = ADASYN(sampling_strategy='minority',random_state=42)
X_rs, y_rs = sampler.fit_sample(X, y)
print('ADASYN {}'.format(Counter(y_rs)))
df = pd.concat([X_rs,y_rs],axis=1, sort=False)
```

    ADASYN Counter({0: 64512, 1: 63268})
    


```python
train,test = train_test_split(df,test_size=0.4,random_state=1234)
df_hex = h2o.H2OFrame(df)
train_hex = h2o.H2OFrame(train)
test_hex = h2o.H2OFrame(test)
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    Parse progress: |█████████████████████████████████████████████████████████| 100%
    


```python
rf_v10 = H2ORandomForestEstimator(
        model_id = 'rf_v10',
        ntrees = 300,
        nfolds=10,
        min_rows=100,
        seed=1234)
rf_v10.train(list(predictors),target,training_frame=train_hex)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
createGains(rf_v10)
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>5112</td>
      <td>5112</td>
      <td>0</td>
      <td>5112</td>
      <td>5112</td>
      <td>0</td>
      <td>0.20</td>
      <td>0.00</td>
      <td>2530.5</td>
      <td>2.02</td>
      <td>20.0</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5111</td>
      <td>5048</td>
      <td>63</td>
      <td>10223</td>
      <td>10160</td>
      <td>63</td>
      <td>0.40</td>
      <td>0.00</td>
      <td>5061.0</td>
      <td>2.01</td>
      <td>40.0</td>
      <td>99.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5111</td>
      <td>4584</td>
      <td>527</td>
      <td>15334</td>
      <td>14744</td>
      <td>590</td>
      <td>0.58</td>
      <td>0.02</td>
      <td>7591.5</td>
      <td>1.94</td>
      <td>56.0</td>
      <td>96.15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5111</td>
      <td>3371</td>
      <td>1740</td>
      <td>20445</td>
      <td>18115</td>
      <td>2330</td>
      <td>0.72</td>
      <td>0.09</td>
      <td>10122.0</td>
      <td>1.79</td>
      <td>63.0</td>
      <td>88.60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5111</td>
      <td>2256</td>
      <td>2855</td>
      <td>25556</td>
      <td>20371</td>
      <td>5185</td>
      <td>0.81</td>
      <td>0.20</td>
      <td>12652.5</td>
      <td>1.61</td>
      <td>61.0</td>
      <td>79.71</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5112</td>
      <td>1600</td>
      <td>3512</td>
      <td>30668</td>
      <td>21971</td>
      <td>8697</td>
      <td>0.87</td>
      <td>0.34</td>
      <td>15183.0</td>
      <td>1.45</td>
      <td>53.0</td>
      <td>71.64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5111</td>
      <td>1224</td>
      <td>3887</td>
      <td>35779</td>
      <td>23195</td>
      <td>12584</td>
      <td>0.92</td>
      <td>0.49</td>
      <td>17713.5</td>
      <td>1.31</td>
      <td>43.0</td>
      <td>64.83</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5111</td>
      <td>1002</td>
      <td>4109</td>
      <td>40890</td>
      <td>24197</td>
      <td>16693</td>
      <td>0.96</td>
      <td>0.65</td>
      <td>20244.0</td>
      <td>1.20</td>
      <td>31.0</td>
      <td>59.18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5111</td>
      <td>700</td>
      <td>4411</td>
      <td>46001</td>
      <td>24897</td>
      <td>21104</td>
      <td>0.98</td>
      <td>0.82</td>
      <td>22774.5</td>
      <td>1.09</td>
      <td>16.0</td>
      <td>54.12</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5111</td>
      <td>408</td>
      <td>4703</td>
      <td>51112</td>
      <td>25305</td>
      <td>25807</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>25305.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>49.51</td>
    </tr>
  </tbody>
</table>
</div>




```python
ROC_AUC(rf_v10,test_hex,'loan_default')
```

    drf prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/sampling_rf/output_142_1.png)


**Oversampling**: Overall, the results of ADASYN and SMOTE are extremely similar in the gains table and AUC and AUCPR scores. The two models here also provided better results than the random oversampling for the minority class approach. However, with oversampling methods, there is a possibility of overfitting, thus the scores might not be reliable to realistic problems. However, I would recommend applying the model to the real world use case and monitor its performance over time. 

### Conclusion

The given dataset has high level of negative-positive impact to users. Based on my observations, the predictive power increases when we apply optimal hyperparementers. Additionally, the dataset has a wide-range of date and time, which may be investigated further because the variable seems like a good choice for prediction. 

Both undersampling and oversampling result in high AUC scores, compared to the best random forest model I generated. However, the lift score in the gain table of the best random forest model is better. This insight leads me to consider what would work best in a real life use case and I would like to further examine the results with newly generated data to gauge further performance before deciding which one stands out as the best.

If I spent more time on generating and exploring the date-time variable, it could support the predictive models performance. Additionally, I would have liked to spend further time in generating features that have predictive power, such as date data. I am curious how it would affect the model performance if we split the data by date. From my assumption, by splitting the data by date, I would be able to understand how the model would predict with future data, thus simulating the result to the real world problem.
