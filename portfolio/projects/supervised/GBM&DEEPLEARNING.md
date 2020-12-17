
# Supervised Machine Learning with GBM and Deep Learning

Tram Duong
<br>November 16, 2020

## Table of Contents:
* [Part 1: EDA and FE](#Part_1)
* [Part 2: Data Preparation](#Part_2)
* [Part 3: Supervised Learning](#Part_3)

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




    <matplotlib.axes._subplots.AxesSubplot at 0x2c2f50f5f28>




![png](/assets/img/gbm_deeplearning/output_45_1.png)


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
    loan_default    80000 non-null int64
    AP001           80000 non-null int64
    AP002           80000 non-null int64
    AP003           80000 non-null int64
    AP006           80000 non-null int32
    AP007           80000 non-null int64
    AP008           80000 non-null int64
    AP009           80000 non-null int64
    TD001           80000 non-null int64
    TD002           80000 non-null int64
    TD005           80000 non-null int64
    TD006           80000 non-null int64
    TD013           80000 non-null int64
    TD014           80000 non-null int64
    TD015           80000 non-null int64
    TD023           80000 non-null float64
    TD024           80000 non-null float64
    TD025           80000 non-null float64
    TD026           80000 non-null float64
    TD027           80000 non-null float64
    TD028           80000 non-null float64
    TD029           80000 non-null float64
    CR004           80000 non-null int64
    CR005           80000 non-null int64
    CR009           80000 non-null int64
    CR012           80000 non-null int64
    CR015           80000 non-null int64
    CR017           80000 non-null int64
    PA022           80000 non-null float64
    PA028           80000 non-null float64
    PA030           80000 non-null float64
    CD008           80000 non-null float64
    CD018           80000 non-null float64
    CD071           80000 non-null float64
    CD072           80000 non-null float64
    CD088           80000 non-null float64
    CD100           80000 non-null float64
    CD113           80000 non-null float64
    CD115           80000 non-null float64
    CD130           80000 non-null float64
    CD131           80000 non-null float64
    CD152           80000 non-null float64
    CD153           80000 non-null float64
    CD160           80000 non-null float64
    CD166           80000 non-null float64
    MB005           80000 non-null float64
    MB007           80000 non-null int32
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


![png](/assets/img/gbm_deeplearning/output_56_0.png)



![png](/assets/img/gbm_deeplearning/output_56_1.png)



![png](/assets/img/gbm_deeplearning/output_56_2.png)



![png](/assets/img/gbm_deeplearning/output_56_3.png)



![png](/assets/img/gbm_deeplearning/output_56_4.png)



![png](/assets/img/gbm_deeplearning/output_56_5.png)



![png](/assets/img/gbm_deeplearning/output_56_6.png)



![png](/assets/img/gbm_deeplearning/output_56_7.png)



![png](/assets/img/gbm_deeplearning/output_56_8.png)



![png](/assets/img/gbm_deeplearning/output_56_9.png)



![png](/assets/img/gbm_deeplearning/output_56_10.png)



![png](/assets/img/gbm_deeplearning/output_56_11.png)



![png](/assets/img/gbm_deeplearning/output_56_12.png)



![png](/assets/img/gbm_deeplearning/output_56_13.png)



![png](/assets/img/gbm_deeplearning/output_56_14.png)



![png](/assets/img/gbm_deeplearning/output_56_15.png)



![png](/assets/img/gbm_deeplearning/output_56_16.png)



![png](/assets/img/gbm_deeplearning/output_56_17.png)



![png](/assets/img/gbm_deeplearning/output_56_18.png)



![png](/assets/img/gbm_deeplearning/output_56_19.png)



![png](/assets/img/gbm_deeplearning/output_56_20.png)



![png](/assets/img/gbm_deeplearning/output_56_21.png)



![png](/assets/img/gbm_deeplearning/output_56_22.png)



![png](/assets/img/gbm_deeplearning/output_56_23.png)


## Part 3: Supervised Learning with Gradient Boosting Machine <a class="anchor" id="Part_3"></a>

GBM constructs a forward stage-wise additive model by implementing gradient descent in function space. GBMs have several hyperparameters that include the number of trees, the depth (or number of leaves), and the shrinkage (or learning rate). GBM is a boosting method, which builds on weak classifiers. The idea is to add a classifier at a time, so that the next classifier is trained to improve the already trained ensemble. 


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
      Ice root: C:\Users\tramh\AppData\Local\Temp\tmpni5k843q
      JVM stdout: C:\Users\tramh\AppData\Local\Temp\tmpni5k843q\h2o_tramh_started_from_python.out
      JVM stderr: C:\Users\tramh\AppData\Local\Temp\tmpni5k843q\h2o_tramh_started_from_python.err
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
<td>1 month and 17 days </td></tr>
<tr><td>H2O_cluster_name:</td>
<td>H2O_from_python_tramh_at4jin</td></tr>
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

Given that we have an highly unbalanced dataset, I'm using the H2O balance_classes. The balance_classes option can be used to balance the class distribution. When enabled, H2O will either undersample the majority classes or oversample the minority classes.

[Reference](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance-and-prediction.html?highlight=AUC)


```python
gbm_v1 = H2OGradientBoostingEstimator(
        model_id = 'gbm_v1',
        max_depth = 7,
        min_rows = 100,
        nfolds = 10,
        ntrees = 10000,
        learn_rate = 0.01,
        stopping_rounds = 15, stopping_tolerance = 1e-5,
        sample_rate = 0.8,
        balance_classes = True,
        col_sample_rate = 0.8,
        score_tree_interval = 10,       
        seed=1234)
```


```python
gbm_v1.train(list(predictors),target,training_frame=train_hex,validation_frame = test_hex)
```

    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
VarImp(gbm_v1)
```


![png](/assets/img/gbm_deeplearning/output_68_0.png)



```python
predictions = gbm_v1.predict(test_hex)
predictions.head()
test_scores = test_hex['loan_default'].cbind(predictions).as_data_frame()
test_scores.head()
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>0.158851</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.139201</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.068588</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.191419</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.210051</td>
    </tr>
  </tbody>
</table>
</div>




```python
createGains(gbm_v1)
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>58</td>
      <td>182</td>
      <td>480</td>
      <td>153</td>
      <td>327</td>
      <td>0.33</td>
      <td>0.17</td>
      <td>93.6</td>
      <td>1.63</td>
      <td>16.0</td>
      <td>31.87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>70</td>
      <td>170</td>
      <td>720</td>
      <td>223</td>
      <td>497</td>
      <td>0.48</td>
      <td>0.26</td>
      <td>140.4</td>
      <td>1.59</td>
      <td>22.0</td>
      <td>30.97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>56</td>
      <td>184</td>
      <td>960</td>
      <td>279</td>
      <td>681</td>
      <td>0.60</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.49</td>
      <td>25.0</td>
      <td>29.06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>48</td>
      <td>192</td>
      <td>1200</td>
      <td>327</td>
      <td>873</td>
      <td>0.70</td>
      <td>0.45</td>
      <td>234.0</td>
      <td>1.40</td>
      <td>25.0</td>
      <td>27.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>41</td>
      <td>199</td>
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
      <td>34</td>
      <td>206</td>
      <td>1680</td>
      <td>402</td>
      <td>1278</td>
      <td>0.86</td>
      <td>0.66</td>
      <td>327.6</td>
      <td>1.23</td>
      <td>20.0</td>
      <td>23.93</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>30</td>
      <td>210</td>
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
      <td>24</td>
      <td>216</td>
      <td>2160</td>
      <td>456</td>
      <td>1704</td>
      <td>0.97</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.08</td>
      <td>9.0</td>
      <td>21.11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>12</td>
      <td>228</td>
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

- Lift = decile 0 of model 1 has 2.03 times greater lift than random selection.
- K_S= abs(cumulative % of total good loan applicants— cumulative % of total bad loan applicants) -> The higher the value, the better the model is at separating the positive cases from negative ones. 


```python
ROC_AUC(gbm_v1,test_hex,'loan_default')
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/gbm_deeplearning/output_72_1.png)


The AUC and PR scores also have positive results (0.6704 and 0.3318). Overall, the parameters used in the first model are more likely to work well.  

### Model 2: 

After testing on small datasets and the results are promising, I apply the same code to the entire dataset. 


```python
gbm_v2 =  H2OGradientBoostingEstimator(
        model_id = 'gbm_v2',
        max_depth = 6,
        min_rows = 50,
        nfolds = 10,
        ntrees = 10000,
        learn_rate = 0.01,
        stopping_rounds = 15, stopping_tolerance = 1e-5,
        sample_rate = 0.8,
        balance_classes = True,
        col_sample_rate = 0.8,
        score_tree_interval = 10,       
        seed=1234)
```


```python
gbm_v2.train(list(predictors),target,training_frame=train_full,validation_frame = test_full)
```

    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
predictions_2 = gbm_v2.predict(test_full)
predictions_2.head()
test_scores_2 = test_full['loan_default'].cbind(predictions_2).as_data_frame()
test_scores_2.head()
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>0.364993</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.140810</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.176016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.123112</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.389896</td>
    </tr>
  </tbody>
</table>
</div>




```python
createGains(gbm_v2)
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>98</td>
      <td>142</td>
      <td>240</td>
      <td>98</td>
      <td>142</td>
      <td>0.21</td>
      <td>0.07</td>
      <td>46.8</td>
      <td>2.09</td>
      <td>14.0</td>
      <td>40.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>73</td>
      <td>167</td>
      <td>480</td>
      <td>171</td>
      <td>309</td>
      <td>0.37</td>
      <td>0.16</td>
      <td>93.6</td>
      <td>1.83</td>
      <td>21.0</td>
      <td>35.62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>61</td>
      <td>179</td>
      <td>720</td>
      <td>232</td>
      <td>488</td>
      <td>0.50</td>
      <td>0.25</td>
      <td>140.4</td>
      <td>1.65</td>
      <td>25.0</td>
      <td>32.22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>56</td>
      <td>184</td>
      <td>960</td>
      <td>288</td>
      <td>672</td>
      <td>0.62</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.54</td>
      <td>27.0</td>
      <td>30.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>42</td>
      <td>198</td>
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
      <td>43</td>
      <td>197</td>
      <td>1440</td>
      <td>373</td>
      <td>1067</td>
      <td>0.80</td>
      <td>0.55</td>
      <td>280.8</td>
      <td>1.33</td>
      <td>25.0</td>
      <td>25.90</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>31</td>
      <td>209</td>
      <td>1680</td>
      <td>404</td>
      <td>1276</td>
      <td>0.86</td>
      <td>0.66</td>
      <td>327.6</td>
      <td>1.23</td>
      <td>20.0</td>
      <td>24.05</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>32</td>
      <td>208</td>
      <td>1920</td>
      <td>436</td>
      <td>1484</td>
      <td>0.93</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.16</td>
      <td>16.0</td>
      <td>22.71</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>21</td>
      <td>219</td>
      <td>2160</td>
      <td>457</td>
      <td>1703</td>
      <td>0.98</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.08</td>
      <td>10.0</td>
      <td>21.16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>11</td>
      <td>229</td>
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



The gain table of model 2 results in slightly better scores compare to model 1, as well as the AUC and PR scores.


```python
ROC_AUC(gbm_v2,test_full,'loan_default')
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/gbm_deeplearning/output_80_1.png)


The model 2 (full data) use the same set of parameters like model 1 (sample data). The results of both models are positive. However, I would like to define the model with Hyperparameter Tuning approach using H2O Grid-search. 

### Model 3: Hyperparameter Tuning 
In this section, I used H2O Grid-search to find the optimal hyper-parameters for the model.

As 'loan_default' contains binary values represented by 0's and 1's, it is assumed to be numeric in H20 and is applied as such. Thus, it would be trained as a regression model instead of a binary classification model, and AUC is not allowed. Therefore, I will define the best model using the rmse and r2 scores. 


```python
hyper_params_tune = {
    'learn_rate':[0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
    'max_depth':[4, 5, 6, 7, 8, 9, 10],
    'ntrees':[300, 500, 700, 900, 1100],
    'stopping_tolerance':[0.0001,0.00001]}

gbm_v3 = H2OGradientBoostingEstimator(
    stopping_rounds = 15,
    col_sample_rate = 0.80,
    sample_rate = 0.80,
    seed = 1234) 

#Build grid search with previously made GBM and hyper parameters
gbm_grid = H2OGridSearch(gbm_v3, hyper_params = hyper_params_tune,
                         search_criteria = {'strategy': "Cartesian"})
```


```python
gbm_grid.train(list(predictors),target,training_frame=train_hex, validation_frame = test_hex)
```

    gbm Grid Build progress: |████████████████████████████████████████████████| 100%
    


```python
# Get the grid results, sorted by validation rmse and r2
sorted_gbm_grid = gbm_grid.get_grid(sort_by='rmse',decreasing=False)
sorted_gbm_grid1 = gbm_grid.get_grid(sort_by='r2',decreasing=True)
print(sorted_gbm_grid)
print(sorted_gbm_grid1)
```

           learn_rate max_depth ntrees stopping_tolerance  \
    0            0.01         4    500             1.0E-5   
    1            0.01         4    700             1.0E-5   
    2            0.01         4    700             1.0E-4   
    3            0.01         5    300             1.0E-5   
    4            0.01         5    500             1.0E-5   
    5            0.01         5    900             1.0E-5   
    6            0.01         5    700             1.0E-5   
    7            0.03         4    300             1.0E-5   
    8            0.03         4    500             1.0E-5   
    9            0.03         4    700             1.0E-5   
    10           0.03         4    900             1.0E-5   
    11           0.03         4   1100             1.0E-5   
    12           0.01         7    500             1.0E-5   
    13           0.01         4    900             1.0E-4   
    14           0.01         4   1100             1.0E-4   
    15           0.01         4    500             1.0E-4   
    16           0.03         4    900             1.0E-4   
    17           0.03         4   1100             1.0E-4   
    18           0.03         4    300             1.0E-4   
    19           0.03         4    500             1.0E-4   
    20           0.03         4    700             1.0E-4   
    21           0.01         7    300             1.0E-5   
    22           0.01         6    300             1.0E-5   
    23           0.01         4    900             1.0E-5   
    24           0.05         5    300             1.0E-4   
    25           0.05         5    900             1.0E-4   
    26           0.05         5   1100             1.0E-4   
    27           0.05         5    300             1.0E-5   
    28           0.05         5    500             1.0E-5   
    29           0.05         5    700             1.0E-5   
    ..  ..        ...       ...    ...                ...   
    320          0.04         9   1100             1.0E-4   
    321          0.04         9    300             1.0E-4   
    322          0.04         9    500             1.0E-4   
    323          0.04         9    700             1.0E-4   
    324          0.04         9    300             1.0E-5   
    325          0.04         9    500             1.0E-5   
    326          0.04         9    700             1.0E-5   
    327          0.04         9    900             1.0E-5   
    328          0.04         9   1100             1.0E-5   
    329          0.05        10    700             1.0E-4   
    330          0.05        10    900             1.0E-4   
    331          0.05        10   1100             1.0E-4   
    332          0.05        10    300             1.0E-4   
    333          0.05        10    500             1.0E-4   
    334          0.01        10   1100             1.0E-5   
    335          0.05        10    300             1.0E-5   
    336          0.05        10    500             1.0E-5   
    337          0.05        10    700             1.0E-5   
    338          0.05        10    900             1.0E-5   
    339          0.05        10   1100             1.0E-5   
    340          0.05         9    700             1.0E-4   
    341          0.05         9    900             1.0E-4   
    342          0.05         9   1100             1.0E-4   
    343          0.05         9    300             1.0E-4   
    344          0.05         9    500             1.0E-4   
    345          0.05         9    300             1.0E-5   
    346          0.05         9    500             1.0E-5   
    347          0.05         9    700             1.0E-5   
    348          0.05         9    900             1.0E-5   
    349          0.05         9   1100             1.0E-5   
    
                                                                     model_ids  \
    0    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    1    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    2    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    3    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    4    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    5    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    6    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    7    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    8    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    9    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    10   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    11   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    12   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    13   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    14   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    15   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    16   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    17   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    18   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    19   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    20   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    21   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    22   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    23   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    24   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    25   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    26   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    27   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    28   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    29   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    ..                                                                     ...   
    320  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    321  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    322  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    323  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    324  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    325  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    326  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    327  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    328  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    329  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    330  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    331  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    332  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    333  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    334  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    335  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    336  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    337  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    338  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    339  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    340  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    341  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    342  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    343  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    344  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    345  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    346  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    347  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    348  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    349  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    
                        rmse  
    0    0.38384619714986123  
    1    0.38407594362426173  
    2    0.38407594362426173  
    3     0.3843114670344269  
    4     0.3843114670344269  
    5     0.3843114670344269  
    6    0.38433921403571697  
    7     0.3843448041609049  
    8     0.3843448041609049  
    9     0.3843448041609049  
    10    0.3843448041609049  
    11    0.3843448041609049  
    12    0.3844281120169422  
    13    0.3844283762195627  
    14    0.3844283762195627  
    15    0.3844283762195627  
    16   0.38448872577587845  
    17   0.38448872577587845  
    18   0.38448872577587845  
    19   0.38448872577587845  
    20   0.38448872577587845  
    21   0.38450167663683427  
    22   0.38450587815199916  
    23   0.38452246802659484  
    24   0.38454722470245184  
    25   0.38454722470245184  
    26   0.38454722470245184  
    27   0.38454722470245184  
    28   0.38454722470245184  
    29   0.38454722470245184  
    ..                   ...  
    320   0.3885178841243181  
    321   0.3885178841243181  
    322   0.3885178841243181  
    323   0.3885178841243181  
    324   0.3885436704771476  
    325   0.3885436704771476  
    326   0.3885436704771476  
    327   0.3885436704771476  
    328   0.3885436704771476  
    329  0.38909489295612776  
    330  0.38909489295612776  
    331  0.38909489295612776  
    332  0.38909489295612776  
    333  0.38909489295612776  
    334  0.38916724361005695  
    335  0.38933313558279387  
    336  0.38933313558279387  
    337  0.38933313558279387  
    338  0.38933313558279387  
    339  0.38933313558279387  
    340   0.3898021823152309  
    341   0.3898021823152309  
    342   0.3898021823152309  
    343   0.3898021823152309  
    344   0.3898021823152309  
    345  0.39039618215896094  
    346  0.39039618215896094  
    347  0.39039618215896094  
    348  0.39039618215896094  
    349  0.39039618215896094  
    
    [350 rows x 7 columns]
    
           learn_rate max_depth ntrees stopping_tolerance  \
    0            0.01         4    500             1.0E-5   
    1            0.01         4    700             1.0E-5   
    2            0.01         4    700             1.0E-4   
    3            0.01         5    300             1.0E-5   
    4            0.01         5    500             1.0E-5   
    5            0.01         5    900             1.0E-5   
    6            0.01         5    700             1.0E-5   
    7            0.03         4    300             1.0E-5   
    8            0.03         4    500             1.0E-5   
    9            0.03         4    700             1.0E-5   
    10           0.03         4    900             1.0E-5   
    11           0.03         4   1100             1.0E-5   
    12           0.01         7    500             1.0E-5   
    13           0.01         4    900             1.0E-4   
    14           0.01         4   1100             1.0E-4   
    15           0.01         4    500             1.0E-4   
    16           0.03         4    900             1.0E-4   
    17           0.03         4   1100             1.0E-4   
    18           0.03         4    300             1.0E-4   
    19           0.03         4    500             1.0E-4   
    20           0.03         4    700             1.0E-4   
    21           0.01         7    300             1.0E-5   
    22           0.01         6    300             1.0E-5   
    23           0.01         4    900             1.0E-5   
    24           0.05         5    300             1.0E-4   
    25           0.05         5    900             1.0E-4   
    26           0.05         5   1100             1.0E-4   
    27           0.05         5    300             1.0E-5   
    28           0.05         5    500             1.0E-5   
    29           0.05         5    700             1.0E-5   
    ..  ..        ...       ...    ...                ...   
    320          0.04         9   1100             1.0E-4   
    321          0.04         9    300             1.0E-4   
    322          0.04         9    500             1.0E-4   
    323          0.04         9    700             1.0E-4   
    324          0.04         9    300             1.0E-5   
    325          0.04         9    500             1.0E-5   
    326          0.04         9    700             1.0E-5   
    327          0.04         9    900             1.0E-5   
    328          0.04         9   1100             1.0E-5   
    329          0.05        10    700             1.0E-4   
    330          0.05        10    900             1.0E-4   
    331          0.05        10   1100             1.0E-4   
    332          0.05        10    300             1.0E-4   
    333          0.05        10    500             1.0E-4   
    334          0.01        10   1100             1.0E-5   
    335          0.05        10    300             1.0E-5   
    336          0.05        10    500             1.0E-5   
    337          0.05        10    700             1.0E-5   
    338          0.05        10    900             1.0E-5   
    339          0.05        10   1100             1.0E-5   
    340          0.05         9    700             1.0E-4   
    341          0.05         9    900             1.0E-4   
    342          0.05         9   1100             1.0E-4   
    343          0.05         9    300             1.0E-4   
    344          0.05         9    500             1.0E-4   
    345          0.05         9    300             1.0E-5   
    346          0.05         9    500             1.0E-5   
    347          0.05         9    700             1.0E-5   
    348          0.05         9    900             1.0E-5   
    349          0.05         9   1100             1.0E-5   
    
                                                                     model_ids  \
    0    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    1    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    2    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    3    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    4    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    5    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    6    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    7    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    8    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    9    Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    10   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    11   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    12   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    13   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    14   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    15   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    16   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    17   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    18   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    19   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    20   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    21   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    22   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    23   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    24   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    25   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    26   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    27   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    28   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    29   Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    ..                                                                     ...   
    320  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    321  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    322  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    323  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    324  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    325  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    326  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    327  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    328  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    329  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    330  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    331  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    332  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    333  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    334  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    335  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    336  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    337  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    338  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    339  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    340  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    341  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    342  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    343  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    344  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    345  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    346  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    347  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    348  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    349  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_mo...   
    
                           r2  
    0     0.06139255890167117  
    1     0.06026863850379305  
    2     0.06026863850379305  
    3    0.059115759234569465  
    4    0.059115759234569465  
    5    0.059115759234569465  
    6    0.058979892049099036  
    7    0.058952518009336874  
    8    0.058952518009336874  
    9    0.058952518009336874  
    10   0.058952518009336874  
    11   0.058952518009336874  
    12   0.058544524230541906  
    13    0.05854323017799212  
    14    0.05854323017799212  
    15    0.05854323017799212  
    16     0.0582476174629164  
    17     0.0582476174629164  
    18     0.0582476174629164  
    19     0.0582476174629164  
    20     0.0582476174629164  
    21    0.05818417368028872  
    22    0.05816359080465017  
    23   0.058082316182425986  
    24    0.05796102547311355  
    25    0.05796102547311355  
    26    0.05796102547311355  
    27    0.05796102547311355  
    28    0.05796102547311355  
    29    0.05796102547311355  
    ..                    ...  
    320   0.03840645781533947  
    321   0.03840645781533947  
    322   0.03840645781533947  
    323   0.03840645781533947  
    324   0.03827880956933083  
    325   0.03827880956933083  
    326   0.03827880956933083  
    327   0.03827880956933083  
    328   0.03827880956933083  
    329   0.03554810814116549  
    330   0.03554810814116549  
    331   0.03554810814116549  
    332   0.03554810814116549  
    333   0.03554810814116549  
    334    0.0351894027771974  
    335   0.03436667964497431  
    336   0.03436667964497431  
    337   0.03436667964497431  
    338   0.03436667964497431  
    339   0.03436667964497431  
    340  0.032038596351542936  
    341  0.032038596351542936  
    342  0.032038596351542936  
    343  0.032038596351542936  
    344  0.032038596351542936  
    345  0.029086293713695643  
    346  0.029086293713695643  
    347  0.029086293713695643  
    348  0.029086293713695643  
    349  0.029086293713695643  
    
    [350 rows x 7 columns]
    
    

Both r2 and rmse result in the same set of optimal hyperparameters as in the model summary above.The model metrics also show high scores on AUC and AUCPR (Both are approximately 0.95). Additionaly the other statistics also provide positive insights.

From the above, I will move forward to train the model with the full data.


```python
best_gbm = sorted_gbm_grid.models[0]
```


```python
print(best_gbm)
```

    Model Details
    =============
    H2OGradientBoostingEstimator :  Gradient Boosting Machine
    Model Key:  Grid_GBM_Key_Frame__upload_a83cd714b687b66823ddb37c3d8a42e8.hex_model_python_1605423428293_1_model_211
    
    
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
      <td>500.0</td>
      <td>500.0</td>
      <td>117809.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>14.056</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    ModelMetricsRegression: gbm
    ** Reported on train data. **
    
    MSE: 0.12525703770052998
    RMSE: 0.3539167101176914
    MAE: 0.27034507821502174
    RMSLE: 0.24615938231141338
    Mean Residual Deviance: 0.12525703770052998
    
    ModelMetricsRegression: gbm
    ** Reported on validation data. **
    
    MSE: 0.14733790306641015
    RMSE: 0.38384619714986123
    MAE: 0.29291711617784505
    RMSLE: 0.26889468971732816
    Mean Residual Deviance: 0.14733790306641015
    
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
      <th>training_mae</th>
      <th>training_deviance</th>
      <th>validation_rmse</th>
      <th>validation_mae</th>
      <th>validation_deviance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>2020-11-15 02:14:43</td>
      <td>6 min 56.088 sec</td>
      <td>0.0</td>
      <td>0.394542</td>
      <td>0.311327</td>
      <td>0.155663</td>
      <td>0.396207</td>
      <td>0.312643</td>
      <td>0.156980</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>2020-11-15 02:14:43</td>
      <td>6 min 56.103 sec</td>
      <td>1.0</td>
      <td>0.394291</td>
      <td>0.311160</td>
      <td>0.155465</td>
      <td>0.396069</td>
      <td>0.312560</td>
      <td>0.156871</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>2020-11-15 02:14:43</td>
      <td>6 min 56.103 sec</td>
      <td>2.0</td>
      <td>0.394070</td>
      <td>0.310968</td>
      <td>0.155291</td>
      <td>0.395926</td>
      <td>0.312409</td>
      <td>0.156758</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>2020-11-15 02:14:43</td>
      <td>6 min 56.119 sec</td>
      <td>3.0</td>
      <td>0.393845</td>
      <td>0.310783</td>
      <td>0.155114</td>
      <td>0.395819</td>
      <td>0.312304</td>
      <td>0.156673</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>2020-11-15 02:14:43</td>
      <td>6 min 56.119 sec</td>
      <td>4.0</td>
      <td>0.393618</td>
      <td>0.310623</td>
      <td>0.154935</td>
      <td>0.395684</td>
      <td>0.312190</td>
      <td>0.156566</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.135 sec</td>
      <td>5.0</td>
      <td>0.393382</td>
      <td>0.310447</td>
      <td>0.154749</td>
      <td>0.395601</td>
      <td>0.312118</td>
      <td>0.156500</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.150 sec</td>
      <td>6.0</td>
      <td>0.393171</td>
      <td>0.310251</td>
      <td>0.154583</td>
      <td>0.395465</td>
      <td>0.311966</td>
      <td>0.156393</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.150 sec</td>
      <td>7.0</td>
      <td>0.392968</td>
      <td>0.310081</td>
      <td>0.154424</td>
      <td>0.395373</td>
      <td>0.311867</td>
      <td>0.156320</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.166 sec</td>
      <td>8.0</td>
      <td>0.392730</td>
      <td>0.309890</td>
      <td>0.154237</td>
      <td>0.395236</td>
      <td>0.311750</td>
      <td>0.156211</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.166 sec</td>
      <td>9.0</td>
      <td>0.392516</td>
      <td>0.309744</td>
      <td>0.154069</td>
      <td>0.395104</td>
      <td>0.311645</td>
      <td>0.156107</td>
    </tr>
    <tr>
      <th>10</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.182 sec</td>
      <td>10.0</td>
      <td>0.392286</td>
      <td>0.309557</td>
      <td>0.153888</td>
      <td>0.394960</td>
      <td>0.311512</td>
      <td>0.155993</td>
    </tr>
    <tr>
      <th>11</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.197 sec</td>
      <td>11.0</td>
      <td>0.392068</td>
      <td>0.309375</td>
      <td>0.153717</td>
      <td>0.394878</td>
      <td>0.311407</td>
      <td>0.155929</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.197 sec</td>
      <td>12.0</td>
      <td>0.391853</td>
      <td>0.309192</td>
      <td>0.153549</td>
      <td>0.394760</td>
      <td>0.311290</td>
      <td>0.155836</td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.213 sec</td>
      <td>13.0</td>
      <td>0.391650</td>
      <td>0.309033</td>
      <td>0.153390</td>
      <td>0.394665</td>
      <td>0.311219</td>
      <td>0.155761</td>
    </tr>
    <tr>
      <th>14</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.213 sec</td>
      <td>14.0</td>
      <td>0.391429</td>
      <td>0.308831</td>
      <td>0.153216</td>
      <td>0.394554</td>
      <td>0.311118</td>
      <td>0.155673</td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.229 sec</td>
      <td>15.0</td>
      <td>0.391232</td>
      <td>0.308695</td>
      <td>0.153063</td>
      <td>0.394424</td>
      <td>0.311034</td>
      <td>0.155570</td>
    </tr>
    <tr>
      <th>16</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.244 sec</td>
      <td>16.0</td>
      <td>0.391034</td>
      <td>0.308544</td>
      <td>0.152908</td>
      <td>0.394305</td>
      <td>0.310944</td>
      <td>0.155476</td>
    </tr>
    <tr>
      <th>17</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.244 sec</td>
      <td>17.0</td>
      <td>0.390847</td>
      <td>0.308356</td>
      <td>0.152761</td>
      <td>0.394196</td>
      <td>0.310801</td>
      <td>0.155391</td>
    </tr>
    <tr>
      <th>18</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.260 sec</td>
      <td>18.0</td>
      <td>0.390672</td>
      <td>0.308232</td>
      <td>0.152624</td>
      <td>0.394100</td>
      <td>0.310716</td>
      <td>0.155315</td>
    </tr>
    <tr>
      <th>19</th>
      <td></td>
      <td>2020-11-15 02:14:44</td>
      <td>6 min 56.260 sec</td>
      <td>19.0</td>
      <td>0.390487</td>
      <td>0.308083</td>
      <td>0.152480</td>
      <td>0.394007</td>
      <td>0.310630</td>
      <td>0.155242</td>
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
      <td>861.343628</td>
      <td>1.000000</td>
      <td>0.101128</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AP003</td>
      <td>554.145630</td>
      <td>0.643350</td>
      <td>0.065061</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PA028</td>
      <td>375.473419</td>
      <td>0.435916</td>
      <td>0.044083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AP006</td>
      <td>349.594452</td>
      <td>0.405871</td>
      <td>0.041045</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TD024</td>
      <td>339.329895</td>
      <td>0.393954</td>
      <td>0.039840</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TD005</td>
      <td>308.948090</td>
      <td>0.358682</td>
      <td>0.036273</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CD113</td>
      <td>301.164825</td>
      <td>0.349645</td>
      <td>0.035359</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MB005</td>
      <td>293.567383</td>
      <td>0.340825</td>
      <td>0.034467</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TD001</td>
      <td>290.571106</td>
      <td>0.337346</td>
      <td>0.034115</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CR015</td>
      <td>288.270782</td>
      <td>0.334676</td>
      <td>0.033845</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CD152</td>
      <td>266.712036</td>
      <td>0.309646</td>
      <td>0.031314</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TD014</td>
      <td>252.524719</td>
      <td>0.293175</td>
      <td>0.029648</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CD166</td>
      <td>249.753067</td>
      <td>0.289958</td>
      <td>0.029323</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PA022</td>
      <td>234.203232</td>
      <td>0.271905</td>
      <td>0.027497</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CD153</td>
      <td>228.389389</td>
      <td>0.265155</td>
      <td>0.026815</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CD100</td>
      <td>223.473206</td>
      <td>0.259447</td>
      <td>0.026237</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CD008</td>
      <td>211.511902</td>
      <td>0.245560</td>
      <td>0.024833</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CD115</td>
      <td>207.579514</td>
      <td>0.240995</td>
      <td>0.024371</td>
    </tr>
    <tr>
      <th>18</th>
      <td>AP009</td>
      <td>188.147995</td>
      <td>0.218435</td>
      <td>0.022090</td>
    </tr>
    <tr>
      <th>19</th>
      <td>AP001</td>
      <td>185.421432</td>
      <td>0.215270</td>
      <td>0.021770</td>
    </tr>
  </tbody>
</table>
</div>


    
    See the whole table with table.as_data_frame()
    
    


```python
best_gbm.train(list(predictors),target,training_frame=train_full)
```

    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    


```python
# Prediction with the best model
predictions_3 = best_gbm.predict(test_full)
predictions_3.head()
test_scores_3 = test_full['loan_default'].cbind(predictions_3).as_data_frame()
test_scores_3.head()
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>0.351301</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.157845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.174167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.148247</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.459149</td>
    </tr>
  </tbody>
</table>
</div>




```python
VarImp(best_gbm)
```


![png](/assets/img/gbm_deeplearning/output_92_0.png)



```python
createGains(best_gbm)
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    




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
      <td>67</td>
      <td>173</td>
      <td>480</td>
      <td>164</td>
      <td>316</td>
      <td>0.35</td>
      <td>0.16</td>
      <td>93.6</td>
      <td>1.75</td>
      <td>19.0</td>
      <td>34.17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>65</td>
      <td>175</td>
      <td>720</td>
      <td>229</td>
      <td>491</td>
      <td>0.49</td>
      <td>0.25</td>
      <td>140.4</td>
      <td>1.63</td>
      <td>24.0</td>
      <td>31.81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>54</td>
      <td>186</td>
      <td>960</td>
      <td>283</td>
      <td>677</td>
      <td>0.60</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.51</td>
      <td>25.0</td>
      <td>29.48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>41</td>
      <td>199</td>
      <td>1200</td>
      <td>324</td>
      <td>876</td>
      <td>0.69</td>
      <td>0.45</td>
      <td>234.0</td>
      <td>1.38</td>
      <td>24.0</td>
      <td>27.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>46</td>
      <td>194</td>
      <td>1440</td>
      <td>370</td>
      <td>1070</td>
      <td>0.79</td>
      <td>0.55</td>
      <td>280.8</td>
      <td>1.32</td>
      <td>24.0</td>
      <td>25.69</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>28</td>
      <td>212</td>
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
      <td>38</td>
      <td>202</td>
      <td>1920</td>
      <td>436</td>
      <td>1484</td>
      <td>0.93</td>
      <td>0.77</td>
      <td>374.4</td>
      <td>1.16</td>
      <td>16.0</td>
      <td>22.71</td>
    </tr>
    <tr>
      <th>8</th>
      <td>240</td>
      <td>20</td>
      <td>220</td>
      <td>2160</td>
      <td>456</td>
      <td>1704</td>
      <td>0.97</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.08</td>
      <td>9.0</td>
      <td>21.11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>12</td>
      <td>228</td>
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
ROC_AUC(best_gbm,test_full,'loan_default')
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/gbm_deeplearning/output_94_1.png)


The best model has a result similar like the other previous models. However, I experienced faster running time with the selected model, providing me similar results in less time. This insight could be due to the much lower number of trees in the best model (500) compared to the other ones (10000).

### Deep Learning

While H2O Deep Learning has many parameters, it was designed to be just as easy to use as the other supervised training methods in H2O. Early stopping, automatic data standardization and handling of categorical variables and missing values and adaptive learning rates (per weight) reduce the amount of parameters the user has to specify. Often, it's just the number and sizes of hidden layers, the number of epochs and the activation function and maybe some regularization techniques. 

* input_dropout_ratio: Specify the input layer dropout ratio to improve generalization. Suggested values are 0.1 or 0.2. This option defaults to 0. 
* hidden: Specify the hidden layer sizes (e.g., 100,100). The value must be positive. This option defaults to (200,200).
* epochs: Specify the number of times to iterate (stream) the dataset. The value can be a fraction. This option defaults to 10.

[References](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html)

The data has: 
* 80000 rows, 47 columns (variables)
* These 46 variables are all numerical 
* 1 target variable (0 or 1)

The H2O Architecture uses the hidden keyword to control model network architecture. Hidden takes a list of integers, representing the number of nodes in each layer.
* 1. Single layer, 1000 nodes
* 2. Two layers, 200 nodes each
* 3. Three layers, 7 nodes each (trying with square root of 47) 
* 4. Four layers,  15 -> 20 -> 25 -> 30

Epochs = 3000. 

[Reference](https://github.com/h2oai/h2o-tutorials)

#### Model 1: 1000 nodes and one layer 


```python
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
DL_modl_1 = H2ODeepLearningEstimator(
    distribution="tweedie",
    activation="RectifierWithDropout",
    hidden=[1000],
    input_dropout_ratio=0.1,
    sparse=True,
    l1=1e-5,
    epochs=3000,
    seed =1234)
```


```python
DL_modl_1.train(
    x=list(predictors),
    y=target,
    training_frame=train_hex,
    validation_frame=test_hex)
```

    deeplearning Model Build progress: |██████████████████████████████████████| 100%
    


```python
createGains(DL_modl_1)
```

    deeplearning prediction progress: |███████████████████████████████████████| 100%
    




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
      <td>71</td>
      <td>169</td>
      <td>480</td>
      <td>160</td>
      <td>320</td>
      <td>0.34</td>
      <td>0.17</td>
      <td>93.6</td>
      <td>1.71</td>
      <td>17.0</td>
      <td>33.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>58</td>
      <td>182</td>
      <td>720</td>
      <td>218</td>
      <td>502</td>
      <td>0.47</td>
      <td>0.26</td>
      <td>140.4</td>
      <td>1.55</td>
      <td>21.0</td>
      <td>30.28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>62</td>
      <td>178</td>
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
      <td>44</td>
      <td>196</td>
      <td>1200</td>
      <td>324</td>
      <td>876</td>
      <td>0.69</td>
      <td>0.45</td>
      <td>234.0</td>
      <td>1.38</td>
      <td>24.0</td>
      <td>27.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>36</td>
      <td>204</td>
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




```python
ROC_AUC(DL_modl_1,test_hex,'loan_default')
```

    deeplearning prediction progress: |███████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/gbm_deeplearning/output_102_1.png)


The first deep learning model has the lowest statistics scores in the gain table and also in AUC score. However, this is a random hidden with only 1 layer and 1000 nodes. I will try different approaches in creating hidden set to determine the best approach for deep learning. 

#### Model 2: 2 layers and 500 nodes each

The approach here is increase number of layers to 2 and decrease number of neurons to 500 for each layer.


```python
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
DL_modl_2 = H2ODeepLearningEstimator(
    distribution="tweedie",
    activation="RectifierWithDropout",
    hidden=[500,500],
    input_dropout_ratio=0.1,
    sparse=True,
    l1=1e-5,
    epochs=3000,
    seed =1234)
```


```python
DL_modl_2.train(
    x=list(predictors),
    y=target,
    training_frame=train_hex,
    validation_frame=test_hex)
```

    deeplearning Model Build progress: |██████████████████████████████████████| 100%
    


```python
createGains(DL_modl_2)
```

    deeplearning prediction progress: |███████████████████████████████████████| 100%
    




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
      <td>66</td>
      <td>174</td>
      <td>480</td>
      <td>163</td>
      <td>317</td>
      <td>0.35</td>
      <td>0.16</td>
      <td>93.6</td>
      <td>1.74</td>
      <td>19.0</td>
      <td>33.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>240</td>
      <td>56</td>
      <td>184</td>
      <td>720</td>
      <td>219</td>
      <td>501</td>
      <td>0.47</td>
      <td>0.26</td>
      <td>140.4</td>
      <td>1.56</td>
      <td>21.0</td>
      <td>30.42</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>51</td>
      <td>189</td>
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
      <td>50</td>
      <td>190</td>
      <td>1200</td>
      <td>320</td>
      <td>880</td>
      <td>0.68</td>
      <td>0.46</td>
      <td>234.0</td>
      <td>1.37</td>
      <td>22.0</td>
      <td>26.67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>51</td>
      <td>189</td>
      <td>1440</td>
      <td>371</td>
      <td>1069</td>
      <td>0.79</td>
      <td>0.55</td>
      <td>280.8</td>
      <td>1.32</td>
      <td>24.0</td>
      <td>25.76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>33</td>
      <td>207</td>
      <td>1680</td>
      <td>404</td>
      <td>1276</td>
      <td>0.86</td>
      <td>0.66</td>
      <td>327.6</td>
      <td>1.23</td>
      <td>20.0</td>
      <td>24.05</td>
    </tr>
    <tr>
      <th>7</th>
      <td>240</td>
      <td>28</td>
      <td>212</td>
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
      <td>27</td>
      <td>213</td>
      <td>2160</td>
      <td>459</td>
      <td>1701</td>
      <td>0.98</td>
      <td>0.88</td>
      <td>421.2</td>
      <td>1.09</td>
      <td>10.0</td>
      <td>21.25</td>
    </tr>
    <tr>
      <th>9</th>
      <td>240</td>
      <td>9</td>
      <td>231</td>
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
ROC_AUC(DL_modl_2,test_hex,'loan_default')
```

    deeplearning prediction progress: |███████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/gbm_deeplearning/output_108_1.png)


From this approach, the lift score increases again, as well as the AUC and PR scores. Therefore, I am going to follow the same approach.

#### Model 3: 3 layers with 7 nodes each (~equals the square root of 47 variables)

Since the data contains 47 variables, I again increase the number of layers to 3 and decreased number of neurons to 7. Considering if it would increase the model performance by using square root of total number of variables


```python
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
DL_modl_3 = H2ODeepLearningEstimator(
    distribution="tweedie",
    activation="RectifierWithDropout",
    hidden=[7,7,7],
    input_dropout_ratio=0.2,
    sparse=True,
    l1=1e-5,
    epochs=3000,
    seed =1234)
```


```python
DL_modl_3.train(
    x=list(predictors),
    y=target,
    training_frame=train_hex,
    validation_frame=test_hex)
```

    deeplearning Model Build progress: |██████████████████████████████████████| 100%
    


```python
createGains(DL_modl_3)
```

    deeplearning prediction progress: |███████████████████████████████████████| 100%
    




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
      <td>101</td>
      <td>139</td>
      <td>240</td>
      <td>101</td>
      <td>139</td>
      <td>0.22</td>
      <td>0.07</td>
      <td>46.8</td>
      <td>2.16</td>
      <td>15.0</td>
      <td>42.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>58</td>
      <td>182</td>
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
      <td>70</td>
      <td>170</td>
      <td>720</td>
      <td>229</td>
      <td>491</td>
      <td>0.49</td>
      <td>0.25</td>
      <td>140.4</td>
      <td>1.63</td>
      <td>24.0</td>
      <td>31.81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>53</td>
      <td>187</td>
      <td>960</td>
      <td>282</td>
      <td>678</td>
      <td>0.60</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.51</td>
      <td>25.0</td>
      <td>29.38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>39</td>
      <td>201</td>
      <td>1200</td>
      <td>321</td>
      <td>879</td>
      <td>0.69</td>
      <td>0.45</td>
      <td>234.0</td>
      <td>1.37</td>
      <td>24.0</td>
      <td>26.75</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>28</td>
      <td>212</td>
      <td>1440</td>
      <td>349</td>
      <td>1091</td>
      <td>0.75</td>
      <td>0.56</td>
      <td>280.8</td>
      <td>1.24</td>
      <td>19.0</td>
      <td>24.24</td>
    </tr>
    <tr>
      <th>6</th>
      <td>240</td>
      <td>47</td>
      <td>193</td>
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
      <td>35</td>
      <td>205</td>
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




```python
ROC_AUC(DL_modl_3,test_hex,'loan_default')
```

    deeplearning prediction progress: |███████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/gbm_deeplearning/output_114_1.png)


This model return as the highest lift score (2.16) so far, as well as the AUC and PR scores. Now, I want to try the same approach with the last model. 

#### Model 4: 4 layers which 15, 20, 25, 30 nodes 

In this model, I increase the number of layers to 4 and randomly picked number of neurons for each layer, starting from 15 and increasing 5 nodes per each layer. 


```python
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
DL_modl_4 = H2ODeepLearningEstimator(
    distribution="tweedie",
    activation="RectifierWithDropout",
    hidden=[15,20,25,30],
    input_dropout_ratio=0.2,
    sparse=True,
    l1=1e-5,
    epochs=3000,
    seed =1234)
```


```python
DL_modl_4.train(
    x=list(predictors),
    y=target,
    training_frame=train_hex,
    validation_frame=test_hex)
```

    deeplearning Model Build progress: |██████████████████████████████████████| 100%
    


```python
createGains(DL_modl_4)
```

    deeplearning prediction progress: |███████████████████████████████████████| 100%
    




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
      <td>91</td>
      <td>149</td>
      <td>240</td>
      <td>91</td>
      <td>149</td>
      <td>0.19</td>
      <td>0.08</td>
      <td>46.8</td>
      <td>1.94</td>
      <td>11.0</td>
      <td>37.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>240</td>
      <td>68</td>
      <td>172</td>
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
      <td>63</td>
      <td>177</td>
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
      <td>53</td>
      <td>187</td>
      <td>960</td>
      <td>275</td>
      <td>685</td>
      <td>0.59</td>
      <td>0.35</td>
      <td>187.2</td>
      <td>1.47</td>
      <td>24.0</td>
      <td>28.65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>41</td>
      <td>199</td>
      <td>1200</td>
      <td>316</td>
      <td>884</td>
      <td>0.68</td>
      <td>0.46</td>
      <td>234.0</td>
      <td>1.35</td>
      <td>22.0</td>
      <td>26.33</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240</td>
      <td>45</td>
      <td>195</td>
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
      <td>37</td>
      <td>203</td>
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
      <td>34</td>
      <td>206</td>
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
      <td>16</td>
      <td>224</td>
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
ROC_AUC(DL_modl_4,test_hex,'loan_default')
```

    deeplearning prediction progress: |███████████████████████████████████████| 100%
    
       * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate
    
    	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy
    
       * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)
    
    


![png](/assets/img/gbm_deeplearning/output_120_1.png)


Even though the result in this model is not bad, it is not good compared to previous ones.  

### Conclusion

For Gradient Boosting models, the predictive power does not increase when we apply optimal hyperparementers as it had similar results to the previous iterations. However, the running time for the best model is faster which should also be considered for optimization. Additionally, from my observations, GBM models aim to find optimal linear combination of trees by training (assume final model is the weighted sum of predictions of individual trees) in relation to given train data, and performed much better than RF models, such as the ones we used in the previous assignment. This leads me to believe that the dataset prefer more extra tunning (GBM) and less claasifier approach (RF).  

For the deep learning model, the predictive power increased more when I increased the number of layers and decreased the number of neurons in each layer. Furthermore, the approach of using the square root of total number of features created the best model out of all iterations. 

Lastly, the dataset has a wide-range of dates and times, which may be investigated further because the variable seems like a good choice for prediction. I also considered the data preprocessing as I believe generating features that have predictive power can improve model performance overall.
