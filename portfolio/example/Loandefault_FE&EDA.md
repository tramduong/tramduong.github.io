
# Feature Engineering and EDA for Loan Default

Tram Duong
<br>September 21, 2020


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
```


```python
# Read data
data = pd.read_csv("C:/github/Data-Science-Portfolio/Feature Engineering & Modeling/Data/XYZloan_default_selected_vars.csv")
```

### Data Preprocessing

#### Exploring the data


```python
#data.info()
#data.describe()
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
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>id</th>
      <th>loan_default</th>
      <th>AP001</th>
      <th>AP002</th>
      <th>AP003</th>
      <th>AP004</th>
      <th>AP005</th>
      <th>AP006</th>
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
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>31</td>
      <td>2</td>
      <td>1</td>
      <td>12</td>
      <td>2017/7/6 10:21</td>
      <td>ios</td>
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
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>2017/4/6 12:51</td>
      <td>h5</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>NaN</td>
      <td>WEB</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>33</td>
      <td>1</td>
      <td>4</td>
      <td>12</td>
      <td>2017/7/1 14:11</td>
      <td>h5</td>
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
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>34</td>
      <td>2</td>
      <td>4</td>
      <td>12</td>
      <td>2017/7/7 10:10</td>
      <td>android</td>
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
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>47</td>
      <td>2</td>
      <td>1</td>
      <td>12</td>
      <td>2017/7/6 14:37</td>
      <td>h5</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>NaN</td>
      <td>WEB</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 89 columns</p>
</div>



#### The two columns  unnamed 0 and unnamed 0.1 do  not contain any important information for our target, thus I am dropping them.


```python
data = data.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
```

#### -99 and -999:

By looking through the dataset, there are numerious amount for "-99" and "-999" in the data. These values are most likely NAs that are handled differently by separate systems and seem to be hold no actual value. Thus, I will replace them with na for futher analysis. 


```python
data = data.replace(-99, np.nan)
data = data.replace(-999, np.nan)
```


```python
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
<p>5 rows × 87 columns</p>
</div>



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



#### From this information, we can see that some features won't be relevant in our analysis as there is too many missing values (over 99% of the data is null). Therefore, I removed those variables


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
#null_cols = null_cols.sort_values(ascending=False)
```

After running some basic functions and checking the data dictionary, the columns that have missing values are all numeric type and are mostly related to the phone info, call details and credit center. Due to the type and the category of these variables, I assume that the missing values were never provided to the company or recorded. 

For this dataset, instead of removing these missing valua, I will impute 0 to all of them in order to not exclude or mispresent any essential data. Thus, with 0 value, I can assume that there is no info for phone, call, bank, or loan to these users. 


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
# convert the categorical columns to a format that works better with classification and regression algorithms:
# One hot encode: pd.getdummies
data = pd.concat([data, pd.get_dummies(data.MB007)], axis=1)
data = pd.concat([data, pd.get_dummies(data.AP006)], axis=1)
```


```python
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
      <th>OPPO</th>
      <th>Other</th>
      <th>SAMSUNG</th>
      <th>VIVO</th>
      <th>WEB</th>
      <th>XIAOMI</th>
      <th>android</th>
      <th>api</th>
      <th>h5</th>
      <th>ios</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 96 columns</p>
</div>




```python
# AP004 is for loan term appication which only contains value of: 3,6,9,12. 
AP004_df = pd.DataFrame(data.AP004.value_counts())
#AP004_df
data = data.drop(columns = "AP004")
```

###### AP004

As the majority of users choose 12 month term, this column does would skew the analysis due to its large proportion value. Therefore, I drop this column. 

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




![png](/assets/img/loandefault/output_41_1.png)


In general, feautures that are too correlated do not improve model efficiency and also affect the performance of linear regression and random forest models, making the learning algorithms slower to create and train. Therefore, I removed highly correlated features to prevent multicollinearity trhough the following function:


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
data_clean = data_new.drop(columns = ["Date", "Time", "AP006", "MB007"])
```

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


![png](/assets/img/loandefault/output_51_0.png)



![png](/assets/img/loandefault/output_51_1.png)



![png](/assets/img/loandefault/output_51_2.png)



![png](/assets/img/loandefault/output_51_3.png)



![png](/assets/img/loandefault/output_51_4.png)



![png](/assets/img/loandefault/output_51_5.png)



![png](/assets/img/loandefault/output_51_6.png)



![png](/assets/img/loandefault/output_51_7.png)



![png](/assets/img/loandefault/output_51_8.png)



![png](/assets/img/loandefault/output_51_9.png)



![png](/assets/img/loandefault/output_51_10.png)



![png](/assets/img/loandefault/output_51_11.png)



![png](/assets/img/loandefault/output_51_12.png)



![png](/assets/img/loandefault/output_51_13.png)



![png](/assets/img/loandefault/output_51_14.png)



![png](/assets/img/loandefault/output_51_15.png)



![png](/assets/img/loandefault/output_51_16.png)



![png](/assets/img/loandefault/output_51_17.png)



![png](/assets/img/loandefault/output_51_18.png)



![png](/assets/img/loandefault/output_51_19.png)



![png](/assets/img/loandefault/output_51_20.png)



![png](/assets/img/loandefault/output_51_21.png)



![png](/assets/img/loandefault/output_51_22.png)



![png](/assets/img/loandefault/output_51_23.png)



![png](/assets/img/loandefault/output_51_24.png)



![png](/assets/img/loandefault/output_51_25.png)



![png](/assets/img/loandefault/output_51_26.png)



![png](/assets/img/loandefault/output_51_27.png)



![png](/assets/img/loandefault/output_51_28.png)



![png](/assets/img/loandefault/output_51_29.png)



![png](/assets/img/loandefault/output_51_30.png)



![png](/assets/img/loandefault/output_51_31.png)



![png](/assets/img/loandefault/output_51_32.png)



![png](/assets/img/loandefault/output_51_33.png)



![png](/assets/img/loandefault/output_51_34.png)



![png](/assets/img/loandefault/output_51_35.png)

