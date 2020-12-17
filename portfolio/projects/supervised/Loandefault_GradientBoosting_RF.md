
# Random Forest and Gradient Boosting for Loan Default

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




![png](/assets/img/gb_rf/output_41_1.png)


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


![png](/assets/img/gb_rf/output_51_0.png)



![png](/assets/img/gb_rf/output_51_1.png)



![png](/assets/img/gb_rf/output_51_2.png)



![png](/assets/img/gb_rf/output_51_3.png)



![png](/assets/img/gb_rf/output_51_4.png)



![png](/assets/img/gb_rf/output_51_5.png)



![png](/assets/img/gb_rf/output_51_6.png)



![png](/assets/img/gb_rf/output_51_7.png)



![png](/assets/img/gb_rf/output_51_8.png)



![png](/assets/img/gb_rf/output_51_9.png)



![png](/assets/img/gb_rf/output_51_10.png)



![png](/assets/img/gb_rf/output_51_11.png)



![png](/assets/img/gb_rf/output_51_12.png)



![png](/assets/img/gb_rf/output_51_13.png)



![png](/assets/img/gb_rf/output_51_14.png)



![png](/assets/img/gb_rf/output_51_15.png)



![png](/assets/img/gb_rf/output_51_16.png)



![png](/assets/img/gb_rf/output_51_17.png)



![png](/assets/img/gb_rf/output_51_18.png)



![png](/assets/img/gb_rf/output_51_19.png)



![png](/assets/img/gb_rf/output_51_20.png)



![png](/assets/img/gb_rf/output_51_21.png)



![png](/assets/img/gb_rf/output_51_22.png)



![png](/assets/img/gb_rf/output_51_23.png)



![png](/assets/img/gb_rf/output_51_24.png)



![png](/assets/img/gb_rf/output_51_25.png)



![png](/assets/img/gb_rf/output_51_26.png)



![png](/assets/img/gb_rf/output_51_27.png)



![png](/assets/img/gb_rf/output_51_28.png)



![png](/assets/img/gb_rf/output_51_29.png)



![png](/assets/img/gb_rf/output_51_30.png)



![png](/assets/img/gb_rf/output_51_31.png)



![png](/assets/img/gb_rf/output_51_32.png)



![png](/assets/img/gb_rf/output_51_33.png)



![png](/assets/img/gb_rf/output_51_34.png)



![png](/assets/img/gb_rf/output_51_35.png)


### Model 1: Logistic regression with sklearn


```python
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
from sklearn.feature_selection import RFE
```


```python
x = data_clean.drop(columns = "loan_default", axis = 1)
y = data_clean.loan_default.values
```


```python
x = x.apply(pd.to_numeric, errors='coerce')
y = y.astype(np.float64)
```


```python
# Split X and y into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```

The model testing will be based on 20% of the dataset, while the model training will be based on 80% of the dataset


```python
lr_model = LogisticRegression().fit(x_train, y_train)
```

    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    


```python
# predict the response
lr_pred = lr_model.predict(x_test)
```


```python
lr_acc = lr_model.score(x_test, y_test)*100
print("Accuracy: "+str(lr_acc) )
```

    Accuracy: 80.8875
    

For the logistic model, I will use Recursive Feature Elimination (RFE) with the Logistic Regression Classifier to select the top 15 features. This method helps to identify the strongest and weakest features for the model perfomance to provide better feature selection overall.


```python
rfe = RFE(lr_model, 15)
fit = rfe.fit(x_train, y_train)
```

    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\utils\validation.py:70: FutureWarning: Pass n_features_to_select=15 as keyword args. From version 0.25 passing these as positional arguments will result in an error
      FutureWarning)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    


```python
top_15_features = x_train.columns[rfe.support_]
print('Chosen best 15 feature by rfe:',top_15_features)
```

    Chosen best 15 feature by rfe: Index(['AP002', 'AP003', 'AP008', 'AP009', 'TD005', 'TD013', 'TD029', 'CR005',
           'CR015', 'IPHONE7', 'IPHONE8', 'IPHONE9', 'Noinfo', 'WEB', 'ios'],
          dtype='object')
    


```python
x_ = data_clean[top_15_features]
y_ = data_clean.loan_default.values
```


```python
x_train,x_test,y_train,y_test=train_test_split(x_,y_,test_size=0.2,random_state=0)
```


```python
lr_model_1 = LogisticRegression().fit(x_train, y_train)
```

    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    


```python
lr_acc_1 = lr_model_1.score(x_test, y_test)*100
print("Accuracy: " + str(lr_acc_1))
```

    Accuracy: 81.2125
    


```python
# predict model
y_pred = lr_model_1.predict(x_test)
```

Confusion Matrix


```python
#Create a confusion matrix table
cm = pd.DataFrame(confusion_matrix(y_test,y_pred))
cm.rename(columns={0:'Predicted Low', 1:'Predicted High'},
         index = {0:'Actual Low',1:'Actual High'},inplace=True)
cm
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
      <th>Predicted Low</th>
      <th>Predicted High</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual Low</th>
      <td>12955</td>
      <td>63</td>
    </tr>
    <tr>
      <th>Actual High</th>
      <td>2943</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>




```python
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix,annot=True,fmt="d")
```




    <AxesSubplot:xlabel='Predicted', ylabel='Actual'>




![png](/assets/img/gb_rf/output_71_1.png)


Some definitions for the matrix are the following:

- TP = True Positives 
- TN = True Negatives
- FP = False Positives
- FN = False Negatives
- Accuracy = (TP+TN)/Total

**ROC Curve**


```python
y_pred_prop = lr_model_1.predict_proba(x_test)[:,1]
```


```python
roc_auc_value = roc_auc_score(y_test,y_pred_prop)
```


```python
fpr, tpr, _ = roc_curve(y_test, y_pred_prop)
```


```python
lw=2
plt.figure(figsize=(6,4))
plt.plot(fpr,tpr, color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' %roc_auc_value)
plt.plot([0,1],[0,1], color='navy',lw=lw,linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()
```


![png](/assets/img/gb_rf/output_77_0.png)


### Gain Tables


```python
def gains_table(a_test,a_pred):
    df_prep = pd.DataFrame(columns = ['actual','pred'])
    df_prep['actual'] = a_test
    df_prep['pred'] = a_pred
    df_prep = df_prep.sort_values(by='pred',ascending=False)
    df_prep['row_id'] = range(0,0+len(df_prep))
    df_prep.head()

    df_prep['decile'] = (df_prep['row_id'] / (len(df_prep)/10)).astype(int)
    df_prep.loc[df_prep['decile'] == 10] =9
    df_prep['decile'].value_counts()

    # Create gains table
    gains = df_prep.groupby('decile')['actual'].agg(['count','sum'])
    gains.columns = ['count','actual']
    gains

    gains['non_actual'] = gains['count'] - gains['actual']
    gains['cum_count'] = gains['count'].cumsum()
    gains['cum_actual'] = gains['actual'].cumsum()
    gains['cum_non_actual'] = gains['non_actual'].cumsum()
    gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
    gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
    gains['if_random'] = np.max(gains['cum_actual']) /10
    gains['if_random'] = gains['if_random'].cumsum()
    gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
    gains['K_S'] = np.abs( gains['percent_cum_actual'] - gains['percent_cum_non_actual']  ) * 100 
    gains['gain'] = (gains['cum_actual'] / gains['cum_count']*100).round(2)
    return(gains)
```


```python
lr_gains = gains_table(y_test,y_pred)
lr_gains['lift'].plot.line()
```




    <AxesSubplot:xlabel='decile'>




![png](/assets/img/gb_rf/output_80_1.png)



```python
lr_gains
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
      <td>1600</td>
      <td>317</td>
      <td>1283</td>
      <td>1600</td>
      <td>317</td>
      <td>1283</td>
      <td>0.11</td>
      <td>0.1</td>
      <td>298.2</td>
      <td>1.06</td>
      <td>1.0</td>
      <td>19.81</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1600</td>
      <td>290</td>
      <td>1310</td>
      <td>3200</td>
      <td>607</td>
      <td>2593</td>
      <td>0.20</td>
      <td>0.2</td>
      <td>596.4</td>
      <td>1.02</td>
      <td>0.0</td>
      <td>18.97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1600</td>
      <td>266</td>
      <td>1334</td>
      <td>4800</td>
      <td>873</td>
      <td>3927</td>
      <td>0.29</td>
      <td>0.3</td>
      <td>894.6</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>18.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1600</td>
      <td>279</td>
      <td>1321</td>
      <td>6400</td>
      <td>1152</td>
      <td>5248</td>
      <td>0.39</td>
      <td>0.4</td>
      <td>1192.8</td>
      <td>0.97</td>
      <td>1.0</td>
      <td>18.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1600</td>
      <td>314</td>
      <td>1286</td>
      <td>8000</td>
      <td>1466</td>
      <td>6534</td>
      <td>0.49</td>
      <td>0.5</td>
      <td>1491.0</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>18.32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1600</td>
      <td>293</td>
      <td>1307</td>
      <td>9600</td>
      <td>1759</td>
      <td>7841</td>
      <td>0.59</td>
      <td>0.6</td>
      <td>1789.2</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>18.32</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1600</td>
      <td>287</td>
      <td>1313</td>
      <td>11200</td>
      <td>2046</td>
      <td>9154</td>
      <td>0.69</td>
      <td>0.7</td>
      <td>2087.4</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>18.27</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1600</td>
      <td>304</td>
      <td>1296</td>
      <td>12800</td>
      <td>2350</td>
      <td>10450</td>
      <td>0.79</td>
      <td>0.8</td>
      <td>2385.6</td>
      <td>0.99</td>
      <td>1.0</td>
      <td>18.36</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1600</td>
      <td>326</td>
      <td>1274</td>
      <td>14400</td>
      <td>2676</td>
      <td>11724</td>
      <td>0.90</td>
      <td>0.9</td>
      <td>2683.8</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>18.58</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1600</td>
      <td>306</td>
      <td>1294</td>
      <td>16000</td>
      <td>2982</td>
      <td>13018</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>2982.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>18.64</td>
    </tr>
  </tbody>
</table>
</div>



## Model 2:  Random Forest


```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
```


```python
rfc = RandomForestClassifier().fit(X_train, Y_train)
rfc_pred = rfc.predict(X_test)
```


```python
rfc_score = rfc.score(X_test, Y_test)*100
print('Accuracy is: ',rfc_score)
```

    Accuracy is:  81.34375
    

In general, the feature importance varies depending on the ML model. <br><br>
For Random Forest, I used feature_importance_ functions to identify the top 15 important variables and check the model again to see if they align with or are different from other models before comparing their performance. 


```python
rf_feature_importance = pd.DataFrame({'Name': X_train.columns,
             'Feature Importances': rfc.feature_importances_})
rf_feature_importance = rf_feature_importance.sort_values(by = 'Feature Importances',ascending=False).reset_index()
rf_feature_importance = rf_feature_importance.drop(columns='index')
top_30 = rf_feature_importance[:30]
```


```python
# Chart to show how the important level for the top 30 variables 
fig = plt.figure(figsize=(19,6))
sns.barplot(x ='Name', y = 'Feature Importances', data=top_30, palette = 'hls')
```




    <AxesSubplot:xlabel='Name', ylabel='Feature Importances'>




![png](/assets/img/gb_rf/output_89_1.png)


The new data contains only the top 15 important features for random forest model


```python
new_data = data_clean.drop(columns = rf_feature_importance[15:]['Name'].values)
```


```python
X = new_data.drop(columns = "loan_default", axis = 1)
Y = new_data.loan_default.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=0)
```


```python
rfc_1 = RandomForestClassifier().fit(X_train, Y_train)
```


```python
Y_pred = rfc_1.predict(X_test)
```


```python
rfc_score_1 = rfc_1.score(X_test, Y_test)*100
print('Accuracy is: ',rfc_score_1)
```

    Accuracy is:  81.1125
    


```python
#Create a confusion matrix table
cm = pd.DataFrame(confusion_matrix(Y_test,Y_pred))
cm.rename(columns={0:'Predicted Low', 1:'Predicted High'},
         index = {0:'Actual Low',1:'Actual High'},inplace=True)
cm
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
      <th>Predicted Low</th>
      <th>Predicted High</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual Low</th>
      <td>12956</td>
      <td>62</td>
    </tr>
    <tr>
      <th>Actual High</th>
      <td>2960</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>




```python
cm = confusion_matrix(Y_test,rfc_1.predict(X_test))
sns.heatmap(cm,annot=True,fmt="d")
```




    <AxesSubplot:>




![png](/assets/img/gb_rf/output_97_1.png)


**Gains Table**


```python
rf_gains = gains_table(Y_test,Y_pred)
rf_gains['lift'].plot.line()
```




    <AxesSubplot:xlabel='decile'>




![png](/assets/img/gb_rf/output_99_1.png)



```python
rf_gains
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
      <td>1600</td>
      <td>307</td>
      <td>1293</td>
      <td>1600</td>
      <td>307</td>
      <td>1293</td>
      <td>0.10</td>
      <td>0.1</td>
      <td>298.2</td>
      <td>1.03</td>
      <td>0.0</td>
      <td>19.19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1600</td>
      <td>290</td>
      <td>1310</td>
      <td>3200</td>
      <td>597</td>
      <td>2603</td>
      <td>0.20</td>
      <td>0.2</td>
      <td>596.4</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>18.66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1600</td>
      <td>265</td>
      <td>1335</td>
      <td>4800</td>
      <td>862</td>
      <td>3938</td>
      <td>0.29</td>
      <td>0.3</td>
      <td>894.6</td>
      <td>0.96</td>
      <td>1.0</td>
      <td>17.96</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1600</td>
      <td>283</td>
      <td>1317</td>
      <td>6400</td>
      <td>1145</td>
      <td>5255</td>
      <td>0.38</td>
      <td>0.4</td>
      <td>1192.8</td>
      <td>0.96</td>
      <td>2.0</td>
      <td>17.89</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1600</td>
      <td>315</td>
      <td>1285</td>
      <td>8000</td>
      <td>1460</td>
      <td>6540</td>
      <td>0.49</td>
      <td>0.5</td>
      <td>1491.0</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>18.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1600</td>
      <td>292</td>
      <td>1308</td>
      <td>9600</td>
      <td>1752</td>
      <td>7848</td>
      <td>0.59</td>
      <td>0.6</td>
      <td>1789.2</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>18.25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1600</td>
      <td>288</td>
      <td>1312</td>
      <td>11200</td>
      <td>2040</td>
      <td>9160</td>
      <td>0.68</td>
      <td>0.7</td>
      <td>2087.4</td>
      <td>0.98</td>
      <td>2.0</td>
      <td>18.21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1600</td>
      <td>307</td>
      <td>1293</td>
      <td>12800</td>
      <td>2347</td>
      <td>10453</td>
      <td>0.79</td>
      <td>0.8</td>
      <td>2385.6</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>18.34</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1600</td>
      <td>325</td>
      <td>1275</td>
      <td>14400</td>
      <td>2672</td>
      <td>11728</td>
      <td>0.90</td>
      <td>0.9</td>
      <td>2683.8</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>18.56</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1600</td>
      <td>310</td>
      <td>1290</td>
      <td>16000</td>
      <td>2982</td>
      <td>13018</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>2982.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>18.64</td>
    </tr>
  </tbody>
</table>
</div>



Even though the assignment asks for 15 features, I tried to find the number of features that bring the most optimal results and what they are. By using RFECV, we can see that the performance of the rf model would be optimized with the 35 variables below. 


```python
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
```


```python
from sklearn.feature_selection import RFECV
# The "accuracy" scoring is proportional to the number of correct classifications
rf_2 = RandomForestClassifier() 
rfecv = RFECV(estimator=rf_2, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, Y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])
```

    Optimal number of features : 35
    Best features : Index(['AP001', 'AP003', 'AP007', 'AP008', 'TD001', 'TD002', 'TD005', 'TD006',
           'TD013', 'TD014', 'TD015', 'TD023', 'TD024', 'CR004', 'CR005', 'CR009',
           'CR015', 'CR017', 'PA022', 'PA028', 'CD008', 'CD018', 'CD071', 'CD072',
           'CD088', 'CD100', 'CD113', 'CD115', 'CD130', 'CD131', 'CD152', 'CD153',
           'CD160', 'CD166', 'MB005'],
          dtype='object')
    

### Model 3: Gradient Boosting


```python
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
```


```python
gbc = GradientBoostingClassifier().fit(X_train, Y_train)
gbc_pred = gbc.predict(X_test)
```


```python
gbc_score = gbc.score(X_test, Y_test)*100
print("Accuracy: " +str(gbc_score))
```

    Accuracy: 81.33749999999999
    


```python
gbc_feature_importance = pd.DataFrame({'Name': X_train.columns,
             'Feature Importances': gbc.feature_importances_})
gbc_feature_importance = gbc_feature_importance.sort_values(by = 'Feature Importances',ascending=False).reset_index()
gbc_feature_importance = gbc_feature_importance.drop(columns='index')
#rf_feature_importance['cum_sum'] = rf_feature_importance['Feature Importances'].cumsum()
top_30 = gbc_feature_importance[:30]
```


```python
# Feature important values
fig = plt.figure(figsize=(19,6))
sns.barplot(x ='Name', y = 'Feature Importances', data=top_30, palette = 'hls')
```




    <AxesSubplot:xlabel='Name', ylabel='Feature Importances'>




![png](/assets/img/gb_rf/output_109_1.png)



```python
gbc_data = data_clean.drop(columns = gbc_feature_importance[15:]['Name'].values)
```


```python
X = gbc_data.drop(columns = "loan_default", axis = 1)
Y = gbc_data.loan_default.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=0)
```


```python
gbc_1 = GradientBoostingClassifier().fit(X_train, Y_train)
```


```python
gbc_pred = gbc_1.predict(X_test)
```


```python
gbc_score_1 = gbc_1.score(X_test, Y_test)*100
print("Accuracy: " +str(gbc_score_1))
```

    Accuracy: 81.35
    


```python
#Create a confusion matrix table
cm = pd.DataFrame(confusion_matrix(Y_test, gbc_1.predict(X_test)))
cm.rename(columns={0:'Predicted Low', 1:'Predicted High'},
         index = {0:'Actual Low',1:'Actual High'},inplace=True)
cm
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
      <th>Predicted Low</th>
      <th>Predicted High</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual Low</th>
      <td>12972</td>
      <td>46</td>
    </tr>
    <tr>
      <th>Actual High</th>
      <td>2938</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>




```python
cm = confusion_matrix(Y_test,gbc_1.predict(X_test))
sns.heatmap(cm,annot=True,fmt="d")
```




    <AxesSubplot:>




![png](/assets/img/gb_rf/output_116_1.png)



```python
gbc_gains = gains_table(Y_test,gbc_pred)
gbc_gains['lift'].plot.line()
```




    <AxesSubplot:xlabel='decile'>




![png](/assets/img/gb_rf/output_117_1.png)



```python
gbc_gains
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
      <td>1600</td>
      <td>324</td>
      <td>1276</td>
      <td>1600</td>
      <td>324</td>
      <td>1276</td>
      <td>0.11</td>
      <td>0.1</td>
      <td>298.2</td>
      <td>1.09</td>
      <td>1.0</td>
      <td>20.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1600</td>
      <td>285</td>
      <td>1315</td>
      <td>3200</td>
      <td>609</td>
      <td>2591</td>
      <td>0.20</td>
      <td>0.2</td>
      <td>596.4</td>
      <td>1.02</td>
      <td>0.0</td>
      <td>19.03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1600</td>
      <td>263</td>
      <td>1337</td>
      <td>4800</td>
      <td>872</td>
      <td>3928</td>
      <td>0.29</td>
      <td>0.3</td>
      <td>894.6</td>
      <td>0.97</td>
      <td>1.0</td>
      <td>18.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1600</td>
      <td>281</td>
      <td>1319</td>
      <td>6400</td>
      <td>1153</td>
      <td>5247</td>
      <td>0.39</td>
      <td>0.4</td>
      <td>1192.8</td>
      <td>0.97</td>
      <td>1.0</td>
      <td>18.02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1600</td>
      <td>317</td>
      <td>1283</td>
      <td>8000</td>
      <td>1470</td>
      <td>6530</td>
      <td>0.49</td>
      <td>0.5</td>
      <td>1491.0</td>
      <td>0.99</td>
      <td>1.0</td>
      <td>18.38</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1600</td>
      <td>291</td>
      <td>1309</td>
      <td>9600</td>
      <td>1761</td>
      <td>7839</td>
      <td>0.59</td>
      <td>0.6</td>
      <td>1789.2</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>18.34</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1600</td>
      <td>286</td>
      <td>1314</td>
      <td>11200</td>
      <td>2047</td>
      <td>9153</td>
      <td>0.69</td>
      <td>0.7</td>
      <td>2087.4</td>
      <td>0.98</td>
      <td>1.0</td>
      <td>18.28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1600</td>
      <td>303</td>
      <td>1297</td>
      <td>12800</td>
      <td>2350</td>
      <td>10450</td>
      <td>0.79</td>
      <td>0.8</td>
      <td>2385.6</td>
      <td>0.99</td>
      <td>1.0</td>
      <td>18.36</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1600</td>
      <td>325</td>
      <td>1275</td>
      <td>14400</td>
      <td>2675</td>
      <td>11725</td>
      <td>0.90</td>
      <td>0.9</td>
      <td>2683.8</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>18.58</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1600</td>
      <td>307</td>
      <td>1293</td>
      <td>16000</td>
      <td>2982</td>
      <td>13018</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>2982.0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>18.64</td>
    </tr>
  </tbody>
</table>
</div>



**Model Comparison**


```python
models_comp = pd.DataFrame({
    'Model' : ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Score' : [lr_acc,rfc_score,gbc_score]})
models_comp.sort_values(by='Score', ascending = True)
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
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>80.88750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gradient Boosting</td>
      <td>81.33750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>81.34375</td>
    </tr>
  </tbody>
</table>
</div>




```python
models_comp_15 = pd.DataFrame({
    'Model' : ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Score' : [lr_acc_1,rfc_score_1,gbc_score_1]})
models_comp_15.sort_values(by='Score', ascending = True)
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
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>81.1125</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>81.2125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gradient Boosting</td>
      <td>81.3500</td>
    </tr>
  </tbody>
</table>
</div>



### Conclusion

The best performing model across all selected features is gradient boosting. However, the difference of their performance scores do not vary greatly.

The best model perform in top 15 features is also gradient boosting. In this case, the difference of performance score is also very close to each other. However, to compare with the perfomance of all selected features models, the performance of top 15 features are higher.

Note: due to the random state in splitting train and test data, the scores vary everytime the code is rerun. But the performance levels tend to be ranked the same and do not vary greatly in their peformance.

The gain tables for all three models shows that the lift, K_S, and gain scores are pretty normal. For example, in the logistic model, decile 0 can get 1.06 times compared to random selection. My observation here is the lower the accuracy score for a model, the higher score the gain tables receive.

The given dataset has high level of negative-positive impact to users. Based on my observations, the predictive power of gradient boosting model is high in an unbalanced dataset. Additionally, the dataset has a wide-range of date and time, which could be investigated further because the variable seems like a good choice for predictors. If I spent more time on generating and exploring the date-time variable, it could support the predictive models performance.
