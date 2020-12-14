
# Unsupervised Learning With KNN and PCA

Tram Duong
<br>October 19, 2020

### Part 1: EDA and FE 
- Data Exploration
- Data Cleaning
- Feature Engineerings


```python
import pandas as pd 
import seaborn as sns  
import matplotlib.pyplot as plt
import numpy as np
import scipy 
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pyod
# Import all models 
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.utils.utility import standardizer
from pyod.models.combination import aom, moa, average, maximization
```

    C:\Users\tramh\.conda\envs\Anomaly\lib\site-packages\sklearn\utils\deprecation.py:143: FutureWarning: The sklearn.utils.testing module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.utils. Anything that cannot be imported from sklearn.utils is now part of the private API.
      warnings.warn(message, FutureWarning)
    


```python
# read data 
payment_data = pd.read_csv("/data/inpatientCharges.csv")
```

A few things I noticed from the data overview:
   - Payments should be in numeric/float format to do statistics 
   - Zipcode column should be in character/object format
   - Provider ID should be in int format


```python
payment_data.head()
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
      <th>DRG Definition</th>
      <th>Provider Id</th>
      <th>Provider Name</th>
      <th>Provider Street Address</th>
      <th>Provider City</th>
      <th>Provider State</th>
      <th>Provider Zip Code</th>
      <th>Hospital Referral Region Description</th>
      <th>Total Discharges</th>
      <th>Average Covered Charges</th>
      <th>Average Total Payments</th>
      <th>Average Medicare Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10001</td>
      <td>SOUTHEAST ALABAMA MEDICAL CENTER</td>
      <td>1108 ROSS CLARK CIRCLE</td>
      <td>DOTHAN</td>
      <td>AL</td>
      <td>36301</td>
      <td>AL - Dothan</td>
      <td>91</td>
      <td>$32963.07</td>
      <td>$5777.24</td>
      <td>$4763.73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10005</td>
      <td>MARSHALL MEDICAL CENTER SOUTH</td>
      <td>2505 U S HIGHWAY 431 NORTH</td>
      <td>BOAZ</td>
      <td>AL</td>
      <td>35957</td>
      <td>AL - Birmingham</td>
      <td>14</td>
      <td>$15131.85</td>
      <td>$5787.57</td>
      <td>$4976.71</td>
    </tr>
    <tr>
      <th>2</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10006</td>
      <td>ELIZA COFFEE MEMORIAL HOSPITAL</td>
      <td>205 MARENGO STREET</td>
      <td>FLORENCE</td>
      <td>AL</td>
      <td>35631</td>
      <td>AL - Birmingham</td>
      <td>24</td>
      <td>$37560.37</td>
      <td>$5434.95</td>
      <td>$4453.79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10011</td>
      <td>ST VINCENT'S EAST</td>
      <td>50 MEDICAL PARK EAST DRIVE</td>
      <td>BIRMINGHAM</td>
      <td>AL</td>
      <td>35235</td>
      <td>AL - Birmingham</td>
      <td>25</td>
      <td>$13998.28</td>
      <td>$5417.56</td>
      <td>$4129.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10016</td>
      <td>SHELBY BAPTIST MEDICAL CENTER</td>
      <td>1000 FIRST STREET NORTH</td>
      <td>ALABASTER</td>
      <td>AL</td>
      <td>35007</td>
      <td>AL - Birmingham</td>
      <td>18</td>
      <td>$31633.27</td>
      <td>$5658.33</td>
      <td>$4851.44</td>
    </tr>
  </tbody>
</table>
</div>




```python
payment_data.describe()
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
      <th>Provider Id</th>
      <th>Provider Zip Code</th>
      <th>Total Discharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>163065.000000</td>
      <td>163065.000000</td>
      <td>163065.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>255569.865428</td>
      <td>47938.121908</td>
      <td>42.776304</td>
    </tr>
    <tr>
      <th>std</th>
      <td>151563.671767</td>
      <td>27854.323080</td>
      <td>51.104042</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10001.000000</td>
      <td>1040.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>110092.000000</td>
      <td>27261.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>250007.000000</td>
      <td>44309.000000</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>380075.000000</td>
      <td>72901.000000</td>
      <td>49.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>670077.000000</td>
      <td>99835.000000</td>
      <td>3383.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
payment_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 163065 entries, 0 to 163064
    Data columns (total 12 columns):
     #   Column                                Non-Null Count   Dtype 
    ---  ------                                --------------   ----- 
     0   DRG Definition                        163065 non-null  object
     1   Provider Id                           163065 non-null  int64 
     2   Provider Name                         163065 non-null  object
     3   Provider Street Address               163065 non-null  object
     4   Provider City                         163065 non-null  object
     5   Provider State                        163065 non-null  object
     6   Provider Zip Code                     163065 non-null  int64 
     7   Hospital Referral Region Description  163065 non-null  object
     8    Total Discharges                     163065 non-null  int64 
     9    Average Covered Charges              163065 non-null  object
     10   Average Total Payments               163065 non-null  object
     11  Average Medicare Payments             163065 non-null  object
    dtypes: int64(3), object(9)
    memory usage: 14.9+ MB
    

According to the link: https://data.cms.gov/Medicare-Inpatient/National-Summary-of-Inpatient-Charge-Data-by-Medic/efwk-h4x3, the dataset description is below:

**DRG Definition** : Classification system that groups similar clinical conditions (diagnoses) and the procedures furnished by the hospital during the stay.

**Total Discharges** : The number of discharges billed by all providers for inpatient hospital services.

**Average Covered Charges** : The average charge of all provider's services covered by Medicare for discharges in the DRG. These will vary from hospital to hospital because of differences in hospital charge structures.

**Average Total Payment**: The average total payments to all providers for the DRG including the MS-DRG amount, teaching, disproportionate share, capital, and outlier payments for all cases. Also included in average total payments are co-payment and deductible amounts that the patient is responsible for and any additional payments by third parties for coordination of benefits.

**Average Medicare Payment**: The average amount that Medicare pays to the provider for Medicare's share of the MS-DRG. Medicare payment amounts include the MS-DRG amount, teaching, disproportionate share, capital, and outlier payments for all cases. Medicare payments DO NOT include beneficiary co-payments and deductible amounts nor any additional payments from third parties for coordination of benefits.


### Preprocessing data

Some columns names have spaces which need to be removed


```python
payment_data.columns
```




    Index(['DRG Definition', 'Provider Id', 'Provider Name',
           'Provider Street Address', 'Provider City', 'Provider State',
           'Provider Zip Code', 'Hospital Referral Region Description',
           ' Total Discharges ', ' Average Covered Charges ',
           ' Average Total Payments ', 'Average Medicare Payments'],
          dtype='object')




```python
payment_data.columns = payment_data.columns.str.strip()
```

All the payment columns include '$' sign which need to be removed and coverted to float type for further analysis


```python
# remove $ sign and convert to float type
payment_data['Average Covered Charges'] = payment_data['Average Covered Charges'].str.strip("$").astype('float')
payment_data['Average Total Payments'] = payment_data['Average Total Payments'].str.strip("$").astype('float')
payment_data['Average Medicare Payments'] = payment_data['Average Medicare Payments'].str.strip("$").astype('float')
```

Zipcode column contain some 4 digits values which need to converted into the right type as they are missing the leading zero.


```python
payment_data['Provider Zip Code'] = payment_data['Provider Zip Code'].astype(str).str.zfill(5)
```

### Exploratory Data Analysis

The dataset contains payments of inpatients in 50 states. Beside making visualizations for comparing the amount of charges in different states, I plan to build some visualization based on regions to gain more insights. 


```python
west = ['WA','OR','CA','ID','NV','MT','WY','UT','AZ','CO','NM']
midwest = ['ND','MN','WI','MI','SD','NE','KS','IA','MO','IL','IN','OH']
south = ['TX','OK','AR','LA','MS','TN','KY','AL','GA','FL','SC','NC','VA','WV','MD','DE']
northeast = ['PA','NJ','NY','CT','MA','RI','VT','NH','ME']
```


```python
s=pd.DataFrame([west,midwest,south,northeast],index=['West','Midwest','South','Northeast'])
s=s.reset_index().melt('index')
payment_data['Region'] = payment_data['Provider State'].map(dict(zip(s['value'],s['index'])))
```


```python
payment_data.head()
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
      <th>DRG Definition</th>
      <th>Provider Id</th>
      <th>Provider Name</th>
      <th>Provider Street Address</th>
      <th>Provider City</th>
      <th>Provider State</th>
      <th>Provider Zip Code</th>
      <th>Hospital Referral Region Description</th>
      <th>Total Discharges</th>
      <th>Average Covered Charges</th>
      <th>Average Total Payments</th>
      <th>Average Medicare Payments</th>
      <th>Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10001</td>
      <td>SOUTHEAST ALABAMA MEDICAL CENTER</td>
      <td>1108 ROSS CLARK CIRCLE</td>
      <td>DOTHAN</td>
      <td>AL</td>
      <td>36301</td>
      <td>AL - Dothan</td>
      <td>91</td>
      <td>32963.07</td>
      <td>5777.24</td>
      <td>4763.73</td>
      <td>South</td>
    </tr>
    <tr>
      <th>1</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10005</td>
      <td>MARSHALL MEDICAL CENTER SOUTH</td>
      <td>2505 U S HIGHWAY 431 NORTH</td>
      <td>BOAZ</td>
      <td>AL</td>
      <td>35957</td>
      <td>AL - Birmingham</td>
      <td>14</td>
      <td>15131.85</td>
      <td>5787.57</td>
      <td>4976.71</td>
      <td>South</td>
    </tr>
    <tr>
      <th>2</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10006</td>
      <td>ELIZA COFFEE MEMORIAL HOSPITAL</td>
      <td>205 MARENGO STREET</td>
      <td>FLORENCE</td>
      <td>AL</td>
      <td>35631</td>
      <td>AL - Birmingham</td>
      <td>24</td>
      <td>37560.37</td>
      <td>5434.95</td>
      <td>4453.79</td>
      <td>South</td>
    </tr>
    <tr>
      <th>3</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10011</td>
      <td>ST VINCENT'S EAST</td>
      <td>50 MEDICAL PARK EAST DRIVE</td>
      <td>BIRMINGHAM</td>
      <td>AL</td>
      <td>35235</td>
      <td>AL - Birmingham</td>
      <td>25</td>
      <td>13998.28</td>
      <td>5417.56</td>
      <td>4129.16</td>
      <td>South</td>
    </tr>
    <tr>
      <th>4</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>10016</td>
      <td>SHELBY BAPTIST MEDICAL CENTER</td>
      <td>1000 FIRST STREET NORTH</td>
      <td>ALABASTER</td>
      <td>AL</td>
      <td>35007</td>
      <td>AL - Birmingham</td>
      <td>18</td>
      <td>31633.27</td>
      <td>5658.33</td>
      <td>4851.44</td>
      <td>South</td>
    </tr>
  </tbody>
</table>
</div>



### Visualization by states


```python
# Make the PairGrid
fig = plt.figure(figsize=(16,10))
g = sns.PairGrid(payment_data,
                 x_vars=['Total Discharges', 'Average Covered Charges', 'Average Total Payments', 'Average Medicare Payments'], 
                 y_vars=["Provider State"],
                 height=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h", linewidth=1)

# Use the same x axis limits on all columns and add better labels
g.set(xlabel="", ylabel="")

# Use semantically meaningful titles for the columns
titles = ["Total Discharges", "Average Coveraged Charges", "Average Total Payments",
          "Average Medicare Paymens"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
plt.tight_layout()
plt.show()
```


    <Figure size 1152x720 with 0 Axes>



![png]![png](/assets/img/pca_knn/output_23_1.png)


### Visualization by regions


```python
fig = plt.figure(figsize=(16,10))
sns.pairplot(payment_data[['Region','Average Total Payments',
                            'Total Discharges','Average Medicare Payments','Average Covered Charges']], hue= 'Region',height = 4)
```




    <seaborn.axisgrid.PairGrid at 0x18c5e049198>




    <Figure size 1152x720 with 0 Axes>



![png]![png](/assets/img/pca_knn/output_25_2.png)


### Feature Correlation


```python
stats_df = pd.DataFrame(payment_data, columns=['Average Total Payments',
                            'Total Discharges','Average Medicare Payments','Average Covered Charges'])
```


```python
x = stats_df
corr = x.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <AxesSubplot:>




![png]![png](/assets/img/pca_knn/output_28_1.png)


The average medicare payments and average covered charges are highly correlated to each other. Furthermore, the average medicare payments are also highly correlated with the average total payments.   

### Common Procedures


```python
fig = plt.figure(figsize=(16,8))
common_drg = payment_data.groupby('DRG Definition').count()['Provider Id'].sort_values(ascending=False)
top_10 = common_drg[:10]
sns.countplot(y='DRG Definition', data=payment_data, palette="Greens_d",
              order=pd.value_counts(payment_data['DRG Definition']).iloc[:10].index)
```




    <AxesSubplot:xlabel='count', ylabel='DRG Definition'>




![png]![png](/assets/img/pca_knn/output_31_1.png)


### Features Engineering 

#### 1:  Patient Average by Provider ID

This feature will provide an estimation of the average amount charges/payment by each provider.


```python
# gorup by id 
patient_avg_id = payment_data.groupby('Provider Id').mean()[['Total Discharges',
                                    'Average Covered Charges','Average Total Payments','Average Medicare Payments']]
patient_avg_id.head()
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
      <th>Total Discharges</th>
      <th>Average Covered Charges</th>
      <th>Average Total Payments</th>
      <th>Average Medicare Payments</th>
    </tr>
    <tr>
      <th>Provider Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10001</th>
      <td>58.750000</td>
      <td>35247.028152</td>
      <td>8749.025109</td>
      <td>7678.214348</td>
    </tr>
    <tr>
      <th>10005</th>
      <td>28.959184</td>
      <td>16451.092041</td>
      <td>6812.131224</td>
      <td>5793.631429</td>
    </tr>
    <tr>
      <th>10006</th>
      <td>45.360465</td>
      <td>36942.357442</td>
      <td>8197.237907</td>
      <td>7145.959535</td>
    </tr>
    <tr>
      <th>10007</th>
      <td>27.409091</td>
      <td>12079.536818</td>
      <td>4860.829091</td>
      <td>4047.025455</td>
    </tr>
    <tr>
      <th>10008</th>
      <td>17.888889</td>
      <td>16148.752222</td>
      <td>5898.136667</td>
      <td>4963.547778</td>
    </tr>
  </tbody>
</table>
</div>



**Conclusion**: This feature shows the average charge/payment of each provider which can be used as a baseline when there is any unsual amount. This insight highlights the business use of this feature and shows how it will be helpful with detecting anomalies.

#### 2: Patient Average by State

This feature will provide an estimation of the average amount charges/payment in each state.


```python
patient_avg_state = payment_data.groupby('Provider State').mean()[['Total Discharges',
                                    'Average Covered Charges','Average Total Payments','Average Medicare Payments']]
patient_avg_state.head()
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
      <th>Total Discharges</th>
      <th>Average Covered Charges</th>
      <th>Average Total Payments</th>
      <th>Average Medicare Payments</th>
    </tr>
    <tr>
      <th>Provider State</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AK</th>
      <td>26.588745</td>
      <td>40348.743333</td>
      <td>14572.391732</td>
      <td>12958.969437</td>
    </tr>
    <tr>
      <th>AL</th>
      <td>39.258322</td>
      <td>31316.462074</td>
      <td>7568.232149</td>
      <td>6418.007120</td>
    </tr>
    <tr>
      <th>AR</th>
      <td>41.978229</td>
      <td>26174.526246</td>
      <td>8019.248805</td>
      <td>6919.720832</td>
    </tr>
    <tr>
      <th>AZ</th>
      <td>36.690284</td>
      <td>41200.063020</td>
      <td>10154.528211</td>
      <td>8825.717240</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>36.357854</td>
      <td>67508.616536</td>
      <td>12629.668472</td>
      <td>11494.381678</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(nrows=4, ncols=1,figsize=(15,10))
plt.subplots_adjust(hspace=1)
for i, ax in enumerate(axes.flatten()):
    patient_avg_state[patient_avg_state.columns[i]].plot(kind='bar',ax=ax)
    ax.set_title(patient_avg_state.columns[i])
```


![png]![png](/assets/img/pca_knn/output_39_0.png)


**Conclusion**: This feature will provide an estimation of the average amount charges/payment in each state which can be used to compare between states and treated as a baseline when there is any unsual amount. This insight highlights the business use of this feature and shows how it will be helpful with detecting anomalies. 

#### 3: Patient Average by Region
This feature displays the average amount charges/payment in each region


```python
patient_avg_reg = payment_data.groupby('Region').mean()[['Total Discharges',
                                    'Average Covered Charges','Average Total Payments','Average Medicare Payments']]
```


```python
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15,10))
plt.subplots_adjust(hspace=0.5)
for i, ax in enumerate(axes.flatten()):
    patient_avg_reg[patient_avg_reg.columns[i]].plot(kind='bar',ax=ax)
    ax.set_title(patient_avg_reg.columns[i])
```


![png]![png](/assets/img/pca_knn/output_43_0.png)


**Conclusion**: This feature will provide an estimation of the average amount charges/payment in each region which can be used to compare between them and treated as a baseline when there is any unsual amount in each region. This insight highlights the business use of this feature and shows how it will be helpful with detecting anomalies.

#### 4: Average out of pocket by provider name
This feature is the amount that patient pay by different provider. It gives us an idea which provider has the greatest charges.


```python
payment_data['Ave Out of Pocket Payment'] = payment_data['Average Total Payments'] - payment_data['Average Medicare Payments']
```


```python
oop_pro= payment_data[['Provider Name', 'Ave Out of Pocket Payment']].groupby(by='Provider Name').agg('mean')
oop_pro = oop_pro.sort_values(('Ave Out of Pocket Payment'), ascending=False)
oop_pro.head()
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
      <th>Ave Out of Pocket Payment</th>
    </tr>
    <tr>
      <th>Provider Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BAYLOR SURGICAL HOSPITAL AT FORT WORTH</th>
      <td>14853.560000</td>
    </tr>
    <tr>
      <th>CANCER TREATMENT CENTERS OF AMERICA</th>
      <td>9613.663333</td>
    </tr>
    <tr>
      <th>USMD HOSPITAL  AT FORT WORTH LP</th>
      <td>9169.745000</td>
    </tr>
    <tr>
      <th>IRVING COPPELL SURGICAL HOSPITAL LLP</th>
      <td>8728.920000</td>
    </tr>
    <tr>
      <th>UVA HEALTH SCIENCES CENTER</th>
      <td>8715.730000</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%capture --no-stdout --no-display output 
#Stop warning from showing

# Top 20 out of pocket per provider

a =  oop_pro[:20]
plt.figure(figsize=(30,20))
fig,ax= plt.subplots()
fig = sns.barplot(a.iloc[:,0],a.index, color="steelblue",alpha=0.8)
ax.set(ylabel=None)
plt.xlabel("Ave OoP per discharge")
```




    Text(0.5, 0, 'Ave OoP per discharge')




    <Figure size 2160x1440 with 0 Axes>



![png]![png](/assets/img/pca_knn/output_48_2.png)


**Conclusion**: Out of pocket is an important indicator for any hospital bill. This feature helps to define the mean of average out of pocket payment in each provider and can be used as a baseline to compare when any out of pocket charges occur.

#### 5: Out of pocket by procedures
This feature is the amount that patient pay by different procedures. It gives us an estimate amount for different procedures


```python
oop_drg= payment_data[['DRG Definition', 'Ave Out of Pocket Payment']].groupby(by='DRG Definition').agg('mean')
oop_drg = oop_drg.sort_values(('Ave Out of Pocket Payment'), ascending=False)
oop_drg.head()
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
      <th>Ave Out of Pocket Payment</th>
    </tr>
    <tr>
      <th>DRG Definition</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>460 - SPINAL FUSION EXCEPT CERVICAL W/O MCC</th>
      <td>3735.070150</td>
    </tr>
    <tr>
      <th>473 - CERVICAL SPINAL FUSION W/O CC/MCC</th>
      <td>2594.714232</td>
    </tr>
    <tr>
      <th>247 - PERC CARDIOVASC PROC W DRUG-ELUTING STENT W/O MCC</th>
      <td>2582.521719</td>
    </tr>
    <tr>
      <th>207 - RESPIRATORY SYSTEM DIAGNOSIS W VENTILATOR SUPPORT 96+ HOURS</th>
      <td>2559.372528</td>
    </tr>
    <tr>
      <th>853 - INFECTIOUS &amp; PARASITIC DISEASES W O.R. PROCEDURE W MCC</th>
      <td>2497.221490</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%capture --no-stdout --no-display output
# prevent warning message from showing

# Top 20 out of pocket by procedure
a =  oop_drg[:20]
plt.figure(figsize=(30,20))
fig,ax= plt.subplots()
fig = sns.barplot(a.iloc[:,0],a.index, color="steelblue",alpha=0.8)
ax.set(ylabel=None)
plt.xlabel("Ave OoP by Procedure")
```




    Text(0.5, 0, 'Ave OoP by Procedure')




    <Figure size 2160x1440 with 0 Axes>



![png]![png](/assets/img/pca_knn/output_52_2.png)


**Conclusion**: This feature helps to define the mean of average out of pocket payment in each procedures and can be used as a baseline to detect anomalies.

#### 6: Ave out of pocket per discharge

This feature shows the average amount of out of pocket per discharge for different procedures. If there are a high amount of charge occuring, they would be captured to possibily be investigated if needed.


```python
payment_data['Ave OoP per discharge'] = payment_data['Ave Out of Pocket Payment']/payment_data['Total Discharges']
```


```python
oop_dis= payment_data[['DRG Definition', 'Ave OoP per discharge']].groupby(by='DRG Definition').agg('mean')
oop_dis = oop_dis.sort_values(('Ave OoP per discharge'), ascending=False)
oop_dis.head()
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
      <th>Ave OoP per discharge</th>
    </tr>
    <tr>
      <th>DRG Definition</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>207 - RESPIRATORY SYSTEM DIAGNOSIS W VENTILATOR SUPPORT 96+ HOURS</th>
      <td>136.913806</td>
    </tr>
    <tr>
      <th>460 - SPINAL FUSION EXCEPT CERVICAL W/O MCC</th>
      <td>128.522544</td>
    </tr>
    <tr>
      <th>473 - CERVICAL SPINAL FUSION W/O CC/MCC</th>
      <td>126.851226</td>
    </tr>
    <tr>
      <th>870 - SEPTICEMIA OR SEVERE SEPSIS W MV 96+ HOURS</th>
      <td>121.497520</td>
    </tr>
    <tr>
      <th>329 - MAJOR SMALL &amp; LARGE BOWEL PROCEDURES W MCC</th>
      <td>113.064950</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%capture --no-stdout --no-display output
# prevent warning message from showing

# Top 20 out of pocket per discharge
a =  oop_dis[:20]
plt.figure(figsize=(30,20))
fig,ax= plt.subplots()
fig = sns.barplot(a.iloc[:,0],a.index, color="steelblue",alpha=0.8)
ax.set(ylabel=None)
plt.xlabel("Ave OoP per discharge")
```




    Text(0.5, 0, 'Ave OoP per discharge')




    <Figure size 2160x1440 with 0 Axes>



![png]![png](/assets/img/pca_knn/output_58_2.png)


**Conclusion**: This feature helps to define the mean of average out of pocket payment per discharge for each procedure and can be used as a baseline to detect anomaly. If a discharge pay a big difference amount from the mean for specific procedure, it would be noticable

#### 7: Percent of payment covered
This feature displays the proportion of the total payment compared to covered charge.


```python
payment_data['Percent of Payment Covered'] = round((payment_data['Average Total Payments'] / 
                                                    payment_data['Average Covered Charges'])*100,2)
payment_data['Percent of Payment Covered'].head()
```




    0    17.53
    1    38.25
    2    14.47
    3    38.70
    4    17.89
    Name: Percent of Payment Covered, dtype: float64




```python
pc_per= payment_data[['DRG Definition', 'Percent of Payment Covered']].groupby(by='DRG Definition').agg('mean')
pc_per = pc_per.sort_values(('Percent of Payment Covered'), ascending=False)
pc_per.head()
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
      <th>Percent of Payment Covered</th>
    </tr>
    <tr>
      <th>DRG Definition</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>885 - PSYCHOSES</th>
      <td>44.798483</td>
    </tr>
    <tr>
      <th>603 - CELLULITIS W/O MCC</th>
      <td>38.141585</td>
    </tr>
    <tr>
      <th>897 - ALCOHOL/DRUG ABUSE OR DEPENDENCE W/O REHABILITATION THERAPY W/O MCC</th>
      <td>37.399592</td>
    </tr>
    <tr>
      <th>292 - HEART FAILURE &amp; SHOCK W CC</th>
      <td>37.206478</td>
    </tr>
    <tr>
      <th>871 - SEPTICEMIA OR SEVERE SEPSIS W/O MV 96+ HOURS W MCC</th>
      <td>37.103869</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%capture --no-stdout --no-display output
# prevent warning message from showing

# Top 20 highest percent of payment covered
a =  pc_per[:20]
plt.figure(figsize=(30,20))
fig,ax= plt.subplots()
fig = sns.barplot(a.iloc[:,0],a.index, color="steelblue",alpha=0.8)
ax.set(ylabel=None)
plt.xlabel("Percent of Payment Covered")
```




    Text(0.5, 0, 'Percent of Payment Covered')




    <Figure size 2160x1440 with 0 Axes>



![png]![png](/assets/img/pca_knn/output_63_2.png)


**Conclusion**: This feature helps to define the percent of payment covered and can be used as a baseline to detect anomaly. If an unsual percentage of payment for a procedure occurs, it would be noticable.

#### 8: Medicare coverage ratio

This feature calculates the proportion covered by medicare for different procedures


```python
payment_data['Medicare Coverage Ratio'] = (payment_data['Average Medicare Payments'] / payment_data['Average Total Payments'])
```


```python
med_cv = payment_data[['DRG Definition', 'Medicare Coverage Ratio']].groupby(by='DRG Definition').agg('mean')
med_cv = med_cv.sort_values(('Medicare Coverage Ratio'), ascending=True)
med_cv.head()
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
      <th>Medicare Coverage Ratio</th>
    </tr>
    <tr>
      <th>DRG Definition</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>310 - CARDIAC ARRHYTHMIA &amp; CONDUCTION DISORDERS W/O CC/MCC</th>
      <td>0.717772</td>
    </tr>
    <tr>
      <th>203 - BRONCHITIS &amp; ASTHMA W/O CC/MCC</th>
      <td>0.719805</td>
    </tr>
    <tr>
      <th>313 - CHEST PAIN</th>
      <td>0.727650</td>
    </tr>
    <tr>
      <th>390 - G.I. OBSTRUCTION W/O CC/MCC</th>
      <td>0.732743</td>
    </tr>
    <tr>
      <th>149 - DYSEQUILIBRIUM</th>
      <td>0.742108</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%capture --no-stdout --no-display output
# prevent warning message from showing

# Top 20 highest medicare covered
a =  med_cv[:20]
plt.figure(figsize=(20,20))
fig,ax= plt.subplots()
fig = sns.barplot(a.iloc[:,0],a.index, color="steelblue",alpha=0.8)
ax.set(ylabel=None)
plt.xlabel("Medicare Coverage Ratio")
```




    Text(0.5, 0, 'Medicare Coverage Ratio')




    <Figure size 1440x1440 with 0 Axes>



![png]![png](/assets/img/pca_knn/output_68_2.png)


**Conclusion**: This feature helps to define the percent of medicare coverage for each procedure and can be used as a baseline to detect anomaly. If an unsual percentage of medicare charges for a procedure occurs, it would be noticable.

#### 9: Medicare coverage ratio by state
This feature displays the average percentage covered by medicare in each state


```python
med_cv_state = payment_data[['Provider State', 'Medicare Coverage Ratio']].groupby(by='Provider State').agg('mean')
med_cv_state = med_cv_state.sort_values(('Medicare Coverage Ratio'), ascending=False)
med_cv_state.head()
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
      <th>Medicare Coverage Ratio</th>
    </tr>
    <tr>
      <th>Provider State</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MD</th>
      <td>0.888943</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>0.885084</td>
    </tr>
    <tr>
      <th>DC</th>
      <td>0.884320</td>
    </tr>
    <tr>
      <th>VT</th>
      <td>0.874861</td>
    </tr>
    <tr>
      <th>MA</th>
      <td>0.872525</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%capture --no-stdout --no-display output
# prevent warning message from showing

fig,ax= plt.subplots()
fig = sns.barplot(med_cv_state.iloc[:,0],med_cv_state.index,color="steelblue",alpha=0.8)
ax.set(ylabel=None)
plt.xlabel("Medicare Coverage Ratio")
```




    Text(0.5, 0, 'Medicare Coverage Ratio')




![png]![png](/assets/img/pca_knn/output_72_1.png)


**Conclusion**: This feature helps to define the percent of medicare coverage in each state and can be used as a baseline to detect anomalies. If an unsual percentage of medicare happens, it would be noticable

###  Z-score 
Z-scores can quantify the unusualness of an observation when your data follow the normal distribution. Z-scores are the number of standard deviations above and below the mean that each value falls. For example, a Z-score of 2 indicates that an observation is two standard deviations above the average while a Z-score of -2 signifies it is two standard deviations below the mean. A Z-score of zero represents a value that equals the mean.

An absolute Z-score greater than 3 are an outlier/anomaly as they fall outside 99.7% of the data.


#### 10: Z-score Average Total Payment 
This feature calculates the z-score for average total payment


```python
payment_data['Z-score Average Total Payments'] = stats.zscore(payment_data['Average Total Payments'])
```


```python
payment_data['Z-score Average Total Payments'].max()
```




    19.10736911457284



**Conclusion**: An absolute Z-score greater than 3 are an outlier/anomaly as they fall outside 99.7% of the data.This feature helps to detect any outlier in total payments. The example above shows that there is a z-score of 19.10 in the average total payment which needs to be investigated.

#### 11: Z-score Average Medicare Payments	
This feature calculates the z-score for average medicare payment


```python
payment_data['Z-score Average Medicare Payments'] = stats.zscore(payment_data['Average Medicare Payments'])
```


```python
payment_data['Z-score Average Medicare Payments'].max()
```




    19.99143875873488



**Conclusion**: An absolute Z-score greater than 3 are an outlier/anomaly as they fall outside 99.7% of the data.This feature helps to detect any outlier in medicare payments. The example above shows that there is a z-score of 19.99 in the average medicare payment which need to be investigated.

#### 12: Average Covered Charges by Procedures
This feature calculates the average cost and number of cases of each procedure


```python
ave_cv = payment_data[['DRG Definition', 'Average Covered Charges']].groupby(by='DRG Definition').agg(['mean','count'])
ave_cv = ave_cv.sort_values(('Average Covered Charges',  'mean'), ascending=False)
ave_cv.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Average Covered Charges</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>DRG Definition</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>870 - SEPTICEMIA OR SEVERE SEPSIS W MV 96+ HOURS</th>
      <td>163227.331725</td>
      <td>939</td>
    </tr>
    <tr>
      <th>207 - RESPIRATORY SYSTEM DIAGNOSIS W VENTILATOR SUPPORT 96+ HOURS</th>
      <td>143428.051066</td>
      <td>1163</td>
    </tr>
    <tr>
      <th>853 - INFECTIOUS &amp; PARASITIC DISEASES W O.R. PROCEDURE W MCC</th>
      <td>139186.350937</td>
      <td>1376</td>
    </tr>
    <tr>
      <th>329 - MAJOR SMALL &amp; LARGE BOWEL PROCEDURES W MCC</th>
      <td>135330.939966</td>
      <td>1476</td>
    </tr>
    <tr>
      <th>246 - PERC CARDIOVASC PROC W DRUG-ELUTING STENT W MCC OR 4+ VESSELS/STENTS</th>
      <td>96348.806707</td>
      <td>917</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 10 highest Average Medicare Payments
a = ave_cv[:10]
```


```python
%%capture --no-stdout --no-display output
# prevent warning message from showing
fig,ax= plt.subplots()
fig = sns.barplot(a.iloc[:,0],a.index, color="steelblue",alpha=0.8)
ax.set(ylabel=None)
plt.xlabel("Average Covered Charges ($)")
plt.figure(figsize=(15,15))
```




    <Figure size 1080x1080 with 0 Axes>




![png]![png](/assets/img/pca_knn/output_86_1.png)



    <Figure size 1080x1080 with 0 Axes>


**Conclusion**: This features provides the insights of hospital bill for different procedures. From that, we could detect any unsual charge for specific procedure.

#### 12: Common Procedures by Region
This feature calculates the total cases of all procedures in each region


```python
common_drug =  payment_data[['Region', 'DRG Definition']].groupby(by=['Region','DRG Definition']).agg({'DRG Definition': 'count'})
```


```python
common_drug[:20]
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
      <th></th>
      <th>DRG Definition</th>
    </tr>
    <tr>
      <th>Region</th>
      <th>DRG Definition</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="20" valign="top">Midwest</th>
      <th>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</th>
      <td>295</td>
    </tr>
    <tr>
      <th>057 - DEGENERATIVE NERVOUS SYSTEM DISORDERS W/O MCC</th>
      <td>292</td>
    </tr>
    <tr>
      <th>064 - INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION W MCC</th>
      <td>396</td>
    </tr>
    <tr>
      <th>065 - INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION W CC</th>
      <td>548</td>
    </tr>
    <tr>
      <th>066 - INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION W/O CC/MCC</th>
      <td>438</td>
    </tr>
    <tr>
      <th>069 - TRANSIENT ISCHEMIA</th>
      <td>468</td>
    </tr>
    <tr>
      <th>074 - CRANIAL &amp; PERIPHERAL NERVE DISORDERS W/O MCC</th>
      <td>245</td>
    </tr>
    <tr>
      <th>101 - SEIZURES W/O MCC</th>
      <td>374</td>
    </tr>
    <tr>
      <th>149 - DYSEQUILIBRIUM</th>
      <td>273</td>
    </tr>
    <tr>
      <th>176 - PULMONARY EMBOLISM W/O MCC</th>
      <td>373</td>
    </tr>
    <tr>
      <th>177 - RESPIRATORY INFECTIONS &amp; INFLAMMATIONS W MCC</th>
      <td>443</td>
    </tr>
    <tr>
      <th>178 - RESPIRATORY INFECTIONS &amp; INFLAMMATIONS W CC</th>
      <td>453</td>
    </tr>
    <tr>
      <th>189 - PULMONARY EDEMA &amp; RESPIRATORY FAILURE</th>
      <td>515</td>
    </tr>
    <tr>
      <th>190 - CHRONIC OBSTRUCTIVE PULMONARY DISEASE W MCC</th>
      <td>637</td>
    </tr>
    <tr>
      <th>191 - CHRONIC OBSTRUCTIVE PULMONARY DISEASE W CC</th>
      <td>643</td>
    </tr>
    <tr>
      <th>192 - CHRONIC OBSTRUCTIVE PULMONARY DISEASE W/O CC/MCC</th>
      <td>621</td>
    </tr>
    <tr>
      <th>193 - SIMPLE PNEUMONIA &amp; PLEURISY W MCC</th>
      <td>618</td>
    </tr>
    <tr>
      <th>194 - SIMPLE PNEUMONIA &amp; PLEURISY W CC</th>
      <td>711</td>
    </tr>
    <tr>
      <th>195 - SIMPLE PNEUMONIA &amp; PLEURISY W/O CC/MCC</th>
      <td>605</td>
    </tr>
    <tr>
      <th>202 - BRONCHITIS &amp; ASTHMA W CC/MCC</th>
      <td>298</td>
    </tr>
  </tbody>
</table>
</div>



**Conclusion**: This feature can be used to identify non-common procedures and common procedures in each region, thus giving us an idea of what procedures are most common in each region. Therefore, we would be able to take closer look to different price point by provider for common procedures and  non-common procedures in order to detect fraud.

#### 13: Differences in Average Total Payment
This feature is the differences between maximum and minimum payments for each procedure


```python
differences = payment_data[['DRG Definition','Average Total Payments']].groupby(by='DRG Definition').agg(['max','min'])
differences['Difference'] = differences[('Average Total Payments','max')] - differences[('Average Total Payments','min')]
differences = differences[:20].sort_values(by='Difference',ascending=False)
# the results were limited to the first 20 values, but can be changed to include as many or as little as needed by adjusting the range
```


```python
%%capture --no-stdout --no-display output
# prevent warning message from showing

sns.set_context("paper")
ax = sns.barplot(differences["Difference"],differences.index, color="steelblue",alpha=0.8)
ax.set(ylabel=None)
plt.xlabel("Differences in Average Total Payment ($)")
plt.figure(figsize=(15,15))
```




    <Figure size 1080x1080 with 0 Axes>




![png]![png](/assets/img/pca_knn/output_94_1.png)



    <Figure size 1080x1080 with 0 Axes>


**Conclusion**: This feature helps to identify the difference in payment for the same procedure. If the difference is high for a procedure, it means that the payment varies largely between different states or different providers. Thus, we need to investigate further for these procedures. 

### Part 2: Data Preparation

Prepairing the dataset for modeling:

    - Drop irrelevant variables
    - Standardization of Numerical / Float variables

#### Payment data contains 19 mixed features of numerical and categorical columns. 


```python
payment_data = payment_data.drop(columns = ['Provider Id'])
features = payment_data
```


```python
%%capture --no-stdout --no-display output
features = features.merge(differences, on = "DRG Definition", how = "left")
```


```python
# dtop Ave max and min here and keep difference
features.head()
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
      <th>DRG Definition</th>
      <th>Provider Name</th>
      <th>Provider Street Address</th>
      <th>Provider City</th>
      <th>Provider State</th>
      <th>Provider Zip Code</th>
      <th>Hospital Referral Region Description</th>
      <th>Total Discharges</th>
      <th>Average Covered Charges</th>
      <th>Average Total Payments</th>
      <th>...</th>
      <th>Region</th>
      <th>Ave Out of Pocket Payment</th>
      <th>Ave OoP per discharge</th>
      <th>Percent of Payment Covered</th>
      <th>Medicare Coverage Ratio</th>
      <th>Z-score Average Total Payments</th>
      <th>Z-score Average Medicare Payments</th>
      <th>(Average Total Payments, max)</th>
      <th>(Average Total Payments, min)</th>
      <th>(Difference, )</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>SOUTHEAST ALABAMA MEDICAL CENTER</td>
      <td>1108 ROSS CLARK CIRCLE</td>
      <td>DOTHAN</td>
      <td>AL</td>
      <td>36301</td>
      <td>AL - Dothan</td>
      <td>91</td>
      <td>32963.07</td>
      <td>5777.24</td>
      <td>...</td>
      <td>South</td>
      <td>1013.51</td>
      <td>11.137473</td>
      <td>17.53</td>
      <td>0.824568</td>
      <td>-0.512776</td>
      <td>-0.510403</td>
      <td>18420.56</td>
      <td>4968.0</td>
      <td>13452.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>MARSHALL MEDICAL CENTER SOUTH</td>
      <td>2505 U S HIGHWAY 431 NORTH</td>
      <td>BOAZ</td>
      <td>AL</td>
      <td>35957</td>
      <td>AL - Birmingham</td>
      <td>14</td>
      <td>15131.85</td>
      <td>5787.57</td>
      <td>...</td>
      <td>South</td>
      <td>810.86</td>
      <td>57.918571</td>
      <td>38.25</td>
      <td>0.859896</td>
      <td>-0.511428</td>
      <td>-0.481265</td>
      <td>18420.56</td>
      <td>4968.0</td>
      <td>13452.56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>ELIZA COFFEE MEMORIAL HOSPITAL</td>
      <td>205 MARENGO STREET</td>
      <td>FLORENCE</td>
      <td>AL</td>
      <td>35631</td>
      <td>AL - Birmingham</td>
      <td>24</td>
      <td>37560.37</td>
      <td>5434.95</td>
      <td>...</td>
      <td>South</td>
      <td>981.16</td>
      <td>40.881667</td>
      <td>14.47</td>
      <td>0.819472</td>
      <td>-0.557435</td>
      <td>-0.552805</td>
      <td>18420.56</td>
      <td>4968.0</td>
      <td>13452.56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>ST VINCENT'S EAST</td>
      <td>50 MEDICAL PARK EAST DRIVE</td>
      <td>BIRMINGHAM</td>
      <td>AL</td>
      <td>35235</td>
      <td>AL - Birmingham</td>
      <td>25</td>
      <td>13998.28</td>
      <td>5417.56</td>
      <td>...</td>
      <td>South</td>
      <td>1288.40</td>
      <td>51.536000</td>
      <td>38.70</td>
      <td>0.762181</td>
      <td>-0.559703</td>
      <td>-0.597218</td>
      <td>18420.56</td>
      <td>4968.0</td>
      <td>13452.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>SHELBY BAPTIST MEDICAL CENTER</td>
      <td>1000 FIRST STREET NORTH</td>
      <td>ALABASTER</td>
      <td>AL</td>
      <td>35007</td>
      <td>AL - Birmingham</td>
      <td>18</td>
      <td>31633.27</td>
      <td>5658.33</td>
      <td>...</td>
      <td>South</td>
      <td>806.89</td>
      <td>44.827222</td>
      <td>17.89</td>
      <td>0.857398</td>
      <td>-0.528290</td>
      <td>-0.498403</td>
      <td>18420.56</td>
      <td>4968.0</td>
      <td>13452.56</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
features.rename(columns={('Average Total Payments', 'max'):'max',
                         ('Average Total Payments', 'min'): 'min',
                         ('Difference', ''): 'Differences'
                        }, inplace=True)
```


```python
features.drop(
    ['min', 'max'],
    axis=1, inplace=True)
```


```python
# not doing this one because the df only has the medicare coverage ratio columns, this one provides information 
## of coverage ratio by state.
features = features.merge(med_cv_state, on = "Provider State", how = "left")
```


```python
features = features.merge(patient_avg_state, on = "Provider State", how= "left")
```


```python
# Rename the last 4 columns as Ave by state
features.head()
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
      <th>DRG Definition</th>
      <th>Provider Name</th>
      <th>Provider Street Address</th>
      <th>Provider City</th>
      <th>Provider State</th>
      <th>Provider Zip Code</th>
      <th>Hospital Referral Region Description</th>
      <th>Total Discharges_x</th>
      <th>Average Covered Charges_x</th>
      <th>Average Total Payments_x</th>
      <th>...</th>
      <th>Percent of Payment Covered</th>
      <th>Medicare Coverage Ratio_x</th>
      <th>Z-score Average Total Payments</th>
      <th>Z-score Average Medicare Payments</th>
      <th>Differences</th>
      <th>Medicare Coverage Ratio_y</th>
      <th>Total Discharges_y</th>
      <th>Average Covered Charges_y</th>
      <th>Average Total Payments_y</th>
      <th>Average Medicare Payments_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>SOUTHEAST ALABAMA MEDICAL CENTER</td>
      <td>1108 ROSS CLARK CIRCLE</td>
      <td>DOTHAN</td>
      <td>AL</td>
      <td>36301</td>
      <td>AL - Dothan</td>
      <td>91</td>
      <td>32963.07</td>
      <td>5777.24</td>
      <td>...</td>
      <td>17.53</td>
      <td>0.824568</td>
      <td>-0.512776</td>
      <td>-0.510403</td>
      <td>13452.56</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
      <td>7568.232149</td>
      <td>6418.00712</td>
    </tr>
    <tr>
      <th>1</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>MARSHALL MEDICAL CENTER SOUTH</td>
      <td>2505 U S HIGHWAY 431 NORTH</td>
      <td>BOAZ</td>
      <td>AL</td>
      <td>35957</td>
      <td>AL - Birmingham</td>
      <td>14</td>
      <td>15131.85</td>
      <td>5787.57</td>
      <td>...</td>
      <td>38.25</td>
      <td>0.859896</td>
      <td>-0.511428</td>
      <td>-0.481265</td>
      <td>13452.56</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
      <td>7568.232149</td>
      <td>6418.00712</td>
    </tr>
    <tr>
      <th>2</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>ELIZA COFFEE MEMORIAL HOSPITAL</td>
      <td>205 MARENGO STREET</td>
      <td>FLORENCE</td>
      <td>AL</td>
      <td>35631</td>
      <td>AL - Birmingham</td>
      <td>24</td>
      <td>37560.37</td>
      <td>5434.95</td>
      <td>...</td>
      <td>14.47</td>
      <td>0.819472</td>
      <td>-0.557435</td>
      <td>-0.552805</td>
      <td>13452.56</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
      <td>7568.232149</td>
      <td>6418.00712</td>
    </tr>
    <tr>
      <th>3</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>ST VINCENT'S EAST</td>
      <td>50 MEDICAL PARK EAST DRIVE</td>
      <td>BIRMINGHAM</td>
      <td>AL</td>
      <td>35235</td>
      <td>AL - Birmingham</td>
      <td>25</td>
      <td>13998.28</td>
      <td>5417.56</td>
      <td>...</td>
      <td>38.70</td>
      <td>0.762181</td>
      <td>-0.559703</td>
      <td>-0.597218</td>
      <td>13452.56</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
      <td>7568.232149</td>
      <td>6418.00712</td>
    </tr>
    <tr>
      <th>4</th>
      <td>039 - EXTRACRANIAL PROCEDURES W/O CC/MCC</td>
      <td>SHELBY BAPTIST MEDICAL CENTER</td>
      <td>1000 FIRST STREET NORTH</td>
      <td>ALABASTER</td>
      <td>AL</td>
      <td>35007</td>
      <td>AL - Birmingham</td>
      <td>18</td>
      <td>31633.27</td>
      <td>5658.33</td>
      <td>...</td>
      <td>17.89</td>
      <td>0.857398</td>
      <td>-0.528290</td>
      <td>-0.498403</td>
      <td>13452.56</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
      <td>7568.232149</td>
      <td>6418.00712</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
features = features.rename(columns={"Total Discharges_x": "Total_Discharges", 
                                   "Average Covered Charges_x": "Ave_Covered_Charges",
                                   "Average Total Payments_x": "Ave_Total_Payments",
                                   "Average Medicare Payments_x":"Aver_Medicare_Payments",
                                   "Ave Out of Pocket Payment":"Ave_OOP",
                                   "Percent of Payment Covered" : "Prop_payment_covered",
                                   "Medicare Coverage Ratio_x":"Medicare_Coverage_Ratio",
                                   "Z-score Average Total Payments": "Zscore_ave_total_payment",
                                   "Z-score Average Medicare Payments": "Zscore_ave_medicare_payment",
                                    "Medicare Coverage Ratio_y": "Medicare_coverage_ratio_bystate",
                                   "Total Discharges_y":"Mean_Total_Discharge_bystate", 
                                   "Average Covered Charges_y":"Mean_Ave_Covered_bystate",
                                  "Average Total Payments_y":"Mean_Ave_Total_Payment_bystate",
                                  "Average Medicare Payments_y":"Mean_Ave_Medicare_bystate"})
```

#### *Note*

KNN and PCA only work well with numerical columns and there are more than 50 unique values in categorical columns like DRG definition, Provider ID, provider name, provider city, provider state, provider zipcode, hospital referral regions. Therefore, one hot encoder doesn't make significant support to clustering method. Additionally, using one hot encoding creates many different new binary features which does not apply to K-mean clustering accurately.  Therefore, all categorical columns are decided to drop.


```python
categorical_col = features.loc[:, features.dtypes == np.object]
a = categorical_col.columns
features = features.drop(columns = a)
```


```python
features.columns
```




    Index(['Total_Discharges', 'Ave_Covered_Charges', 'Ave_Total_Payments',
           'Aver_Medicare_Payments', 'Ave_OOP', 'Ave OoP per discharge',
           'Prop_payment_covered', 'Medicare_Coverage_Ratio',
           'Zscore_ave_total_payment', 'Zscore_ave_medicare_payment',
           'Differences', 'Medicare_coverage_ratio_bystate',
           'Mean_Total_Discharge_bystate', 'Mean_Ave_Covered_bystate',
           'Mean_Ave_Total_Payment_bystate', 'Mean_Ave_Medicare_bystate'],
          dtype='object')




```python
features.isnull().sum()
```




    Total_Discharges                        0
    Ave_Covered_Charges                     0
    Ave_Total_Payments                      0
    Aver_Medicare_Payments                  0
    Ave_OOP                                 0
    Ave OoP per discharge                   0
    Prop_payment_covered                    0
    Medicare_Coverage_Ratio                 0
    Zscore_ave_total_payment                0
    Zscore_ave_medicare_payment             0
    Differences                        124528
    Medicare_coverage_ratio_bystate         0
    Mean_Total_Discharge_bystate            0
    Mean_Ave_Covered_bystate                0
    Mean_Ave_Total_Payment_bystate          0
    Mean_Ave_Medicare_bystate               0
    dtype: int64



With a large amount of missing values in differences column, it does not have significant impact to the model after imputing missing values


```python
features['Differences'].isnull().sum()/len(features)
```




    0.7636709287707356




```python
features = features.drop(columns = ['Differences'])
```

### Correlation


```python
corr =features.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <AxesSubplot:>




![png]![png](/assets/img/pca_knn/output_115_1.png)


In general, feautures that are too correlated do not improve model efficiency and also affect the performance of linear regression and random forest models, making the learning algorithms slower to create and train. Therefore, I removed highly correlated features to prevent multicollinearity trhough the following function:


```python
# Function to remove columns with high correlation value
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
features = correlation(features, 0.9)
```


```python
features.isna().sum()
```




    Total_Discharges                   0
    Ave_Covered_Charges                0
    Ave_Total_Payments                 0
    Ave_OOP                            0
    Ave OoP per discharge              0
    Prop_payment_covered               0
    Medicare_Coverage_Ratio            0
    Medicare_coverage_ratio_bystate    0
    Mean_Total_Discharge_bystate       0
    Mean_Ave_Covered_bystate           0
    Mean_Ave_Total_Payment_bystate     0
    dtype: int64



## Part 2: Unsupervised Learning for Anomalies Detection

### Principal Component Analysis (PCA) 

PCA is a linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. The PCA-Based Anomaly Detection module solves the problem by analyzing available features to determine what constitutes a "normal" class, and applying distance metrics to identify cases that represent anomalies. 

Split dataset into train and test before standardizing the data


```python
from sklearn.model_selection import train_test_split as tts
X_train, X_test = tts(features, test_size=0.30)
from pyod.utils.utility import standardizer
# Standardize data
X_train, X_test= standardizer(X_train, X_test)
```


```python
pca = PCA()
pca_fit = pca.fit(X_train)  
```


```python
# For the predictions of the training data:
y_train_scores = pca.decision_scores_
```


```python
# Now we have the trained PCA model, let's apply to the test data to get the predictions
y_test_pred = pca.predict(X_test) # outlier labels (0 or 1)
# Because it is '0' and '1', we can run a count statistic. There are 44070 '1's and 456 '4850's. 
#The number of anomalies is approximately ten percent, as we have generated before:
unique, counts = np.unique(y_test_pred, return_counts=True)
dict(zip(unique, counts))
#{0: 44070, 1: 4850}
# Generate the anomaly score using pca.decision_function:
y_test_scores = pca.decision_function(X_test)
```

By using the function, explained_variance_ratio, we can see that the variance score for all components under the PCA model.


```python
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Cumulative sum of explained ratio')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.tight_layout()
plt.show()
```


![png]![png](/assets/img/pca_knn/output_128_0.png)



```python
np.cumsum(pca.explained_variance_ratio_)
```




    array([0.24448861, 0.45108973, 0.6031976 , 0.73915498, 0.83877332,
           0.91347225, 0.9431591 , 0.96761278, 0.9807727 , 0.99132136,
           1.        ])



The PCA indicates that 99.13% of the of the variables can be captured by 10 components.Thus, I have the option to choose 10 as it gives the principal components an incremental value of 99.13%.

However, as discussed in class, it is suggested that to keep the same number of components. Therefore, I would keep all 11 features here as recommended by the professor.


```python
import matplotlib.pyplot as plt
plt.hist(y_test_scores, bins='auto', color='mediumvioletred', lw=0)  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
```


![png]![png](/assets/img/pca_knn/output_131_0.png)


Create a dataframe for grouping data points using the results from y_test_scores. After generating different scores ranges, 
the code below make 3 different clusters where one cluster has about 6% of the data points.  


```python
df_test = pd.DataFrame(X_test)
df_test['score'] = y_test_scores
df_test['cluster'] = np.where(df_test['score']<700, 2,
                              (np.where(df_test['score']<1500, 0, 1)))
df_test['cluster'].value_counts()
```




    0    34794
    1    11113
    2     3013
    Name: cluster, dtype: int64




```python
df_test['score'].describe()
```




    count    48920.000000
    mean      1244.023329
    std        694.477713
    min        476.225151
    25%        862.061369
    50%       1055.764673
    75%       1438.135050
    max      27662.721271
    Name: score, dtype: float64




```python
df_test.groupby('cluster').mean()
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
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>score</th>
    </tr>
    <tr>
      <th>cluster</th>
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
      <td>-0.072755</td>
      <td>-0.208975</td>
      <td>-0.224965</td>
      <td>-0.142122</td>
      <td>-0.117929</td>
      <td>-0.043599</td>
      <td>-0.037450</td>
      <td>-0.256868</td>
      <td>0.055712</td>
      <td>-0.226813</td>
      <td>-0.298481</td>
      <td>1012.957203</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.319755</td>
      <td>0.721435</td>
      <td>0.767749</td>
      <td>0.515540</td>
      <td>0.452200</td>
      <td>0.203777</td>
      <td>0.041871</td>
      <td>0.901751</td>
      <td>-0.123544</td>
      <td>0.723400</td>
      <td>1.007677</td>
      <td>2128.565872</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.173436</td>
      <td>-0.253020</td>
      <td>-0.241316</td>
      <td>-0.163855</td>
      <td>-0.190221</td>
      <td>-0.186745</td>
      <td>0.217144</td>
      <td>-0.375160</td>
      <td>-0.150630</td>
      <td>-0.022109</td>
      <td>-0.289592</td>
      <td>649.862520</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.groupby('cluster')['score'].mean().plot.bar(fontsize=12, color='mediumvioletred')
```




    <AxesSubplot:xlabel='cluster'>




![png]![png](/assets/img/pca_knn/output_136_1.png)


From the table and chart above, the average anomaly score in Cluster 2 is much lower than that of Cluster 0 and cluster 1. The summary statistics also show dramatic differences between them in every feature. Additionally, cluster 2 only takes about 6% of the dataset (3013 data points). Therefore, I believe that the data points in Cluster 2 could be anomalous and deserve further inspection.

By aggregating multiple models, the chance of overfitting is greatly reduced and the prediction accuracy will be improved.The PyOD module offers four methods to aggregate the outcome, which are Average, Maximum of Maximum (MOM),Average of Maximum (AOM), Maximum of Average (MOA). Normally, we can choose one method and aggregate the outcome. However, I am going to use the Average and Maximun of maximum (MOM) methods today, so that we can see the differences and benefits in different approaches with the given dataset. 

#### Average


```python
# Combination by average
y_by_average = average(X_test)
import matplotlib.pyplot as plt
plt.hist(y_by_average, bins='auto',color='mediumvioletred', lw=0) # arguments are passed to np.histogram
plt.title("Combination by average")
plt.show()
```


![png]![png](/assets/img/pca_knn/output_140_0.png)


Create a dataframe for grouping data points using the results from X_test. After generating different scores ranges, the code below make 3 different clusters where one cluster has about 6% of the data points.


```python
# Combination by mom
y_by_maximization = maximization(X_test)
df_test['y_by_average_score'] = y_by_average
df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']< -0.1, 0,
                                            (np.where(df_test['y_by_average_score']< -0, 2, 1)))
df_test['y_by_average_cluster'].value_counts()
```




    0    24416
    1    19968
    2     4536
    Name: y_by_average_cluster, dtype: int64




```python
df_test['y_by_average_score'].describe()
```




    count    48920.000000
    mean         0.002208
    std          0.422596
    min         -0.919141
    25%         -0.293772
    50%         -0.099129
    75%          0.242968
    max          9.799379
    Name: y_by_average_score, dtype: float64




```python
df_test.groupby('y_by_average_cluster').mean()
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
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>score</th>
      <th>cluster</th>
      <th>y_by_average_score</th>
    </tr>
    <tr>
      <th>y_by_average_cluster</th>
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
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.152548</td>
      <td>-0.370799</td>
      <td>-0.457816</td>
      <td>-0.156949</td>
      <td>-0.088994</td>
      <td>-0.140696</td>
      <td>-0.346260</td>
      <td>-0.585800</td>
      <td>-0.298915</td>
      <td>-0.258783</td>
      <td>-0.557212</td>
      <td>960.287413</td>
      <td>0.227269</td>
      <td>-0.310434</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.201765</td>
      <td>0.468644</td>
      <td>0.569591</td>
      <td>0.227712</td>
      <td>0.148938</td>
      <td>0.176918</td>
      <td>0.355280</td>
      <td>0.783578</td>
      <td>0.339706</td>
      <td>0.347302</td>
      <td>0.747008</td>
      <td>1653.371704</td>
      <td>0.516226</td>
      <td>0.396949</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.043027</td>
      <td>-0.070675</td>
      <td>-0.048074</td>
      <td>-0.093554</td>
      <td>-0.099686</td>
      <td>0.019278</td>
      <td>0.259394</td>
      <td>-0.306492</td>
      <td>0.138163</td>
      <td>-0.118098</td>
      <td>-0.302234</td>
      <td>969.293995</td>
      <td>0.282628</td>
      <td>-0.052632</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.groupby('y_by_average_cluster')['y_by_average_score'].mean().plot.bar(fontsize=12, color='mediumvioletred')
```




    <AxesSubplot:xlabel='y_by_average_cluster'>




![png]![png](/assets/img/pca_knn/output_145_1.png)


The average anomaly score in Cluster 2 is very different from Cluster 0 and cluster 1. The summary statistics also show large differences between these clusters in every feature. Additionally, cluster 2 only takes approximately 10% of the dataset (4549 data points). Therefore, I believe that the data points in Cluster 2 could be anomalous and deserve further inspection.

#### Maximum of Maximum (MOM)


```python
# Combination by mom
y_by_maximization = maximization(X_test)
df_test['y_by_maximization_score'] = y_by_maximization
df_test['y_by_maximization_cluster'] = np.where(df_test['y_by_maximization_score']<0, 0, 1)
df_test['y_by_maximization_cluster'].value_counts()
```




    1    48121
    0      799
    Name: y_by_maximization_cluster, dtype: int64



When we use the Maximum of Maximum method, we get 799 data points that have an outlier scores higher than 0 with 2 clusters. While we used 3 clusters above, there was a clear delineation with just 2 clusters for this measure, so I reduced the number as performance did not improve with 3. Next, we use the following code to produce the summary statistics by cluster.


```python
df_test['y_by_maximization_score'].describe()
```




    count    48920.000000
    mean         1.535629
    std          1.547922
    min         -0.316791
    25%          0.617005
    50%          1.179640
    75%          2.152001
    max         67.873187
    Name: y_by_maximization_score, dtype: float64




```python
# Combination by mom
y_by_maximization = maximization(X_test)
             
import matplotlib.pyplot as plt
plt.hist(y_by_maximization, bins='auto',color='mediumvioletred', lw=0)  # arguments are passed to np.histogram
plt.title("Combination by max")
plt.show()
```


![png]![png](/assets/img/pca_knn/output_151_0.png)



```python
df_test.groupby('y_by_maximization_cluster').mean()
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
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>score</th>
      <th>cluster</th>
      <th>y_by_average_score</th>
      <th>y_by_average_cluster</th>
      <th>y_by_maximization_score</th>
    </tr>
    <tr>
      <th>y_by_maximization_cluster</th>
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
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.288531</td>
      <td>-0.501673</td>
      <td>-0.684760</td>
      <td>-0.250434</td>
      <td>-0.225714</td>
      <td>-0.472337</td>
      <td>-0.670406</td>
      <td>-0.641649</td>
      <td>-0.642467</td>
      <td>-0.463731</td>
      <td>-0.787492</td>
      <td>923.958567</td>
      <td>0.133917</td>
      <td>-0.511745</td>
      <td>0.000000</td>
      <td>-0.094860</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.015169</td>
      <td>0.007995</td>
      <td>0.010902</td>
      <td>0.010195</td>
      <td>0.010999</td>
      <td>0.011685</td>
      <td>0.007319</td>
      <td>0.009685</td>
      <td>0.012988</td>
      <td>0.009379</td>
      <td>0.011838</td>
      <td>1249.337677</td>
      <td>0.353941</td>
      <td>0.010741</td>
      <td>0.603479</td>
      <td>1.562702</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.groupby('y_by_maximization_cluster')['y_by_maximization_score'].mean().plot.bar(fontsize=12, color='mediumvioletred')
```




    <AxesSubplot:xlabel='y_by_maximization_cluster'>




![png]![png](/assets/img/pca_knn/output_153_1.png)


By using MOM method, we can see that there are dramatic differences between cluster 0 and cluster 1 in the summary statistic table above. Furthermore, the average maximization score of cluster 0 is significantly lower than cluster 1. Therefore, I believe that the data points in Cluster 0 could be anomalous and deserves further inspection.

### K-nearest neighbor: KNN  Model:

KNN-based anomaly detection methods relies on neighbors search to decide whether a data point is an outlier. The method computes the distance of an observation called the Euclidean distance, which compares observations against each other. Thus, an isolated data point has a large distance to other observations and it can be seen as an outlier through KNN. 

Split train test datasets and standardize the data


```python
from sklearn.model_selection import train_test_split as tts

X_train, X_test = tts(features, test_size=0.3)
from pyod.utils.utility import standardizer
# Standardize data
X_train, X_test= standardizer(X_train, X_test)
```


```python
# train kNN detector
from pyod.models.knn import KNN
clf_name = 'KNN'
clf = KNN()
clf.fit(X_train)
# Outlier scores:
y_train_scores = clf.decision_scores_
```


```python
# Now we have the trained K-NN model, let's apply to the test data to get the predictions
y_test_pred = clf.predict(X_test) # outlier labels (0 or 1)
# Because it is '0' and '1', we can run a count statistic. There are 44 '1's and 456 '0's. The number of anomalies is roughly ten percent, as we have generated before:
unique, counts = np.unique(y_test_pred, return_counts=True)
dict(zip(unique, counts))
#{0: 456, 1: 44}
# And you can generate the anomaly score using clf.decision_function:
y_test_scores = clf.decision_function(X_test)
```


```python
import matplotlib.pyplot as plt
plt.hist(y_test_scores, bins='auto',lw=0)  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
```


![png]![png](/assets/img/pca_knn/output_160_0.png)


Create a dataframe for grouping clusters. 


```python
df_test = pd.DataFrame(X_test)
df_test['score'] = y_test_scores
df_test['cluster'] = np.where(df_test['score']<1, 0, 1)
df_test['cluster'].value_counts()
```




    0    46112
    1     2808
    Name: cluster, dtype: int64




```python
df_test.groupby('cluster').mean()
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
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>score</th>
    </tr>
    <tr>
      <th>cluster</th>
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
      <td>-0.053490</td>
      <td>-0.080232</td>
      <td>-0.113099</td>
      <td>-0.138047</td>
      <td>-0.104031</td>
      <td>-0.018746</td>
      <td>0.034724</td>
      <td>-0.001532</td>
      <td>0.022909</td>
      <td>0.006031</td>
      <td>-0.012583</td>
      <td>0.373883</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.858927</td>
      <td>1.395625</td>
      <td>1.882697</td>
      <td>2.208307</td>
      <td>1.711775</td>
      <td>0.264079</td>
      <td>-0.551027</td>
      <td>-0.016674</td>
      <td>-0.136797</td>
      <td>-0.059699</td>
      <td>0.150461</td>
      <td>1.568175</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.groupby('cluster')['score'].mean().plot.bar(fontsize=12, color='mediumvioletred')
```




    <AxesSubplot:xlabel='cluster'>




![png]![png](/assets/img/pca_knn/output_164_1.png)


The average anomaly score in Cluster 1 is much higher than Cluster 0. The summary statistics also show large differences between the two clusters in every feature. Additionally, cluster 1 only takes about 6% of the dataset (2808 data points). Therefore, I believe that the data points in Cluster 1 could be anomalous and deserve further inspection.

### Achieve Model Stability by Aggregating Multiple Models

This approach is to reduce the chance of overfitting and improve the prediction accuracy.


```python
X_train, X_test = tts(features, test_size=0.3)
# Standardize data
X_train_norm, X_test_norm = standardizer(X_train, X_test)
# Test a range of k-neighbors from 10 to 200. There will be 20 k-NN models.
n_clf = 20
k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 
 120, 130, 140, 150, 160, 170, 180, 190, 200]
# Just prepare data frames so we can store the model results
train_scores = np.zeros([X_train.shape[0], n_clf])
test_scores = np.zeros([X_test.shape[0], n_clf])
train_scores.shape
# Modeling
for i in range(n_clf):
    k = k_list[i]
    clf = KNN(n_neighbors=k, method='largest')
    clf.fit(X_train_norm)

    # Store the results in each column:
    train_scores[:, i] = clf.decision_scores_
    test_scores[:, i] = clf.decision_function(X_test_norm) 
# Decision scores have to be normalized before combination
train_scores_norm, test_scores_norm = standardizer(train_scores,test_scores)
```

From all four PyOD methods, I choose Average and Maximum of Maximum methods to find anomalous clusters. With different approaches, we will see a small change in total number of anomalous values. 

### Method 1: Average


```python
# Combination by average
y_by_average = average(test_scores_norm)
import matplotlib.pyplot as plt
plt.hist(y_by_average, bins='auto',color='mediumvioletred', lw=0) # arguments are passed to np.histogram
plt.title("Combination by average")
plt.show()
```


![png]![png](/assets/img/pca_knn/output_170_0.png)


Create a dataframe for grouping data points using the results from y_test_scores. After generating different scores ranges, the code below make 3 different clusters where one cluster has about 6% of the data points.


```python
df_test = pd.DataFrame(X_test)
df_test['y_by_average_score'] = y_by_average
df_test['y_by_average_cluster'] =  np.where(df_test['y_by_average_score']< -0.1, 0,
                                            (np.where(df_test['y_by_average_score']< -0, 2, 1)))
df_test['y_by_average_cluster'].value_counts()
```




    0    30219
    1    15535
    2     3166
    Name: y_by_average_cluster, dtype: int64




```python
df_test['y_by_average_score'].describe()
```




    count    48920.000000
    mean         0.001864
    std          1.024943
    min         -0.758232
    25%         -0.432940
    50%         -0.235889
    75%          0.139228
    max         70.302526
    Name: y_by_average_score, dtype: float64




```python
df_test.groupby('y_by_average_cluster').mean()
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
      <th>Total_Discharges</th>
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Total_Discharge_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
      <th>Mean_Ave_Total_Payment_bystate</th>
      <th>y_by_average_score</th>
    </tr>
    <tr>
      <th>y_by_average_cluster</th>
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
      <td>33.162249</td>
      <td>28040.104088</td>
      <td>7293.289628</td>
      <td>896.176893</td>
      <td>37.152267</td>
      <td>31.721087</td>
      <td>0.855126</td>
      <td>0.847397</td>
      <td>43.099472</td>
      <td>36983.937343</td>
      <td>9700.512995</td>
      <td>-0.390124</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60.514837</td>
      <td>53133.309223</td>
      <td>14668.412225</td>
      <td>1867.328584</td>
      <td>76.289759</td>
      <td>36.553877</td>
      <td>0.830196</td>
      <td>0.845619</td>
      <td>42.263915</td>
      <td>34973.442101</td>
      <td>9773.504899</td>
      <td>0.775300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46.185092</td>
      <td>32417.838234</td>
      <td>8903.654318</td>
      <td>1072.961611</td>
      <td>40.074716</td>
      <td>34.901901</td>
      <td>0.845802</td>
      <td>0.843813</td>
      <td>42.442675</td>
      <td>33436.026414</td>
      <td>9465.212670</td>
      <td>-0.051783</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.groupby('y_by_average_cluster')['y_by_average_score'].mean().plot.bar(fontsize=12, color='mediumvioletred')
```




    <AxesSubplot:xlabel='y_by_average_cluster'>




![png]![png](/assets/img/pca_knn/output_175_1.png)


The absolute score of y_by_average_score in Cluster 2 is significantly lower than the absolute score of Cluster 0 and cluster 1. The summary statistics also show large differences between the two clusters in every feature. Additionally, cluster 2 only takes about 6% of the dataset (3166 data points). Therefore, I believe that the data points in Cluster 2 could be anomalous and deserve further inspection.

### Method 2: The Maximum of Maximum (MoM)


```python
# Combination by mom
y_by_maximization = maximization(test_scores_norm)
             
import matplotlib.pyplot as plt
plt.hist(y_by_maximization, bins='auto', lw=0)  # arguments are passed to np.histogram
plt.title("Combination by max")
plt.show()
```


![png]![png](/assets/img/pca_knn/output_178_0.png)


Create a dataframe for grouping data points using the results from y_test_scores. After generating different scores ranges, the code below make 3 different clusters where one cluster has about 6% of the data points.


```python
df_test = pd.DataFrame(X_test)
df_test['y_by_maximization_score'] = y_by_maximization
df_test['y_by_maximization_cluster'] =  np.where(df_test['y_by_maximization_score']< -0.1, 0,
                                                 (np.where(df_test['y_by_maximization_score']<0, 2, 1)))
df_test['y_by_maximization_cluster'].value_counts()
```




    0    27340
    1    18081
    2     3499
    Name: y_by_maximization_cluster, dtype: int64




```python
df_test.groupby('y_by_maximization_cluster').mean()
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
      <th>Total_Discharges</th>
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Total_Discharge_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
      <th>Mean_Ave_Total_Payment_bystate</th>
      <th>y_by_average_score</th>
      <th>y_by_average_cluster</th>
      <th>y_by_maximization_score</th>
    </tr>
    <tr>
      <th>y_by_maximization_cluster</th>
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
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.376993</td>
      <td>27778.831031</td>
      <td>7166.175707</td>
      <td>883.232781</td>
      <td>36.933977</td>
      <td>31.416214</td>
      <td>0.855728</td>
      <td>0.847885</td>
      <td>43.203059</td>
      <td>37377.232546</td>
      <td>9733.059564</td>
      <td>-0.415359</td>
      <td>0.000000</td>
      <td>-0.373306</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58.424700</td>
      <td>50232.169230</td>
      <td>13857.820603</td>
      <td>1756.494024</td>
      <td>71.221979</td>
      <td>36.306667</td>
      <td>0.832438</td>
      <td>0.845116</td>
      <td>42.232634</td>
      <td>34679.116796</td>
      <td>9712.145539</td>
      <td>0.657857</td>
      <td>1.088048</td>
      <td>0.829459</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41.979423</td>
      <td>30775.549534</td>
      <td>8565.925862</td>
      <td>1023.367494</td>
      <td>39.212002</td>
      <td>34.742295</td>
      <td>0.848535</td>
      <td>0.844235</td>
      <td>42.465413</td>
      <td>33684.434593</td>
      <td>9497.259835</td>
      <td>-0.127925</td>
      <td>0.627036</td>
      <td>-0.051398</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.groupby('y_by_maximization_cluster')['y_by_maximization_score'].mean().plot.bar(fontsize=12, color='mediumvioletred')
```




    <AxesSubplot:xlabel='y_by_maximization_cluster'>




![png]![png](/assets/img/pca_knn/output_182_1.png)


The y_by_maximization_score in Cluster 2 is much lower than that of Cluster 0 and cluster 1. The summary statistics also show large differences between the two clusters in every feature. Additionally, cluster 2 only takes about 6% of the dataset (3499 data points). Therefore, I believe that the data points in Cluster 2 could be anomalous and deserve further inspection.

### Summary:

Overall, recorgnizing potentially frauds in heathcare is not an easy task. Each provider has a different population and price tag for their drug. By applying PCA and KNN methods, anomalous charges can be detected, avoiding fraud. With the given dataset, there are some takeaways from the report as follows:

    - PCA method:
        - With the anomaly score, there are 3 clusters created where one cluster contains about 6% of the data points. In this cluster, the summary statistic also shows that its average anomaly score is much lower than the other two clusters. 
        - With the Average method, there are also 3 clusters created where one cluster contains aproximately 10% of the data points. In this cluster, the summary statistic also shows that its average score is much lower than the other two.
        - With The Maximum of Maximum method, there are 2 clusters created where one cluster contains about 2% of the data points. In this cluster, the summary statistic also shows that its maximization score is significantly lower than the other cluster.
    - KNN method:
        - With the anomaly score, there are 2 clusters created where one cluster contains about 5% of the data points. In this cluster, the summary statistic also shows that its average anomaly score is much lower than the other two clusters. 
        - With the Average method, there are 3 clusters created where one cluster contains about 6% of the data points. In this cluster, the summary statistic also shows that its average anomaly score is much lower than the other two clusters. 
        - With The Maximum of Maximum method, there are 3 clusters created where one cluster contains about 6% of the data points. In this cluster, the summary statistic also shows that its average anomaly score is much lower than the other two clusters. 
        
**Conclusion:** Overall, both of the approaches provide insights to detect anomalies. In PCA, the MOM method defines 2 clusters where the statistics calculation shows a significant difference between them. However, the other approaches required smaller breakdowns with 3 clusters to notice the anomalous activities. In KNN, the anomaly score defined 2 clusters where the statistics calculation showed a significant difference between them. However, the other approaches required smaller breakdowns to 3 clusters to notice the anomalous activities. Therefore, it is essential to try different modelling and approaches from PyOD to compare the anomaly detection effectiveness.
