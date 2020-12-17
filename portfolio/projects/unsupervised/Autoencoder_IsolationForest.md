
# Unspervised Learning: Autoencoder and Isolation Forest

Tram Duong
<br>October 24, 2020

## Table of Contents:
* [Part 1: EDA and FE](#Part_1)
* [Part 2: Data Preparation](#Part_2)
* [Part 3: Unsupervised Learning](#Part_3)

## Part 1: EDA and FE <a class="anchor" id="Part_1"></a>
- Data Exploration
- Data Cleaning
- Feature Engineerings


```python
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy
from scipy import stats
sns.set(rc={'figure.figsize':(10,15)})
```


```python
# read data
payment_data = pd.read_csv("C:/Github/inpatientCharges.csv")
```

Few notices from the data overview:
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



![png](/assets/img/Autoencoder_Isolation/output_24_1.png)


### Visualization by regions


```python
fig = plt.figure(figsize=(16,10))
sns.pairplot(payment_data[['Region','Average Total Payments',
                            'Total Discharges','Average Medicare Payments','Average Covered Charges']], hue= 'Region',height = 4)
```




    <seaborn.axisgrid.PairGrid at 0x26303433898>




    <Figure size 1152x720 with 0 Axes>



![png](/assets/img/Autoencoder_Isolation/output_26_2.png)


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




![png](/assets/img/Autoencoder_Isolation/output_29_1.png)


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




![png](/assets/img/Autoencoder_Isolation/output_32_1.png)


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


![png](/assets/img/Autoencoder_Isolation/output_40_0.png)


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


![png](/assets/img/Autoencoder_Isolation/output_44_0.png)


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



![png](/assets/img/Autoencoder_Isolation/output_49_2.png)


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



![png](/assets/img/Autoencoder_Isolation/output_53_2.png)


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



![png](/assets/img/Autoencoder_Isolation/output_59_2.png)


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



![png](/assets/img/Autoencoder_Isolation/output_64_2.png)


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



![png](/assets/img/Autoencoder_Isolation/output_69_2.png)


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




![png](/assets/img/Autoencoder_Isolation/output_73_1.png)


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




![png](/assets/img/Autoencoder_Isolation/output_87_1.png)



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




![png](/assets/img/Autoencoder_Isolation/output_95_1.png)



    <Figure size 1080x1080 with 0 Axes>


**Conclusion**: This feature helps to identify the difference in payment for the same procedure. If the difference is high for a procedure, it means that the payment varies largely between different states or different providers. Thus, we need to investigate further for these procedures.

## Part 2: Data Preparation <a class="anchor" id="Part_2"></a>
In this part, I use K-means clustering algorithm to explore the dataset, following the steps below:

    - Drop irrelevant variables
    - Standardization of Numerical / Float variables

#### Payment data contains 19 mixed features of numerical and categorical columns.


```python
payment_data = payment_data.drop(columns = ['Provider Id'])
```


```python
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
K-means only works with numerical columns and there are more than 50 unique values in categorical columns like DRG definition, Provider ID, provider name, provider city, provider state, provider zipcode, hospital referral regions. Therefore, one hot encoder doesn't make significant support to clustering method. Additionally, using one hot encoding creates many different new binary features which does not apply to K-mean clustering accurately.  Therefore, all categorical columns are decided to drop.


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
corr = features.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <AxesSubplot:>




![png](/assets/img/Autoencoder_Isolation/output_117_1.png)


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
features = correlation(features, 0.8)
```


```python
features.columns
```




    Index(['Total_Discharges', 'Ave_Covered_Charges', 'Ave_Total_Payments',
           'Ave_OOP', 'Ave OoP per discharge', 'Prop_payment_covered',
           'Medicare_Coverage_Ratio', 'Medicare_coverage_ratio_bystate',
           'Mean_Total_Discharge_bystate', 'Mean_Ave_Covered_bystate'],
          dtype='object')



As total discharges amount varies depending on the hospital location, size, treatement provided. This feature would not support the model performance and might affect the accuracy of the model. Thus, I will drop both of the discharges features before further analysis.


```python
features = features.drop(columns = ['Total_Discharges','Mean_Total_Discharge_bystate'])
```

#### Splitting train and test dataset


```python
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
X_train, X_test = tts(features, test_size=0.30)
```

#### Standardize data


```python
X_train = StandardScaler().fit_transform(X_train)
X_train = pd.DataFrame(X_train)
X_test = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(X_test)
```

## Part 3: Unsupervised Learning for Anomalies Detection <a class="anchor" id="Part_3"></a>

In this part. I will use autoencoder and isolation forest algorithm to detect outliers. I will try 3 different models, the average and the maximun of maximun methods for both of the approaches. Furthermore, the feature selection plays an important role to the success of a model, beside their statistical support, it is important to understand their business insights and how they would affect the model. Below is a briefly explaination of the selected features and why they are important:

   - **Ave_Covered_Charges**: The average charge of all provider's services covered by Medicare for discharges in the DRG, providng the differences in coverage by different hospital charge structures.
   - **Ave_Total_Payments**: The average total payments to all providers for the DRG, providing the difference of total payments by different providers for the drug.
   - **Ave_OOP**: The average out of pocket payment by different provider, providing an idea of the cost by different providers. This feature helps to define the mean of average out of pocket payment in each provider and can be used as a baseline to compare when any out of pocket charges occur.
   - **Ave OoP per discharge**: This feature helps to define the mean of average out of pocket payment per discharge for each procedure and can be used as a baseline to detect anomaly. If a discharge pay a big difference amount from the mean for specific procedure, it would be noticable.
   - **Prop_payment_covered**: This feature displays the proportion of the total payment compared to covered charge.If an unsual percentage of payment for a procedure occurs, it would be noticable. If an unsual percentage of payment for a procedure occurs, it would be noticable.
   - **Medicare_Coverage_Ratio**: This feature represents the proportion covered by medicare for different procedures and can be used as a baseline to detect anomaly. If an unsual percentage of medicare charges for a procedure occurs, it would be noticable.
   - **Medicare_coverage_ratio_bystate**: This feature displays the average percentage covered by medicare in each state and can be used as a baseline to detect anomalies. If an unsual percentage of medicare happens, it would be noticable.
   - **Mean_Ave_Covered_bystate**: This feature will provide an estimation of the average covered charge in each state which can be used to compare between states and treated as a baseline when there is any unsual amount within a state.

### Autoencoder

Autoencoder techniques can perform non-linear transformations with their non-linear activation function and multiple layers.It is more efficient to train several layers with an autoencoder, rather than training one huge transformation with PCA.

Autoencoder is an unsupervised learning of artificial neural network that copies the input values (input layer) to the output values (output layer) throughout many hidden layers. The hidden layers in an autoencoder model must have fewers dimensions that those of the input or outpu layers. If the number of neurons is more than the input and output layers, the model will be given too much capacity to learn the data and simply return the output values as the input values, without extracting any essential information.

#### Model 1


```python
from pyod.models.auto_encoder import AutoEncoder
```


```python
#from pyod.models.auto_encoder import AutoEncoder
clf1 = AutoEncoder(hidden_neurons =[9, 2, 2, 9])
clf1.fit(X_train)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 8)                 72        
    _________________________________________________________________
    dropout (Dropout)            (None, 8)                 0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 8)                 72        
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 8)                 0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 9)                 81        
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 9)                 0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 20        
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 2)                 0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 2)                 6         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 2)                 0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 9)                 27        
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 9)                 0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 8)                 80        
    =================================================================
    Total params: 358
    Trainable params: 358
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/100
    3211/3211 [==============================] - 3s 822us/step - loss: 1.1948 - val_loss: 0.9717
    Epoch 2/100
    3211/3211 [==============================] - 3s 806us/step - loss: 1.0200 - val_loss: 0.9468
    Epoch 3/100
    3211/3211 [==============================] - 3s 849us/step - loss: 1.0093 - val_loss: 0.9427
    Epoch 4/100
    3211/3211 [==============================] - 3s 907us/step - loss: 1.0074 - val_loss: 0.9417
    Epoch 5/100
    3211/3211 [==============================] - 3s 893us/step - loss: 1.0069 - val_loss: 0.9414
    Epoch 6/100
    3211/3211 [==============================] - 3s 849us/step - loss: 1.0067 - val_loss: 0.9412
    Epoch 7/100
    3211/3211 [==============================] - 3s 849us/step - loss: 1.0067 - val_loss: 0.9412
    Epoch 8/100
    3211/3211 [==============================] - 3s 853us/step - loss: 1.0066 - val_loss: 0.9411
    Epoch 9/100
    3211/3211 [==============================] - 3s 860us/step - loss: 1.0066 - val_loss: 0.9411
    Epoch 10/100
    3211/3211 [==============================] - 3s 919us/step - loss: 1.0066 - val_loss: 0.9411
    Epoch 11/100
    3211/3211 [==============================] - 3s 887us/step - loss: 1.0066 - val_loss: 0.9411
    Epoch 12/100
    3211/3211 [==============================] - 3s 861us/step - loss: 1.0066 - val_loss: 0.9411
    Epoch 13/100
    3211/3211 [==============================] - 3s 861us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 14/100
    3211/3211 [==============================] - 3s 851us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 15/100
    3211/3211 [==============================] - 3s 852us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 16/100
    3211/3211 [==============================] - 3s 904us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 17/100
    3211/3211 [==============================] - 3s 887us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 18/100
    3211/3211 [==============================] - 3s 866us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 19/100
    3211/3211 [==============================] - 3s 873us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 20/100
    3211/3211 [==============================] - 3s 861us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 21/100
    3211/3211 [==============================] - 3s 918us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 22/100
    3211/3211 [==============================] - 3s 888us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 23/100
    3211/3211 [==============================] - 3s 866us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 24/100
    3211/3211 [==============================] - 3s 860us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 25/100
    3211/3211 [==============================] - 3s 850us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 26/100
    3211/3211 [==============================] - 3s 891us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 27/100
    3211/3211 [==============================] - 3s 928us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 28/100
    3211/3211 [==============================] - 3s 907us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 29/100
    3211/3211 [==============================] - 3s 861us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 30/100
    3211/3211 [==============================] - 3s 867us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 31/100
    3211/3211 [==============================] - 3s 876us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 32/100
    3211/3211 [==============================] - 3s 907us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 33/100
    3211/3211 [==============================] - 3s 909us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 34/100
    3211/3211 [==============================] - 3s 897us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 35/100
    3211/3211 [==============================] - 3s 860us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 36/100
    3211/3211 [==============================] - 3s 864us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 37/100
    3211/3211 [==============================] - 3s 862us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 38/100
    3211/3211 [==============================] - 3s 921us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 39/100
    3211/3211 [==============================] - 3s 892us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 40/100
    3211/3211 [==============================] - 3s 852us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 41/100
    3211/3211 [==============================] - 3s 864us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 42/100
    3211/3211 [==============================] - 3s 857us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 43/100
    3211/3211 [==============================] - 3s 851us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 44/100
    3211/3211 [==============================] - 3s 910us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 45/100
    3211/3211 [==============================] - 3s 917us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 46/100
    3211/3211 [==============================] - 3s 852us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 47/100
    3211/3211 [==============================] - 3s 969us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 48/100
    3211/3211 [==============================] - 3s 866us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 49/100
    3211/3211 [==============================] - 3s 915us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 50/100
    3211/3211 [==============================] - 3s 913us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 51/100
    3211/3211 [==============================] - 3s 883us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 52/100
    3211/3211 [==============================] - 3s 881us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 53/100
    3211/3211 [==============================] - 3s 874us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 54/100
    3211/3211 [==============================] - 3s 877us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 55/100
    3211/3211 [==============================] - 3s 927us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 56/100
    3211/3211 [==============================] - 3s 909us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 57/100
    3211/3211 [==============================] - 3s 873us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 58/100
    3211/3211 [==============================] - 3s 870us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 59/100
    3211/3211 [==============================] - 3s 858us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 60/100
    3211/3211 [==============================] - 3s 878us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 61/100
    3211/3211 [==============================] - 3s 902us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 62/100
    3211/3211 [==============================] - 3s 903us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 63/100
    3211/3211 [==============================] - 3s 966us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 64/100
    3211/3211 [==============================] - 3s 860us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 65/100
    3211/3211 [==============================] - 3s 863us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 66/100
    3211/3211 [==============================] - 3s 929us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 67/100
    3211/3211 [==============================] - 3s 905us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 68/100
    3211/3211 [==============================] - 3s 861us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 69/100
    3211/3211 [==============================] - 3s 868us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 70/100
    3211/3211 [==============================] - 3s 874us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 71/100
    3211/3211 [==============================] - 3s 893us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 72/100
    3211/3211 [==============================] - 3s 921us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 73/100
    3211/3211 [==============================] - 3s 905us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 74/100
    3211/3211 [==============================] - 3s 898us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 75/100
    3211/3211 [==============================] - 3s 881us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 76/100
    3211/3211 [==============================] - 3s 873us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 77/100
    3211/3211 [==============================] - 3s 928us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 78/100
    3211/3211 [==============================] - 3s 909us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 79/100
    3211/3211 [==============================] - 3s 862us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 80/100
    3211/3211 [==============================] - 3s 864us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 81/100
    3211/3211 [==============================] - 3s 872us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 82/100
    3211/3211 [==============================] - 3s 865us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 83/100
    3211/3211 [==============================] - 3s 922us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 84/100
    3211/3211 [==============================] - 3s 914us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 85/100
    3211/3211 [==============================] - 3s 867us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 86/100
    3211/3211 [==============================] - 3s 882us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 87/100
    3211/3211 [==============================] - 3s 879us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 88/100
    3211/3211 [==============================] - 3s 901us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 89/100
    3211/3211 [==============================] - 3s 907us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 90/100
    3211/3211 [==============================] - 3s 892us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 91/100
    3211/3211 [==============================] - 3s 873us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 92/100
    3211/3211 [==============================] - 3s 877us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 93/100
    3211/3211 [==============================] - 3s 868us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 94/100
    3211/3211 [==============================] - 3s 927us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 95/100
    3211/3211 [==============================] - 3s 898us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 96/100
    3211/3211 [==============================] - 3s 877us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 97/100
    3211/3211 [==============================] - 3s 870us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 98/100
    3211/3211 [==============================] - 3s 867us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 99/100
    3211/3211 [==============================] - 3s 868us/step - loss: 1.0066 - val_loss: 0.9410
    Epoch 100/100
    3211/3211 [==============================] - 3s 923us/step - loss: 1.0066 - val_loss: 0.9410





    AutoEncoder(batch_size=32, contamination=0.1, dropout_rate=0.2, epochs=100,
          hidden_activation='relu', hidden_neurons=[9, 2, 2, 9],
          l2_regularizer=0.1,
          loss=<function mean_squared_error at 0x000002632C152598>,
          optimizer='adam', output_activation='sigmoid', preprocessing=True,
          random_state=None, validation_size=0.1, verbose=1)




```python
# Get the outlier scores for the train data
y_train_scores_1 = clf1.decision_scores_  

# Predict the anomaly scores
y_test_scores_1 = clf1.decision_function(X_test)  # outlier scores
y_test_scores_1 = pd.Series(y_test_scores_1)

# Plot it!
import matplotlib.pyplot as plt
plt.figure(figsize=(30,20))
plt.hist(y_test_scores_1, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title("Histogram for Model Clf1 Anomaly Scores",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_134_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the score starts going down around 8 to 10, I choose 8 to be the cur point and >= 8 to be the outliers.


```python
#Get the Summary Statistics by Cluster
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['score'] = y_test_scores_1
df_test['cluster'] = np.where(df_test['score']<8, 'Normal', 'Outlier')
df_test['cluster'].value_counts()
```




    Normal     48378
    Outlier      542
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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.042881</td>
      <td>-0.044647</td>
      <td>-0.053812</td>
      <td>-0.055129</td>
      <td>0.000381</td>
      <td>0.010403</td>
      <td>-0.003704</td>
      <td>-0.006172</td>
      <td>2.225357</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>3.827513</td>
      <td>3.985085</td>
      <td>4.803138</td>
      <td>4.920745</td>
      <td>-0.034044</td>
      <td>-0.928540</td>
      <td>0.330630</td>
      <td>0.550895</td>
      <td>11.504062</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x2632e92c588>




![png](/assets/img/Autoencoder_Isolation/output_138_1.png)


**Note**: From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is dramatically large (6 times difference). Additionally, the average values of other features also show big differences between them.

#### Model 2:


```python
clf2 = AutoEncoder(hidden_neurons =[9, 10,2, 10, 9])
clf2.fit(X_train)
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_7 (Dense)              (None, 8)                 72        
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 8)                 0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 8)                 72        
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 8)                 0         
    _________________________________________________________________
    dense_9 (Dense)              (None, 9)                 81        
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 9)                 0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 10)                100       
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 10)                0         
    _________________________________________________________________
    dense_11 (Dense)             (None, 2)                 22        
    _________________________________________________________________
    dropout_10 (Dropout)         (None, 2)                 0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 10)                30        
    _________________________________________________________________
    dropout_11 (Dropout)         (None, 10)                0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 9)                 99        
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 9)                 0         
    _________________________________________________________________
    dense_14 (Dense)             (None, 8)                 80        
    =================================================================
    Total params: 556
    Trainable params: 556
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 1.1055 - val_loss: 1.0391
    Epoch 2/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 1.0107 - val_loss: 1.0236
    Epoch 3/100
    3211/3211 [==============================] - 3s 979us/step - loss: 1.0017 - val_loss: 1.0197
    Epoch 4/100
    3211/3211 [==============================] - 3s 992us/step - loss: 0.9993 - val_loss: 1.0186
    Epoch 5/100
    3211/3211 [==============================] - 3s 986us/step - loss: 0.9986 - val_loss: 1.0182
    Epoch 6/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9983 - val_loss: 1.0181
    Epoch 7/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9982 - val_loss: 1.0180
    Epoch 8/100
    3211/3211 [==============================] - 3s 993us/step - loss: 0.9981 - val_loss: 1.0180
    Epoch 9/100
    3211/3211 [==============================] - 3s 976us/step - loss: 0.9981 - val_loss: 1.0180
    Epoch 10/100
    3211/3211 [==============================] - 3s 979us/step - loss: 0.9981 - val_loss: 1.0180
    Epoch 11/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 12/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 13/100
    3211/3211 [==============================] - 3s 969us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 14/100
    3211/3211 [==============================] - 3s 978us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 15/100
    3211/3211 [==============================] - 3s 997us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 16/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 17/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 18/100
    3211/3211 [==============================] - 3s 991us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 19/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 20/100
    3211/3211 [==============================] - 3s 976us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 21/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 22/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 23/100
    3211/3211 [==============================] - 3s 983us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 24/100
    3211/3211 [==============================] - 3s 988us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 25/100
    3211/3211 [==============================] - 3s 982us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 26/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 27/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 28/100
    3211/3211 [==============================] - 3s 982us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 29/100
    3211/3211 [==============================] - 3s 990us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 30/100
    3211/3211 [==============================] - 3s 989us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 31/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 32/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 33/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 34/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 35/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 36/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 37/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 38/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 39/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 40/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 41/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 42/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 43/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 44/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 45/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 46/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 47/100
    3211/3211 [==============================] - 3s 987us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 48/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 49/100
    3211/3211 [==============================] - 3s 999us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 50/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 51/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 52/100
    3211/3211 [==============================] - 3s 991us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 53/100
    3211/3211 [==============================] - 3s 994us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 54/100
    3211/3211 [==============================] - 3s 993us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 55/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 56/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 57/100
    3211/3211 [==============================] - 3s 980us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 58/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 59/100
    3211/3211 [==============================] - 3s 992us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 60/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 61/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 62/100
    3211/3211 [==============================] - 3s 987us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 63/100
    3211/3211 [==============================] - 3s 996us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 64/100
    3211/3211 [==============================] - 3s 991us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 65/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 66/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 67/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 68/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 69/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 70/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 0.9980 - val_loss: 1.0180
    Epoch 71/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 72/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 73/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 74/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 75/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 76/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 77/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 78/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 79/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 80/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 81/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 82/100
    3211/3211 [==============================] - 3s 997us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 83/100
    3211/3211 [==============================] - 3s 995us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 84/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 85/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 86/100
    3211/3211 [==============================] - 3s 977us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 87/100
    3211/3211 [==============================] - 3s 985us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 88/100
    3211/3211 [==============================] - 3s 993us/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 89/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 90/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 91/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 92/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0179
    Epoch 93/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0180
    Epoch 94/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0180
    Epoch 95/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0180
    Epoch 96/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0180
    Epoch 97/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0180
    Epoch 98/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0180
    Epoch 99/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 0.9980 - val_loss: 1.0180
    Epoch 100/100
    3211/3211 [==============================] - 3s 1ms/step - loss: 0.9980 - val_loss: 1.0180





    AutoEncoder(batch_size=32, contamination=0.1, dropout_rate=0.2, epochs=100,
          hidden_activation='relu', hidden_neurons=[9, 10, 2, 10, 9],
          l2_regularizer=0.1,
          loss=<function mean_squared_error at 0x000002632C152598>,
          optimizer='adam', output_activation='sigmoid', preprocessing=True,
          random_state=None, validation_size=0.1, verbose=1)




```python
# Predict the anomaly scores
y_test_scores_2 = clf2.decision_function(X_test)  
y_test_scores_2 = pd.Series(y_test_scores_2)

# Plot the histogram
import matplotlib.pyplot as plt
plt.hist(y_test_scores_2, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.title("Histogram for Model Clf2 Anomaly Scores",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_142_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the frequency starts going down around 8 to 10, I choose 8 to be the cur point and >= 8 to be the outliers.


```python
#Get the Summary Statistics by Cluster
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['score'] = y_test_scores_2
df_test['cluster'] = np.where(df_test['score']<8, 'Normal', 'Outlier')
df_test['cluster'].value_counts()
```




    Normal     48378
    Outlier      542
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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.042881</td>
      <td>-0.044647</td>
      <td>-0.053812</td>
      <td>-0.055129</td>
      <td>0.000381</td>
      <td>0.010403</td>
      <td>-0.003704</td>
      <td>-0.006172</td>
      <td>2.225198</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>3.827513</td>
      <td>3.985085</td>
      <td>4.803138</td>
      <td>4.920745</td>
      <td>-0.034044</td>
      <td>-0.928540</td>
      <td>0.330630</td>
      <td>0.550895</td>
      <td>11.504619</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x2632e941cf8>




![png](/assets/img/Autoencoder_Isolation/output_146_1.png)


**Note**: The results in model 2 are similar like model 1. From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is dramatically large (6 times difference). Additionally, the average values of other features also show big differences between them.

#### Model 3


```python
# Step 1: Build the model
clf3 = AutoEncoder(hidden_neurons =[9, 15, 10, 2, 10,15,9])
clf3.fit(X_train)
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_15 (Dense)             (None, 8)                 72        
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 8)                 0         
    _________________________________________________________________
    dense_16 (Dense)             (None, 8)                 72        
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 8)                 0         
    _________________________________________________________________
    dense_17 (Dense)             (None, 9)                 81        
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 9)                 0         
    _________________________________________________________________
    dense_18 (Dense)             (None, 15)                150       
    _________________________________________________________________
    dropout_16 (Dropout)         (None, 15)                0         
    _________________________________________________________________
    dense_19 (Dense)             (None, 10)                160       
    _________________________________________________________________
    dropout_17 (Dropout)         (None, 10)                0         
    _________________________________________________________________
    dense_20 (Dense)             (None, 2)                 22        
    _________________________________________________________________
    dropout_18 (Dropout)         (None, 2)                 0         
    _________________________________________________________________
    dense_21 (Dense)             (None, 10)                30        
    _________________________________________________________________
    dropout_19 (Dropout)         (None, 10)                0         
    _________________________________________________________________
    dense_22 (Dense)             (None, 15)                165       
    _________________________________________________________________
    dropout_20 (Dropout)         (None, 15)                0         
    _________________________________________________________________
    dense_23 (Dense)             (None, 9)                 144       
    _________________________________________________________________
    dropout_21 (Dropout)         (None, 9)                 0         
    _________________________________________________________________
    dense_24 (Dense)             (None, 8)                 80        
    =================================================================
    Total params: 976
    Trainable params: 976
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.1892 - val_loss: 1.0020
    Epoch 2/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0165 - val_loss: 0.9833
    Epoch 3/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0061 - val_loss: 0.9791
    Epoch 4/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0037 - val_loss: 0.9780
    Epoch 5/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0030 - val_loss: 0.9776
    Epoch 6/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0028 - val_loss: 0.9774
    Epoch 7/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0027 - val_loss: 0.9774
    Epoch 8/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0026 - val_loss: 0.9773
    Epoch 9/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0026 - val_loss: 0.9773
    Epoch 10/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0026 - val_loss: 0.9773
    Epoch 11/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0026 - val_loss: 0.9773
    Epoch 12/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0026 - val_loss: 0.9773
    Epoch 13/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0026 - val_loss: 0.9773
    Epoch 14/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 15/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 16/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 17/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 18/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 19/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 20/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 21/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 22/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 23/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 24/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 25/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 26/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 27/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 28/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 29/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 30/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 31/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 32/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 33/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 34/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 35/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 36/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 37/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 38/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 39/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 40/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 41/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 42/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 43/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 44/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 45/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 46/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 47/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 48/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 49/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 50/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 51/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 52/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 53/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 54/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 55/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 56/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 57/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 58/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 59/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 60/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 61/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 62/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 63/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 64/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 65/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 66/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 67/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 68/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 69/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 70/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 71/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 72/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 73/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 74/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 75/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 76/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 77/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 78/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 79/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 80/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 81/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 82/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 83/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 84/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 85/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 86/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 87/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 88/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 89/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 90/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 91/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 92/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 93/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 94/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 95/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 96/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 97/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 98/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 99/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772
    Epoch 100/100
    3211/3211 [==============================] - 4s 1ms/step - loss: 1.0025 - val_loss: 0.9772





    AutoEncoder(batch_size=32, contamination=0.1, dropout_rate=0.2, epochs=100,
          hidden_activation='relu', hidden_neurons=[9, 15, 10, 2, 10, 15, 9],
          l2_regularizer=0.1,
          loss=<function mean_squared_error at 0x000002632C152598>,
          optimizer='adam', output_activation='sigmoid', preprocessing=True,
          random_state=None, validation_size=0.1, verbose=1)




```python
# Predict the anomaly scores
y_test_scores_3 = clf3.decision_function(X_test)  
y_test_scores_3 = pd.Series(y_test_scores_3)

# Step 2: Determine the cut point
import matplotlib.pyplot as plt
plt.hist(y_test_scores_3, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title("Histogram for Model Clf3 Anomaly Scores",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_150_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the frequency starts going down around 8 to 10, I choose 8 to be the cur point and >= 8 to be the outliers.


```python
#Get the Summary Statistics by Cluster
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['score'] = y_test_scores_3
df_test['cluster'] = np.where(df_test['score']<8, 'Normal', 'Outlier')
df_test['cluster'].value_counts()
```




    Normal     48378
    Outlier      542
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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.042881</td>
      <td>-0.044647</td>
      <td>-0.053812</td>
      <td>-0.055129</td>
      <td>0.000381</td>
      <td>0.010403</td>
      <td>-0.003704</td>
      <td>-0.006172</td>
      <td>2.225281</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>3.827513</td>
      <td>3.985085</td>
      <td>4.803138</td>
      <td>4.920745</td>
      <td>-0.034044</td>
      <td>-0.928540</td>
      <td>0.330630</td>
      <td>0.550895</td>
      <td>11.504405</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x26337999f60>




![png](/assets/img/Autoencoder_Isolation/output_154_1.png)


**Note**: The results in model 3 are similar to model 1 and model 2, including the number of outliers and the summary statistic table. From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is dramatically large (6 times difference). Additionally, the average values of other features also show big differences between them.

#### Aggregate to Achieve Model Stability

Although unsupervised techniques are powerful in detecting outliers, they are prone to overfitting and unstable results. The solution is to train multiple models then aggregate the scores


```python
%%capture --no-stdout --no-display output
#Stop warning from showing
# Put all the predictions in a data frame
from pyod.models.combination import aom, moa, average, maximization

# Put all the predictions in a data frame
train_scores = pd.DataFrame({'clf1': clf1.decision_scores_,
                             'clf2': clf2.decision_scores_,
                             'clf3': clf3.decision_scores_
                            })

test_scores  = pd.DataFrame({'clf1': clf1.decision_function(X_test),
                             'clf2': clf2.decision_function(X_test),
                             'clf3': clf3.decision_function(X_test)
                            })
```


```python
from pyod.models.combination import aom, moa, average, maximization
from pyod.utils.utility import standardizer
```


```python
# Although we did standardization before, it was for the variables.
# Now we do the standardization for the decision scores
train_scores_norm, test_scores_norm = standardizer(train_scores,test_scores)
```

##### Average Method


```python
# Combination by average
y_by_average = average(test_scores_norm)

import matplotlib.pyplot as plt
plt.hist(y_by_average, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title("Histogram for Average Method",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_161_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the fequency starts going down somewhere below 5, I choose 2 to be the cur point and >= 2 to be the outliers.


```python
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['y_by_average_score'] = y_by_average
df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<2,'Normal', 'Outlier')
df_test['y_by_average_cluster'].value_counts()
```




    Normal     47460
    Outlier     1460
    Name: y_by_average_cluster, dtype: int64




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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.087677</td>
      <td>-0.096259</td>
      <td>-0.091224</td>
      <td>-0.087021</td>
      <td>-0.003402</td>
      <td>0.014987</td>
      <td>-0.010536</td>
      <td>-0.013194</td>
      <td>-0.106002</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>2.850087</td>
      <td>3.129089</td>
      <td>2.965398</td>
      <td>2.828771</td>
      <td>0.110586</td>
      <td>-0.487177</td>
      <td>0.342501</td>
      <td>0.428905</td>
      <td>3.724213</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'y_by_average_cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x2633794fcf8>




![png](/assets/img/Autoencoder_Isolation/output_165_1.png)


**Note**: From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is high (3.8 compare to -0.12). Additionally, the average values of other features also show big differences between them. This indicates that the data points in the outlier cluster is anomolous and need to be investigated.

#### Maximum of Maximum Method


```python
# Combination by max
y_by_maximization = maximization(test_scores_norm)

import matplotlib.pyplot as plt
plt.hist(y_by_maximization, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title("Histogram for MOM Method",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_168_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the fequency starts going down somewhere below 5, I choose 3 to be the cur point and >= 3 to be the outliers.


```python
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['y_by_maximization_score'] = y_by_maximization
df_test['y_by_maximization_cluster'] = np.where(df_test['y_by_maximization_score']<3, 'Normal', 'Outlier')
df_test['y_by_maximization_cluster'].value_counts()
```




    Normal     48181
    Outlier      739
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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.054218</td>
      <td>-0.057347</td>
      <td>-0.064919</td>
      <td>-0.064573</td>
      <td>0.000395</td>
      <td>0.012656</td>
      <td>-0.004994</td>
      <td>-0.007799</td>
      <td>-0.068231</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>3.534876</td>
      <td>3.738857</td>
      <td>4.232565</td>
      <td>4.209992</td>
      <td>-0.025736</td>
      <td>-0.825148</td>
      <td>0.325596</td>
      <td>0.508460</td>
      <td>5.010593</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'y_by_maximization_cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x263460d96a0>




![png](/assets/img/Autoencoder_Isolation/output_172_1.png)


**Note**: The result of MOM is similar to the average method. From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is high (3.8 compare to -0.12). Additionally, the average values of other features also show big differences between them. This indicates that the data points in the outlier cluster is anomolous and need to be investigated.

### Isolation Forest

Isolation algorithm is useful for both supervised and unsupervised learning. Isolation Forest is an unsupervised learning that calculates an anomaly score and seperates into binary based on an anomaly threshhold. The way that the algorithm constructs the separation is by first creating isolation trees or random decision trees. Then, the score is calculated as the path length to isolate the observation. Isolation forest randomly choose split points among ranomly choosen variables which help to reduce the chance of overfitting.

Isolation forest is an advanced outlier dection that delects anomalies based on the concept of isolation instead of distance or density measurment. It is different from other methods like KNN or PCA in anomalies dectection and is knowns as an effective method at reducing frauds.

#### Model 1: {Samples: Auto}


```python
from pyod.models.iforest import IForest
clf1 = IForest(behaviour="new", bootstrap=True,n_jobs=-1,)
clf1.fit(X_train)
```




    IForest(behaviour='new', bootstrap=True, contamination=0.1, max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=-1, random_state=None,
        verbose=0)




```python
# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
# We apply the model to the test data X_test to get the outlier scores.
y_test_scores_1 = clf1.decision_function(X_test)  # outlier scores
y_test_scores_1 = pd.Series(y_test_scores_1)
```


```python
import matplotlib.pyplot as plt
plt.hist(y_test_scores_1, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title("Histogram with 'auto' bins (Model 1)",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_178_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the fequency starts going down around 0.1, I choose 0.1 to be the cur point and >= 0.1 to be the outliers.


```python
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['score'] = y_test_scores_1
df_test['cluster'] = np.where(df_test['score']<0.15, 'Normal', 'Outlier')
df_test['cluster'].value_counts()
```




    Normal     48493
    Outlier      427
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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.026142</td>
      <td>-0.031459</td>
      <td>-0.054863</td>
      <td>-0.053461</td>
      <td>-0.000656</td>
      <td>0.014086</td>
      <td>-0.002030</td>
      <td>-0.003021</td>
      <td>-0.085794</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>2.968895</td>
      <td>3.572701</td>
      <td>6.230646</td>
      <td>6.071338</td>
      <td>0.074541</td>
      <td>-1.599683</td>
      <td>0.230486</td>
      <td>0.343133</td>
      <td>0.188436</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x2633971b588>




![png](/assets/img/Autoencoder_Isolation/output_182_1.png)


**Note**: From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is high. Additionally, the average values of other features also show big differences between them. This indicates that the data points in the outlier cluster is anomolous and need to be investigated.

#### Model 2: Samples: 50


```python
clf2 = IForest(max_samples=50, bootstrap=True,n_jobs=-1,)
clf2.fit(X_train)
```




    IForest(behaviour='old', bootstrap=True, contamination=0.1, max_features=1.0,
        max_samples=50, n_estimators=100, n_jobs=-1, random_state=None,
        verbose=0)




```python
# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
# We apply the model to the test data X_test to get the outlier scores.
y_test_scores_2 = clf2.decision_function(X_test)  # outlier scores
y_test_scores_2 = pd.Series(y_test_scores_2)
y_test_scores_2.head()
```




    0   -0.120533
    1   -0.059649
    2   -0.077699
    3   -0.090892
    4   -0.146159
    dtype: float64




```python
plt.hist(y_test_scores_2, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title("Histogram with 'auto' bins (Model 2)",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_187_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the fequency starts going down around 0.1, I choose 0.1 to be the cur point and >= 0.1 to be the outliers.


```python
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['score'] = y_test_scores_2
df_test['cluster'] = np.where(df_test['score']<0.1, 'Normal', 'Outlier')
df_test['cluster'].value_counts()
```




    Normal     48393
    Outlier      527
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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.036478</td>
      <td>-0.039582</td>
      <td>-0.058009</td>
      <td>-0.053923</td>
      <td>0.001716</td>
      <td>0.010206</td>
      <td>-0.004756</td>
      <td>-0.006885</td>
      <td>-0.086998</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>3.349670</td>
      <td>3.634723</td>
      <td>5.326820</td>
      <td>4.951623</td>
      <td>-0.157540</td>
      <td>-0.937174</td>
      <td>0.436728</td>
      <td>0.632238</td>
      <td>0.122749</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x26351584588>




![png](/assets/img/Autoencoder_Isolation/output_191_1.png)


**Note**: The results in model 2 are similar to model 1. From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is high. Additionally, the average values of other features also show big differences between them. This indicates that the data points in the outlier cluster is anomolous and need to be investigated.

#### Model 3: Sample: 100


```python
clf3 = IForest(behaviour="new", max_samples=100, bootstrap=True,n_jobs=-1,)
clf3.fit(X_train)
```




    IForest(behaviour='new', bootstrap=True, contamination=0.1, max_features=1.0,
        max_samples=100, n_estimators=100, n_jobs=-1, random_state=None,
        verbose=0)




```python
# clf.decision_function: Predict raw anomaly score of X using the fitted detector.
# We apply the model to the test data X_test to get the outlier scores.
y_test_scores_3 = clf3.decision_function(X_test)  # outlier scores
y_test_scores_3 = pd.Series(y_test_scores_3)
y_test_scores_3.head()
```




    0   -0.138078
    1   -0.079561
    2   -0.091734
    3   -0.072862
    4   -0.152575
    dtype: float64




```python
plt.hist(y_test_scores_3, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title("Histogram with 'auto' bins (Model 3)",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_196_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the fequency starts going down around 0.1, I choose 0.1 to be the cur point and >= 0.1 to be the outliers.


```python
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['score'] = y_test_scores_3
df_test['cluster'] = np.where(df_test['score']<0.1, 'Normal', 'Outlier')
df_test['cluster'].value_counts()
```




    Normal     48056
    Outlier      864
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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.052835</td>
      <td>-0.059139</td>
      <td>-0.077434</td>
      <td>-0.07491</td>
      <td>-0.000069</td>
      <td>0.016678</td>
      <td>-0.007143</td>
      <td>-0.009232</td>
      <td>-0.097827</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>2.938695</td>
      <td>3.289350</td>
      <td>4.306935</td>
      <td>4.16652</td>
      <td>0.003846</td>
      <td>-0.927660</td>
      <td>0.397287</td>
      <td>0.513484</td>
      <td>0.131902</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x2635c54e860>




![png](/assets/img/Autoencoder_Isolation/output_200_1.png)


**Note**: The results in model 3 are similar to model 1 and model 2. From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is high. Additionally, the average values of other features also show big differences between them. This indicates that the data points in the outlier cluster is anomolous and need to be investigated.

#### Aggregate to Achieve Model Stability


```python
# The predictions of the training data can be obtained by clf.decision_scores_.
# It is already generated during the model building process.
train_scores = pd.DataFrame({'clf1': clf1.decision_scores_,
                             'clf2': clf2.decision_scores_,
                             'clf3': clf3.decision_scores_
                            })

# The predictions of the test data need to be predicted using clf.decision_function(X_test)
test_scores  = pd.DataFrame({'clf1': clf1.decision_function(X_test),
                             'clf2': clf2.decision_function(X_test),
                             'clf3': clf3.decision_function(X_test)
                            })
```


```python
# Although we did standardization before, it was for the variables.
# Now we do the standardization for the decision scores
train_scores_norm, test_scores_norm = standardizer(train_scores,test_scores)
```

#### Average Method


```python
# Combination by average
y_by_average = average(test_scores_norm)

import matplotlib.pyplot as plt
plt.hist(y_by_average, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title("Histogram for Average Method",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_206_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the fequency starts going down around 3, I choose 3 to be the cur point and >= 3 to be the outliers.


```python
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['y_by_average_score'] = y_by_average
df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<3,'Normal', 'Outlier')
df_test['y_by_average_cluster'].value_counts()
```




    Normal     48116
    Outlier      804
    Name: y_by_average_cluster, dtype: int64




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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.049303</td>
      <td>-0.055297</td>
      <td>-0.075570</td>
      <td>-0.072741</td>
      <td>0.000168</td>
      <td>0.017258</td>
      <td>-0.006548</td>
      <td>-0.008200</td>
      <td>-0.043871</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>2.950552</td>
      <td>3.309267</td>
      <td>4.522553</td>
      <td>4.353235</td>
      <td>-0.010076</td>
      <td>-1.032837</td>
      <td>0.391848</td>
      <td>0.490754</td>
      <td>3.564904</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'y_by_average_cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x26337de1a58>




![png](/assets/img/Autoencoder_Isolation/output_210_1.png)


**Note**: From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is high. Additionally, the average values of other features also show big differences between them. This indicates that the data points in the outlier cluster is anomolous and need to be investigated.

#### Maximum of maximum method


```python
# Combination by max
y_by_maximization = maximization(test_scores_norm)

import matplotlib.pyplot as plt
plt.hist(y_by_maximization, bins='auto', lw=0,  color='mediumvioletred')  
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title("Histogram for MOM Method",fontsize = 20)
plt.show()
```


![png](/assets/img/Autoencoder_Isolation/output_213_0.png)


By using histogram to count the frequency by the anomaly score, it indicates that the higher the score is, the lower the frequency is, thus providing the group of outliers. As the chart shows that the fequency starts going down around 3, I choose 3 to be the cur point and >= 3 to be the outliers.


```python
df_test = X_test.copy()
old_col = df_test.columns[:9]
feature_name = features.columns
df_test.rename(columns=dict(zip(old_col, feature_name)), inplace=True)
df_test['y_by_maximization_score'] = y_by_maximization
df_test['y_by_maximization_cluster'] = np.where(df_test['y_by_maximization_score']<3,'Normal', 'Outlier')
df_test['y_by_maximization_cluster'].value_counts()
```




    Normal     47889
    Outlier     1031
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
      <th>Ave_Covered_Charges</th>
      <th>Ave_Total_Payments</th>
      <th>Ave_OOP</th>
      <th>Ave OoP per discharge</th>
      <th>Prop_payment_covered</th>
      <th>Medicare_Coverage_Ratio</th>
      <th>Medicare_coverage_ratio_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Normal</th>
      <td>-0.059620</td>
      <td>-0.068439</td>
      <td>-0.085933</td>
      <td>-0.081852</td>
      <td>-0.001153</td>
      <td>0.021232</td>
      <td>-0.007112</td>
      <td>-0.008843</td>
      <td>0.070621</td>
    </tr>
    <tr>
      <th>Outlier</th>
      <td>2.769287</td>
      <td>3.178911</td>
      <td>3.991489</td>
      <td>3.801928</td>
      <td>0.053541</td>
      <td>-0.986216</td>
      <td>0.330359</td>
      <td>0.410750</td>
      <td>3.771988</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df_test.iloc[:,[1,2,3,4,9]], hue = 'y_by_maximization_cluster', palette='husl')
```




    <seaborn.axisgrid.PairGrid at 0x26357c8e4e0>




![png](/assets/img/Autoencoder_Isolation/output_217_1.png)


**Note**: The results of MOM method is similar to the results in the average method. From the summary statistic table above, the difference of the average anomaly score between the outlier cluster and the normal cluater is high. Additionally, the average values of other features also show big differences between them. This indicates that the data points in the outlier cluster is anomolous and need to be investigated.

### Conclusion:

After trying both autoencoder and isolation techniques, I think combining different approaches/method and comparing results increases prediction confidence, and reduce bias. From that, I believe it creates a higher level of confidence, especially since each parameter selection contains bias. As the resulls come out similar within each algorithm, my takeaways would be investigating the accounts that are defined as outliers in the techniques utilized.  


```python

```
