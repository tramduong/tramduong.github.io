
# Feature Engineering and EDA For Hospital Charge Data

Tram Duong
<br>October 5, 2020


```python
import pandas as pd 
import seaborn as sns 
import numpy as np 
import plotly
#import vincent
import matplotlib.pyplot as plt
import scipy 
from scipy import stats
sns.set(rc={'figure.figsize':(10,15)})
```


```python
# read data 
payment_data = pd.read_csv("C:/github/Data-Science-Portfolio/Feature Engineering & Modeling/Data/inpatientCharges.csv")
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



![png](/assets/img/hospitalfe/output_22_1.png)


### Visualization by regions


```python
fig = plt.figure(figsize=(16,10))
sns.pairplot(payment_data[['Region','Average Total Payments',
                            'Total Discharges','Average Medicare Payments','Average Covered Charges']], hue= 'Region',height = 4)
```




    <seaborn.axisgrid.PairGrid at 0x1cefb7fd978>




    <Figure size 1152x720 with 0 Axes>



![png](/assets/img/hospitalfe/output_24_2.png)


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




![png](/assets/img/hospitalfe/output_27_1.png)


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




![png](/assets/img/hospitalfe/output_30_1.png)


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


![png](/assets/img/hospitalfe/output_38_0.png)


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


![png](/assets/img/hospitalfe/output_42_0.png)


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



![png](/assets/img/hospitalfe/output_47_2.png)


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



![png](/assets/img/hospitalfe/output_51_2.png)


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



![png](/assets/img/hospitalfe/output_57_2.png)


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



![png](/assets/img/hospitalfe/output_62_2.png)


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



![png](/assets/img/hospitalfe/output_67_2.png)


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




![png](/assets/img/hospitalfe/output_71_1.png)


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




![png](/assets/img/hospitalfe/output_85_1.png)



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




![png](/assets/img/hospitalfe/output_93_1.png)



    <Figure size 1080x1080 with 0 Axes>


**Conclusion**: This feature helps to identify the difference in payment for the same procedure. If the difference is high for a procedure, it means that the payment varies largely between different states or different providers. Thus, we need to investigate further for these procedures. 
