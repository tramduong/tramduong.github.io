
# K-Means with Elbows and Silhouette Score

Tram Duong
<br>October 12, 2020


### Part 1: EDA and FE
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
payment_data = pd.read_csv("/data/inpatientCharges.csv")
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
    DRG Definition                          163065 non-null object
    Provider Id                             163065 non-null int64
    Provider Name                           163065 non-null object
    Provider Street Address                 163065 non-null object
    Provider City                           163065 non-null object
    Provider State                          163065 non-null object
    Provider Zip Code                       163065 non-null int64
    Hospital Referral Region Description    163065 non-null object
     Total Discharges                       163065 non-null int64
     Average Covered Charges                163065 non-null object
     Average Total Payments                 163065 non-null object
    Average Medicare Payments               163065 non-null object
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



![png](/assets/img/Kmeans/output_23_1.png)


### Visualization by regions


```python
fig = plt.figure(figsize=(16,10))
sns.pairplot(payment_data[['Region','Average Total Payments',
                            'Total Discharges','Average Medicare Payments','Average Covered Charges']], hue= 'Region',height = 4)
```




    <seaborn.axisgrid.PairGrid at 0x2a0b7b13080>




    <Figure size 1152x720 with 0 Axes>



![png](/assets/img/Kmeans/output_25_2.png)


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




    <matplotlib.axes._subplots.AxesSubplot at 0x2a0b9ec0a90>




![png](/assets/img/Kmeans/output_28_1.png)


The average medicare payments and average covered charges are highly correlated to each other. Furthermore, the average medicare payments are also highly correlated with the average total payments.   

### Common Procedures


```python
fig = plt.figure(figsize=(16,8))
common_drg = payment_data.groupby('DRG Definition').count()['Provider Id'].sort_values(ascending=False)
top_10 = common_drg[:10]
sns.countplot(y='DRG Definition', data=payment_data, palette="Greens_d",
              order=pd.value_counts(payment_data['DRG Definition']).iloc[:10].index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2a0b9ec73c8>




![png](/assets/img/Kmeans/output_31_1.png)


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


![png](/assets/img/Kmeans/output_39_0.png)


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


![png](/assets/img/Kmeans/output_43_0.png)


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



![png](/assets/img/Kmeans/output_48_2.png)


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



![png](/assets/img/Kmeans/output_52_2.png)


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



![png](/assets/img/Kmeans/output_58_2.png)


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



![png](/assets/img/Kmeans/output_63_2.png)


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



![png](/assets/img/Kmeans/output_68_2.png)


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




![png](/assets/img/Kmeans/output_72_1.png)


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




![png](/assets/img/Kmeans/output_86_1.png)



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




![png](/assets/img/Kmeans/output_94_1.png)



    <Figure size 1080x1080 with 0 Axes>


**Conclusion**: This feature helps to identify the difference in payment for the same procedure. If the difference is high for a procedure, it means that the payment varies largely between different states or different providers. Thus, we need to investigate further for these procedures.

### Part 2: Clustering
In this part, I use K-means clustering algorithm to explore the dataset, following the steps below:

    - Drop irrelevant variables
    - Standardization of Numerical / Float variables
    - k-means Clustering
    - Visualizing the results
    - Explain the anomalies

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
           'Differences', 'Medicare Coverage Ratio_y',
           'Mean_Total_Discharge_bystate', 'Mean_Ave_Covered_bystate',
           'Mean_Ave_Total_Payment_bystate', 'Mean_Ave_Medicare_bystate'],
          dtype='object')




```python
features.isnull().sum()
```




    Total_Discharges                       0
    Ave_Covered_Charges                    0
    Ave_Total_Payments                     0
    Aver_Medicare_Payments                 0
    Ave_OOP                                0
    Ave OoP per discharge                  0
    Prop_payment_covered                   0
    Medicare_Coverage_Ratio                0
    Zscore_ave_total_payment               0
    Zscore_ave_medicare_payment            0
    Differences                       124528
    Medicare Coverage Ratio_y              0
    Mean_Total_Discharge_bystate           0
    Mean_Ave_Covered_bystate               0
    Mean_Ave_Total_Payment_bystate         0
    Mean_Ave_Medicare_bystate              0
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




    <matplotlib.axes._subplots.AxesSubplot at 0x2a0c3f655f8>




![png](/assets/img/Kmeans/output_116_1.png)


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
           'Medicare_Coverage_Ratio', 'Medicare Coverage Ratio_y',
           'Mean_Total_Discharge_bystate', 'Mean_Ave_Covered_bystate'],
          dtype='object')



#### Standardize all features


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(features)
```


```python
X = pd.DataFrame(data = features, columns = features.columns)
X.head()

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
      <th>Medicare Coverage Ratio_y</th>
      <th>Mean_Total_Discharge_bystate</th>
      <th>Mean_Ave_Covered_bystate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>91</td>
      <td>32963.07</td>
      <td>5777.24</td>
      <td>1013.51</td>
      <td>11.137473</td>
      <td>17.53</td>
      <td>0.824568</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14</td>
      <td>15131.85</td>
      <td>5787.57</td>
      <td>810.86</td>
      <td>57.918571</td>
      <td>38.25</td>
      <td>0.859896</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24</td>
      <td>37560.37</td>
      <td>5434.95</td>
      <td>981.16</td>
      <td>40.881667</td>
      <td>14.47</td>
      <td>0.819472</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>13998.28</td>
      <td>5417.56</td>
      <td>1288.40</td>
      <td>51.536000</td>
      <td>38.70</td>
      <td>0.762181</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>31633.27</td>
      <td>5658.33</td>
      <td>806.89</td>
      <td>44.827222</td>
      <td>17.89</td>
      <td>0.857398</td>
      <td>0.816622</td>
      <td>39.258322</td>
      <td>31316.462074</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.cluster import KMeans

k_range = range(2,11)
sse = []
for k in k_range:
  clusterer = KMeans(n_clusters=k, random_state=0)
  cluster_labels = clusterer.fit(X)
  sse.append(clusterer.inertia_)

fig = plt.figure(figsize=(20,10))
plt.plot(k_range, sse, "-o", label = "SSE at Cluster Size", )
plt.xlabel("Clusters")
plt.ylabel("SSE")
plt.title("Elbow for KMeans Clustering")
plt.legend()
plt.show()
```


![png](/assets/img/Kmeans/output_124_0.png)


From the elbow chart above, the ideal number of clusters is somewhat hard to determine. Both 6 and 8 seems like good places to start.

*Note* - the SSE values are scaled, i.e. 0.8 on the chart is equal to 0.8\*1e14


```python
# generating scatter plot for predictions
km5 = KMeans(n_clusters = 5, init ='k-means++', max_iter=300, n_init=10,random_state=0)
y_mean = km5.fit_predict(X)

plt.figure(figsize=(20, 10))  
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_mean, cmap='Paired', s=50)
plt.title('Customer Segmentation with 5 Clusters')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])

centers = km5.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
```




    <matplotlib.collections.PathCollection at 0x2a0c8326518>




![png](/assets/img/Kmeans/output_126_1.png)



```python
# generating scatter plot for predictions
km1 = KMeans(n_clusters = 6, init ='k-means++', max_iter=300, n_init=10,random_state=0)
y_mean = km1.fit_predict(X)

plt.figure(figsize=(20, 10))  
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_mean, cmap='Paired', s=50)
plt.title('Customer Segmentation with 6 Clusters')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])

centers = km1.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
```




    <matplotlib.collections.PathCollection at 0x2a0c84df320>




![png](/assets/img/Kmeans/output_127_1.png)



```python
# generating scatter plot for predictions
km2 = KMeans(n_clusters = 7, init ='k-means++', max_iter=300, n_init=10,random_state=0)
y_mean = km2.fit_predict(X)

plt.figure(figsize=(20, 10))  
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_mean, cmap='Paired', s=50)
plt.title('Customer Segmentation with 7 Clusters')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])

centers = km2.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
```




    <matplotlib.collections.PathCollection at 0x2a0c9bd8470>




![png](/assets/img/Kmeans/output_128_1.png)



```python
# generating scatter plot for predictions
km3 = KMeans(n_clusters = 8, init ='k-means++', max_iter=300, n_init=10,random_state=0)
y_mean = km3.fit_predict(X)

plt.figure(figsize=(20, 10))  
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_mean, cmap='Paired', s=50)
plt.title('Customer Segmentation with 8 Clusters')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])

centers = km3.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
```




    <matplotlib.collections.PathCollection at 0x2a0cde1fef0>




![png](/assets/img/Kmeans/output_129_1.png)



```python
# generating scatter plot for predictions
km3 = KMeans(n_clusters = 9, init ='k-means++', max_iter=300, n_init=10,random_state=0)
y_mean = km3.fit_predict(X)

plt.figure(figsize=(20, 10))  
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_mean, cmap='Paired', s=50)
plt.title('Customer Segmentation with 9 Clusters')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])

centers = km3.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
```




    <matplotlib.collections.PathCollection at 0x2a0ce253c88>




![png](/assets/img/Kmeans/output_130_1.png)



```python
# generating scatter plot for predictions
km3 = KMeans(n_clusters = 10, init ='k-means++', max_iter=300, n_init=10,random_state=0)
y_mean = km3.fit_predict(X)

plt.figure(figsize=(20, 10))  
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_mean, cmap='Paired', s=50)
plt.title('Customer Segmentation with 10 Clusters')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])

centers = km3.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
```




    <matplotlib.collections.PathCollection at 0x2a0ce40ca20>




![png](/assets/img/Kmeans/output_131_1.png)


#### Indecipherable Visualizations

As the model takes 11 features and creates results in an 11-dimensional space, it does not translate well when attempting to display it through a 2-d graph. Therefore, I will use silhouette scores to help determine the best number of clusters.

##### *Note*
The ranges for the silhouette scores were spread across 3 cells:
    - first range [2:4]
    - second range [5:10]
    - third range [11:12]
The code takes some time to run for every K number and I wanted to explore the data further without spending time rerunning code.


```python
# Choose range off elbows in above chart and optimized by observing generated charts
range_n_clusters = [2,3,4]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # No silhouette coefficient was smaller than -0.3, so limited graph to this range for easier viewing
    ax1.set_xlim([-0.4, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(X)

    # Determing the silouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # iterate through number of clusters
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate values
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14)

plt.show()
```

    For n_clusters = 2 The average silhouette_score is : 0.6754436345864707
    For n_clusters = 3 The average silhouette_score is : 0.5604495050464645
    For n_clusters = 4 The average silhouette_score is : 0.4821517741483471



![png](/assets/img/Kmeans/output_134_1.png)



![png](/assets/img/Kmeans/output_134_2.png)



![png](/assets/img/Kmeans/output_134_3.png)



```python
# Adapted from SKLearn silhouette docs

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# Choose range off elbows in above chart and optimized by observing generated charts
range_n_clusters = [5, 6, 7, 8, 9, 10]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # No silhouette coefficient was smaller than -0.3, so limited graph to this range for easier viewing
    ax1.set_xlim([-0.4, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(X)

    # Determing the silouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # iterate through number of clusters
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate values
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14)

plt.show()
```

    For n_clusters = 5 The average silhouette_score is : 0.5069573045577488
    For n_clusters = 6 The average silhouette_score is : 0.46466229986346946
    For n_clusters = 7 The average silhouette_score is : 0.3977982689708172
    For n_clusters = 8 The average silhouette_score is : 0.37274497497270376
    For n_clusters = 9 The average silhouette_score is : 0.364913379892289
    For n_clusters = 10 The average silhouette_score is : 0.35349254345463216



![png](/assets/img/Kmeans/output_135_1.png)



![png](/assets/img/Kmeans/output_135_2.png)



![png](/assets/img/Kmeans/output_135_3.png)



![png](/assets/img/Kmeans/output_135_4.png)



![png](/assets/img/Kmeans/output_135_5.png)



![png](/assets/img/Kmeans/output_135_6.png)



```python
# Choose range off elbows in above chart and optimized by observing generated charts
range_n_clusters = [11,12]

for n_clusters in range_n_clusters:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # No silhouette coefficient was smaller than -0.3, so limited graph to this range for easier viewing
    ax1.set_xlim([-0.4, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(X)

    # Determing the silouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # iterate through number of clusters
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate values
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14)

plt.show()
```

    For n_clusters = 11 The average silhouette_score is : 0.3746648376888015
    For n_clusters = 12 The average silhouette_score is : 0.3746603450407584



![png](/assets/img/Kmeans/output_136_1.png)



![png](/assets/img/Kmeans/output_136_2.png)


Overall, there is not a sample that overwhelmingly stands out from the others. When using silhouette scores, it is important to note the height represented for each cluster as they should have similar sizes and that they should at least the average silhouette score.
<br><br>
From the clusters that range from 2 to 12, all of the clusters at least meet the average silhoutte score, but size is a less clear metric. In all of the cluster sizes, there is a disproportionately large cluster in either cluster 0 or 1. As this characteristic is noted in all the clusters, it may mean that the data is just weighted more heavily to create it. However, as the data set is not neatly balanced, it makes sense that there may be a major change in size from one cluster to another.
<br><br>
Instead of relying on one method solely, using a hybrid approach and choosing off the silhouettes and the k-means elbows will provide a better indication of what cluster to use. Therefore, 6 clusters will be used moving forward.


### Analysis for k=6 Clusters


```python
km1 = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
km1.fit(X)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=6, n_init=10, n_jobs=None, precompute_distances='auto',
           random_state=0, tol=0.0001, verbose=0)




```python
cluster = km1.labels_
features['cluster'] = km1.labels_
payment_data['cluster'] = km1.labels_
```


```python
cluster_eda = payment_data.groupby('cluster').mean().reset_index()
```


```python
cluster_eda['Proportion_by_cluster'] = (payment_data['cluster'].value_counts() / payment_data['cluster'].value_counts().sum()) * 100
```


```python
cluster_eda
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
      <th>cluster</th>
      <th>Total Discharges</th>
      <th>Average Covered Charges</th>
      <th>Average Total Payments</th>
      <th>Average Medicare Payments</th>
      <th>Ave Out of Pocket Payment</th>
      <th>Ave OoP per discharge</th>
      <th>Percent of Payment Covered</th>
      <th>Medicare Coverage Ratio</th>
      <th>Z-score Average Total Payments</th>
      <th>Z-score Average Medicare Payments</th>
      <th>Proportion_by_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>41.769673</td>
      <td>17435.133711</td>
      <td>6418.865009</td>
      <td>5395.872294</td>
      <td>1022.992715</td>
      <td>42.062905</td>
      <td>40.278640</td>
      <td>0.822227</td>
      <td>-0.429064</td>
      <td>-0.423920</td>
      <td>56.365866</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>31.422819</td>
      <td>151401.620988</td>
      <td>31154.406327</td>
      <td>28715.057334</td>
      <td>2439.348993</td>
      <td>113.093816</td>
      <td>20.853819</td>
      <td>0.920858</td>
      <td>2.798173</td>
      <td>2.766361</td>
      <td>2.649864</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>40.188552</td>
      <td>40706.009778</td>
      <td>8065.669566</td>
      <td>7052.031639</td>
      <td>1013.637927</td>
      <td>42.735734</td>
      <td>21.259141</td>
      <td>0.858268</td>
      <td>-0.214206</td>
      <td>-0.197342</td>
      <td>8.013982</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>41.350454</td>
      <td>84755.816998</td>
      <td>18753.381037</td>
      <td>16986.323877</td>
      <td>1767.057161</td>
      <td>73.137638</td>
      <td>22.391985</td>
      <td>0.899876</td>
      <td>1.180216</td>
      <td>1.161762</td>
      <td>9.109864</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>48.309773</td>
      <td>42256.437151</td>
      <td>11441.088421</td>
      <td>10086.390332</td>
      <td>1354.698089</td>
      <td>52.409169</td>
      <td>27.414836</td>
      <td>0.869899</td>
      <td>0.226184</td>
      <td>0.217787</td>
      <td>23.374115</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>24.706179</td>
      <td>294909.832560</td>
      <td>48288.973745</td>
      <td>45643.175259</td>
      <td>2645.798487</td>
      <td>134.121650</td>
      <td>16.731160</td>
      <td>0.945507</td>
      <td>5.033714</td>
      <td>5.082285</td>
      <td>0.486309</td>
    </tr>
  </tbody>
</table>
</div>



#### Anomaly Detection


```python
fig, axarr = plt.subplots(4, 3, figsize=(15, 15))

cluster_eda['Total Discharges'].plot.bar(
    ax=axarr[0][0], fontsize=12, color='mediumvioletred'
)
axarr[0][0].set_title("Total Discharges", fontsize=14)

cluster_eda['Average Covered Charges'].plot.bar(
    ax=axarr[0][1], fontsize=12, color='mediumvioletred'
)
axarr[0][1].set_title("Average Covered Charges", fontsize=14)

cluster_eda['Average Total Payments'].plot.bar(
    ax=axarr[0][2], fontsize=12, color='mediumvioletred'
)
axarr[0][2].set_title("Average Total Payments", fontsize=14)

cluster_eda['Average Medicare Payments'].plot.bar(
    ax=axarr[1][0], fontsize=12, color='mediumvioletred'
)
axarr[1][0].set_title("Average Medicare Payments", fontsize=14)

cluster_eda['Ave Out of Pocket Payment'].plot.bar(
    ax=axarr[1][1], fontsize=12, color='mediumvioletred'
)
axarr[1][1].set_title("Ave Out of Pocket Payment", fontsize=14)

cluster_eda['Ave OoP per discharge'].plot.bar(
    ax=axarr[1][2], fontsize=12, color='mediumvioletred'
)
axarr[1][2].set_title("Ave OoP per discharge", fontsize=14)

cluster_eda['Percent of Payment Covered'].plot.bar(
    ax=axarr[2][0], fontsize=12, color='mediumvioletred'
)
axarr[2][0].set_title("Percent of Payment Covered", fontsize=14)

cluster_eda['Medicare Coverage Ratio'].plot.bar(
    ax=axarr[2][1], fontsize=12, color='mediumvioletred'
)
axarr[2][1].set_title("Medicare Coverage Ratio", fontsize=14)
plt.subplots_adjust(hspace=.3)

cluster_eda['Z-score Average Total Payments'].plot.bar(
    ax=axarr[2][2], fontsize=12, color='mediumvioletred'
)
axarr[2][2].set_title("Z-score Average Total Payments", fontsize=14)

cluster_eda['Z-score Average Medicare Payments'].plot.bar(
    ax=axarr[3][0], fontsize=12, color='mediumvioletred'
)
axarr[3][0].set_title("Z-score Average Medicare Payments", fontsize=14)

cluster_eda['Proportion_by_cluster'].plot.bar(
    ax=axarr[3][1], fontsize=12, color='mediumvioletred'
)
axarr[3][1].set_title("Proportion_by_cluster", fontsize=14)

plt.subplots_adjust(hspace=.4)

import seaborn as sns
sns.despine()
```


![png](/assets/img/Kmeans/output_145_0.png)


From the 11 features used for clustering, 6 different clusters were created. The charts above detail information regarding to mean of all numeric features from the dataframe for each cluster. The key takeaways are as followings:

**Total Discharges**: Cluster 5 has a low mean value here compared to the others which should be a concern to detect anomalies. However, this cluster contains about 0.5% of the data which may explain its low result for total discharge. Similalry, but not to the same extent, cluster 1 is also affected through its low proportion of data.

**Average Covered Charges**: Cluster 5 has an extremely high value for the mean of average covered charge, totaling of approximately \\$300k which is concerning when comparing its size with other clusters. Cluster 1 also has a high value here which of about \\$150k. These two clusters seem to contain anomalies/supicious transactions.

**Average Total Payments**: Similarly to *Average Covered Charges*, Cluster 5 an Cluster 1 have high values of \\$50k and \\$30k respectively. These facts are suspicious considering their size in the dataset.

**Average Medicare Payments**: Again, Cluster 5 has a large difference compared to the others clusters and almost doubles the amount for cluster 1 and is 10 times larger than cluster 0. This is a great concern as cluster 5 contains a small amount of the data points but its average medicare payments is significantly higher. Cluster 1 should also be treated similarly.

**Average Out of Pocket Payments**: Cluster 5 and cluster 1 also have high values for out of pocket payment here but not significantly different from others. Other clusters pay around \\$1000-1500 while these two pay around \\$2500. However, as these amounts are still higher than the average, they should still be investigated.

**Average OOP per Discharge**: Cluster 5 has an extremely high value for the mean of average covered charge, at approximately \\$135, which is a large concern compared to other clusters. Cluster 1 also has a high value here which is about \\$115. These two clusters seem to contain anomalies or supicious transactions.

**Percent of Payment Covered**: Cluster 0 has a high value which is almost double the values of the other clusters. This group indicates about 56% of the dataset however and has not been suspicious within other features, meaning it should be considered for anomalies, but may just be the shape of the data.

**Medicare Coverage Ratio**: Overall, this feature is fairly even across all clusters and therefore has little to investigate. By observing how it affects the data and models, including it later on should be reconsidered.

**Z-score Average Total Payments**: An absolute Z-score greater than 3 are an outliers or anomalies as they fall outside 99.7% of the data and a Z-score of 2 representing 95% of the data . Group 5 (0.5% data point) has an extremely high z-score with a value around 5 and group 1 (2.6% data point) reaches right below 3. Therefore, greater examinination is needed for these clusters as they are anomalous.

**Z Score Average Medicare Payments**: The pattern for the Z-scores of cluster 1 and cluster 5 follow the same pattern as their previous Z-scores and they should be investigated fully to determine the cause.

**Proportion per Cluster**: Overall, cluster 0 makes up over 50% of the data while cluster 4 makes up around 23%. When re-analyzing the clusters, further feature engineering should be considered to find ways to break up these groups. However, this may simple be the shape of the data and is not something that can be altered.







#### Conclusion

Overall, the cluster model contains 11 final features and generates 6 clusters. From those clusters, there are 2  that standout for anomaly detection: group 1 and group 5. As their means for most features are incredibly high and have z-scores that fall outside the normal range, they are highly suspect and may contain anomalous behaviour or outliers that need to be identified. Through monitoring these variables with these features and clusters, hopefully greater detection is realized to prevent any fraud.
