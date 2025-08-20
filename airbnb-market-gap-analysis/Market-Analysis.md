## Table of Contents



## Contents
- [Assignment](#assignment)
- [Data Exploration](#data-exploration)
- [Searches Dataset](#searches-dataset)
- [Distributions](#distributions)
- [Contacts Dataset](#contacts-dataset)



**• What guests are searching in Dublin**  
**• Which inquiries do hosts tend to accept**  
**• What gaps exist between guest demand and host supply**  
**• Any other information that deepens the understanding of the data**  



<h2 id="assignment">Assignment</h2>



The goal is to analyze, understand, visualize, and communicate the demand/supply of the market in Dublin.


<h2 id="data-exploration">Data Exploration</h2>




```python
#Import libraries/dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

contacts_file = ("contacts.tsv")
contacts = pd.read_csv(contacts_file, sep="\t")

searches_file = ("searches.tsv")
searches = pd.read_csv(searches_file, sep="\t")
```

    The history saving thread hit an unexpected error (OperationalError('attempt to write a readonly database')).History will not be written to the database.



```python
#Find % of null values in datasets 
print('Contacts')
print(contacts.isna().sum()/len(contacts), '\n')
print('Searches')
print(searches.isna().sum()/len(searches))
```

    Contacts
    id_guest          0.000000
    id_host           0.000000
    id_listing        0.000000
    ts_contact_at     0.000000
    ts_reply_at       0.077208
    ts_accepted_at    0.536367
    ts_booking_at     0.722101
    ds_checkin        0.000000
    ds_checkout       0.000000
    n_guests          0.000000
    n_messages        0.000000
    dtype: float64 
    
    Searches
    ds                      0.000000
    id_user                 0.000000
    ds_checkin              0.331561
    ds_checkout             0.331561
    n_searches              0.000000
    n_nights                0.331561
    n_guests_min            0.000000
    n_guests_max            0.000000
    origin_country          0.000000
    filter_price_min        0.627221
    filter_price_max        0.627221
    filter_room_types       0.546940
    filter_neighborhoods    0.962336
    dtype: float64


The neighborhood column in searches has 96.2336% of null values. This could lead to inaccurate assumptions about the demand from people. When looking through the column, 'City Centre' was a common choice, so this should be investigated further with more data.

<h2 id="searches-dataset">Searches Dataset</h2>





```python
#Drop filter_neighborhoods column

searches = searches.drop(columns=['filter_neighborhoods'])
```


```python
#Manipulation of searches dataset

#Convert date column to datetime data type for easier analysis
searches['ds'] = pd.to_datetime(searches['ds'])
searches['ds_checkin'] = pd.to_datetime(searches['ds_checkin'])
searches['ds_checkout'] = pd.to_datetime(searches['ds_checkout'])

#How soon they want the room
searches['length_preperation'] = searches['ds_checkin'] - searches['ds']
```


```python
#Describe searches dataset

#Helps understand the dataset and its distribution of values within columns better
display(searches.describe())
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
      <th>ds</th>
      <th>ds_checkin</th>
      <th>ds_checkout</th>
      <th>n_searches</th>
      <th>n_nights</th>
      <th>n_guests_min</th>
      <th>n_guests_max</th>
      <th>filter_price_min</th>
      <th>filter_price_max</th>
      <th>length_preperation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>35737</td>
      <td>23888</td>
      <td>23888</td>
      <td>35737.000000</td>
      <td>23888.000000</td>
      <td>35737.000000</td>
      <td>35737.000000</td>
      <td>13322.000000</td>
      <td>1.332200e+04</td>
      <td>23888</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014-10-07 14:32:34.478551552</td>
      <td>2014-11-27 22:42:10.609511168</td>
      <td>2014-12-05 14:50:57.468184832</td>
      <td>9.206565</td>
      <td>7.672765</td>
      <td>1.742955</td>
      <td>2.105857</td>
      <td>8.470200</td>
      <td>9.019063e+07</td>
      <td>51 days 08:11:53.730743469</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2014-10-01 00:00:00</td>
      <td>2014-10-01 00:00:00</td>
      <td>2014-10-02 00:00:00</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>9.000000e+00</td>
      <td>-1 days +00:00:00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2014-10-04 00:00:00</td>
      <td>2014-10-17 00:00:00</td>
      <td>2014-10-23 00:00:00</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>8.600000e+01</td>
      <td>10 days 00:00:00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2014-10-08 00:00:00</td>
      <td>2014-11-03 00:00:00</td>
      <td>2014-11-09 00:00:00</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.390000e+02</td>
      <td>26 days 00:00:00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2014-10-11 00:00:00</td>
      <td>2014-12-13 00:00:00</td>
      <td>2014-12-24 00:00:00</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.010000e+02</td>
      <td>67 days 00:00:00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2014-10-14 00:00:00</td>
      <td>2016-06-02 00:00:00</td>
      <td>2016-08-17 00:00:00</td>
      <td>448.000000</td>
      <td>399.000000</td>
      <td>16.000000</td>
      <td>16.000000</td>
      <td>1250.000000</td>
      <td>1.073742e+09</td>
      <td>604 days 00:00:00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.348746</td>
      <td>21.557614</td>
      <td>1.460440</td>
      <td>1.817358</td>
      <td>53.987679</td>
      <td>2.978482e+08</td>
      <td>65 days 18:56:19.491940518</td>
    </tr>
  </tbody>
</table>
</div>


This shows that the number of guests is usually 1 or 2. This can be understood since even at 75% the n_guests_min and n_guests_max are 2 and at 25% is 1. Leads to believe that smaller accommodations are preferred


```python
#Calculate skewness in searches dataset

display(searches.skew(axis = 0, numeric_only = True, skipna = True))
```


    n_searches           7.509258
    n_nights             9.942364
    n_guests_min         3.583798
    n_guests_max         3.148548
    filter_price_min    11.087745
    filter_price_max     2.999833
    dtype: float64


All numeric columns have a Fisher-Pearson coefficient value greater than 1. This results in a positive skewness. With more time, I would have used a transformation method such as log transformation to reduce the skewness

<h2 id="distributions">Distributions</h2>





```python
#Distribution plot of n_guests_min and n_guests_max
sns.displot(searches, x = 'n_guests_min', color = 'brown')
sns.displot(searches, x = 'n_guests_max', color = 'black')
plt.show()
```


    
![png](README_files/README_17_0.png)
    



    
![png](README_files/README_17_1.png)
    


Both have similar distributions with 1 being the most popular option and 2 being the next popular option




```python
#When were searches conducted

ax = sns.displot(searches, x = 'ds', color = 'brown')
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in ax.axes.flat]

```




    [[None, None, None, None, None, None, None]]




    
![png](README_files/README_19_1.png)
    


Noticed all date searches were between October 1st to October 14th. No major variation in when search was conducted between these dates




```python
#Percentage of dataset with a filter_price_max above 600

print(len(searches[searches['filter_price_max'] > 600])/len(searches['filter_price_max'])*100, '%')

```

    5.311022189887232 %



```python
#Distribution of filter_price_max of searches

#Removing the set upper limit
searches_maxprice_removed = searches[searches['filter_price_max'] <= 600]

#Distribution plot of filter_price_max column
sns.displot(x=searches_maxprice_removed["filter_price_max"], color = 'blue')
plt.show()
```


    
![png](README_files/README_22_0.png)
    


Filter_price_max was chosen instead of filter_price_min due to the min usually being set at $0

To further help better visualize the trend we set the filter price max as less or equal to 600. 600 was chosen as the limit since only 14.25% of the dataset has values greater than 600


```python
#Distribution of length_preperation of searches

#Percentage of dataset beyond 100 days
distribution = searches["length_preperation"] / np.timedelta64(1, 'D')
print(len(distribution[distribution > 100])/len(distribution)*100, '% \n')

#Remove values beyond 100 days
distribution = distribution[distribution < 100]

#Distribution plot of length_preperation column
sns.displot(x=distribution, color = 'green')
plt.show()
```

    9.396423874415872 % 
    



    
![png](README_files/README_24_1.png)
    


100 days was chosen as the limit since only 14.06% of the dataset exists beyond that




```python
#Distribution of n_nights of searches

#Percentage of dataset beyond 20 nights
print(len(searches[searches['n_nights'] > 20])/len(searches['n_nights'])*100, '% \n')

#Remove n_nights beyond 20 days
searches_within_twenty = searches[searches['n_nights'] < 20]

#Distribution plot of length_preperation column
sns.displot(searches_within_twenty, x='n_nights', color = 'red')
plt.show()

```

    4.737387021854101 % 
    



    
![png](README_files/README_26_1.png)
    


Removing n_nights beyond 20 days since only 7.3% of the dataset exists beyond 20 days




```python
#Distribution of months of ds_checkin of searches

checkin_month = pd.DatetimeIndex(searches['ds_checkin']).month

#Distribution plot of length_preperation column
sns.displot(checkin_month, color = 'yellow')
plt.show()

```


    
![png](README_files/README_28_0.png)
    


Used only the check-in month, cause checkout is usually within 5/6 days. The mean of n_nights after removing the upper outlier limit is 5.6, so assumed 5 or 6 days after the check-in date people usually checkout


```python
#Types of rooms searched for

searches['filter_room_types'].unique()[0:15] #Display first 15 unique values
```




    array([',Entire home/apt,Entire home/apt,Private room,Private room', nan,
           ',Entire home/apt',
           'Entire home/apt,Entire home/apt,Private room,Private room',
           'Entire home/apt', ',Shared room,Private room',
           'Entire home/apt,Private room,Shared room,Private room,Shared room',
           'Private room', 'Entire home/apt,Private room', ',Private room',
           ',Entire home/apt,Private room',
           ',Entire home/apt,Private room,Private room',
           'Entire home/apt,Private room,Shared room',
           ',Entire home/apt,Entire home/apt,Private room',
           ',Entire home/apt,Entire home/apt,Shared room,Shared room'],
          dtype=object)



"""
Most of the room types requested were entire home/apt and private rooms, sometimes shared rooms.
I would have cleaned this column since most filter values are repeated within the same cell.
On the Airbnb website, there are only 4 values in the type of place:

- Entire Place
- Private Room
- Hotel Room
- Shared Room

So searching how often these 4 strings occur would be how I go about it.
"""


```python
#Find top 15 countries where searches originate from

#Group by origin country and finding the count of each country
search_origin = searches.groupby("origin_country").agg({'origin_country' : 'count'})
search_origin.columns = ['count']

search_origin = search_origin.sort_values('count', ascending = False) #Sort count in descending order
search_origin.nlargest(15, 'count') #Find the 15 largest values
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
    </tr>
    <tr>
      <th>origin_country</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IE</th>
      <td>6608</td>
    </tr>
    <tr>
      <th>US</th>
      <td>5811</td>
    </tr>
    <tr>
      <th>GB</th>
      <td>4832</td>
    </tr>
    <tr>
      <th>FR</th>
      <td>3444</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>2333</td>
    </tr>
    <tr>
      <th>DE</th>
      <td>2170</td>
    </tr>
    <tr>
      <th>ES</th>
      <td>1759</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>1085</td>
    </tr>
    <tr>
      <th>AU</th>
      <td>962</td>
    </tr>
    <tr>
      <th>NL</th>
      <td>843</td>
    </tr>
    <tr>
      <th>BR</th>
      <td>636</td>
    </tr>
    <tr>
      <th>CH</th>
      <td>535</td>
    </tr>
    <tr>
      <th>BE</th>
      <td>386</td>
    </tr>
    <tr>
      <th>AT</th>
      <td>320</td>
    </tr>
    <tr>
      <th>RU</th>
      <td>274</td>
    </tr>
  </tbody>
</table>
</div>



<h2 id="contacts-dataset">Contacts Dataset</h2>


```python

contacts_file = ("contacts.tsv")
contacts = pd.read_csv(contacts_file, sep="\t")

searches_file = ("searches.tsv")
searches = pd.read_csv(searches_file, sep="\t")
```


```python
contacts.columns
```




    Index(['id_guest', 'id_host', 'id_listing', 'ts_contact_at', 'ts_reply_at',
           'ts_accepted_at', 'ts_booking_at', 'ds_checkin', 'ds_checkout',
           'n_guests', 'n_messages'],
          dtype='object')




```python
searches.columns
```




    Index(['ds', 'id_user', 'ds_checkin', 'ds_checkout', 'n_searches', 'n_nights',
           'n_guests_min', 'n_guests_max', 'origin_country', 'filter_price_min',
           'filter_price_max', 'filter_room_types', 'filter_neighborhoods'],
          dtype='object')




```python
# Merge contacts and searches datasets
merged_df = pd.merge(
    contacts,
    searches,
    left_on='id_guest',
    right_on='id_user',
    how='inner'  # Use 'left' if you want all contacts, 'inner' for only matches
)

# Check the shape and preview
print("Merged shape:", merged_df.shape)
merged_df.head()

# Optional: Save merged dataset for Tableau
merged_df.to_csv("merged_airbnb.csv", index=False)
print("Merged dataset saved as merged_airbnb.csv")

```

    Merged shape: (28536, 24)
    Merged dataset saved as merged_airbnb.csv



```python
merged_df.head()
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
      <th>id_guest</th>
      <th>id_host</th>
      <th>id_listing</th>
      <th>ts_contact_at</th>
      <th>ts_reply_at</th>
      <th>ts_accepted_at</th>
      <th>ts_booking_at</th>
      <th>ds_checkin_x</th>
      <th>ds_checkout_x</th>
      <th>n_guests</th>
      <th>...</th>
      <th>ds_checkout_y</th>
      <th>n_searches</th>
      <th>n_nights</th>
      <th>n_guests_min</th>
      <th>n_guests_max</th>
      <th>origin_country</th>
      <th>filter_price_min</th>
      <th>filter_price_max</th>
      <th>filter_room_types</th>
      <th>filter_neighborhoods</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000dfad9-459b-4f0b-8310-3d6ab34e4f57</td>
      <td>13bb24b8-d432-43a2-9755-5ea11b43bb69</td>
      <td>21d2b1a2-fdc3-4b4c-a1f0-0eaf0cc02370</td>
      <td>2014-10-04 16:26:28.0</td>
      <td>2014-10-04 16:26:28.0</td>
      <td>2014-10-04 16:26:28.0</td>
      <td>2014-10-04 16:26:28.0</td>
      <td>2014-10-13</td>
      <td>2014-10-15</td>
      <td>2</td>
      <td>...</td>
      <td>2014-10-15</td>
      <td>6</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>CA</td>
      <td>0.0</td>
      <td>130.0</td>
      <td>,Entire home/apt</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00197051-c6cb-4c3a-99e9-86615b819874</td>
      <td>46aa3897-9c00-4d76-ac66-a307593d0675</td>
      <td>fb5ed09a-9848-4f2c-b2ef-34deb62164fb</td>
      <td>2014-11-04 09:10:03.0</td>
      <td>2014-11-04 09:45:50.0</td>
      <td>2014-11-04 09:45:50.0</td>
      <td>2014-11-04 12:20:46.0</td>
      <td>2014-11-27</td>
      <td>2014-11-30</td>
      <td>1</td>
      <td>...</td>
      <td>2014-11-30</td>
      <td>13</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>DK</td>
      <td>0.0</td>
      <td>336.0</td>
      <td>,Entire home/apt,Private room,Private room</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0027538e-aa9e-4a02-8979-b8397e5d4cba</td>
      <td>6bbb88ca-db66-48c5-9c4b-862f7706284a</td>
      <td>d3871da6-8012-4dc4-b508-c91f2c10c297</td>
      <td>2014-10-10 12:02:50.0</td>
      <td>2014-10-10 15:07:01.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014-10-17</td>
      <td>2014-10-19</td>
      <td>2</td>
      <td>...</td>
      <td>2014-10-19</td>
      <td>21</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>CY</td>
      <td>0.0</td>
      <td>1258.0</td>
      <td>,Entire home/apt,Entire home/apt,Private room,...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0027538e-aa9e-4a02-8979-b8397e5d4cba</td>
      <td>6bbb88ca-db66-48c5-9c4b-862f7706284a</td>
      <td>d3871da6-8012-4dc4-b508-c91f2c10c297</td>
      <td>2014-10-10 12:02:50.0</td>
      <td>2014-10-10 15:07:01.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014-10-17</td>
      <td>2014-10-19</td>
      <td>2</td>
      <td>...</td>
      <td>2014-10-19</td>
      <td>44</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>CY</td>
      <td>0.0</td>
      <td>214.0</td>
      <td>,Entire home/apt</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0027538e-aa9e-4a02-8979-b8397e5d4cba</td>
      <td>8772bc85-a9b7-4d85-a52d-41f3620c2912</td>
      <td>0d9b5583-8053-4b67-adfe-8c29eb12efed</td>
      <td>2014-10-10 15:23:53.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014-10-17</td>
      <td>2014-10-19</td>
      <td>2</td>
      <td>...</td>
      <td>2014-10-19</td>
      <td>21</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>CY</td>
      <td>0.0</td>
      <td>1258.0</td>
      <td>,Entire home/apt,Entire home/apt,Private room,...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
#Manipulation of contacts dataset

#Convert date columns to datetime data type 
contacts['ts_contact_at'] = pd.to_datetime(contacts['ts_contact_at'])
contacts['ts_reply_at'] = pd.to_datetime(contacts['ts_reply_at'])
contacts['ts_accepted_at'] = pd.to_datetime(contacts['ts_accepted_at'])
contacts['ts_booking_at'] = pd.to_datetime(contacts['ts_booking_at'])
contacts['ds_checkin'] = pd.to_datetime(contacts['ds_checkin'])
contacts['ds_checkout'] = pd.to_datetime(contacts['ds_checkout'])
contacts['accepted'] = np.where(np.isnan(contacts['ts_accepted_at']), False, True)

contacts['length_stay'] = contacts['ds_checkout'] - contacts['ds_checkin']

#Understand dataset
display(contacts.dtypes)
display(contacts.describe())
```


    id_guest                   object
    id_host                    object
    id_listing                 object
    ts_contact_at      datetime64[ns]
    ts_reply_at        datetime64[ns]
    ts_accepted_at     datetime64[ns]
    ts_booking_at      datetime64[ns]
    ds_checkin         datetime64[ns]
    ds_checkout        datetime64[ns]
    n_guests                    int64
    n_messages                  int64
    accepted                     bool
    length_stay       timedelta64[ns]
    dtype: object



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
      <th>ts_contact_at</th>
      <th>ts_reply_at</th>
      <th>ts_accepted_at</th>
      <th>ts_booking_at</th>
      <th>ds_checkin</th>
      <th>ds_checkout</th>
      <th>n_guests</th>
      <th>n_messages</th>
      <th>length_stay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7823</td>
      <td>7219</td>
      <td>3627</td>
      <td>2174</td>
      <td>7823</td>
      <td>7823</td>
      <td>7823.000000</td>
      <td>7823.000000</td>
      <td>7823</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014-10-10 11:59:33.640802816</td>
      <td>2014-10-10 22:42:36.511982336</td>
      <td>2014-10-11 14:07:07.993382912</td>
      <td>2014-10-13 07:54:43.307267840</td>
      <td>2014-11-16 09:47:00.452511744</td>
      <td>2014-11-22 05:12:33.317141760</td>
      <td>2.422600</td>
      <td>6.319954</td>
      <td>5 days 19:25:32.864629937</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2014-03-04 11:08:13</td>
      <td>2014-04-18 09:39:06</td>
      <td>2014-05-21 16:51:54</td>
      <td>2014-05-21 16:51:54</td>
      <td>2014-10-01 00:00:00</td>
      <td>2014-10-02 00:00:00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1 days 00:00:00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2014-10-03 16:28:52</td>
      <td>2014-10-03 23:07:32</td>
      <td>2014-10-04 13:45:13</td>
      <td>2014-10-05 13:53:46</td>
      <td>2014-10-15 00:00:00</td>
      <td>2014-10-19 00:00:00</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2 days 00:00:00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2014-10-08 17:34:36</td>
      <td>2014-10-09 00:16:21</td>
      <td>2014-10-09 13:46:50</td>
      <td>2014-10-10 11:15:13</td>
      <td>2014-10-31 00:00:00</td>
      <td>2014-11-06 00:00:00</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>3 days 00:00:00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2014-10-13 21:28:51.500000</td>
      <td>2014-10-14 10:00:04.500000</td>
      <td>2014-10-14 16:28:27.500000</td>
      <td>2014-10-15 13:16:17.249999872</td>
      <td>2014-11-22 00:00:00</td>
      <td>2014-11-30 00:00:00</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>5 days 00:00:00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015-02-02 08:45:18</td>
      <td>2015-02-02 23:18:25</td>
      <td>2015-02-03 14:16:42</td>
      <td>2015-01-21 16:16:29</td>
      <td>2015-10-24 00:00:00</td>
      <td>2015-12-01 00:00:00</td>
      <td>16.000000</td>
      <td>102.000000</td>
      <td>334 days 00:00:00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.617347</td>
      <td>6.472827</td>
      <td>14 days 23:45:24.447710564</td>
    </tr>
  </tbody>
</table>
</div>



```python
#Calculate skewness in contacts dataset

display(contacts.skew(axis = 0, numeric_only = True, skipna = True))
```


    n_guests      2.441468
    n_messages    3.696440
    accepted      0.145883
    dtype: float64


All columns have a Fisher-Pearson coefficient value greater than 1. Except for accepted, which could be due to it being derived from an existing column. With more time, I would have used a transformation method such as box-cox to reduce the skewness.


```python
#Number of guests stayed

contacts_less8 = contacts[contacts['n_guests'] < 8]
sns.displot(contacts_less8, x = 'n_guests', hue = 'accepted', multiple="dodge")
plt.show()
```


    
![png](README_files/README_42_0.png)
    


Choosing less than 8 guests, since only 1.46% (114 values) of the contacts dataset has 8 or more guests. To better visualize the majority distribution we removed rows with 8 or more guests.

2 guests is the most popular option to book, but 1 guest is the most popularly searched option. This leads me to believe there is a lack of supply of viable single guest rooms.


```python
#Conversion rate from accepting to booking

contacts['ts_booking_at'].count()/contacts['ts_accepted_at'].count()
```




    0.5993934381031155




```python
#Timeframe of when guests or accepted vs rejected

contacts['month_checkin'] = contacts['ds_checkin'].dt.month #Extract month from checkin date
contacts_checkin = contacts[contacts['month_checkin'] > 9] #Use only peak season months (Oct, Nov, Dec)

#Distribution of checkin among October, November, and December and split by acceptance
sns.displot(contacts_checkin, x='month_checkin', hue = 'accepted', multiple="dodge")
plt.xticks([10, 11, 12])
plt.show()

```


    
![png](README_files/README_45_0.png)
    



```python
#Merge datasets for more analysis

merged_datasets = contacts.merge(searches, left_on='id_guest', right_on='id_user')
```


```python
#Check difference between prices searched between accepted/rejected applicants

merged_pricemax_filter = merged_datasets.loc[(merged_datasets['filter_price_max'] <= 600)]

sns.displot(merged_pricemax_filter, x="filter_price_max", hue="accepted", multiple="dodge")
plt.show()
```


    
![png](README_files/README_47_0.png)
    


To further help better visualize the trend we set the filter price max as less or equal to 600. 600 was chosen as the limit since only 14.25% of the dataset has values greater than 600.

As seen, more people are rejected compared than accepted with an average acceptance rate of 43%




```python
#Classify dataset based on filter_price_max

def label_price (row):
    if (row['filter_price_max'] >= 0) & (row['filter_price_max'] < 100):
        return '0-100'
    
    elif (row['filter_price_max'] >= 100) & (row['filter_price_max'] < 200):
        return '100-200'

    elif (row['filter_price_max'] >= 200) & (row['filter_price_max'] < 300):
        return '200-300'
    
    elif (row['filter_price_max'] >= 300) & (row['filter_price_max'] < 400):
        return '300-400'

    elif (row['filter_price_max'] >= 400) & (row['filter_price_max'] < 500):
        return '400-500'
    
    elif (row['filter_price_max'] >= 500) & (row['filter_price_max'] < 600):
        return '500-600'
    
    else:
        return '600+'

merged_datasets['classification_max_price'] = merged_datasets.apply(lambda row: label_price(row), axis=1)

merged_datasets.groupby('classification_max_price').agg({'accepted': 'mean'})
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
      <th>accepted</th>
    </tr>
    <tr>
      <th>classification_max_price</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0-100</th>
      <td>0.411160</td>
    </tr>
    <tr>
      <th>100-200</th>
      <td>0.430308</td>
    </tr>
    <tr>
      <th>200-300</th>
      <td>0.431149</td>
    </tr>
    <tr>
      <th>300-400</th>
      <td>0.450488</td>
    </tr>
    <tr>
      <th>400-500</th>
      <td>0.485549</td>
    </tr>
    <tr>
      <th>500-600</th>
      <td>0.422297</td>
    </tr>
    <tr>
      <th>600+</th>
      <td>0.433122</td>
    </tr>
  </tbody>
</table>
</div>



Based on this table, it can be seen that regardless of max_filter_price, people are rejected at similar rates




```python
#Find the acceptance rate by country

dataset_country = merged_datasets[['origin_country', 'accepted']]

#Find acceptance count by country and accepted
accepted_count = dataset_country.groupby(['origin_country', 'accepted']).agg({'origin_country':'count'})
accepted_count.columns = ['count_accepted']

#Find acceptance count by country
country_count = dataset_country.groupby(['origin_country']).agg({'origin_country':'count'})
country_count.columns = ['count_country']

#Merge datasets for easier manipulation 
acceptance_country = pd.merge(dataset_country, accepted_count,  how='left', on=['origin_country','accepted']) #Merge accepted count
acceptance_country = acceptance_country.drop_duplicates()

acceptance_country = pd.merge(acceptance_country, country_count, how='left', on=['origin_country']) #Merge total country count
acceptance_country = acceptance_country.sort_values(['count_country', 'accepted'], ascending = [False, True])
acceptance_country = acceptance_country[acceptance_country['count_country'] >= 100] #100 is used so there is a good amount of data to make assumptions
acceptance_country = acceptance_country[acceptance_country['accepted'] == True]

#Divide count_accepted column by count_country column to find acceptance rate by country
acceptance_country['acceptance_rate'] = acceptance_country['count_accepted']/acceptance_country['count_country']
acceptance_country.sort_values(['acceptance_rate'], ascending = True)
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
      <th>origin_country</th>
      <th>accepted</th>
      <th>count_accepted</th>
      <th>count_country</th>
      <th>acceptance_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>IN</td>
      <td>True</td>
      <td>138</td>
      <td>874</td>
      <td>0.157895</td>
    </tr>
    <tr>
      <th>55</th>
      <td>HR</td>
      <td>True</td>
      <td>159</td>
      <td>530</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>72</th>
      <td>AT</td>
      <td>True</td>
      <td>83</td>
      <td>239</td>
      <td>0.347280</td>
    </tr>
    <tr>
      <th>54</th>
      <td>RU</td>
      <td>True</td>
      <td>83</td>
      <td>239</td>
      <td>0.347280</td>
    </tr>
    <tr>
      <th>11</th>
      <td>IT</td>
      <td>True</td>
      <td>1183</td>
      <td>3137</td>
      <td>0.377112</td>
    </tr>
    <tr>
      <th>100</th>
      <td>AE</td>
      <td>True</td>
      <td>59</td>
      <td>154</td>
      <td>0.383117</td>
    </tr>
    <tr>
      <th>0</th>
      <td>CA</td>
      <td>True</td>
      <td>407</td>
      <td>993</td>
      <td>0.409869</td>
    </tr>
    <tr>
      <th>13</th>
      <td>IE</td>
      <td>True</td>
      <td>1217</td>
      <td>2951</td>
      <td>0.412403</td>
    </tr>
    <tr>
      <th>24</th>
      <td>ES</td>
      <td>True</td>
      <td>794</td>
      <td>1914</td>
      <td>0.414838</td>
    </tr>
    <tr>
      <th>49</th>
      <td>RO</td>
      <td>True</td>
      <td>50</td>
      <td>118</td>
      <td>0.423729</td>
    </tr>
    <tr>
      <th>78</th>
      <td>CR</td>
      <td>True</td>
      <td>82</td>
      <td>188</td>
      <td>0.436170</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GB</td>
      <td>True</td>
      <td>1610</td>
      <td>3667</td>
      <td>0.439051</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BE</td>
      <td>True</td>
      <td>134</td>
      <td>304</td>
      <td>0.440789</td>
    </tr>
    <tr>
      <th>38</th>
      <td>BR</td>
      <td>True</td>
      <td>215</td>
      <td>482</td>
      <td>0.446058</td>
    </tr>
    <tr>
      <th>27</th>
      <td>AU</td>
      <td>True</td>
      <td>268</td>
      <td>590</td>
      <td>0.454237</td>
    </tr>
    <tr>
      <th>17</th>
      <td>FR</td>
      <td>True</td>
      <td>1526</td>
      <td>3232</td>
      <td>0.472153</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CH</td>
      <td>True</td>
      <td>279</td>
      <td>585</td>
      <td>0.476923</td>
    </tr>
    <tr>
      <th>7</th>
      <td>US</td>
      <td>True</td>
      <td>2050</td>
      <td>4298</td>
      <td>0.476966</td>
    </tr>
    <tr>
      <th>14</th>
      <td>DE</td>
      <td>True</td>
      <td>745</td>
      <td>1535</td>
      <td>0.485342</td>
    </tr>
    <tr>
      <th>31</th>
      <td>NL</td>
      <td>True</td>
      <td>212</td>
      <td>433</td>
      <td>0.489607</td>
    </tr>
    <tr>
      <th>46</th>
      <td>SG</td>
      <td>True</td>
      <td>115</td>
      <td>232</td>
      <td>0.495690</td>
    </tr>
    <tr>
      <th>65</th>
      <td>PT</td>
      <td>True</td>
      <td>101</td>
      <td>203</td>
      <td>0.497537</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DK</td>
      <td>True</td>
      <td>86</td>
      <td>125</td>
      <td>0.688000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset_country.columns

```




    Index(['origin_country', 'accepted'], dtype='object')



An interesting point is that India only has the lowest acceptance rate of 15%, which is half of the acceptance rate compared to the second lowest accepted country. Needs to be investigated further


```python

```
