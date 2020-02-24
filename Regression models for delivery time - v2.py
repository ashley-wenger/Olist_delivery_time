# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:34:49 2019

@author: ashley
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from math import pi, sin, cos, acos
import seaborn as sns  # pretty plotting, including heat map

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import average_precision_score
from inspect import signature



#set some Pandas options
pd.set_option('display.max_columns', 65) 
pd.set_option('display.max_rows', 20) 
pd.set_option('display.width', 160)



wkg_dir = 'C:/Users/ashle/Documents/Personal Data/Northwestern/2019-04  fall MSDS498_Sec56 Capstone/Git_repo/Predictive_models/'
os.chdir(wkg_dir)


#read in the data
orders_df = pd.read_csv('../DataSet/Order_level_dataset.csv')
orders_df.head()
orders_df.info()

#convert fields to more specific datatypes (esp. date fields)
orders_df.order_estimated_delivery_date = pd.to_datetime(orders_df.order_estimated_delivery_date)  
orders_df.order_purchase_timestamp = pd.to_datetime(orders_df.order_purchase_timestamp)  
orders_df.order_approved_at = pd.to_datetime(orders_df.order_approved_at)  
orders_df.order_delivered_carrier_date = pd.to_datetime(orders_df.order_delivered_carrier_date)  
orders_df.order_delivered_customer_date = pd.to_datetime(orders_df.order_delivered_customer_date)  
orders_df.ship_limit_initial = pd.to_datetime(orders_df.ship_limit_initial)  
orders_df.ship_limit_final = pd.to_datetime(orders_df.ship_limit_final)  
orders_df.earliest_review_dt = pd.to_datetime(orders_df.earliest_review_dt)  
orders_df.latest_review_dt = pd.to_datetime(orders_df.latest_review_dt)  




# =================
#   Prep dataset
# =================
#our predicted variable!
orders_df['fulfill_duration'] = (orders_df.order_delivered_customer_date - orders_df.order_purchase_timestamp).dt.total_seconds()/86400



# Create calc'd variables that are suspected to be valuable for prediction
# ------------------------------------------------------------------------
orders_df['late_delivery_flag'] = (orders_df.order_delivered_customer_date >= orders_df.order_estimated_delivery_date + timedelta(days=1)).astype(int)
#sum(orders_df.late_delivery_flag)  #6535

# estimated delivery time (in days)
orders_df['est_delivery_time_days'] = (orders_df.order_estimated_delivery_date - orders_df.order_purchase_timestamp).dt.total_seconds()/86400
orders_df['est_delivery_time_days'] = np.ceil(orders_df['est_delivery_time_days']).astype(int)

#approval time
orders_df['approval_time_days'] = (orders_df.order_approved_at - orders_df.order_purchase_timestamp).dt.total_seconds()/86400



    
# distance/geography related aspects
# ----------------------------------
def distance_calcn_rough(start_lat, start_long, end_lat, end_long):
    #check for missing values and quit if any are found
    if ( np.isnan(start_lat) | np.isnan(start_long) | np.isnan(end_lat) | np.isnan(end_long) ):
        return None

    #convert from radians to radians
    start_lat = start_lat * pi / 180.0
    start_long = start_long * pi / 180.0
    end_lat = end_lat * pi / 180.0
    end_long = end_long * pi / 180.0
  
    cosine_val = sin(start_lat)*sin(end_lat) + cos(start_lat)*cos(end_lat)*cos(start_long - end_long)
  
    if (cosine_val > 1) | (cosine_val < -1): 
        cosine_val = round(cosine_val, 6)      #round it off so we aren't losing cases due to machine precision issues (esp. when cosine_val ~1 b/c they are at the exact same location)

    if (cosine_val > 1) | (cosine_val < -1):
        rtrn_val = None
    else: 
        rtrn_val = 6371.01*acos(cosine_val)

    return rtrn_val


orders_df['distance_km'] = orders_df.apply(lambda rww:  distance_calcn_rough(rww['lat_seller'], rww['long_seller'], rww['lat_customer'], rww['long_customer']), axis=1)
#sum(pd.isnull(orders_df.distance_km))  #1267 - not as many nulls as I thought

orders_df['states_same_or_diff'] = orders_df.customer_state.astype(str) == orders_df.seller_state.astype(str)
orders_df['states_same_or_diff'] = orders_df.states_same_or_diff.astype(int)

orders_df['state_pair'] = orders_df.customer_state.astype(str) + '-' + orders_df.seller_state.astype(str)
#orders_df.info()




#look for impacts by month, yr-and-mo, ....  of the purchase
orders_df['purchase_mo'] = orders_df.order_purchase_timestamp.dt.month_name()
orders_df['purchase_yr_and_mo'] = orders_df.order_purchase_timestamp.dt.year + (orders_df.order_purchase_timestamp.dt.month-1)/12
orders_df['purchase_day_of_wk'] = orders_df.order_purchase_timestamp.dt.weekday_name


#look for impacts by month, yr-and-mo, ....  of the estimated delivery date
orders_df['est_delivery_mo'] = orders_df.order_estimated_delivery_date.dt.month_name()
orders_df['est_delivery_yr_and_mo'] = orders_df.order_estimated_delivery_date.dt.year + (orders_df.order_estimated_delivery_date.dt.month-1)/12
orders_df['est_delivery_day_of_wk'] = orders_df.order_estimated_delivery_date.dt.weekday_name

#these return integers, and I don't want the model to think these are ordinal values as I don't think that's the case
#orders_df['est_delivery_mo'] = orders_df.order_estimated_delivery_date.dt.month
#orders_df['est_delivery_yr_and_mo'] = orders_df.order_estimated_delivery_date.dt.year*100 + orders_df.order_estimated_delivery_date.dt.month
#orders_df['est_delivery_day_of_wk'] = orders_df.order_estimated_delivery_date.dt.weekday



#  hopefully we don't need these for good predictions, as they aren't known until late in the game.  Ideally make a good prediction up front at time of purchase
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# shipping limit missed
orders_df['shipping_limit_missed'] = (orders_df.order_delivered_carrier_date > orders_df.ship_limit_final).astype(int)
orders_df['shipping_limit_miss_amt'] = (orders_df.order_delivered_carrier_date - orders_df.ship_limit_final).dt.total_seconds()/86400


# days remaining?
def days_remaining(start_dt1, start_dt2, end_dt):
    '''Calc time delta between end_dt and min(start_dt1, start_dt2)
    All three inputs are expected to be datetimes.'''
    
    if start_dt1 <= start_dt2:
        start_dt = start_dt1
    else:
        start_dt = start_dt2
        
    days_remng = (end_dt - start_dt).total_seconds()/86400
    return days_remng

    
orders_df['days_remaining'] = orders_df.apply(lambda rww: days_remaining(rww['order_delivered_carrier_date'], rww['ship_limit_final'], rww['order_estimated_delivery_date']), axis=1)
#orders_df[['order_delivered_carrier_date', 'ship_limit_final', 'order_estimated_delivery_date', 'days_remaining']]


# end of "hopefully we don't need these" vars








# outlier and missing handling
# ----------------------------
# drop rows with missing values as some techniques are sensitive to this (logistic regression)
orders_df2 = orders_df.loc[ ( pd.notnull(orders_df.order_purchase_timestamp) ) &
                            ( pd.notnull(orders_df.order_delivered_customer_date) ) &
                            ( pd.notnull(orders_df.distance_km) )  &
                            ( pd.notnull(orders_df.order_approved_at) ) &
                            ( pd.notnull(orders_df.order_delivered_carrier_date) ) &
                            ( pd.notnull(orders_df.ttl_pd) )  ]
orders_df2.info()    #95,981   
    #96,476 w/ just requiring the 1st 2 fields (essential to getting fulfill duration) so the extra cols with pruning for nulls make very little difference

    #distance_km is the main one, but this only drops another ~500 rows, seems certain we'll need this as a predictor

for col in orders_df2.columns:
    nbr_nulls = sum(pd.isnull(orders_df2[col]))
    if nbr_nulls > 0:
        print(f'{col} has {nbr_nulls} nulls still.')

#product_ctgry_mfu has 1344 nulls still.     but we should be able to impute a value to this if necessary (leave all one hot encodings zero??) so just move on
        




# ============================
# Split into test and train
# ============================
#need to start doing EDA to look for important predictors, volumes, etc.
#  before doing that, split the data into training and test so that we are not data snooping into the test set.
#  need to do the EDA on just the training or it's cheating to consider the test set an unknown dataset

predicted_col = ['fulfill_duration']

RANDOM_SEED=42

orders_train, orders_test, y_train, y_test = train_test_split(orders_df2, orders_df2[predicted_col], test_size=0.25, random_state=RANDOM_SEED)


    


# =============================================================
# EDA for important trends, insights, and explanatory variables
# =============================================================

#orders_train.info()
#orders_train.select_dtypes(exclude=['datetime', 'object']).columns
#
#'fulfill_duration', 
# 'est_delivery_time_days', 'distance_km', 'approval_time_days', 
#
#'ttl_pd', 'ttl_price', 'ttl_freight', 'pmt_mthds_used', 'installments_used_ttl', 'payment_types_used', 
#'nbr_items', 'nbr_sellers', 'nbr_products', 'nbr_photos', 
#'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width
#   
#     'purchase_yr_and_mo', 'est_delivery_yr_and_mo', 
# 'lat_customer', 'long_customer', 'lat_seller', 'long_seller', 
#
#       'shipping_limit_miss_amt', 'days_remaining'],
#
#'states_same_or_diff',     'late_delivery_flag', 'shipping_limit_missed',
#numerical, but don't use in correlation analysis as I don't think the "order" of zips has meaning so correlation has no meaning:  customer_zip_code_prefix', 'seller_zip_code_prefix', 

#check these:   orders_train.select_dtypes(include=['datetime', 'object']).columns



#look at correlations, scatter plots of fulfill_duration vs. these vars
contnuous_vars = ['fulfill_duration', 'est_delivery_time_days', 'distance_km', 'approval_time_days', 'ttl_pd', 'ttl_price', 'ttl_freight', 'pmt_mthds_used', 'installments_used_ttl', 'payment_types_used', 'nbr_items', 'nbr_sellers', 'nbr_products', 'nbr_photos', 'ttl_wt', 'ttl_length', 'ttl_height', 'ttl_width', 'purchase_yr_and_mo', 'est_delivery_yr_and_mo', 'states_same_or_diff', 'lat_customer', 'long_customer', 'lat_seller', 'long_seller', 'shipping_limit_miss_amt', 'days_remaining', 'late_delivery_flag', 'shipping_limit_missed']
        
def corr_chart(df_corr, fig_size=(12,12), file_nm='plot-corr-map.jpg'):
    corr_matrix=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr_matrix, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(corr_matrix, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=90) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig(file_nm, 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)  
    
    return plt, corr_matrix

corr_plt, corr_mtrx = corr_chart(orders_train[contnuous_vars], (20,20), 'output/FulfillDurnModel_EDA_correlations.jpg')
 
corrs_w_fulfill_dur = corr_mtrx.iloc[0, ].reset_index()        
corrs_w_fulfill_dur.columns = ['predictor', 'corr_w_fulfill_dur']
corrs_w_fulfill_dur['corr_strength'] = abs(corrs_w_fulfill_dur.corr_w_fulfill_dur)
corrs_w_fulfill_dur = corrs_w_fulfill_dur.sort_values('corr_strength', ascending=False)

corr_plt.close()

#remove these, as we want to predict earlier on, before these are known
corrs_w_fulfill_dur = corrs_w_fulfill_dur[ ~ corrs_w_fulfill_dur.predictor.isin(['late_delivery_flag', 'days_remaining', 'shipping_limit_missed', 'shipping_limit_miss_amt'])]
#ditto for 'approval_time_days',  but if this seems REALLY helpful might want to add it in as a second round "updated delivery" estimate
 
#might want to leave out lat_customer (esp.) as it's highly correlated with distance_km - ditto for other lat/long fields
list(corrs_w_fulfill_dur.head(25)['predictor'])
top_contnous_pred = ['distance_km', 'est_delivery_time_days', 'states_same_or_diff', 'ttl_freight', 'purchase_yr_and_mo', 'est_delivery_yr_and_mo', 'ttl_pd', 'ttl_wt', 'ttl_price', 'installments_used_ttl', 'ttl_height', 'nbr_sellers', 'ttl_length', 'nbr_photos', 'nbr_products', 'nbr_items', 'ttl_width', 'pmt_mthds_used', 'payment_types_used']
top_contnous_pred2 = ['lat_customer', 'long_customer', 'lat_seller', 'long_seller', 'approval_time_days']
 

#look at scatter plots
for var in top_contnous_pred:
#for var in top_contnous_pred2:
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    plt.title(f'Fulfill duration vs {var}')   
    plt.xlabel(var)
    plt.ylabel('Fulfill duration (days)')
    plt.scatter(orders_train[var], orders_train['fulfill_duration']) 
    file_nm = 'output/Rgrsn_scatter_' + var + '.jpg'
    plt.savefig(file_nm, 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)  





# look at categorical vars
# ========================
# do box plots of distros by categorical var when the var is low cardinality
      
catgrcl_vars = ['customer_zip_code_prefix', 'customer_city', 'customer_state', 'seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state', 'state_pair', 'product_ctgry_mfu',  'payment_type_mfu', 'purchase_mo', 'purchase_yr_and_mo', 'purchase_day_of_wk', 'est_delivery_mo', 'est_delivery_day_of_wk']
# check this out??       'ship_limit_final', 

#orders_train[catgrcl_vars].nunique()
catgrcl_vars_lowC = ['customer_state', 'seller_state', 'product_ctgry_mfu',  'payment_type_mfu', 'purchase_mo', 'purchase_yr_and_mo', 'purchase_day_of_wk', 'est_delivery_mo', 'est_delivery_day_of_wk']
catgrcl_vars_hiC = ['customer_zip_code_prefix', 'customer_city', 'seller_id', 'seller_zip_code_prefix', 'seller_city', 'state_pair']

for var in catgrcl_vars_lowC:
    orders_train.boxplot(column='fulfill_duration', by=var)

#cust state seems to be the most interesting
fig1, ax1 = plt.subplots(figsize=(10,7))
#rotn = 45 if var in ['seller_city', 'state_pair', 'customer_zip_code_prefix'] else 90
rotn = 0
orders_train.boxplot(column='fulfill_duration', by='customer_state', ax=ax1, rot=rotn, vert=False)
plt.suptitle(f'Boxplot of fulfill_duration, grouped by customer state')
ax1.set_title('')
ax1.set_xlim(0, 50)
plt.ylabel('Customer state')
plt.xlabel('Fulfill duration (days)')
file_nm = 'output/FulfillDurnModel_EDA_boxplot_by_cust_state.jpg'
plt.savefig(file_nm, 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

plt.close()
    
    
df_stats_by_var_and_lvl = pd.DataFrame()

for var in catgrcl_vars_hiC:
    lvl_cnts = []
    lvl_means = []
    lvl_std = []
    lvl_min = []
    lvl_max = []
    lvl_1qtle = []
    lvl_3qtle = []
    nbr_lvls = len(pd.unique(orders_train[var]))
    print(f'\nNow calculating stats for the {var} field; ({nbr_lvls} distinct values therein)')

    lvl_lst = list(pd.unique(orders_train[var]))
    for lvl in lvl_lst:
        df_lvl = orders_train.loc[orders_train[var]==lvl, [var, 'fulfill_duration']]
        lvl_cnts.append( len(df_lvl) )
        lvl_means.append( df_lvl['fulfill_duration'].mean() ) 
        lvl_std.append( df_lvl['fulfill_duration'].std() ) 
        lvl_min.append( df_lvl['fulfill_duration'].min() ) 
        lvl_max.append( df_lvl['fulfill_duration'].max() ) 
        lvl_1qtle.append( df_lvl['fulfill_duration'].quantile(q=0.25) ) 
        lvl_3qtle.append( df_lvl['fulfill_duration'].quantile(q=0.75) ) 

    #add results to a dataframe
    df_var = pd.DataFrame( {'variabl' : [var for i in range(nbr_lvls)], 'ctgry_lvl': lvl_lst, 'nbr':lvl_cnts, 'lvl_mean':lvl_means, 'lvl_std':lvl_std, 'lvl_min':lvl_min, 'lvl_max':lvl_max, 'lvl_1qtle':lvl_1qtle, 'lvl_3qtle':lvl_3qtle } )
    #add the dataframe for this var to the master df
    df_stats_by_var_and_lvl = df_stats_by_var_and_lvl.append(df_var)


#df_stats_by_var_and_lvl.shape
#df_lvl.shape
#df_stats_by_var_and_lvl.head(20)

#extract levels of the variables that are a) decent sized (at least 50 obs) and b) higher or lower than avg  (1st quartile is above the overall mean, or 3rd quartile is below the overall mean)
big_movers = df_stats_by_var_and_lvl[ (df_stats_by_var_and_lvl.nbr > 50 ) &
                                      (   ( df_stats_by_var_and_lvl.lvl_1qtle > np.mean(orders_train.fulfill_duration) )
                                        | ( df_stats_by_var_and_lvl.lvl_3qtle < np.mean(orders_train.fulfill_duration) )
                                      ) ]
#282 rows
big_movers['variabl'].value_counts()
#customer_city               88
#seller_zip_code_prefix      70
#seller_id                   69
#state_pair                  27
#seller_city                 22
#customer_zip_code_prefix     6



def flag_big_mover_record(var_name, level_name):
    rcds_fnd = big_movers[ ( big_movers['variabl'] ==var ) & 
                           ( big_movers['ctgry_lvl'] ==level_name ) ]

    if (len(rcds_fnd) > 0):
        return level_name
    else:
        return ' ALL OTHERS'
 
orders_train2 = orders_train.copy()    



#orders_test.drop('customer_city__grpd', axis='columns', inplace=True)
#orders_test.drop('seller_zip_code_prefix__grpd', axis='columns', inplace=True)
#orders_test.drop('seller_id__grpd', axis='columns', inplace=True)
#orders_test.drop('state_pair__grpd', axis='columns', inplace=True)
#orders_test.drop('seller_city__grpd', axis='columns', inplace=True)
#orders_test.drop('customer_zip_code_prefix__grpd', axis='columns', inplace=True)
orders_test2 = orders_test.copy()    



cols_to_grp = list(pd.unique(big_movers['variabl']))
#already done during QA, just copy it over
#orders_train2['customer_zip_code_prefix__grpd'] = df_plt['grp_col']
#cols_to_grp.remove('customer_zip_code_prefix')

#fix typo in _grpd col
#newcolnames = list(orders_train2.columns)
#newcolnames.index('customer_zip_code_prefix_grpd')
#Out[67]: 62
#
#newcolnames[62]
#Out[68]: 'customer_zip_code_prefix_grpd'
#
#newcolnames[62] = 'customer_zip_code_prefix__grpd'
#
#newcolnames[62]
#Out[70]: 'customer_zip_code_prefix__grpd'
#orders_train2.columns = newcolnames


#create new columns for prediction that leave the big movers specified and lump everything else into one "others" bucket
for var in cols_to_grp:
    new_col_nm = var + '__grpd'
    orders_train2[new_col_nm] = orders_train.apply(lambda rww: flag_big_mover_record(var, rww[var]), axis=1)

#repeat for test dataset
for var in cols_to_grp:
    new_col_nm = var + '__grpd'
    orders_test2[new_col_nm] = orders_test.apply(lambda rww: flag_big_mover_record(var, rww[var]), axis=1)



orders_train2.info()
orders_train2.head()

#orders_train2.loc[orders_train2.customer_zip_code_prefix__grpd == 'All others', 'customer_zip_code_prefix__grpd'] = ' ALL OTHERS'
#orders_train2.loc[orders_train2.customer_city__grpd == 'All others', 'customer_city__grpd'] = ' ALL OTHERS'
#orders_train2.loc[orders_train2.seller_id__grpd == 'All others', 'seller_id__grpd'] = ' ALL OTHERS'
#orders_train2.loc[orders_train2.seller_zip_code_prefix__grpd == 'All others', 'seller_zip_code_prefix__grpd'] = ' ALL OTHERS'
#orders_train2.loc[orders_train2.seller_city__grpd == 'All others', 'seller_city__grpd'] = ' ALL OTHERS'
#orders_train2.loc[orders_train2.state_pair__grpd == 'All others', 'state_pair__grpd'] = ' ALL OTHERS'

#plot boxplots for the big movers and see how they compare to the "all others"
for var in pd.unique(big_movers['variabl']):
    fig1, ax1 = plt.subplots(figsize=(16,9))
    byvar = var +'__grpd'
    #rotn = 45 if var in ['seller_city', 'state_pair', 'customer_zip_code_prefix'] else 90
    rotn = 0
    orders_train2.boxplot(column='fulfill_duration', by=byvar, ax=ax1, rot=rotn, vert=False)
    plt.suptitle(f'Boxplot of fulfill_duration, grouped by {var}')
#    plt.title(var)
    ax1.set_title('')
    ax1.set_xlim(0, 50)
    plt.ylabel(var)
    plt.xlabel('Fulfill duration (days)')
    file_nm = 'output/FulfillDurnModel_EDA_boxplot_by' + var + '.jpg'
    plt.savefig(file_nm, 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)  


#'customer_zip_code_prefix'  == no appreciable diff
#'seller_city', 'state_pair',  





#one hot encoding
ohe_cols = []
grpd_cols = ['customer_city__grpd', 'seller_zip_code_prefix__grpd', 'seller_id__grpd', 'state_pair__grpd', 'seller_city__grpd', 'customer_zip_code_prefix__grpd']
for col in grpd_cols:
    print(f'Encoding column {col}')
    for lvl in orders_train2[col].unique():
        new_col_nm = col[:-6] + '__' + str(lvl).replace(' ', '_')
        orders_train2[new_col_nm] = ( orders_train2[col] == lvl ).astype(int)
        ohe_cols.append(new_col_nm)
        

#repeat for test dataset
for col in grpd_cols:
    print(f'Encoding column {col}')
    for lvl in orders_train2[col].unique():
        new_col_nm = col[:-6] + '__' + str(lvl).replace(' ', '_')
        orders_test2[new_col_nm] = ( orders_test2[col] == lvl ).astype(int)




#cols_to_get_ohe = pd.DataFrame(list(orders_train2.columns))
#cols_to_get_ohe 
#ohe_cols = list(orders_train2.columns)[69:]


#scale the continuous vars
#don't scale boolean vars (dummies from categorical predictors)

orders_train2.to_csv('orders_train2.csv')
orders_test2.to_csv('orders_test2.csv')


# resuming work, read these back in

orders_train2 = pd.read_csv('orders_train2.csv')
orders_test2 = pd.read_csv('orders_test2.csv')


orders_train2.order_estimated_delivery_date = pd.to_datetime(orders_train2.order_estimated_delivery_date)  
orders_train2.order_purchase_timestamp = pd.to_datetime(orders_train2.order_purchase_timestamp)  
orders_train2.order_approved_at = pd.to_datetime(orders_train2.order_approved_at)  
orders_train2.order_delivered_carrier_date = pd.to_datetime(orders_train2.order_delivered_carrier_date)  
orders_train2.order_delivered_customer_date = pd.to_datetime(orders_train2.order_delivered_customer_date)  
orders_train2.ship_limit_initial = pd.to_datetime(orders_train2.ship_limit_initial)  
orders_train2.ship_limit_final = pd.to_datetime(orders_train2.ship_limit_final)  
orders_train2.earliest_review_dt = pd.to_datetime(orders_train2.earliest_review_dt)  
orders_train2.latest_review_dt = pd.to_datetime(orders_train2.latest_review_dt)  

orders_test2.order_estimated_delivery_date = pd.to_datetime(orders_test2.order_estimated_delivery_date)  
orders_test2.order_purchase_timestamp = pd.to_datetime(orders_test2.order_purchase_timestamp)  
orders_test2.order_approved_at = pd.to_datetime(orders_test2.order_approved_at)  
orders_test2.order_delivered_carrier_date = pd.to_datetime(orders_test2.order_delivered_carrier_date)  
orders_test2.order_delivered_customer_date = pd.to_datetime(orders_test2.order_delivered_customer_date)  
orders_test2.ship_limit_initial = pd.to_datetime(orders_test2.ship_limit_initial)  
orders_test2.ship_limit_final = pd.to_datetime(orders_test2.ship_limit_final)  
orders_test2.earliest_review_dt = pd.to_datetime(orders_test2.earliest_review_dt)  
orders_test2.latest_review_dt = pd.to_datetime(orders_test2.latest_review_dt)  

#cols_to_get_ohe = pd.DataFrame(list(orders_train2.columns))
#cols_to_get_ohe 
ohe_cols = list(orders_train2.columns)[69:]


x_train = orders_train2.copy()
y_train = orders_train2[['fulfill_duration']].copy()


x_test = orders_test2.copy()
y_test = orders_test2[['fulfill_duration']].copy()



#x_train = x_train.loc[y_train.fulfill_duration <= 50, :]
#71484
#plt.hist(x_train.loc[y_train.fulfill_duration > 50, 'fulfill_duration'])
#501


x_train['Cust_st_SP'] = (x_train.customer_state == 'SP').astype(int) 
x_train['Cust_st_PA'] = (x_train.customer_state == 'PA').astype(int) 
x_train['Cust_st_BA'] = (x_train.customer_state == 'BA').astype(int) 
x_train['Cust_st_CE'] = (x_train.customer_state == 'CE').astype(int) 
x_train['order_estimated_delivery_mo'] = x_train.order_estimated_delivery_date.dt.month


x_test['Cust_st_SP'] = (x_test.customer_state == 'SP').astype(int) 
x_test['Cust_st_PA'] = (x_test.customer_state == 'PA').astype(int) 
x_test['Cust_st_BA'] = (x_test.customer_state == 'BA').astype(int) 
x_test['Cust_st_CE'] = (x_test.customer_state == 'CE').astype(int) 
x_test['order_estimated_delivery_mo'] = x_test.order_estimated_delivery_date.dt.month


#sum(x_test.customer_state == 'SP')
#x_test.describe()

#top_contnous_pred = ['distance_km', 'est_delivery_time_days', 'states_same_or_diff', 'ttl_freight', 'purchase_yr_and_mo', 'est_delivery_yr_and_mo', 'ttl_pd', 'ttl_wt', 'ttl_price', 'installments_used_ttl', 'ttl_height', 'nbr_sellers', 'ttl_length', 'nbr_photos', 'nbr_products', 'nbr_items', 'ttl_width', 'pmt_mthds_used', 'payment_types_used']
#top_contnous_pred2 = ['lat_customer', 'long_customer', 'lat_seller', 'long_seller', 'approval_time_days']

from sklearn.linear_model import LinearRegression

# ------------------------
# build regression model 1
# ------------------------
pred_cols_lr1 = top_contnous_pred

#x_lr1 = x_train[pred_cols_lr1]
x_train_lr1 = x_train.loc[y_train.fulfill_duration <= 50, pred_cols_lr1]
y_train_lr1= y_train.loc[y_train.fulfill_duration <= 50, :]


mdl_lr1 = LinearRegression(fit_intercept = True)
mdl_lr1.fit(x_train_lr1, y_train_lr1)

mdl_lr1.score(x_train_lr1, y_train_lr1)
#0.272

x_test_lr1 = x_test.loc[:, pred_cols_lr1]
y_test_lr1 = y_test.loc[:, 'fulfill_duration']

mdl_lr1.score(x_test_lr1, y_test_lr1)
#0.228



#mdl_lr1.get_params()
#mdl_lr1.coef_


# -----------------------------------------
# build regression model 2 (Renato's model)
# -----------------------------------------

#pred_rs_mdl = ['Cust_st_SP', 'lat_customer', 'freight_value', 'Cust_st_BA', 'Cust_st_PA', 'long_customer', 'Cust_st_CE', 'order_estimated_delivery_mo']
#tweak the name of the total freight column slightly
pred_rs_mdl = ['Cust_st_SP', 'lat_customer', 'ttl_freight', 'Cust_st_BA', 'Cust_st_PA', 'long_customer', 'Cust_st_CE', 'order_estimated_delivery_mo']
x_train[pred_rs_mdl]


x_train_lr2 = x_train.loc[y_train.fulfill_duration <= 50, pred_rs_mdl]
y_train_lr2= y_train.loc[y_train.fulfill_duration <= 50, :]


mdl_lr2 = LinearRegression(fit_intercept = True)
mdl_lr2.fit(x_train_lr2, y_train_lr2)

mdl_lr2.score(x_train_lr2, y_train_lr2)
#0.198

x_test_lr2 = x_test.loc[:, pred_rs_mdl]
y_test_lr2 = y_test.loc[:, 'fulfill_duration']

mdl_lr2.score(x_test_lr2, y_test_lr2)
#0.166



# -----------------------------------------------------------------------
# build regression model 3 (baseline model, with their current prediction
# -----------------------------------------------------------------------


pred_base_mdl = ['est_delivery_time_days']


x_train_lr3 = x_train.loc[y_train.fulfill_duration <= 50, pred_base_mdl]
y_train_lr3= y_train.loc[y_train.fulfill_duration <= 50, :]


mdl_lr3 = LinearRegression(fit_intercept = True)
mdl_lr3.fit(x_train_lr3, y_train_lr3)

mdl_lr3.score(x_train_lr3, y_train_lr3)
#0.183

x_test_lr3 = x_test.loc[:, pred_base_mdl]
y_test_lr3 = y_test.loc[:, 'fulfill_duration']

mdl_lr3.score(x_test_lr3, y_test_lr3)
#0.154


y_pred_lr1 = mdl_lr1.predict(x_test_lr1)
mse_lr1 = mean_squared_error(y_test_lr1, y_pred_lr1)
#68.8
np.sqrt(mse_lr1)
#8.30    


y_pred_lr3 = mdl_lr3.predict(x_test_lr3)
mse_lr1 = mean_squared_error(y_test_lr3, y_pred_lr1)
#68.8
np.sqrt(mse_lr1)
#8.30    



# --------------------------------------------------------------------
# build random forest regression model with just continuous predictors
# --------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

pred_cols_rf1 = top_contnous_pred

#x_rf1 = x_train[pred_cols_rf1]
x_train_rf1 = x_train.loc[y_train.fulfill_duration <= 50, pred_cols_rf1]
y_train_rf1= y_train.loc[y_train.fulfill_duration <= 50, :]

x_test_rf1 = x_test.loc[:, pred_cols_rf1]
y_test_rf1 = y_test.loc[:, 'fulfill_duration']


# check out max depths for one that doesn't overfit
train_scores = []
test_scores = []
train_rmse = []
test_rmse = []

max_depths = list(range(1, 11)) 
max_depths = max_depths + list(range(12, 32, 2))

for dpth in max_depths:
    print(f'Calculating results for max depth of {dpth}')
    mdl_rf1 = RandomForestRegressor(n_estimators=20, max_depth = dpth, random_state = RANDOM_SEED)
    mdl_rf1.fit(x_train_rf1, y_train_rf1)

    train_scores.append(mdl_rf1.score(x_train_rf1, y_train_rf1))
    test_scores.append(mdl_rf1.score(x_test_rf1, y_test_rf1))

    train_rmse.append( np.sqrt( mean_squared_error(y_train_rf1, mdl_rf1.predict(x_train_rf1) ) ) )
    test_rmse.append( np.sqrt( mean_squared_error(y_test_rf1, mdl_rf1.predict(x_test_rf1) ) ) )


from matplotlib.legend_handler import HandlerLine2D

#plot the RMSEs for training and test, see if we see train keep going down but test rmse level off
line1, = plt.plot(max_depths, train_rmse, 'b', label='Training Data RMSE')
line2, = plt.plot(max_depths, test_rmse, 'r', label='Test Data RMSE')

plt.ylabel('RMSE')
plt.xlabel('max depth')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.show()

test_scores[5:8]
#~0.31 -- better!

plt.plot(max_depths, train_scores, 'b', label='Training Data Score')
plt.plot(max_depths, test_scores, 'r', label='Test Data Score')
plt.ylabel('Score')
plt.xlabel('max depth')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.show()







#mdl_rf2.feature_importances_



# -------------------------------------------------------------------------------
# build random forest regression model with continuous AND categorical predictors
# -------------------------------------------------------------------------------

pred_cols_rf2 = top_contnous_pred + ohe_cols

x_train_rf2 = x_train.loc[y_train.fulfill_duration <= 50, pred_cols_rf2]
y_train_rf2= y_train.loc[y_train.fulfill_duration <= 50, :]

x_test_rf2 = x_test.loc[:, pred_cols_rf2]
y_test_rf2 = y_test.loc[:, 'fulfill_duration']


# check out n_estimators for one that doesn't overfit
train_scores_rf2 = []
test_scores_rf2 = []
train_rmse_rf2 = []
test_rmse_rf2 = []

n_trees = list(range(20, 220, 20)) 

n_trees = [40]

for n_est in n_trees:
    print(f'Calculating results for n_estimators of {n_est}')
    mdl_rf2 = RandomForestRegressor(n_estimators=n_est, max_depth=10, random_state = RANDOM_SEED)
    mdl_rf2.fit(x_train_rf2, y_train_rf2)

    train_scores_rf2.append(mdl_rf2.score(x_train_rf2, y_train_rf2))
    test_scores_rf2.append(mdl_rf2.score(x_test_rf2, y_test_rf2))

    train_rmse_rf2.append( np.sqrt( mean_squared_error(y_train_rf2, mdl_rf2.predict(x_train_rf2) ) ) )
    test_rmse_rf2.append( np.sqrt( mean_squared_error(y_test_rf2, mdl_rf2.predict(x_test_rf2) ) ) )



#baseline using the expected delivery days
np.sqrt(mean_squared_error(y_test_rf2, orders_test2.est_delivery_time_days))

#get feature importances
feature_importances_rf2 = pd.DataFrame( {'Feature_name': list(x_train_rf2.columns), 'Importance': list(mdl_rf2.feature_importances_) } )
feature_importances_rf2 = feature_importances_rf2.sort_values('Importance', ascending=False)
feature_importances_rf2.head(20)
feature_importances_rf2.info()




#for col in x_train_rf2.columns:
#    nbr_nulls = sum(pd.isnull(x_train_rf2[col]))
#    if (nbr_nulls != 0) :
#        print(f'{col} has {nbr_nulls} null values')
#
#for col in x_test_rf2.columns:
#    nbr_nulls = sum(pd.isnull(x_test_rf2[col]))
#    if (nbr_nulls != 0) :
#        print(f'{col} has {nbr_nulls} null values')


#plot the RMSEs for training and test, see if we see train keep going down but test rmse level off
line1, = plt.plot(n_trees, train_rmse_rf2, 'b', label='Training Data RMSE')
line2, = plt.plot(n_trees, test_rmse_rf2, 'r', label='Test Data RMSE')

plt.ylabel('RMSE')
plt.xlabel('# of trees')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.show()

test_scores[5:8]
#~0.31 -- better!

plt.plot(n_trees, train_scores_rf2, 'b', label='Training Data Score')
plt.plot(n_trees, test_scores_rf2, 'r', label='Test Data Score')
plt.ylabel('Score')
plt.xlabel('# of trees')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.show()

    