library(dplyr)
library(data.table)
library(stats)


#set params
# -------------
setwd("C:/Users/ashle/Documents/Personal Data/Northwestern/2019-04  fall MSDS498_Sec56 Capstone/Git_repo/Predictive_models")

flnm_raw_data <- 'Merged_dataset_w_LatLong.csv'
# -------------
# end of params


df_raw_data <- read.csv(flnm_raw_data, stringsAsFactors = FALSE)

#convert dates from factor to date
df_raw_data$order_purchase_timestamp = as.POSIXct(df_raw_data$order_purchase_timestamp, format="%Y-%m-%d %H:%M:%S")    
df_raw_data$order_approved_at = as.POSIXct(df_raw_data$order_approved_at, format="%Y-%m-%d %H:%M:%S")
df_raw_data$order_delivered_carrier_date = as.POSIXct(df_raw_data$order_delivered_carrier_date, format="%Y-%m-%d %H:%M:%S")
df_raw_data$order_delivered_customer_date = as.POSIXct(df_raw_data$order_delivered_customer_date, format="%Y-%m-%d %H:%M:%S")
df_raw_data$order_estimated_delivery_date = as.POSIXct(df_raw_data$order_estimated_delivery_date, format="%Y-%m-%d %H:%M:%S")
df_raw_data$shipping_limit_date = as.POSIXct(df_raw_data$shipping_limit_date, format="%Y-%m-%d %H:%M:%S")

#colnames(df_raw_data)
#head(df_raw_data$order_purchase_timestamp)
dt_cols <- c('order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date')
#df_raw_data[1:10,dt_cols]
#str(df_raw_data)

# strptime(df_raw_data[1:10, 'order_purchase_timestamp'], "%Y-%m-%d %H:%M:%S")
# strptime(x, "%Y-%m-%d %H:%M:S")

dt_raw_data <- data.table(df_raw_data)
glimpse(dt_raw_data)
str(dt_raw_data)
dt_raw_data$X <- NULL   #delete this column; it is junk left over from the index when exporting the dataset


#checks on approval date 
#approved dates before order purchase?
dt_raw_data[order_purchase_timestamp> order_approved_at]
#0 rows found; perfect



#checks on carrier delivery date 
#delivered to carrier before approved?
dt_to_carrier_b4_approval <- dt_raw_data[order_approved_at > order_delivered_carrier_date]
nrow(dt_to_carrier_b4_approval)  #1603 rows found  (only 819 rows found when just considering dates) 
  #TODO?? treat these as outliers and exclude from training&test sets??

#delivered to carrier before purchased?
dt_to_carrier_b4_purchased <- dt_raw_data[order_purchase_timestamp > order_delivered_carrier_date]
nrow(dt_to_carrier_b4_purchased)  #189 rows found  (only 2 when ignoring time) 
  #TODO treat these as outliers and exclude from training&test sets



#checks on customer delivery date 
#delivered to customer before delivered to carrier?
dt_to_cust_b4_carrier <- dt_raw_data[order_delivered_carrier_date >= order_delivered_customer_date]
nrow(dt_to_cust_b4_carrier)   #69 rows found if >= is used, only 59 rows if > is used
  #TODO treat these as outliers and exclude from training&test sets

#delivered to customer before approved?
dt_to_cust_b4_approval <- dt_raw_data[order_approved_at > order_delivered_customer_date]
nrow(dt_to_cust_b4_approval)   #73 rows
  #TODO?? treat these as outliers and exclude from training&test sets??

#delivered to customer before purchased?
dt_raw_data[order_purchase_timestamp > order_delivered_customer_date]
#0 rows found; perfect


nrow(dt_raw_data)  #118,318


sum(is.na(dt_raw_data$order_approved_at))  #17
sum(is.na(dt_raw_data$order_delivered_customer_date))  #2588
#get counts by order status  (where the delivered to cust date is not null)   
dt_raw_data[ ! is.na(order_delivered_customer_date), .N, by=order_status]
  #115,723 of the recrds w/ cust_deliv date are "delivered" status.   7 others are cancelled - not too bad, just ignore.

#how often is the status delivered but we don't have a delivered to cust date
dt_raw_data[ order_status == 'delivered' & is.na(order_delivered_customer_date)] %>% nrow()
#only 8 times; no big deal


#focus on the delivered orders only (exclude shipped, approved, unavailable, ....)
dt_raw_2 <- dt_raw_data[ ! is.na(order_delivered_customer_date) ]   #115,730

#remove ones w/ no carrier delivered date, or funky carrier date vs. delivery date
dt_raw_2 <- dt_raw_2[ ! is.na(order_delivered_carrier_date) ]   #1 row removed
dt_raw_2 <- dt_raw_2[ order_delivered_carrier_date <= order_delivered_customer_date ]  #115,670

#remove ones w/ no approval date or funky approval date relative to delivery date
dt_raw_2 <- dt_raw_2[ ! is.na(order_approved_at) ]   #17 rows removed

#investigate further??
#dt_raw_2[ order_approved_at > order_delivered_customer_date ] %>% nrow()     #73 rows
#dt_raw_2[ order_approved_at > order_delivered_carrier_date ] %>% nrow()      #1593 rows

dt_funky_approval_dts <- dt_raw_2[ order_approved_at > order_delivered_customer_date | order_approved_at > order_delivered_carrier_date ]  #1593 rows

#are these particularly interesting or overindexing in some way that would be important to look into?
dt_funky_approval_dts %>% dplyr::filter( order_delivered_customer_date > order_estimated_delivery_date ) %>% nrow()
#35 out of 1593 missed their targeted delivery date

#vs 9032 out of 115k missed their delivery date in the nonfunky set
dt_raw_2[ order_approved_at <= order_delivered_customer_date & order_approved_at <= order_delivered_carrier_date ] %>% dplyr::filter( order_delivered_customer_date > order_estimated_delivery_date ) %>% nrow()

#9032/115661  #7.8%
#35/1593  #2.2%  ==> just skip these for now


dt_raw_2 <- dt_raw_2[ order_approved_at <= order_delivered_customer_date & order_approved_at <= order_delivered_carrier_date ]   #114,060




#check that dates are fully populated now
for (col in dt_cols){
  print(paste0(col, ' has ', sum(is.na(dt_raw_2[,..col])), ' NAs'))}
#good, they are all populated





# add field for distance between seller and customer
# ==================================================

#this function does spherical geometry to return the distance between the 2 pts in kilometers
distance_calcn_rough <- function(start_lat, start_long, end_lat, end_long){

  #check for missing values and quit if any are found
  if (is.na(start_lat) | is.na(start_long) | is.na(end_lat) | is.na(end_long) )
    return(NaN)
  
  #convert from radians to radians
  start_lat <- start_lat * pi / 180.0
  start_long <- start_long * pi / 180.0
  end_lat <- end_lat * pi / 180.0
  end_long <- end_long * pi / 180.0
  
  cosine_val <- sin(start_lat)*sin(end_lat) + cos(start_lat)*cos(end_lat)*cos(start_long - end_long)
  
  if (cosine_val > 1 | cosine_val < -1){ cosine_val <- round(cosine_val, 6) }     #round it off so we aren't losing cases due to machine precision issues (esp. when cosine_val ~1 b/c they are at the exact same location)

  if (cosine_val > 1 | cosine_val < -1){
    rtrn_val <-NaN}
  else {
    rtrn_val <- 6371.01*acos(cosine_val)}
  
  return(rtrn_val)
}      

# printrough <- function(start_lat, start_long, end_lat, end_long){
#   
#   rtrn_val <- paste0(start_lat, ', ', start_long, ', ', end_lat, ', ', end_long)
#   rtrn_val <- paste0(rtrn_val, '\n', typeof(start_lat), ', ', typeof(start_long), ', ',typeof(end_lat), ', ', typeof(end_long))
#   return(rtrn_val)
# }      


#calculate the distance
dt_raw_2$distance_calcn_rough <- apply(dt_raw_2[, c('lat_seller', 'long_seller', 'lat_customer', 'long_customer')], MARGIN=1, FUN=function(rww){distance_calcn_rough(rww[1], rww[2], rww[3], rww[4])} )

#checks
# sum(is.na(dt_raw_2$lat_customer))  #297
# sum(is.na(dt_raw_2$long_customer))  #297
# sum(is.na(dt_raw_2$lat_seller))    #259
# sum(is.na(dt_raw_2$long_seller))   #259
# 
# sum(is.na(dt_raw_2$distance_calcn_rough))  #558

hist(dt_raw_2$distance_calcn_rough)
#vary up to a point or two over 3000k but most are 1500 or less.   Brazil is 4,395 km (2,731 mi) N – S and 4,320 km (2,684 mi) E – W  so this sounds reasonable except maybe a few data errors
  #Read more: https://www.nationsencyclopedia.com/Americas/Brazil-LOCATION-SIZE-AND-EXTENT.html#ixzz62rU23H86

boxplot(dt_raw_2$distance_calcn_rough)
#looks like 5 pts over 3500;   were some outliers on the map so get rid of them

par(mfrow=c(2,2))
boxplot(dt_raw_2$lat_customer, main='Lat_Customer')
boxplot(dt_raw_2$long_customer, main='Long_Customer')
boxplot(dt_raw_2$lat_seller, main='Lat_Seller')
boxplot(dt_raw_2$long_seller, main='Long_Seller')

#Brazil covers "every latitude you can name between roughly 5.2Â° north to 33.7Â° south, and every longitude between about 34.8Â° and 73.9Â° west"  ==> ones outside of this are errors?  intl custs??
  #the sellers are all in the ranges expected for Brazil.  The customers have a few outliers.  Might want to exclude them or flag them w/ a categorical variable b/c their shipping times are probably very different (by necessity, will be by boat or  by plane, which would be a lot longer for boat and ?? for plane)
  #sellers outside of expected locations
  # dt_raw_2[ lat_seller>6 | lat_seller < -34 ]  #0, as noted on boxplot
  # dt_raw_2[ long_seller>-34 | long_seller < -74 ]  #0, as noted on boxplot

  #custs outside of expected locations
  # dt_raw_2[ long_customer>-34 | long_customer < -74 ]  #9, as noted on boxplot
  # dt_raw_2[ lat_customer>6 | lat_customer < -34 ]  #8, as noted on boxplot
  # dt_raw_2[ long_customer>-34 | long_customer < -74 | lat_customer>6 | lat_customer < -34 ]  #9, so the exceptional longitudes include all the exceptional latitudes

dt_raw_2[ long_customer>-34 | long_customer < -74 | lat_customer>6 | lat_customer < -34 ]$distance_calcn_rough <- NaN
hist(dt_raw_2$distance_calcn_rough)   #now it maxes out at 3500, much cleaner





#misc
#dt_raw_2 %>% colnames()
dt_raw_2[, .N, by=product_category_name_english] %>% arrange(N)   #72 categories, with 2 to 11k order-items
dt_raw_2[, .N, by=seller_state] %>% arrange(N)  #22 states, with SP way dominant at 81k, MG second biggest at 8.9k, ...
dt_raw_2[, .N, by=customer_state] %>% arrange(N)  #27 states, with SP still leading but less so (48k); RJ next at 14.8k, MG at 13k, ...






# ========================================================================================
#    try some predictive models for delivery time ttl, delivery time carrier to cust, etc
# ========================================================================================


#calculate the elapsed time for delivery, from purchase to delivery
#-------------------------------------------------------------------
dt_raw_2$delivery_time_ttl <- dt_raw_2$order_delivered_customer_date - dt_raw_2$order_purchase_timestamp
#dt_raw_2[1:10, c('order_delivered_customer_date', 'order_purchase_timestamp', 'delivery_time_ttl')]
#convert from a "difftime" datatype to an integer value (holding a # of hours)
dt_raw_2$delivery_time_ttl <- as.numeric(dt_raw_2$delivery_time_ttl, units="hours")
str(dt_raw_2)



#create a flag for the cases where the estimated delivery date was missed.  Would be good to do a logistic regression/binomial classifier on missing the date
#adding 86400 (seconds, ie. one day) b/c the estimate has no time on it, and a delivery at some point on that day would be considered meeting the prediction
dt_raw_2$estimated_delivery_wrong <- dt_raw_2$order_delivered_customer_date > dt_raw_2$order_estimated_delivery_date+86400

sum(dt_raw_2$estimated_delivery_wrong)  #7532 records (note the distinct orders will be a little lower)
dt_raw_2 %>% filter( estimated_delivery_wrong==TRUE) %>% select(order_id) %>% unique() %>% nrow()   #6510 distinct orders



numeric_cols <- select_if(dt_raw_2, is.numeric) %>% colnames()
numeric_cols <- numeric_cols[ ! ( numeric_cols %in% c('Rww_ID', 'order_item_id', 'payment_sequential', 'product_name_lenght', 'product_description_lenght') ) ]


corr_mtrx <- cor(dt_raw_2[, ..numeric_cols], use="complete.obs")
par(mfrow=c(1,1))
corrplot::corrplot(corr_mtrx)

dlvry_time_corrs <- corr_mtrx[17,]
nrow(corr_mtrx)


#create a simple regression model predicting ttl delivery time
mdl_ttl_reg <- stats::lm(delivery_time_ttl ~ distance_calcn_rough + review_score, data=dt_raw_2)

summary(mdl_ttl_reg)
par(mfrow=c(2,2))
plot(mdl_ttl_reg)
par(mfrow=c(1,1))


plot(mdl_ttl_reg$residuals)
coef(mdl_ttl_reg)

#residuals not distributed as desired; way to 
mdl_ttl_reg2 <- stats::lm(log(delivery_time_ttl) ~ distance_calcn_rough + review_score, data=dt_raw_2)

summary(mdl_ttl_reg2)
coef(mdl_ttl_reg2)
par(mfrow=c(2,2))
plot(mdl_ttl_reg2)
par(mfrow=c(1,1))

#much better looking residual plots for residuals vs fitted, normality (QQplot)
#still very low R2  ==> need more info, more predictors to account for addnl sources of variability





# predict elapsed time for the carrier's part of the process (from delivery to carrier to delivery to customer)
# =============================================================================================================

#calculate the elapsed time for the carrier's part of the process (from delivery to carrier to delivery to customer)
# -------------------------------------------------------------------------------------------------------------------
dt_raw_2$delivery_time_carrier_to_cust <- dt_raw_2$order_delivered_customer_date - dt_raw_2$order_delivered_carrier_date
#dt_raw_2[1:10, c('order_delivered_customer_date', 'order_delivered_carrier_date', 'delivery_time_carrier_to_cust', 'delivery_time_ttl')]
#convert from seconds to hours (including fractions of hours)
dt_raw_2$delivery_time_carrier_to_cust <- as.numeric(dt_raw_2$delivery_time_carrier_to_cust, units="secs")/3600.00
#str(dt_raw_2)
# dt_raw_2[1:10, 'delivery_time_carrier_to_cust']

# amt of time for other steps of process
dt_raw_2$delivery_time_order_to_carrier <- dt_raw_2$delivery_time_ttl - dt_raw_2$delivery_time_carrier_to_cust
summary(dt_raw_2[,c('delivery_time_order_to_carrier', 'delivery_time_carrier_to_cust', 'delivery_time_ttl' )])
  #notice 75% of them (3rd quartile) are at carrier in 99hrs, at cust in 376  ==> LOT of orders with a really long tail


par(mfrow=c(1,3))
boxplot(dt_raw_2$delivery_time_order_to_carrier, main='From purchase to carrier pickup', ylim=c(0, 5000))
boxplot(dt_raw_2$delivery_time_carrier_to_cust, main='From carrier pickup to cust delivery', ylim=c(0, 5000))
boxplot(dt_raw_2$delivery_time_ttl, main='From start to finish', ylim=c(0, 5000))
par(mfrow=c(1,1))

par(mfrow=c(3,1))
hist(dt_raw_2$delivery_time_order_to_carrier, main='From purchase to carrier pickup', xlim=c(0, 5000))
hist(dt_raw_2$delivery_time_carrier_to_cust, main='From carrier pickup to cust delivery', xlim=c(0, 5000))
hist(dt_raw_2$delivery_time_ttl, main='From start to finish', xlim=c(0, 5000))
par(mfrow=c(1,1))


#factors affecting delivery time 
    # multiple items per order- do all order items have the same carrier pickup date?
    # seller?  --  seller location, if sellers have multiple and we can figure out the location for this order??
    # shipper (don't know, seller might be a proxy for this)
    # customer (esp. location)
    # product category?
    # pay in installments?



dt_raw_2[, .(avg_ttl=mean(delivery_time_ttl), sd_ttl=sd(delivery_time_ttl), n_ttl=.N), by=customer_state] %>% arrange(desc(avg_ttl))

boxplot(delivery_time_ttl ~ product_category_name_english, data=dt_raw_2)  #only one or two that seem unusually hi or low
  boxplot(delivery_time_order_to_carrier ~ product_category_name_english, data=dt_raw_2)  #only one or two that seem unusually hi or low
  boxplot(delivery_time_carrier_to_cust ~ product_category_name_english, data=dt_raw_2)  #even more consistent
boxplot(delivery_time_ttl ~ customer_state, data=dt_raw_2)
boxplot(delivery_time_ttl ~ seller_state, data=dt_raw_2)   #often similar, maybe create a related var like "AM", "RO", "MG", "" (these 2 have significantly longer tails), all others"
#too nitty gritty to see much     boxplot(delivery_time_ttl ~ customer_city, data=dt_raw_2)
#this plot is too busy but try making a table and categorizing seller into "quick",.., "fast";  "quick but variable", quick and dependable; med and variable, ...; boxplot(delivery_time_ttl ~ seller_id, data=dt_raw_2)   #often similar, maybe create a related var like "AM", "RO", "MG", "" (these 2 have significantly longer tails), all others"
#product_id, seller_id, 

#bucket them by value  and then try
#boxplot(delivery_time_ttl ~ payment_value, data=dt_raw_2)  #only one or two that seem unusually hi or low





#check delivery variation for items within a single order?
order_delivery <- dt_raw_2 %>% group_by(order_id) %>% dplyr::summarise(min_delivery_dt=min(order_delivered_customer_date),
                                                                       max_delivery_dt=max(order_delivered_customer_date)) %>%
                          filter(min_delivery_dt!=max_delivery_dt)
#0 rows found; perfect, all items are delivered at the same time 
  #TODO:   recommend to them to split up orders??   are the approved dates diff?

#dt_raw_2 %>% group_by(order_id) %>% dplyr::summarise(min_delivery_dt=min(order_delivered_customer_date),
#                                                     max_delivery_dt=max(order_delivered_customer_date)) %>% head(30)


#dt_raw_2 %>% colnames()


#Shows the seller shipping limit date for handling the order over to the logistic partner.
#TODO:  does missing the shipping limit date ever happen?   if so, how does it impact the ontime delivery of the order
          #primarily for the shipping limit_final, but check initial, and/or multiple misses???
#dt_raw_2$missed_shipping_limit_dt_flag <- 
shipping_limit_probs <- dt_raw_2 %>% group_by(order_id) %>% 
                summarize(max_shipping_limit=max(shipping_limit_date),
                          delivered_to_carrier_dt=max(order_delivered_carrier_date),
                          delivered_to_cust_dt=max(order_delivered_customer_date),
                          est_cust_delivery_dt=max(order_estimated_delivery_date)) %>%
                    mutate(shipping_limit_problem_flag=case_when( max_shipping_limit < delivered_to_carrier_dt ~ 1, TRUE ~ 0),
                           late_delivery_flag=case_when( delivered_to_cust_dt > est_cust_delivery_dt+86400 ~ 1, TRUE ~ 0))

sum(shipping_limit_probs$shipping_limit_problem_flag)
#8690 orders had at least one item miss their shipping limit (cutoff)

sum(shipping_limit_probs$late_delivery_flag)
#6510 orders were delivered late


#odds of delivering late if shipping limit is missed?
shipping_limit_probs %>% filter(shipping_limit_problem_flag == 1 & late_delivery_flag == 1 ) %>% nrow()
#1814 / 8690  # 20.87%

#vs odds of late delivery for an order received at the shipper on time
shipping_limit_probs %>% filter(shipping_limit_problem_flag == 0) %>% nrow()   #86,396
shipping_limit_probs %>% filter(shipping_limit_problem_flag == 0 & late_delivery_flag == 1 ) %>% nrow()  #4696
#4695/86396   # 5.4%


#TODO:  check if multiple installments/multiple pmt methods slows down approval    orders.order_approval defn: Shows the payment approval timestamp.
multiple_items_probs <- dt_raw_2 %>% group_by(order_id) %>%
                            summarize(
                                  nbr_items=n(),
                                  purchase_dt=max(order_purchase_timestamp),
                                  approval_dt=max(order_approved_at),
                                  delivered_to_carrier_dt=max(order_delivered_carrier_date),
                                  delivered_to_cust_dt=max(order_delivered_customer_date),
                                  est_cust_delivery_dt=max(order_estimated_delivery_date)) %>%
                              mutate(approval_time = approval_dt - purchase_dt,
                                     late_delivery_flag=case_when( delivered_to_cust_dt > est_cust_delivery_dt+86400 ~ 1, TRUE ~ 0),
                                     est_time = est_cust_delivery_dt - purchase_dt)

multiple_items_probs$approval_time <- as.numeric(multiple_items_probs$approval_time, units='secs')/86400
multiple_items_probs$est_time <- as.numeric(multiple_items_probs$est_time, units='days')
boxplot(approval_time ~ nbr_items, data=multiple_items_probs, main='Approval time as a function of item count', ylab='Approval time (days)', xlab='Items per order')

summary(multiple_items_probs)
  #amazing, mean and median for est delivery time is ~23 DAYS!  not exactly Amazon-like turnaround!!
  #note the approval time is less than a day in 75% of the cases
hist(multiple_items_probs$approval_time, breaks=seq(0,31,1/24))   #with 20 divisions per day (0.05 gap size), we get ~ hourly buckets
#heavily skewed to an hour or two
ecdf(multiple_items_probs$approval_time)(1)  #empirical cumulative distribution.   1 day is the 83rd percentile of the time distro (83% of the orders have approval time of 1 day or less)

#odds of late delivery with approval time > 1 day vs. <1 day  (90? 95% of cases)
multiple_items_probs %>% filter(approval_time <= 1) %>% nrow()   #79,181
multiple_items_probs %>% filter(approval_time <= 1 & late_delivery_flag==1) %>% nrow()   #5,161
#5161/79181 #  6.52%

multiple_items_probs %>% filter(approval_time > 1) %>% nrow()   #15,905
multiple_items_probs %>% filter(approval_time > 1 & late_delivery_flag==1) %>% nrow()   #1,349
#1349/15905 #  8.48%

  #some increase in odds of being late for orders that take more than 1 day to approve



# check if multiple items increases the odds of late delivery
late_delivery_odds_by_item_cnt <- multiple_items_probs %>% group_by(nbr_items) %>%
                                      summarize(ttl_orders=n(),
                                                orders_late=sum(late_delivery_flag)) %>%
                                        mutate(pct_late=orders_late/ttl_orders)

plot(late_delivery_odds_by_item_cnt$nbr_items, late_delivery_odds_by_item_cnt$pct_late)
View(late_delivery_odds_by_item_cnt)
#actually it's the opposite - more items tends to have higher?!?! ontime delivery - is est delivery date padded more??
  
boxplot(est_time ~ nbr_items, data=multiple_items_probs, main='Estimated delivery time vs items per order', ylab='Estimated delivery time (days)', xlab='Items per order')
#very slight increase in estimated time for higher # of items but surprisingly little


multiple_items_probs$est_time_daybuckets <- ceiling(multiple_items_probs$est_time)

late_deliv_by_est_time <- multiple_items_probs %>% group_by(est_time_daybuckets) %>%
                            summarize(orders=n(),
                                      orders_late=sum(late_delivery_flag)) %>%
                              mutate(pct_late=orders_late/orders)

plot(late_deliv_by_est_time$est_time_daybuckets, late_deliv_by_est_time$pct_late, main='% Late Delivery by Estimated Delivery Time', xlab='Estimated Delivery Time (days)', ylab='Percent of orders delivered late')
  #not surprisingly, the longer cushion is given on the est time to deliver, the less likely it is late




#TODO:  can we predict low sentiment / review score  as a function of ontime delivery, seller, # of photos, product, price, # of reviews, avg score on other reviews, "buy local" (same state/city seller-cust), avg score on other reviews, ...???
#TODO:  can we look at all first time custs and compare the odds of a 2nd order for happy reviewers vs. unhappy reviewers?
#TODO:  can we do time series analysis and see how a negative review changes trajectory????  (quantify $ value of a negative review!!)
#TODo:  if the review answer date is after the delivery, then it seems more likely to reflect the delivery performance than predict it




numeric_cols2 <- c(numeric_cols, 'delivery_time_carrier_to_cust')
# numeric_cols3 <- append(numeric_cols, 'delivery_time_carrier_to_cust')
# numeric_cols3 == numeric_cols2

# customer_zip_code_prefix
# customer_city
# customer_state
# payment_installments


corr_mtrx <- cor(dt_raw_2[, ..numeric_cols2], use="complete.obs")
par(mfrow=c(1,1))
corrplot::corrplot(corr_mtrx)
corr_mtrx[19,]

hist(dt_raw_2$delivery_time_carrier_to_cust)



mdl_c2c_reg <- lm(delivery_time_carrier_to_cust ~ distance_calcn_rough + review_score, data=dt_raw_2)

#freight_value, price vs. sum pmt values - do these always balance out?
#distro of order status     "customer_id"                   "order_status" - hist?
