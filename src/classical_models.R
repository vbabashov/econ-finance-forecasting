install.packages("fpp3", dependencies = TRUE)
library(tidyverse)
library(lubridate)
library(tsibble)
library(feasts)
library(tsfeatures)
library(fable)
library(timeDate)
library(chron)
library(RQuantLib)
library(sugrrants)
library(purrr)
library(broom)
library(glue)
library(GGally)
library(stringr)
library(urca)
library(future)
library(fable.prophet)
library(dplyr)



#read csv file in
raw <- readr::read_csv("/Users/vusalbabashov/Desktop/econ-finance-forecasting/data/data.csv") 
raw


raw %>%
  select(-Site_Coded) %>%
  gather(key='Product', value = "Value", 'ANW':'EFD') %>%
  mutate(Date = dmy(Date)) %>%
  as_tsibble(index=Date, key = c(Region, Site, Product), regular=FALSE) ->raw1
raw1




# Start the Time Series from the January 1st of 2014 and add NA values for missing dates, exclude the last day, December 13 
raw1%>%  
 # filter_index("2014-01-02" ~ "2019-12-13") %>%
  group_by_key() %>%
  complete(Date = seq.Date(ymd("2014-01-01"), ymd("2019-12-13"), by="day")) %>%
  ungroup() ->raw2
raw2


# Create a new tsibble, replace NAs with zeros, and scale by values by million
raw2 %>%
  as_tsibble(index = Date,key = c(Region, Site, Product), regular = TRUE) %>%
  mutate(Value = replace_na(Value,0)) %>%
  select(Date, Region, Site, Product, Value) -> ts_BNDS
ts_BNDS


is_regular(ts_BNDS)
has_gaps(ts_BNDS,  .full = TRUE)
count_gaps(ts_BNDS,.full = TRUE)
scan_gaps(ts_BNDS, .full = TRUE)


# listHolidays()
# stat_holidays = getHolidayList("Canada", as.Date("2014-01-01"), as.Date("2019-12-31"), includeWeekends=FALSE) 
# stat_holidays


# years = c(2014:2019)
# holiday_calendar <- timeDate (c(NewYearsDay(year = years), GoodFriday(year = years), EasterMonday(year = years),
#                                   CAVictoriaDay(year = years), CACanadaDay(year = years), CALabourDay(year = years), 
#                                   CAThanksgivingDay(year = years), ChristmasEve(year = years), ChristmasDay(year = years), BoxingDay(year = years)))


hlist <- c("NewYearsDay","GoodFriday", "Easter", "EasterMonday","CACanadaDay", "CALabourDay", "CAThanksgivingDay", "ChristmasDay", "BoxingDay")
myholidays  <- dates(as.character(holiday(2014:2019,hlist)),format="Y-m-d")
holiday_weeks<-yearweek(as_date(myholidays))



# Add whether day is holiday or not, day of the week, day of the week code, year and week to the tsibble
ts_BNDS %>%
  mutate(DayofWeek = wday(Date, label = TRUE, week_start = 1), DayofWeekNo = wday(Date, week_start = 1)) %>%
  #mutate(Holiday = ts_BNDS_rdc$Date %in% stat_holidays) %>% 
  #mutate(Holiday = is.holiday(ts_BNDS_rdc$Date, stat_holidays)) %>%
  mutate(Date_Holiday = is.holiday(ts_BNDS$Date,myholidays)) %>% 
  mutate(Week = yearweek(Date)) %>%
  mutate(Week_Holiday = if_else(yearweek(Date) %in% holiday_weeks, 1, 0)) %>%
  mutate(Quarter = yearquarter(Date))-> new
new


# summarize weekly total for each RRP
new %>%
  select(-DayofWeek,-DayofWeekNo,-Quarter, -Date_Holiday) %>%
  group_by(Region, Product) %>%
  index_by(year_week = ~ yearweek(.)) %>% # weekly aggregates for each RDP
  summarise (
    weekly_total = sum(Value, na.rm=TRUE),
    weekly_hol   = mean(Week_Holiday)
    ) -> new3

new3 %>%
  ungroup() %>%
  update_tsibble(index=year_week, key = c(Region, Product), regular=TRUE) %>% mutate(Week_Start = as.Date(year_week)) -> weekly_tssible_rdp
weekly_tssible_rdp



# summarize weekly total for each RDC
new %>%
  select(-DayofWeek,-DayofWeekNo,-Quarter, -Date_Holiday) %>%
  group_by_key() %>%
  index_by(year_week = ~ yearweek(.)) %>% # weekly aggregates for each RDC
  summarise (
    weekly_total = sum(Value, na.rm=TRUE),
    weekly_hol   = mean(Week_Holiday)
  ) -> new4
new4 %>%
  ungroup() %>%
  update_tsibble(index=year_week, key = c(Region, Site, Product), regular=TRUE) %>% mutate(Week_Start = as.Date(year_week))-> weekly_tssible_rdc
weekly_tssible_rdc



# scaled_logit <- new_transformation(
#   transformation = function(x, lower=0, upper=1){
#     log((x-lower)/(upper-x))
#   },
#   inverse = function(x, lower=0, upper=1){
#     (upper-lower)*exp(x)/(1+exp(x)) + lower
#   }
# )



## Build Generic Non-seasonal and Seasonal Arima Model for each region

train <- weekly_tssible_rdp %>%
  filter_index("2017 W01" ~ "2019 W20")
train

test <- weekly_tssible_rdp %>%
  filter_index("2019 W21" ~ .)
test



###SeasonaL, and Non-Seasonal Arima, Dynamic Harmonic Regression, SNAIVE

fit_all_v1 <- train %>%
  model(
    #arima = ARIMA(log(weekly_total+1) ~ pdq (d = 0:1) + PDQ (D = 0:1, period = 52) + weekly_hol),
    snaive= SNAIVE(weekly_total ~ lag(52)),
    #harmonic2 = ARIMA(log(weekly_total+1) ~ fourier(K = 2) + PDQ(0, 0, 0) + weekly_hol),
    #harmonic4 = ARIMA(log(weekly_total+1) ~ fourier(K = 4) + PDQ(0, 0, 0) + weekly_hol),
    #harmonic6 = ARIMA(log(weekly_total+1) ~ fourier(K = 6) + PDQ(0, 0, 0) + weekly_hol),
    #harmonic8 = ARIMA(log(weekly_total+1) ~ fourier(K = 8) + PDQ(0, 0, 0) + weekly_hol)
  )
  
fc_all_v1 <- fit_all_v1 %>% forecast(test) 
#fc_all_v1 <- fit_all_v1 %>% forecast(test) %>% hilo(level = c(80, 95))

fc_all_v1 %>% accuracy(weekly_tssible_rdp) %>% select(.model, Region, Product, RMSE)-> xxx
write.csv(xxx,"/Users/vusalbabashov/Desktop/baseline.csv") 


# Auto Plot
fit_all_v1 %>% 
  forecast(test) %>%
  filter(Region=="R1" & Product == "EFD") %>%
  autoplot(filter_index(weekly_tssible_rdp, "2016 W01"~.), level=80) +
  #ggtitle("Forecasts for production") +
  xlab("Year") + ylab("Total Number of Notes")
  #guides(colour = guide_legend(title = "Model"))


### STL Decomposition and Forecasting ####
# First Approach
dcmp <-train %>%
  model(STL(log(weekly_total+1) ~ trend (window=7), robust=TRUE)) %>%
  components() %>%
  select (-.model)

dcmp %>% 
  #model(SNAIVE(season_year)) %>%
  model(NAIVE(season_adjust)) %>%
  forecast(test)

#Second Approach
train %>%
  model(stlf=decomposition_model(
    #STL(log(weekly_total+1) ~ trend (window=7), robust=TRUE),
    #NAIVE(season_adjust)
    STL(log(weekly_total+1) ~ trend(window = 7),robust = TRUE),
    #RW(season_adjust ~ drift()), SNAIVE(season_year)
    ARIMA(season_adjust ~ pdq (d = 0:1) + PDQ (0,0,0)), SNAIVE(season_year)
  )) %>%
  forecast(test) -> fc_dcmp

fc_dcmp %>% accuracy(weekly_tssible_rdp) %>% select(.model, Region, Product, MAE)-> x
write.csv(x,"/Users/vusalbabashov/Desktop/MAE_dcmp.csv") 

#Plot
fc_dcmp %>%
  filter(Region=="R1" & Product == "EFD") %>%
  autoplot(filter_index(weekly_tssible_rdp, "2016 W01"~.), level=80) +
  #ggtitle("Forecasts for production") +
  xlab("Year") + ylab("Total Number of Notes")
  #guides(colour = guide_legend(title = "Model"))



### Hierarchical TS Forecasting ####
# Reconciliation of non-normal forecasts is not yet supported
train <- weekly_tssible_rdc %>%
  filter_index("2016 W01" ~ "2018 W52")
train

test <- weekly_tssible_rdc %>%
  filter_index("2019 W01" ~ .)
test


bnds_agg <- train %>%
  aggregate_key(Product*(Region/Site), Value = sum(weekly_total)) %>%
  print(n = 100)


fit_agg <- bnds_agg %>%
  model(arima = ARIMA(log(Value+1) ~ pdq (d=0:1) + PDQ (D=0:1, period = 52))) %>%
  reconcile(arima_adjusted = min_trace(arima))

fit_agg %>%
  forecast(h="50 weeks", bootstrap=TRUE) -> fc_agg

fc_agg %>% accuracy(weekly_tssible_rdc) %>% select(.model, Region, Product, MAE)-> xxx
write.csv(xxx,"/Users/vusalbabashov/Desktop/MAE_agg.csv") 




#######################################################################################

# fc <- fit_all_v1%>%   
#   forecast(h = "2 weeks") %>%
#   hilo(level = c(80, 95)) %>%
#   print(n=500)
# 
# augment(fit_all_v1) %>%
#   filter(Region == "R1"& Product == "50FD") %>%
#   ggplot(aes(x = year_week)) +
#   geom_line(aes(y = weekly_total)) +
#   geom_line(aes(y = .fitted)) +
#   xlab("Year") + ylab(NULL) +
#   ggtitle("Total Value in $") +
#   guides(colour=guide_legend(title="Forecast"))





