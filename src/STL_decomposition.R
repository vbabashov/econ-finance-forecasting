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
raw <- readr::read_csv("/Users/vusalbabashov/Desktop/data.csv") 
raw


raw %>%
  gather(key='Product', value = "Value", '5NW':'100FD') %>%
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



weekly_tssible_rdp %>%
  model(STL(log1p(weekly_total)~ season(window=13))) %>%
  components() %>% temp

write.csv(temp,"/Users/vusalbabashov/Desktop/stl_decomp.csv") 
