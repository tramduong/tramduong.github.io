---
layout: page
title: Stock Analysis Trading Strategies
description: >
  This page contains the stock strategies and analysis
hide_description: true
sitemap: false
---

<style>

.banner {
  box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
  center;
}

.justify {
  text-align: justify;
}

.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}

* {
  box-sizing: border-box;
}

.column25 {
  float: left;
  width: 25%;
  padding: 10px;
}

.column30 {
  float: left;
  width: 30%;
  padding: 10px;
}

.column40 {
  float: left;
  width: 40%;
  padding: 10px;
}

.column50 {
  float: left;
  width: 50%;
  padding: 10px;
}

.column60 {
  float: left;
  width: 60%;
  padding: 10px;
}

.column70 {
  float: left;
  width: 70%;
  padding: 10px;
}

.column75 {
  float: left;
  width: 75%;
  padding: 10px;
}

.row:after {
  content: "";
  display: table;
  clear: both;
}

@media screen and (max-width: 600px) {
  .column25 {
    width: 100%;
  }
  .column30 {
    width: 100%;
  }
  .column40 {
    width: 100%;
  }
  .column50 {
    width: 100%;
  }
  .column60 {
    width: 100%;
  }
  .column70 {
    width: 100%;
  }
  .column75 {
    width: 100%;
  }
}

.button {
  display: block;
  margin-left: auto;
  margin-right: auto;
  center;
  width: 175px;
}

.button:hover{
  position: relative;
  top: -1px;
  box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 10px 0 rgba(0, 0, 0, 0.15);
}

.button_smaller {
  display: block;
  margin-left: auto;
  margin-right: auto;
  center;
  width: 150px;
}

.button_smaller:hover{
  position: relative;
  top: -1px;
  box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 10px 0 rgba(0, 0, 0, 0.15);
}

.button_smallest {
  display: block;
  margin-left: auto;
  margin-right: auto;
  center;
  width: 110px;
}

.button_smallest:hover{
  position: relative;
  top: -1px;
  box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 10px 0 rgba(0, 0, 0, 0.15);
}

</style>

* This unordered list will be replaced by the table of contents
{:toc}


In this project, I focus on Analysis of trending stocks including change in price over time, daily returns, and define trading strategies for different stock, using sharpe ratio,macd, oversold, overbought, and buy&hold.<br>
For a more detailed explanation, check out [My Github](https://github.com/tramduong/Data-Science-Portfolio/blob/master/Feature%20Engineering%20Projects/Doc/Stock_FE%26TradingStrategy.ipynb)
{:.note}

<br>


<img src="/assets/img/data_exploratory/stock.jpg"  alt="Portfolio Banner" class="banner">


<br>

##  Stock Tranding Strategies & EDA

Tram Duong
<br>September 28, 2020

### Load the necessary packages and modules


```R
#install.packages("PerformanceAnalytics")
```


```R
library(dplyr) # or just dplyr
library(quantmod)
library(TTR)
library(PerformanceAnalytics)
library("IRdisplay")
library(ggplot2)
Sys.setenv(TZ = "UTC")
```


    Attaching package: 'dplyr'

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

    Warning message:
    "package 'quantmod' was built under R version 3.6.3"Loading required package: xts
    Warning message:
    "package 'xts' was built under R version 3.6.3"Loading required package: zoo
    Warning message:
    "package 'zoo' was built under R version 3.6.2"
    Attaching package: 'zoo'

    The following objects are masked from 'package:base':

        as.Date, as.Date.numeric


    Attaching package: 'xts'

    The following objects are masked from 'package:dplyr':

        first, last

    Loading required package: TTR
    Warning message:
    "package 'TTR' was built under R version 3.6.3"Registered S3 method overwritten by 'quantmod':
      method            from
      as.zoo.data.frame zoo
    Version 0.4-0 included new data defaults. See ?getSymbols.
    Warning message:
    "package 'PerformanceAnalytics' was built under R version 3.6.3"
    Attaching package: 'PerformanceAnalytics'

    The following object is masked from 'package:graphics':

        legend



I selected 3 companies from the most active stock list in yahoo finance: General Electric, Tesla and Apple.


```R
getSymbols(c("GE","TSLA","AAPL"))
```

    'getSymbols' currently uses auto.assign=TRUE by default, but will
    use auto.assign=FALSE in 0.5-0. You will still be able to use
    'loadSymbols' to automatically load data. getOption("getSymbols.env")
    and getOption("getSymbols.auto.assign") will still be checked for
    alternate defaults.

    This message is shown once per session and may be disabled by setting
    options("getSymbols.warning4.0"=FALSE). See ?getSymbols for details.




<ol class=list-inline>
	<li>'GE'</li>
	<li>'TSLA'</li>
	<li>'AAPL'</li>
</ol>




```R
GE_df <- GE
TSLA_df <- TSLA
APPL_df <- AAPL
```

### Stock data transformation
* These common stock data transformation can be handled easily by the functions in the quantmod library


```R
transformation <- function(df){
        # Returns from Open to Close, Hi to Close, or Close to Close
        df$OpCl <- OpCl(df)
        df$OpOp <- OpOp(df)
        df$HiCl <- HiCl(df)
        df$ClCl <- ClCl(df)

        df$pcntOpCl1 <- Delt(Op(df),Cl(df),k=1)
        df$pcntOpCl2 <- Delt(Op(df),Cl(df),k=2)
        df$pcntOpCl3 <- Delt(Op(df),Cl(df),k=3)

        #One period lag of the close
        df$lagCl <- Lag(Cl(df))
        df$lag2Cl <- Lag(Cl(df),2)  
        df$lag3Cl <- Lag(Cl(df),3)

        # Move up the OpCl by one period
        df$nextOpCl <- Next(OpCl(df))
    return(df)
}

```


```R
GE_df1<-transformation(GE_df)
TSLA_df1<-transformation(TSLA_df)
APPL_df1<-transformation(APPL_df)
```


```R
head(GE_df1)
```


<table>
<thead><tr><th scope=col>GE.Open</th><th scope=col>GE.High</th><th scope=col>GE.Low</th><th scope=col>GE.Close</th><th scope=col>GE.Volume</th><th scope=col>GE.Adjusted</th><th scope=col>OpCl</th><th scope=col>OpOp</th><th scope=col>HiCl</th><th scope=col>ClCl</th><th scope=col>pcntOpCl1</th><th scope=col>pcntOpCl2</th><th scope=col>pcntOpCl3</th><th scope=col>lagCl</th><th scope=col>lag2Cl</th><th scope=col>lag3Cl</th><th scope=col>nextOpCl</th></tr></thead>
<tbody>
	<tr><td>35.97115     </td><td>36.68269     </td><td>35.94231     </td><td>36.50962     </td><td>44951700     </td><td>23.33968     </td><td> 0.0149693283</td><td>          NA </td><td>-0.004718192 </td><td>           NA</td><td>           NA</td><td>           NA</td><td>           NA</td><td>      NA     </td><td>      NA     </td><td>      NA     </td><td>-0.0057940898</td></tr>
	<tr><td>36.50962     </td><td>36.53846     </td><td>36.00962     </td><td>36.29808     </td><td>32540300     </td><td>23.20444     </td><td>-0.0057940898</td><td> 0.014969328 </td><td>-0.006578903 </td><td>-0.0057940898</td><td> 0.0090885049</td><td>           NA</td><td>           NA</td><td>36.50962     </td><td>      NA     </td><td>      NA     </td><td>-0.0002662145</td></tr>
	<tr><td>36.12500     </td><td>36.30769     </td><td>35.87500     </td><td>36.11538     </td><td>28108200     </td><td>23.08765     </td><td>-0.0002662145</td><td>-0.010534676 </td><td>-0.005296674 </td><td>-0.0050331592</td><td>-0.0107980864</td><td> 0.0040096018</td><td>           NA</td><td>36.29808     </td><td>36.50962     </td><td>      NA     </td><td> 0.0026702034</td></tr>
	<tr><td>36.00962     </td><td>36.22115     </td><td>35.81731     </td><td>36.10577     </td><td>24662200     </td><td>23.08150     </td><td> 0.0026702034</td><td>-0.003193993 </td><td>-0.003185514 </td><td>-0.0002661747</td><td>-0.0005323183</td><td>-0.0110613869</td><td> 0.0037423599</td><td>36.11538     </td><td>36.29808     </td><td>36.50962     </td><td>-0.0060878539</td></tr>
	<tr><td>36.32692     </td><td>36.52885     </td><td>35.92308     </td><td>36.10577     </td><td>25581200     </td><td>23.08150     </td><td>-0.0060878539</td><td> 0.008811702 </td><td>-0.011581997 </td><td> 0.0000000000</td><td> 0.0026702034</td><td>-0.0005323183</td><td>-0.0110613869</td><td>36.10577     </td><td>36.11538     </td><td>36.29808     </td><td> 0.0048153837</td></tr>
	<tr><td>35.94231     </td><td>36.16346     </td><td>35.90385     </td><td>36.11538     </td><td>24956400     </td><td>23.08765     </td><td> 0.0048153837</td><td>-0.010587629 </td><td>-0.001329436 </td><td> 0.0002662455</td><td>-0.0058232292</td><td> 0.0029371598</td><td>-0.0002662145</td><td>36.10577     </td><td>36.10577     </td><td>36.11538     </td><td> 0.0112000233</td></tr>
</tbody>
</table>



### First company stock: General Electric (GE)



```R
GE_df.monthly <- to.monthly(GE_df)
GE_df.monthly$month <- format(index(GE_df.monthly),"%Y%m")
GE_df.monthly$year <- format(index(GE_df.monthly),"%Y")
#head(GE_df.monthly)
```


```R
rtn.daily <- dailyReturn(GE_df) # returns by day
rtn.weekly <- weeklyReturn(GE_df) # returns by week
rtn.monthly <- monthlyReturn(GE_df) # returns by month, indexed by yearmon
# daily,weekly,monthly,quarterly, and yearly
rtn.allperiods <- allReturns(GE_df) # note the plural
#head(rtn.daily)
```

#### The basic characteristics of stock returns
- A standard normal distribution has 0 mean, 1 standard deviation, and 0 excess [kurtosis](http://www.r-tutor.com/elementary-statistics/numerical-measures/kurtosis)
- The ditribution of a typical stock returns has small standard deviation and positive excess kurtosis


```R
# Generate a standard normal distribution
rn <- rnorm(100000)
print(paste0("standard deviation: ", sd(rn)))
print(paste0("Kurtosis: ", round(kurtosis(rn),2)))
options(repr.plot.width = 4, repr.plot.height = 4)
hist(rn,breaks=100,prob=TRUE)
curve(dnorm(x, mean=0, sd=1), col="darkblue", lwd=2, add=TRUE ) # Overlay a standard normal distribution
```

    [1] "standard deviation: 0.998766119015111"
    [1] "Kurtosis: 0.01"



![png](/assets/img/stockstrategies/output_15_1.png)



```R
print(paste0("standard deviation: ", sd(rtn.daily)))
print(paste0("Kurtosis: ", round(kurtosis(rtn.daily),2)))

options(repr.plot.width = 4, repr.plot.height = 4)
hist(rtn.daily, breaks=100, prob=TRUE) # Make it a probability distribution

m<-mean(rtn.daily)
std<-sqrt(var(rtn.daily))
m
std
# Overlay a standard normal distribution
curve(dnorm(x, mean=m, sd=std), col="darkblue", lwd=2, add=TRUE )
```

    [1] "standard deviation: 0.022288482105975"
    [1] "Kurtosis: 9.13"



-0.000260234672118874



<table>
<thead><tr><th></th><th scope=col>daily.returns</th></tr></thead>
<tbody>
	<tr><th scope=row>daily.returns</th><td>0.02228848</td></tr>
</tbody>
</table>




![png](/assets/img/stockstrategies/output_16_3.png)



```R
# A really basic boxplot.
GE_df$year <- format(index(GE_df),"%Y")
GE_df$month <- format(index(GE_df),"%Y%m")
GE_df3 <- data.frame(GE_df) %>% filter(year==2019)
GE_df3$GE.Volume <- as.numeric(GE_df3$GE.Volume)

options(repr.plot.width = 6, repr.plot.height = 3)

# Basic plot
p <-ggplot(GE_df3, aes(x=as.factor(month), y=GE.Volume))

p + geom_boxplot(fill="slateblue", alpha=0.2) +  xlab("Month")
```


![png](/assets/img/stockstrategies/output_17_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
# Change outlier, color, shape and size
p2 <- p  + geom_boxplot(outlier.colour="red", outlier.shape=8,
                outlier.size=1) +     xlab("Month")
p2
```


![png](/assets/img/stockstrategies/output_18_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 4)
GE_df <- GE
GE_df$OpCl <- OpCl(GE_df)
GE_df$OpOp <- OpOp(GE_df)
GE_df$HiCl <- HiCl(GE_df)
GE_df$month <- format(index(GE_df),"%Y%m")
GE_df$year <- format(index(GE_df),"%Y")
GE_df_hiCl <- GE_df[GE_df$year==2019,]
```


```R
boxplot(HiCl~month, data=GE_df_hiCl, notch=TRUE,
  col=(c("gold","darkgreen")),
  main="Hi-Closed", xlab="Month")

boxplot(OpCl~month, data=GE_df_hiCl, notch=TRUE,
  col=(c("gold","darkgreen")),
  main="Open-Closed", xlab="Month")
```

    Warning message in bxp(list(stats = structure(c(-0.0367074873404006, -0.022396337150083, :
    "some notches went outside hinges ('box'): maybe set notch=FALSE"Warning message in bxp(list(stats = structure(c(-0.0356347098278267, -0.00234732414202743, :
    "some notches went outside hinges ('box'): maybe set notch=FALSE"


![png](/assets/img/stockstrategies/output_20_1.png)



![png](/assets/img/stockstrategies/output_20_2.png)


### Common technical indicators

#### MACD
* MACD=12-Period EMA − 26-Period EMA, or "fast EMA - slow FMA"
* The MACD was developed by Gerald Appel and is probably the most popular price oscillator.
* It can be used as a generic oscillator for any univariate series, not only price.
* The MACD has a positive value whenever the 12-period EMA is above the 26-period EMA and a negative value when the 12-period EMA is below the 26-period EMA. The more distant the MACD is above or below its baseline indicates that the distance between the two EMAs is growing.

#### RSI
* Introduced by Welles Wilder Jr. in his seminal 1978 book "New Concepts in Technical Trading Systems", the relative strength index (RSI) is a popular momentum indicator.
* It measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
* The RSI is displayed as an oscillator and can have a reading from 0 to 100.  
* RSI >= 70: a security is overbought or overvalued and may be primed for a trend reversal or corrective pullback in price.
* RSI <= 30: an oversold or undervalued condition.
* It can be used in the price of a stock or other asset.

#### Bollinger Bands
* Bollinger Bands are a type of price envelope developed by John Bollinger
* Bollinger Bands are envelopes plotted at a standard deviation level above and below a simple moving average of the price. Because the distance of the bands is based on standard deviation, they adjust to volatility swings in the underlying price.
* Bollinger Bands use 2 parameters, Period and Standard Deviations, StdDev. The default values are 20 for period, and 2 for standard deviations, although you may customize the combinations.
* Bollinger bands help determine whether prices are high or low on a relative basis. They are used in pairs, both upper and lower bands and in conjunction with a moving average. Further, the pair of bands is not intended to be used on its own. Use the pair to confirm signals given with other indicators.
* "Distance from a moving average" or "standard deviation" apply the same concept
* Click [here](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands#:~:text=Bollinger%20Bands%20are%20envelopes%20plotted,Period%20and%20Standard%20Deviations%2C%20StdDev.) for more detail


```R
v <- Delt(Op(GE_df),Cl(GE_df),k=1:3)
colnames(v) <-c("pcntOpCl1","pcntOpCl2","pcntOpCl3")
GE_df1 <- cbind(GE_df,v)
#head(GE_df1)
```


```R
macd <- MACD(GE_df1$GE.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "SMA", percent = FALSE)
rsi <- RSI(GE_df1$GE.Adjusted, n = 14, maType = "SMA")

#tail(macd)
```


```R
d <- cbind(GE,macd,rsi)
d$SMA12 <- SMA(d$GE.Adjusted,12)
d$SMA26 <- SMA(d$GE.Adjusted,26)
d <- subset(d, select = -c(GE.Open,GE.High,GE.Low,GE.Close,GE.Volume))
d[50:60]
```


               GE.Adjusted        macd     signal      rsi    SMA12    SMA26
    2007-03-15    21.38575 -0.31889936 -0.2858283 41.52020 21.39969 21.71859
    2007-03-16    21.28663 -0.31254067 -0.3003253 35.32895 21.37130 21.68384
    2007-03-19    21.47868 -0.31072138 -0.3109206 50.16835 21.35426 21.66498
    2007-03-20    21.54063 -0.30437621 -0.3177587 47.51760 21.34910 21.65347
    2007-03-21    21.98048 -0.25917304 -0.3151064 56.97691 21.39711 21.65628
    2007-03-22    22.18492 -0.21049935 -0.3033675 62.91216 21.45338 21.66388
    2007-03-23    22.19112 -0.12486392 -0.2805977 69.06927 21.53031 21.65517
    2007-03-26    22.30263 -0.04822158 -0.2485414 69.16187 21.61033 21.65855
    2007-03-27    22.17253  0.02291088 -0.2073761 73.10118 21.68622 21.66331
    2007-03-28    22.02386  0.08685072 -0.1622927 66.76845 21.74352 21.65667
    2007-03-29    22.02386  0.16413141 -0.1093292 69.52395 21.81890 21.65476


### Plot technical charts


```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(GE, subset = "last 3 months")
```


![png](/assets/img/stockstrategies/output_26_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(GE, subset = "2018::2020-01")
```


![png](/assets/img/stockstrategies/output_27_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(GE, theme = chartTheme("white"))
```


![png](/assets/img/stockstrategies/output_28_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(GE, subset = "2018::2020-09", TA = c(addVo(), addBBands()))  #add volume and Bollinger Bands from TTR
```


![png](/assets/img/stockstrategies/output_29_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(GE, subset = "2019::2019-12",bar.type='hlc',
            TA = c(addSMA(n=12,col="green"),addSMA(n=26,col="red"),
                addMACD(),addRSI()),
            theme = chartTheme("white"))  
```


![png](/assets/img/stockstrategies/output_30_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
barChart(GE,subset = "2019::2019-12",bar.type='hlc')
```


![png](/assets/img/stockstrategies/output_31_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
candleChart(GE,subset = "2020::2020-06",multi.col=TRUE, theme='white')
```


![png](/assets/img/stockstrategies/output_32_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(GE, subset = "2020::2020-06",
            theme="white",  
            TA="addVo();addBBands();addCCI();
                addTA(OpCl(GE),col='blue', type='h')  ")
```


![png](/assets/img/stockstrategies/output_33_0.png)


### Trading strategy & signals

#### MACD & RSI trading rule



```R
macd <- MACD(GE$GE.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "SMA", percent = FALSE)
rsi <- RSI(GE$GE.Adjusted, n = 14, maType = "SMA")
#tail(macd)
#tail(rsi)
```

Here we assume no transaction cost.


```R
macd <- MACD(GE$GE.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "SMA", percent = FALSE)
rsi <- RSI(GE$GE.Adjusted, n = 14, maType = "SMA")

# Strategy 1: if macd>signal, enter and stay in the market. If macd<signal, exit the market.
strategy1 <- ifelse ((macd$signal < macd$macd) , 1, 0)
strategy1[is.na(strategy1)] <-0

# Strategy 2: if overbought, enter and stay in the market.
strategy2 <- ifelse ((macd$signal < macd$macd) & (rsi$rsi > 70), 1, 0)
strategy2[is.na(strategy2)] <-0

# Strategy 3: if oversold, enter and stay in the market.
strategy3 <- ifelse ((macd$signal > macd$macd) & (rsi$rsi < 30), 1, 0)
strategy3[is.na(strategy3)] <-0


# Buy-and-hold: keep it all time. So "1", not "0"
bh_strategy <- rep(1,dim(macd)[1])
```

### Backtesting

#### Annualized return
* An annualized total return is the average amount earned by an investment each year over a given time period.

#### Sharpe Ratio
* [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)
* [Annualized Sharpe Ratio](https://www.rdocumentation.org/packages/PerformanceAnalytics/versions/2.0.4/topics/SharpeRatio.annualized#:~:text=The%20annualized%20Sharpe%20ratio%20is,standard%20deviation%20of%20excess%20return.)
* Usually, any Sharpe ratio greater than 1.0 is considered acceptable to good by investors. A ratio higher than 2.0 is rated as very good. A ratio of 3.0 or higher is considered excellent. A ratio under 1.0 is considered sub-optimal.
* "Lag": Since we are working with Closing prices, we can BUY or SELL on our signal the next day only

#### Strategy for GE


```R
# Put in a function
backtest <- function(df,from_date,to_date,strategy,strategy_name){
    rtn.daily <- dailyReturn(df) # returns by day
    trade_return <- rtn.daily[index(rtn.daily)<=to_date & index(rtn.daily)>=from_date]*lag(strategy, na.pad = FALSE)
    cumm_return <- Return.cumulative(trade_return)
    annual_return <- Return.annualized(trade_return)
    summary(as.ts(trade_return))
    SharpeRatio <- SharpeRatio(as.ts(trade_return), Rf = 0, p = 0.95, FUN = "StdDev")
    SharpeRatioAnnualized <- SharpeRatio.annualized(trade_return, Rf = 0)
    out <- as.data.frame(c(cumm_return,annual_return,SharpeRatio,SharpeRatioAnnualized))
    out <- round(out,2)
    colnames(out) <- strategy_name
    row.names(out) <- c('Cumulative Return','Annualized Return','Sharpe Ratio','Annualized Sharpe Ratio')

  return( out )
    }

# Strategy 1
strategy1_performance <- backtest(GE, from_date = '2015-06-01', to_date = '2020-05-31', strategy1,"Strategy1")
strategy1_performance

# Strategy 2
strategy2_performance <- backtest(GE, from_date = '2015-06-01', to_date = '2020-05-31', strategy2,"Strategy2")
strategy2_performance

# Strategy 3
strategy3_performance <- backtest(GE, from_date = '2015-06-01', to_date = '2020-05-31', strategy3,"Strategy3")
strategy3_performance


# Buy-and-hold strategy
BH_backtest <- function(df,from_date,to_date,strategy_name){
    rtn.daily <- dailyReturn(df) # returns by day
    trade_return <- rtn.daily[index(rtn.daily)<=to_date & index(rtn.daily)>=from_date]
    cumm_return <- Return.cumulative(trade_return)
    annual_return <- Return.annualized(trade_return)
    summary(as.ts(trade_return))
    SharpeRatio <- SharpeRatio(as.ts(trade_return), Rf = 0, p = 0.95, FUN = "StdDev")
    SharpeRatioAnnualized <- SharpeRatio.annualized(trade_return, Rf = 0)
    out <- as.data.frame(c(cumm_return,annual_return,SharpeRatio,SharpeRatioAnnualized))
    out <- round(out,2)
    colnames(out) <- strategy_name
    row.names(out) <- c('Cumulative Return','Annualized Return','Sharpe Ratio','Annualized Sharpe Ratio')

  return( out )
    }

buy_and_hold_performance <- BH_backtest(GE, from_date = '2015-06-01', to_date = '2020-05-31',"Buy & Hold Strategy")
buy_and_hold_performance
```


<table>
<thead><tr><th></th><th scope=col>Strategy1</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>-0.29</td></tr>
	<tr><th scope=row>Annualized Return</th><td>-0.07</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>-0.01</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>-0.29</td></tr>
</tbody>
</table>




<table>
<thead><tr><th></th><th scope=col>Strategy2</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>-0.22</td></tr>
	<tr><th scope=row>Annualized Return</th><td>-0.05</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>-0.03</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>-0.48</td></tr>
</tbody>
</table>




<table>
<thead><tr><th></th><th scope=col>Strategy3</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>-0.63</td></tr>
	<tr><th scope=row>Annualized Return</th><td>-0.18</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>-0.05</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>-0.85</td></tr>
</tbody>
</table>




<table>
<thead><tr><th></th><th scope=col>Buy &amp; Hold Strategy</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>-0.75</td></tr>
	<tr><th scope=row>Annualized Return</th><td>-0.24</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>-0.04</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>-0.65</td></tr>
</tbody>
</table>



Out of the above strategies, the best performance was achieved with strategy 1. However, this stock is not ideal as every strategy resulted in negative returns and negative sharpe ratios. As GE has mostly declined since its peak in 2000, it is not surprising that they have such poor performance. However, if one strategy must be picked, it should be strategy 1 for this stock. In addition, strategy 2 should be explorered with this stock as its returns are better than those of strategy 1 although its sharpe ratio is worse. This insight could help improve performance, but dropping the stock is most likely the optimal solution for this case.

### Second company stock: Tesla


```R
TSLA_df.monthly <- to.monthly(TSLA_df)
TSLA_df.monthly$month <- format(index(TSLA_df.monthly),"%Y%m")
TSLA_df.monthly$year <- format(index(TSLA_df.monthly),"%Y")
#head(TSLA_df.monthly)
```


```R
rtn.daily <- dailyReturn(TSLA_df) # returns by day
rtn.weekly <- weeklyReturn(TSLA_df) # returns by week
rtn.monthly <- monthlyReturn(TSLA_df) # returns by month, indexed by yearmon
# daily,weekly,monthly,quarterly, and yearly
rtn.allperiods <- allReturns(TSLA_df) # note the plural
#head(rtn.daily)
```

#### The basic characteristics of stock returns


```R
# Generate a standard normal distribution
rn <- rnorm(100000)
print(paste0("standard deviation: ", sd(rn)))
print(paste0("Kurtosis: ", round(kurtosis(rn),2)))
options(repr.plot.width = 4, repr.plot.height = 4)
hist(rn,breaks=100,prob=TRUE)
curve(dnorm(x, mean=0, sd=1), col="darkblue", lwd=2, add=TRUE ) # Overlay a standard normal distribution
```

    [1] "standard deviation: 1.00563106954407"
    [1] "Kurtosis: -0.01"



![png](/assets/img/stockstrategies/output_46_1.png)



```R
print(paste0("standard deviation: ", sd(rtn.daily)))
print(paste0("Kurtosis: ", round(kurtosis(rtn.daily),2)))

options(repr.plot.width = 4, repr.plot.height = 4)
hist(rtn.daily, prob=TRUE) # Make it a probability distribution

m<-mean(rtn.daily)
std<-sqrt(var(rtn.daily))
m
std
# Overlay a standard normal distribution
curve(dnorm(x, mean=m, sd=std), col="darkblue", lwd=2, add=TRUE )
```

    [1] "standard deviation: 0.0358103255386379"
    [1] "Kurtosis: 6.79"



0.00246103951064709



<table>
<thead><tr><th></th><th scope=col>daily.returns</th></tr></thead>
<tbody>
	<tr><th scope=row>daily.returns</th><td>0.03581033</td></tr>
</tbody>
</table>




![png](/assets/img/stockstrategies/output_47_3.png)



```R
# A really basic boxplot.
TSLA_df$year <- format(index(TSLA_df),"%Y")
TSLA_df$month <- format(index(TSLA_df),"%Y%m")
TSLA_df3 <- data.frame(TSLA_df) %>% filter(year==2019)
TSLA_df3$TSLA.Volume <- as.numeric(TSLA_df3$TSLA.Volume)

options(repr.plot.width = 6, repr.plot.height = 3)

# Basic plot
p <-ggplot(TSLA_df3, aes(x=as.factor(month), y=TSLA.Volume))

p + geom_boxplot(fill="slateblue", alpha=0.2) +  xlab("Month")
```


![png](/assets/img/stockstrategies/output_48_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
# Change outlier, color, shape and size
p2 <- p  + geom_boxplot(outlier.colour="red", outlier.shape=8,
                outlier.size=1) +     xlab("Month")
p2
```


![png](/assets/img/stockstrategies/output_49_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 4)
TSLA_df <- TSLA
TSLA_df$OpCl <- OpCl(TSLA_df)
TSLA_df$OpOp <- OpOp(TSLA_df)
TSLA_df$HiCl <- HiCl(TSLA_df)
TSLA_df$month <- format(index(TSLA_df),"%Y%m")
TSLA_df$year <- format(index(TSLA_df),"%Y")
TSLA_df_hiCl <- TSLA_df[TSLA_df$year==2019,]

```


```R
boxplot(HiCl~month, data=TSLA_df_hiCl, notch=TRUE,
  col=(c("gold","darkgreen")),
  main="Hi-Closed", xlab="Month")

boxplot(OpCl~month, data=TSLA_df_hiCl, notch=TRUE,
  col=(c("gold","darkgreen")),
  main="Open-Closed", xlab="Month")
```

    Warning message in bxp(list(stats = structure(c(-0.0294804879701457, -0.0234634966565875, :
    "some notches went outside hinges ('box'): maybe set notch=FALSE"Warning message in bxp(list(stats = structure(c(-0.0642104973439782, -0.0167862905982905, :
    "some notches went outside hinges ('box'): maybe set notch=FALSE"


![png](/assets/img/stockstrategies/output_51_1.png)



![png](/assets/img/stockstrategies/output_51_2.png)


#### Common technical indicators


```R
v <- Delt(Op(TSLA_df),Cl(TSLA_df),k=1:3)
colnames(v) <-c("pcntOpCl1","pcntOpCl2","pcntOpCl3")
TSLA_df2 <- cbind(TSLA_df,v)
#head(TSLA_df2)
```


```R
macd <- MACD(TSLA_df2$TSLA.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "SMA", percent = FALSE)
rsi <- RSI(TSLA_df2$TSLA.Adjusted, n = 14, maType = "SMA")

#tail(macd)
```


```R
d <- cbind(TSLA,macd,rsi)
d$SMA12 <- SMA(d$TSLA.Adjusted,12)
d$SMA26 <- SMA(d$TSLA.Adjusted,26)
d <- subset(d, select = -c(TSLA.Open,TSLA.High,TSLA.Low,TSLA.Close,TSLA.Volume))
#d[50:60]
```

#### Plot technical charts


```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(TSLA, subset = "last 3 months")
```


![png](/assets/img/stockstrategies/output_57_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(TSLA, subset = "2018::2020-01")
```


![png](/assets/img/stockstrategies/output_58_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(TSLA, theme = chartTheme("white"))
```


![png](/assets/img/stockstrategies/output_59_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(TSLA, subset = "2018::2020-09", TA = c(addVo(), addBBands()))  #add volume and Bollinger Bands from TTR
```


![png](/assets/img/stockstrategies/output_60_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(TSLA, subset = "2019::2019-12",bar.type='hlc',
            TA = c(addSMA(n=12,col="green"),addSMA(n=26,col="red"),
                addMACD(),addRSI()),
            theme = chartTheme("white"))  
```


![png](/assets/img/stockstrategies/output_61_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
barChart(TSLA,subset = "2019::2019-12",bar.type='hlc')
```


![png](/assets/img/stockstrategies/output_62_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
candleChart(TSLA,subset = "2020::2020-06",multi.col=TRUE, theme='white')
```


![png](/assets/img/stockstrategies/output_63_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(TSLA, subset = "2020::2020-06",
            theme="white",  
            TA="addVo();addBBands();addCCI();
                addTA(OpCl(GE),col='blue', type='h')  ")
```


![png](/assets/img/stockstrategies/output_64_0.png)


#### Develop your trading strategy & signals


```R
macd <- MACD(TSLA$TSLA.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "SMA", percent = FALSE)
rsi <- RSI(TSLA$TSLA.Adjusted, n = 14, maType = "SMA")
#tail(macd)
#tail(rsi)
```


```R
# Strategy 1
strategy1_performance <- backtest(TSLA, from_date = '2015-06-01', to_date = '2020-05-31', strategy1,"Strategy1")
strategy1_performance

# Strategy 2
strategy2_performance <- backtest(TSLA, from_date = '2015-06-01', to_date = '2020-05-31', strategy2,"Strategy2")
strategy2_performance

# Strategy 3
strategy3_performance <- backtest(TSLA, from_date = '2015-06-01', to_date = '2020-05-31', strategy3,"Strategy3")
strategy3_performance


buy_and_hold_performance <- BH_backtest(TSLA, from_date = '2015-06-01', to_date = '2020-05-31',"Buy & Hold Strategy")
buy_and_hold_performance
```


<table>
<thead><tr><th></th><th scope=col>Strategy1</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>1.14</td></tr>
	<tr><th scope=row>Annualized Return</th><td>0.16</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>0.04</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>0.49</td></tr>
</tbody>
</table>




<table>
<thead><tr><th></th><th scope=col>Strategy2</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>-0.02</td></tr>
	<tr><th scope=row>Annualized Return</th><td> 0.00</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td> 0.00</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>-0.03</td></tr>
</tbody>
</table>




<table>
<thead><tr><th></th><th scope=col>Strategy3</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>-0.29</td></tr>
	<tr><th scope=row>Annualized Return</th><td>-0.07</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>-0.01</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>-0.28</td></tr>
</tbody>
</table>




<table>
<thead><tr><th></th><th scope=col>Buy &amp; Hold Strategy</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>2.33</td></tr>
	<tr><th scope=row>Annualized Return</th><td>0.27</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>0.05</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>0.51</td></tr>
</tbody>
</table>



For Tesla, the best strategy was to buy & hold the stock followed by strategy 1. As the company has consistently increased in value, it makes sense that the best option would be to hold onto it. As strategy 1 follows a similar plan with buy & hold and only sells if the MACD is less than the signal, it highlights how their Sharpe Ratios are also similar and only their returns differ by a large margin.

### APPLE


```R
APPL_df.monthly <- to.monthly(APPL_df)
APPL_df.monthly$month <- format(index(APPL_df.monthly),"%Y%m")
APPL_df.monthly$year <- format(index(APPL_df.monthly),"%Y")
#head(APPL_df.monthly)
```


```R
rtn.daily <- dailyReturn(APPL_df) # returns by day
rtn.weekly <- weeklyReturn(APPL_df) # returns by week
rtn.monthly <- monthlyReturn(APPL_df) # returns by month, indexed by yearmon
# daily,weekly,monthly,quarterly, and yearly
rtn.allperiods <- allReturns(APPL_df) # note the plural
#head(rtn.daily)
```

#### The basic characteristics of stock returns


```R
# Generate a standard normal distribution
rn <- rnorm(100000)
print(paste0("standard deviation: ", sd(rn)))
print(paste0("Kurtosis: ", round(kurtosis(rn),2)))
options(repr.plot.width = 4, repr.plot.height = 4)
hist(rn,breaks=100,prob=TRUE)
curve(dnorm(x, mean=0, sd=1), col="darkblue", lwd=2, add=TRUE ) # Overlay a standard normal distribution
```

    [1] "standard deviation: 1.00218344334141"
    [1] "Kurtosis: 0"



![png](/assets/img/stockstrategies/output_73_1.png)



```R
print(paste0("standard deviation: ", sd(rtn.daily)))
print(paste0("Kurtosis: ", round(kurtosis(rtn.daily),2)))

options(repr.plot.width = 4, repr.plot.height = 4)
hist(rtn.daily, prob=TRUE) # Make it a probability distribution

m<-mean(rtn.daily)
std<-sqrt(var(rtn.daily))
m
std
# Overlay a standard normal distribution
curve(dnorm(x, mean=m, sd=std), col="darkblue", lwd=2, add=TRUE )
```

    [1] "standard deviation: 0.0205005712951042"
    [1] "Kurtosis: 6.41"



0.00125752987770374



<table>
<thead><tr><th></th><th scope=col>daily.returns</th></tr></thead>
<tbody>
	<tr><th scope=row>daily.returns</th><td>0.02050057</td></tr>
</tbody>
</table>




![png](/assets/img/stockstrategies/output_74_3.png)



```R
# A really basic boxplot.
APPL_df$year <- format(index(APPL_df),"%Y")
APPL_df$month <- format(index(APPL_df),"%Y%m")
APPL_df3 <- data.frame(APPL_df) %>% filter(year==2019)
APPL_df3$AAPL.Volume <- as.numeric(APPL_df3$AAPL.Volume)

options(repr.plot.width = 6, repr.plot.height = 3)

# Basic plot
p <-ggplot(APPL_df3, aes(x=as.factor(month), y=AAPL.Volume))

p + geom_boxplot(fill="slateblue", alpha=0.2) +  xlab("Month")
```


![png](/assets/img/stockstrategies/output_75_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
# Change outlier, color, shape and size
p2 <- p  + geom_boxplot(outlier.colour="red", outlier.shape=8,
                outlier.size=1) +     xlab("Month")
p2
```


![png](/assets/img/stockstrategies/output_76_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 4)
APPL_df <- AAPL
APPL_df$OpCl <- OpCl(APPL_df)
APPL_df$OpOp <- OpOp(APPL_df)
APPL_df$HiCl <- HiCl(APPL_df)
APPL_df$month <- format(index(APPL_df),"%Y%m")
APPL_df$year <- format(index(APPL_df),"%Y")
APPL_df_hiCL <- APPL_df[APPL_df$year==2019,]

```


```R
boxplot(HiCl~month, data=APPL_df_hiCL, notch=TRUE,
  col=(c("gold","darkgreen")),
  main="Hi-Closed", xlab="Month")

boxplot(OpCl~month, data=APPL_df_hiCL, notch=TRUE,
  col=(c("gold","darkgreen")),
  main="Open-Closed", xlab="Month")
```

    Warning message in bxp(list(stats = structure(c(-0.0151479053254439, -0.0114169983149307, :
    "some notches went outside hinges ('box'): maybe set notch=FALSE"Warning message in bxp(list(stats = structure(c(-0.0198836386450063, -0.00517823820250807, :
    "some notches went outside hinges ('box'): maybe set notch=FALSE"


![png](/assets/img/stockstrategies/output_78_1.png)



![png](/assets/img/stockstrategies/output_78_2.png)


#### Common technical indicators


```R
v <- Delt(Op(APPL_df),Cl(APPL_df),k=1:3)
colnames(v) <-c("pcntOpCl1","pcntOpCl2","pcntOpCl3")
APPL_df2 <- cbind(APPL_df,v)
#head(APPL_df2)
```


```R
macd <- MACD(APPL_df2$AAPL.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "SMA", percent = FALSE)
rsi <- RSI(APPL_df2$AAPL.Adjusted, n = 14, maType = "SMA")

#tail(macd)
```


```R
d <- cbind(AAPL,macd,rsi)
d$SMA12 <- SMA(d$AAPL.Adjusted,12)
d$SMA26 <- SMA(d$AAPL.Adjusted,26)
d <- subset(d, select = -c(AAPL.Open,AAPL.High,AAPL.Low,AAPL.Close,AAPL.Volume))
#d[50:60]
```

#### Plot technical charts


```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(AAPL, subset = "last 3 months")
```


![png](/assets/img/stockstrategies/output_84_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(AAPL, subset = "2014::2020-01")
```


![png](/assets/img/stockstrategies/output_85_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(AAPL, theme = chartTheme("white"))
```


![png](/assets/img/stockstrategies/output_86_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(AAPL, subset = "2014::2020-09", TA = c(addVo(), addBBands()))  #add volume and Bollinger Bands from TTR
```


![png](/assets/img/stockstrategies/output_87_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(AAPL, subset = "2019::2019-12",bar.type='hlc',
            TA = c(addSMA(n=12,col="green"),addSMA(n=26,col="red"),
                addMACD(),addRSI()),
            theme = chartTheme("white"))  
```


![png](/assets/img/stockstrategies/output_88_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
barChart(AAPL,subset = "2019::2019-12",bar.type='hlc')
```


![png](/assets/img/stockstrategies/output_89_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
candleChart(AAPL,subset = "2020::2020-06",multi.col=TRUE, theme='white')
```


![png](/assets/img/stockstrategies/output_90_0.png)



```R
options(repr.plot.width = 6, repr.plot.height = 3)
chartSeries(AAPL, subset = "2020::2020-06",
            theme="white",  
            TA="addVo();addBBands();addCCI();
                addTA(OpCl(GE),col='blue', type='h')  ")
```


![png](/assets/img/stockstrategies/output_91_0.png)


#### Develop your trading strategy & signals


```R
macd <- MACD(AAPL$AAPL.Adjusted, nFast = 12, nSlow = 26, nSig = 9, maType = "SMA", percent = FALSE)
rsi <- RSI(AAPL$AAPL.Adjusted, n = 14, maType = "SMA")
#tail(macd)
#tail(rsi)
```

#### Backtesting


```R
# Strategy 1
strategy1_performance <- backtest(AAPL, from_date = '2015-06-01', to_date = '2020-05-31', strategy1,"Strategy1")
strategy1_performance

# Strategy 2
strategy2_performance <- backtest(AAPL, from_date = '2015-06-01', to_date = '2020-05-31', strategy2,"Strategy2")
strategy2_performance

# Strategy 3
strategy3_performance <- backtest(AAPL,from_date = '2015-06-01', to_date = '2020-05-31', strategy3,"Strategy3")
strategy3_performance


buy_and_hold_performance <- BH_backtest(AAPL, from_date = '2015-06-01', to_date = '2020-05-31',"Buy & Hold Strategy")
buy_and_hold_performance
```


<table>
<thead><tr><th></th><th scope=col>Strategy1</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>0.02</td></tr>
	<tr><th scope=row>Annualized Return</th><td>0.00</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>0.01</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>0.02</td></tr>
</tbody>
</table>




<table>
<thead><tr><th></th><th scope=col>Strategy2</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>0.29</td></tr>
	<tr><th scope=row>Annualized Return</th><td>0.05</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>0.05</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>0.77</td></tr>
</tbody>
</table>




<table>
<thead><tr><th></th><th scope=col>Strategy3</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>0.13</td></tr>
	<tr><th scope=row>Annualized Return</th><td>0.03</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>0.01</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>0.15</td></tr>
</tbody>
</table>




<table>
<thead><tr><th></th><th scope=col>Buy &amp; Hold Strategy</th></tr></thead>
<tbody>
	<tr><th scope=row>Cumulative Return</th><td>1.44</td></tr>
	<tr><th scope=row>Annualized Return</th><td>0.20</td></tr>
	<tr><th scope=row>Sharpe Ratio</th><td>0.05</td></tr>
	<tr><th scope=row>Annualized Sharpe Ratio</th><td>0.68</td></tr>
</tbody>
</table>



For Apple, the best strategy was stategy 2 followed by buy & hold according to their Sharpe Ratios. As strategy 2 outlines buying stock and staying in the market if it is overbought, it makes sense that this option would be optimal provided past performance. Before earnings reports and product reveals, Apple stock is typically highly sought after causing it to be considered as overbought. However, provided the event was received positively, the price most likely continues to rise indicating that short sellers were wrong with their predictions. However, an interesting area to explore further would be comparing the returns. While strategy 2 has the best ratios, buy & hold has the best returns, suggesting that it may be better to simply keep the stock and hold it as it has performed well.
