
## Feature Engineering & EDA  For Credit Card Data
    Tram Duong
    September 14, 2020 

## Table of Contents:
* [Introduction](#intro)
* [Initial EDA](#initial_eda)
* [Feature 1](#feature1)
* [Feature 2](#feature2)
* [Feature 3](#feature3)
* [Feature 4](#feature4)
* [Feature 5](#feature5)
* [Feature 6](#feature6)
* [Feature 7](#feature7)
* [Feature 8](#feature8)
* [Feature 9](#feature9)
* [Feature 10](#feature10)
* [Feature 11](#feature11)
* [Feature 12](#feature12)
* [Feature 13](#feature13)
* [Feature 14](#feature14)
* [Feature 15](#feature15)
* [Feature 16](#feature16)
* [Feature 17](#feature17)
* [Feature 18](#feature18)
* [Feature 19](#feature19)
* [Feature 20](#feature20)

### Introduction <a class="anchor" id="intro"></a>

This dataset contains information on purchases made through the purchase card programs administered by the State of Oklahoma and higher education institutions. 

In this report, I will complete the following tasks.

- Create new features and conduct exploratory data analysis. 
- Each feature or discussion is a new lead. Structure your EDA for different leads with sub-sections. Each sub-section will cover the following:
    - Lead: Write what you are going to do in two to three sentences.
    - Analysis: your EDA
    - Conclusion: What is the business insight? How can this feature help prediction? Write a short conclusion in the end of each sub-section. 

#### Load Data and Libraries


```python
#install.packages("plotly")
#install.packages("DataExplorer")
#install.packages("githubinstall")
#library(githubinstall)
#githubinstall("xda")
```


```python
library(dplyr, quietly = TRUE)
library(DataExplorer, quietly = TRUE)
library(xda, quietly = TRUE)
library(ggplot2, quietly = TRUE)
library(plotly, quietly = TRUE)
library(lubridate, quietly = TRUE)
```

    
    Attaching package: 'dplyr'
    
    The following objects are masked from 'package:stats':
    
        filter, lag
    
    The following objects are masked from 'package:base':
    
        intersect, setdiff, setequal, union
    
    Warning message:
    "package 'DataExplorer' was built under R version 3.6.3"Warning message:
    "package 'plotly' was built under R version 3.6.3"
    Attaching package: 'plotly'
    
    The following object is masked from 'package:ggplot2':
    
        last_plot
    
    The following object is masked from 'package:stats':
    
        filter
    
    The following object is masked from 'package:graphics':
    
        layout
    
    
    Attaching package: 'lubridate'
    
    The following object is masked from 'package:base':
    
        date
    
    


```python
setwd("C:/github/Data-Science-Portfolio/Feature Engineering & ModelingDoc/")
ccard <- read.csv("res_purchase_card_(pcard)_fiscal_year_2014_3pcd-aiuu.csv")
```


```python
colnames(ccard)
```


<ol class=list-inline>
	<li>'Year.Month'</li>
	<li>'Agency.Number'</li>
	<li>'Agency.Name'</li>
	<li>'Cardholder.Last.Name'</li>
	<li>'Cardholder.First.Initial'</li>
	<li>'Description'</li>
	<li>'Amount'</li>
	<li>'Vendor'</li>
	<li>'Transaction.Date'</li>
	<li>'Posted.Date'</li>
	<li>'Merchant.Category.Code..MCC.'</li>
</ol>




```python
colnames(ccard)<-c('Year_Month', 'Agency_Number', 'Agency_Name', 'Cardholder_Last_Name',
      'Cardholder_First_Initial', 'Description', 'Amount', 'Vendor', 'Transaction_Date',
      'Posted_Date', 'Merchant_Category')
```


```python
nrow(ccard)
head(ccard)
```


442458



<table>
<thead><tr><th scope=col>Year_Month</th><th scope=col>Agency_Number</th><th scope=col>Agency_Name</th><th scope=col>Cardholder_Last_Name</th><th scope=col>Cardholder_First_Initial</th><th scope=col>Description</th><th scope=col>Amount</th><th scope=col>Vendor</th><th scope=col>Transaction_Date</th><th scope=col>Posted_Date</th><th scope=col>Merchant_Category</th></tr></thead>
<tbody>
	<tr><td>201307                                                   </td><td>1000                                                     </td><td>OKLAHOMA STATE UNIVERSITY                                </td><td>Mason                                                    </td><td>C                                                        </td><td>GENERAL PURCHASE                                         </td><td>890.00                                                   </td><td>NACAS                                                    </td><td>07/30/2013 12:00:00 AM                                   </td><td>07/31/2013 12:00:00 AM                                   </td><td>CHARITABLE AND SOCIAL SERVICE ORGANIZATIONS              </td></tr>
	<tr><td>201307                                                   </td><td>1000                                                     </td><td>OKLAHOMA STATE UNIVERSITY                                </td><td>Mason                                                    </td><td>C                                                        </td><td>ROOM CHARGES                                             </td><td>368.96                                                   </td><td>SHERATON HOTEL                                           </td><td>07/30/2013 12:00:00 AM                                   </td><td>07/31/2013 12:00:00 AM                                   </td><td>SHERATON                                                 </td></tr>
	<tr><td>201307                                                   </td><td>1000                                                     </td><td>OKLAHOMA STATE UNIVERSITY                                </td><td>Massey                                                   </td><td>J                                                        </td><td>GENERAL PURCHASE                                         </td><td>165.82                                                   </td><td>SEARS.COM 9300                                           </td><td>07/29/2013 12:00:00 AM                                   </td><td>07/31/2013 12:00:00 AM                                   </td><td>DIRCT MARKETING/DIRCT MARKETERS--NOT ELSEWHERE CLASSIFIED</td></tr>
	<tr><td>201307                                                   </td><td>1000                                                     </td><td>OKLAHOMA STATE UNIVERSITY                                </td><td>Massey                                                   </td><td>T                                                        </td><td>GENERAL PURCHASE                                         </td><td> 96.39                                                   </td><td>WAL-MART #0137                                           </td><td>07/30/2013 12:00:00 AM                                   </td><td>07/31/2013 12:00:00 AM                                   </td><td>GROCERY STORES,AND SUPERMARKETS                          </td></tr>
	<tr><td>201307                                                   </td><td>1000                                                     </td><td>OKLAHOMA STATE UNIVERSITY                                </td><td>Mauro-Herrera                                            </td><td>M                                                        </td><td>HAMMERMILL COPY PLUS COPY EA                             </td><td>125.96                                                   </td><td>STAPLES DIRECT                                           </td><td>07/30/2013 12:00:00 AM                                   </td><td>07/31/2013 12:00:00 AM                                   </td><td>STATIONERY, OFFICE SUPPLIES, PRINTING AND WRITING PAPER  </td></tr>
	<tr><td>201307                                                   </td><td>1000                                                     </td><td>OKLAHOMA STATE UNIVERSITY                                </td><td>Mauro-Herrera                                            </td><td>M                                                        </td><td>GENERAL PURCHASE                                         </td><td>394.28                                                   </td><td>KYOCERA DOCUMENT SOLUTION                                </td><td>07/29/2013 12:00:00 AM                                   </td><td>07/31/2013 12:00:00 AM                                   </td><td>OFFICE, PHOTOGRAPHIC, PHOTOCOPY, AND MICROFILM EQUIPMENT </td></tr>
</tbody>
</table>




```python
summary(ccard)
```


       Year_Month     Agency_Number  
     Min.   :201307   Min.   : 1000  
     1st Qu.:201309   1st Qu.: 1000  
     Median :201401   Median :47700  
     Mean   :201357   Mean   :42786  
     3rd Qu.:201404   3rd Qu.:76000  
     Max.   :201406   Max.   :98000  
                                     
                                    Agency_Name    
     OKLAHOMA STATE UNIVERSITY            :115995  
     UNIVERSITY OF OKLAHOMA               : 76143  
     UNIV. OF OKLA. HEALTH SCIENCES CENTER: 58247  
     DEPARTMENT OF CORRECTIONS            : 22322  
     DEPARTMENT OF TOURISM AND RECREATION : 17232  
     DEPARTMENT OF TRANSPORTATION         : 15689  
     (Other)                              :136830  
                   Cardholder_Last_Name Cardholder_First_Initial
     JOURNEY HOUSE TRAVEL INC: 10137    J      : 55031          
     UNIVERSITY AMERICAN     :  7219    G      : 42251          
     JOURNEY HOUSE TRAVEL    :  4693    D      : 38120          
     Heusel                  :  4212    M      : 35352          
     Hines                   :  3423    S      : 34698          
     Bowers                  :  2448    C      : 33213          
     (Other)                 :410326    (Other):203793          
                            Description         Amount         
     GENERAL PURCHASE             :247187   Min.   : -42863.0  
     AIR TRAVEL                   : 29584   1st Qu.:     30.9  
     ROOM CHARGES                 : 18120   Median :    104.9  
     AT&T SERVICE PAYMENT ITM     :  2657   Mean   :    425.0  
     001 Priority          1LB PCE:  2005   3rd Qu.:    345.0  
     000000000000000000000000     :  1828   Max.   :1903858.4  
     (Other)                      :141077                      
                           Vendor                     Transaction_Date 
     STAPLES                  : 14842   09/11/2013 12:00:00 AM:  2122  
     AMAZON MKTPLACE PMTS     : 12197   08/07/2013 12:00:00 AM:  2108  
     WW GRAINGER              : 12076   01/14/2014 12:00:00 AM:  2059  
     Amazon.com               : 10766   01/16/2014 12:00:00 AM:  2009  
     BILL WARREN OFFICE PRODUC:  4479   09/05/2013 12:00:00 AM:  1999  
     LOWES #00241             :  4231   10/01/2013 12:00:00 AM:  1996  
     (Other)                  :383867   (Other)               :430165  
                     Posted_Date    
     01/13/2014 12:00:00 AM:  3256  
     04/14/2014 12:00:00 AM:  3163  
     03/10/2014 12:00:00 AM:  3139  
     03/03/2014 12:00:00 AM:  3101  
     09/16/2013 12:00:00 AM:  3062  
     01/20/2014 12:00:00 AM:  3032  
     (Other)               :423705  
                                                   Merchant_Category 
     STATIONERY, OFFICE SUPPLIES, PRINTING AND WRITING PAPER: 24860  
     BOOK STORES                                            : 21981  
     INDUSTRIAL SUPPLIES NOT ELSEWHERE CLASSIFIED           : 21669  
     DENTAL/LABORATORY/MEDICAL/OPHTHALMIC HOSP EQIP AND SUP.: 20183  
     GROCERY STORES,AND SUPERMARKETS                        : 17152  
     MISCELLANEOUS AND SPECIALTY RETAIL STORES              : 13335  
     (Other)                                                :323278  


### Exploratory Data Analysis (EDA) <a class="anchor" id="initial_eda"></a>

#### Visualization by Agency Name

I started by looking at the agency name and calculating the statistical values for each agency, then created visualizations for some important statistical values, such as mean, sum, min, max, etc.


```python
# Create statistic tables by agency
stat_by_agency <- ccard %>% group_by(Agency_Name) %>%
    summarise(count = n(),
              amount = sum(Amount),
              mean = mean(Amount),
              min = min(Amount),
              max = max(Amount)
             ) %>%
    arrange(desc(amount)) %>% ungroup()

# Calculate the percentage of total values of transactions per agency
stat_by_agency <- stat_by_agency %>%
    mutate(row = rep(1:nrow(stat_by_agency)),
          Agency_Name_ind = paste(row,Agency_Name,sep="_"),
          percent = amount/sum(amount)) %>%
          arrange(desc(percent)) %>%
    select(Agency_Name_ind,count, amount, percent,mean, min, max)
```

The stat_by_agency shows the percent of total amount of transactions of each agency in descending order. 


```python
head(stat_by_agency)
```


<table>
<thead><tr><th scope=col>Agency_Name_ind</th><th scope=col>count</th><th scope=col>amount</th><th scope=col>percent</th><th scope=col>mean</th><th scope=col>min</th><th scope=col>max</th></tr></thead>
<tbody>
	<tr><td>1_OKLAHOMA STATE UNIVERSITY            </td><td>115995                                 </td><td>33778840                               </td><td>0.17963575                             </td><td> 291.2094                              </td><td> -6266.53                              </td><td>  27967.38                             </td></tr>
	<tr><td>2_UNIVERSITY OF OKLAHOMA               </td><td> 76143                                 </td><td>24886383                               </td><td>0.13234570                             </td><td> 326.8374                              </td><td>-41740.00                              </td><td> 114203.17                             </td></tr>
	<tr><td>3_UNIV. OF OKLA. HEALTH SCIENCES CENTER</td><td> 58247                                 </td><td>24527325                               </td><td>0.13043623                             </td><td> 421.0916                              </td><td> -7188.61                              </td><td>1903858.37                             </td></tr>
	<tr><td>4_GRAND RIVER DAM AUTH.                </td><td> 10427                                 </td><td>22213829                               </td><td>0.11813306                             </td><td>2130.4142                              </td><td> -9000.00                              </td><td>1089180.00                             </td></tr>
	<tr><td>5_DEPARTMENT OF TRANSPORTATION         </td><td> 15689                                 </td><td>14399262                               </td><td>0.07657522                             </td><td> 917.7935                              </td><td>-34108.00                              </td><td> 348053.75                             </td></tr>
	<tr><td>6_DEPARTMENT OF CORRECTIONS            </td><td> 22322                                 </td><td>13988872                               </td><td>0.07439277                             </td><td> 626.6854                              </td><td>-20000.00                              </td><td>  96190.38                             </td></tr>
</tbody>
</table>




```python
# Subtract the table 
temp <-stat_by_agency[1:30,]
```


```python
gg<- ggplot() + geom_bar(aes(reorder(temp$Agency_Name_ind,temp$percent),temp$percent), stat = 'identity',
                        color = "blue" , fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + labs(
    title = "Percentage of Total Expense by Agency",
    x = "Agency Name",
    y = "Percentage") 

gg + coord_flip()
```


![png](/assets/img/creditcard/output_17_0.png)


#### Feature Selection & Engineering

The dataset contains of information of agency number, agency name, cardholder, purchase description, amount, vendor, transaction, merchant category. The variables we are going to use are:
- Agency_number: unique ID of each agency
- Agency_name: Name of agency
- Cardholder_Last_Name: Last name of cardholder under each business
- Cardholder.First.Initial: First initial name of cardholder under each business
- Description: Informaion of expense
- Amount: The amount spend for each transaction. 
- Vendor: The recipient of the expense  
- Transaction_date: The transaction's date
- Posted_date: The date that the transaction went through
- Merchant_Category: Expense category

**First Step: Data Preprocessing**
  - Setting format for Transaction_Date and Posted_Date variables.
  - Creating Time variable to see the duration from transaction date to post date for each transaction. 
  - Creating a month_yr variables.


```python
time_by_agency <- ccard %>% group_by(Agency_Name) %>%
    mutate(Transaction_Date=as.Date(Transaction_Date,format="%m/%d/%Y %H:%M")) %>%
    mutate(Posted_Date=as.Date(Posted_Date,format="%m/%d/%Y %H:%M")) %>%
    arrange(Agency_Name,Transaction_Date, Posted_Date) %>%
    mutate(Time = Posted_Date-Transaction_Date)

# Adding month_yr column into the dataframe
time_by_agency$month_yr <- format(as.Date(time_by_agency$Transaction_Date), "%Y-%m")

time_by_agency[,c("Agency_Number","Agency_Name", "Transaction_Date","Posted_Date","Vendor","Merchant_Category",
                  "Description","Amount","Time", "month_yr")][,1:10]
```


<table>
<thead><tr><th scope=col>Agency_Number</th><th scope=col>Agency_Name</th><th scope=col>Transaction_Date</th><th scope=col>Posted_Date</th><th scope=col>Vendor</th><th scope=col>Merchant_Category</th><th scope=col>Description</th><th scope=col>Amount</th><th scope=col>Time</th><th scope=col>month_yr</th></tr></thead>
<tbody>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-06-29                                             </td><td>2013-07-01                                             </td><td>FACEBK  CK7ZD4WK52                                     </td><td>ADVERTISING SERVICES                                   </td><td>GENERAL PURCHASE                                       </td><td> 415.85                                                </td><td>2 days                                                 </td><td>2013-06                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-01                                             </td><td>2013-07-01                                             </td><td>FACEBK  MB2EF4WL52                                     </td><td>ADVERTISING SERVICES                                   </td><td>GENERAL PURCHASE                                       </td><td>  96.14                                                </td><td>0 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-01                                             </td><td>2013-07-02                                             </td><td>Amazon.com                                             </td><td>BOOK STORES                                            </td><td>Magna Cart Personal Hand T PCE                         </td><td>  68.00                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-04                                             </td><td>WAL-MART #2804                                         </td><td>GROCERY STORES,AND SUPERMARKETS                        </td><td>GENERAL PURCHASE                                       </td><td>  82.28                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-04                                             </td><td>WAL-MART #2804                                         </td><td>GROCERY STORES,AND SUPERMARKETS                        </td><td>GENERAL PURCHASE                                       </td><td>  -2.58                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-04                                             </td><td>TECH-LOCK INC                                          </td><td>BUSINESS SERVICES NOT ELSEWHERE CLASSIFIED             </td><td>GENERAL PURCHASE                                       </td><td>   9.50                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-04                                             </td><td>JOURNYHSE   HUGHES                                     </td><td>TRAVEL AGENCIES                                        </td><td>GENERAL PURCHASE                                       </td><td>  25.00                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-04                                             </td><td>JOURNYHSE   KNIGHT                                     </td><td>TRAVEL AGENCIES                                        </td><td>GENERAL PURCHASE                                       </td><td>  25.00                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-04                                             </td><td>JOURNYHSE   HONEYSUCKL                                 </td><td>TRAVEL AGENCIES                                        </td><td>GENERAL PURCHASE                                       </td><td>  25.00                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-05                                             </td><td>AMERICAN AI 0017289901645                              </td><td>AMERICAN AIRLINES                                      </td><td>AIR TRAVEL                                             </td><td> 721.60                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-05                                             </td><td>AMERICAN AI 0017289901648                              </td><td>AMERICAN AIRLINES                                      </td><td>AIR TRAVEL                                             </td><td> 851.60                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-05                                             </td><td>GAYLORD NATIONAL F/D                                   </td><td>GAYLORD OPRYLAND                                       </td><td>GENERAL PURCHASE                                       </td><td> -76.32                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-05                                             </td><td>OFFICE DEPOT #1079                                     </td><td>COMBINATION CATALOG AND RETAIL MERCHANT                </td><td>CHAIRMIDBACKLEATHERBL NMB                              </td><td> 179.99                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-05                                             </td><td>AMERICAN AI 0017289901722                              </td><td>AMERICAN AIRLINES                                      </td><td>AIR TRAVEL                                             </td><td> 721.60                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-03                                             </td><td>2013-07-09                                             </td><td>TOYS FOR SPECIAL CHILDREN                              </td><td>HOBBY,TOY,AND GAME STORES                              </td><td>GENERAL PURCHASE                                       </td><td>1005.55                                                </td><td>6 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-05                                             </td><td>2013-07-08                                             </td><td>AMERICAN AI 0017290513281                              </td><td>AMERICAN AIRLINES                                      </td><td>AIR TRAVEL                                             </td><td> 620.60                                                </td><td>3 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-05                                             </td><td>2013-07-08                                             </td><td>JOURNYHSE   BLUNDELL                                   </td><td>TRAVEL AGENCIES                                        </td><td>GENERAL PURCHASE                                       </td><td>  25.00                                                </td><td>3 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-08                                             </td><td>2013-07-09                                             </td><td>JOURNYHSE   SHARP                                      </td><td>TRAVEL AGENCIES                                        </td><td>GENERAL PURCHASE                                       </td><td>  25.00                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-08                                             </td><td>2013-07-09                                             </td><td>JOURNYHSE   REEVES                                     </td><td>TRAVEL AGENCIES                                        </td><td>GENERAL PURCHASE                                       </td><td>  25.00                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-08                                             </td><td>2013-07-09                                             </td><td>JOURNYHSE   VALENZUELA                                 </td><td>TRAVEL AGENCIES                                        </td><td>GENERAL PURCHASE                                       </td><td>  25.00                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-08                                             </td><td>2013-07-09                                             </td><td>JOURNYHSE   LOFTIN                                     </td><td>TRAVEL AGENCIES                                        </td><td>GENERAL PURCHASE                                       </td><td>  25.00                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-08                                             </td><td>2013-07-10                                             </td><td>AMERICAN AI 0017290513339                              </td><td>AMERICAN AIRLINES                                      </td><td>AIR TRAVEL                                             </td><td> 721.60                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-08                                             </td><td>2013-07-10                                             </td><td>AMERICAN AI 0017290513361                              </td><td>AMERICAN AIRLINES                                      </td><td>AIR TRAVEL                                             </td><td> 721.60                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-08                                             </td><td>2013-07-10                                             </td><td>AMERICAN AI 0017290513338                              </td><td>AMERICAN AIRLINES                                      </td><td>AIR TRAVEL                                             </td><td> 721.60                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-08                                             </td><td>2013-07-10                                             </td><td>AMERICAN AI 0017290513360                              </td><td>AMERICAN AIRLINES                                      </td><td>AIR TRAVEL                                             </td><td> 721.60                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-09                                             </td><td>2013-07-10                                             </td><td>STAPLES                                                </td><td>STATIONERY, OFFICE SUPPLIES, PRINTING AND WRITING PAPER</td><td>CLEANING PAD CRT SCREEN TW BX                          </td><td>  17.38                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-09                                             </td><td>2013-07-10                                             </td><td>JOURNYHSE   BARRESI                                    </td><td>TRAVEL AGENCIES                                        </td><td>GENERAL PURCHASE                                       </td><td>  25.00                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-09                                             </td><td>2013-07-10                                             </td><td>STAPLES                                                </td><td>STATIONERY, OFFICE SUPPLIES, PRINTING AND WRITING PAPER</td><td>BATTERY ALKALN D 12/PK PK|STAPLES PAD PERF LTR WHI     </td><td>  35.09                                                </td><td>1 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-09                                             </td><td>2013-07-11                                             </td><td>HOBBY LOBBY #124                                       </td><td>HOBBY,TOY,AND GAME STORES                              </td><td>GENERAL PURCHASE                                       </td><td> 114.47                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>26500                                                  </td><td>`DEPARTMENT OF EDUCATION                               </td><td>2013-07-09                                             </td><td>2013-07-11                                             </td><td>AMERICAN AI 0017290513398                              </td><td>AMERICAN AIRLINES                                      </td><td>AIR TRAVEL                                             </td><td> 583.60                                                </td><td>2 days                                                 </td><td>2013-07                                                </td></tr>
	<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-03-11                                                   </td><td>2014-03-12                                                   </td><td>AMAZON MKTPLACE PMTS                                         </td><td>BOOK STORES                                                  </td><td>VIZIO E241-A1 24-inch 1080 PCE                               </td><td>225.49                                                       </td><td>1 days                                                       </td><td>2014-03                                                      </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-03-11                                                   </td><td>2014-03-12                                                   </td><td>Amazon.com                                                   </td><td>COMPUTER NETWORK/INFORMATION SERVICES                        </td><td>VIZIO E320i-B2 32-Inch 720 PCE                               </td><td>294.74                                                       </td><td>1 days                                                       </td><td>2014-03                                                      </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-03-19                                                   </td><td>2014-03-20                                                   </td><td>PRYOR PRINTING INC                                           </td><td>MISCELLANEOUS PUBLISHING AND PRINTING SERVICES               </td><td>GENERAL PURCHASE                                             </td><td>294.89                                                       </td><td>1 days                                                       </td><td>2014-03                                                      </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-03-21                                                   </td><td>2014-03-24                                                   </td><td>BACKBLAZE BACKUP                                             </td><td>COMPUTERS, COMPUTER PERIPHERAL EQUIPMENT, SOFTWARE           </td><td>GENERAL PURCHASE                                             </td><td>450.00                                                       </td><td>3 days                                                       </td><td>2014-03                                                      </td></tr>
	<tr><td>88000                                                                                                </td><td>WILL ROGERS MEMORIAL COMMISSION                                                                      </td><td>2014-03-24                                                                                           </td><td>2014-03-24                                                                                           </td><td><span style=white-space:pre-wrap>RACKSPACE EMAIL &amp; APPS   </span>                                </td><td><span style=white-space:pre-wrap>COMPUTER NETWORK/INFORMATION SERVICES                        </span></td><td><span style=white-space:pre-wrap>GENERAL PURCHASE                                  </span>           </td><td>109.00                                                                                               </td><td>0 days                                                                                               </td><td>2014-03                                                                                              </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-04-09                                                   </td><td>2014-04-09                                                   </td><td>GOTPRINT.COM                                                 </td><td>MISCELLANEOUS PUBLISHING AND PRINTING SERVICES               </td><td>GENERAL PURCHASE                                             </td><td> 55.29                                                       </td><td>0 days                                                       </td><td>2014-04                                                      </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-04-22                                                   </td><td>2014-04-23                                                   </td><td>USPS 39174902033603861                                       </td><td>POSTAGE STAMPS                                               </td><td>GENERAL PURCHASE                                             </td><td> 51.00                                                       </td><td>1 days                                                       </td><td>2014-04                                                      </td></tr>
	<tr><td>88000                                                                                                </td><td>WILL ROGERS MEMORIAL COMMISSION                                                                      </td><td>2014-04-24                                                                                           </td><td>2014-04-24                                                                                           </td><td><span style=white-space:pre-wrap>RACKSPACE EMAIL &amp; APPS   </span>                                </td><td><span style=white-space:pre-wrap>COMPUTER NETWORK/INFORMATION SERVICES                        </span></td><td><span style=white-space:pre-wrap>GENERAL PURCHASE                                  </span>           </td><td>109.00                                                                                               </td><td>0 days                                                                                               </td><td>2014-04                                                                                              </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-04-25                                                   </td><td>2014-04-28                                                   </td><td>TULSAWORLD.COM                                               </td><td>BUSINESS SERVICES NOT ELSEWHERE CLASSIFIED                   </td><td>30 DAY JOB POSTING   30 DA                                   </td><td>350.00                                                       </td><td>3 days                                                       </td><td>2014-04                                                      </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-04-25                                                   </td><td>2014-04-28                                                   </td><td>JOBTARGET LLC                                                </td><td>EMPLOYMENT AGENCIES AND TEMPORARY HELP SERVICES              </td><td>GENERAL PURCHASE                                             </td><td>250.00                                                       </td><td>3 days                                                       </td><td>2014-04                                                      </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-05-01                                                   </td><td>2014-05-05                                                   </td><td>SOUTH CENTRAL INDUSTRIES                                     </td><td>DIRCT MARKETING/DIRCT MARKETERS--NOT ELSEWHERE CLASSIFIED    </td><td>GENERAL PURCHASE                                             </td><td>503.84                                                       </td><td>4 days                                                       </td><td>2014-05                                                      </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-05-06                                                   </td><td>2014-05-07                                                   </td><td>AMAZON MKTPLACE PMTS                                         </td><td>BOOK STORES                                                  </td><td>Cutler Hammer BAB3050H 50A PCE                               </td><td>120.98                                                       </td><td>1 days                                                       </td><td>2014-05                                                      </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-05-14                                                   </td><td>2014-05-15                                                   </td><td>ADVENTURE AWNINGS AND SIG                                    </td><td>DURABLE GOODS, NOT ELSEWHERE CLASSIFIED                      </td><td>GENERAL PURCHASE                                             </td><td> 40.00                                                       </td><td>1 days                                                       </td><td>2014-05                                                      </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-05-15                                                   </td><td>2014-05-16                                                   </td><td>Amazon.com                                                   </td><td>BOOK STORES                                                  </td><td>VIZIO E420i-B0 42-Inch 108 PCE                               </td><td>428.00                                                       </td><td>1 days                                                       </td><td>2014-05                                                      </td></tr>
	<tr><td>88000                                                                                                </td><td>WILL ROGERS MEMORIAL COMMISSION                                                                      </td><td>2014-05-24                                                                                           </td><td>2014-05-26                                                                                           </td><td><span style=white-space:pre-wrap>RACKSPACE EMAIL &amp; APPS   </span>                                </td><td><span style=white-space:pre-wrap>COMPUTER NETWORK/INFORMATION SERVICES                        </span></td><td><span style=white-space:pre-wrap>GENERAL PURCHASE                                  </span>           </td><td>109.00                                                                                               </td><td>2 days                                                                                               </td><td>2014-05                                                                                              </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-05-30                                                   </td><td>2014-06-02                                                   </td><td>BUDDY'S GRILL                                                </td><td>EATING PLACES AND RESTAURANTS                                </td><td>GENERAL PURCHASE                                             </td><td>494.00                                                       </td><td>3 days                                                       </td><td>2014-05                                                      </td></tr>
	<tr><td>88000                                                                                                </td><td>WILL ROGERS MEMORIAL COMMISSION                                                                      </td><td>2014-06-24                                                                                           </td><td>2014-06-24                                                                                           </td><td><span style=white-space:pre-wrap>RACKSPACE EMAIL &amp; APPS   </span>                                </td><td><span style=white-space:pre-wrap>COMPUTER NETWORK/INFORMATION SERVICES                        </span></td><td><span style=white-space:pre-wrap>GENERAL PURCHASE                                  </span>           </td><td>109.00                                                                                               </td><td>0 days                                                                                               </td><td>2014-06                                                                                              </td></tr>
	<tr><td>88000                                                        </td><td>WILL ROGERS MEMORIAL COMMISSION                              </td><td>2014-06-28                                                   </td><td>2014-06-30                                                   </td><td>ATT BILL PAYMENT                                             </td><td>CABLE, SATELLITE, AND OTHER PAY TELEVISION AND RADIO SERVICES</td><td>132186573 ITM                                                </td><td> 98.00                                                       </td><td>2 days                                                       </td><td>2014-06                                                      </td></tr>
	<tr><td>86500                                                        </td><td>WORKER'S COMP. COMMISSION                                    </td><td>2014-03-12                                                   </td><td>2014-03-14                                                   </td><td>STONECE.COM                                                  </td><td>BUSINESS AND SECRETARIAL SCHOOLS                             </td><td>GENERAL PURCHASE                                             </td><td> 60.00                                                       </td><td>2 days                                                       </td><td>2014-03                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2013-07-24                                                   </td><td>2013-07-24                                                   </td><td>DMI  DELL K-12/GOVT                                          </td><td>COMPUTERS, COMPUTER PERIPHERAL EQUIPMENT, SOFTWARE           </td><td>EAR CUSHION FOAM 2 CS351 C PCE|EAR CUSHIONS LTHR 2           </td><td> 41.76                                                       </td><td>0 days                                                       </td><td>2013-07                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2013-07-24                                                   </td><td>2013-07-24                                                   </td><td>DMI  DELL K-12/GOVT                                          </td><td>COMPUTERS, COMPUTER PERIPHERAL EQUIPMENT, SOFTWARE           </td><td>American Power Conversion PCE                                </td><td>130.82                                                       </td><td>0 days                                                       </td><td>2013-07                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2013-08-02                                                   </td><td>2013-08-05                                                   </td><td>SHI CORP                                                     </td><td>COMPUTERS, COMPUTER PERIPHERAL EQUIPMENT, SOFTWARE           </td><td>GENERAL PURCHASE                                             </td><td>217.00                                                       </td><td>3 days                                                       </td><td>2013-08                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2013-08-16                                                   </td><td>2013-08-19                                                   </td><td>DMI  DELL K-12/GOVT                                          </td><td>COMPUTERS, COMPUTER PERIPHERAL EQUIPMENT, SOFTWARE           </td><td>EAR CUSHION FOAM 2 CS351 C PCE|SPAREEAR CUSHIONLEA           </td><td> 83.52                                                       </td><td>3 days                                                       </td><td>2013-08                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2013-08-27                                                   </td><td>2013-08-29                                                   </td><td>SHI CORP                                                     </td><td>COMPUTERS, COMPUTER PERIPHERAL EQUIPMENT, SOFTWARE           </td><td>GENERAL PURCHASE                                             </td><td> 87.00                                                       </td><td>2 days                                                       </td><td>2013-08                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2013-08-28                                                   </td><td>2013-08-28                                                   </td><td>DMI  DELL K-12/GOVT                                          </td><td>COMPUTERS, COMPUTER PERIPHERAL EQUIPMENT, SOFTWARE           </td><td>Dell Wireless Mouse Blue PCE                                 </td><td>107.95                                                       </td><td>0 days                                                       </td><td>2013-08                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2013-09-26                                                   </td><td>2013-09-27                                                   </td><td>CDW GOVERNMENT                                               </td><td>CATALOG MERCHANTS                                            </td><td>KINGSTON 8GB DT 101 GEN PCB                                  </td><td> 31.00                                                       </td><td>1 days                                                       </td><td>2013-09                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2013-10-22                                                   </td><td>2013-10-23                                                   </td><td>CDW GOVERNMENT                                               </td><td>CATALOG MERCHANTS                                            </td><td>APC BATTERY UPS SURGE 10 PCB                                 </td><td>195.00                                                       </td><td>1 days                                                       </td><td>2013-10                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2014-01-07                                                   </td><td>2014-01-08                                                   </td><td>SHI CORP                                                     </td><td>COMPUTERS, COMPUTER PERIPHERAL EQUIPMENT, SOFTWARE           </td><td>GENERAL PURCHASE                                             </td><td>217.00                                                       </td><td>1 days                                                       </td><td>2014-01                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2014-01-08                                                   </td><td>2014-01-09                                                   </td><td>CDW GOVERNMENT                                               </td><td>CATALOG MERCHANTS                                            </td><td>STARTECH 6FT DVI DUAL LI PCB                                 </td><td> 50.00                                                       </td><td>1 days                                                       </td><td>2014-01                                                      </td></tr>
	<tr><td>36900                                                        </td><td>WORKERS COMPENSATION COURT                                   </td><td>2014-01-27                                                   </td><td>2014-01-28                                                   </td><td>SYNERGY DATACOM SUPPLY                                       </td><td>TELECOMMUNICATION EQUIPMENT AND TELEPHONE SALES              </td><td>Telecommunication Services EA                                </td><td>324.71                                                       </td><td>1 days                                                       </td><td>2014-01                                                      </td></tr>
</tbody>
</table>



### 1: Average amount charged per month by agency<a class="anchor" id="feature1"></a>

This feature will provide an estimation of the average amount spent per month for each agency. If there are any anomalies present, such as unusual high charges per month, it would be noticed and investigated further. 


```python
mean_by_agency <- time_by_agency %>%
          group_by(Agency_Name, month_yr) %>%
          summarise(month_average = mean(Amount))
```

From the output result, we will be able to track the monthly payment of each agency, below is an example of using scatter chart to identify monthly spending


```python
sample_1 <- mean_by_agency %>% filter(Agency_Name == "ATTORNEY GENERAL")
```


```python
ggplot(sample_1, aes(x = month_yr , y = month_average)) +
        geom_point(alpha = 0.5 , size = 1.5, color = "blue", fill = "blue") + 
        labs(x = 'Time' , y = 'Monthly Spending')
```


![png](/assets/img/creditcard/output_25_0.png)


**Conclusion**: The chart above shows that agency Attorney General made an unusually large payment in the month of June of 2013. However, their monthly spending after that seems very regular. Additionally, this feature shows that the agency payment would be predictable, such as decreasing spend every 3 months. This insight highlights the business use of this feature and shows how it will be helpful with detecting anomalies.

### 2: Total number of transactions per month by agency<a class="anchor" id="feature2"></a>

This feature helps detecting if an agency has made more transactions than they previously had, indicating potential of fraud. It filters by agency name and shows the number of transactions over time. 


```python
count_by_agency  <- time_by_agency %>%
          group_by(Agency_Name, month_yr) %>%
          summarise(count = n())
```


```python
sample_2 <- count_by_agency %>% filter(Agency_Name == "ATTORNEY GENERAL")
nrow(sample_2)
```


13



```python
ggplot() + geom_bar(aes(reorder(sample_2$month_yr,sample_2$count),sample_2$count), stat = 'identity',
                    color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + labs(
    title = "Total Number of Transactions per Month by Agency",
    x = "Time",
    y = "Number of Transactions") 
```


![png](/assets/img/creditcard/output_30_0.png)


**Conclusion**: Through the chart, it shows that there is an upward trend of transactions, which is likely to not be problematic. If there are sudden increases, it may predict the possibility of fraud and that further exploration is needed.

### 3: Average spending amount per transaction for each agency<a class="anchor" id="feature3"></a>
This feature shows the average amount spend in the transaction by an agency. It helps to show if there is a sudden increase compared to all agencies or if tracked monthly, shows how the transactions change.


```python
# Sort and subtract mean value in descending order
mean_data <- ccard %>% group_by(Agency_Name) %>%
    summarise(mean_amount = mean(Amount)) %>%
    arrange(desc(mean_amount)) %>% ungroup() 
# Subtract top-30 highest mean values
sample_3 <- mean_data[1:15,]
```


```python
gg2 <- ggplot() + geom_bar(aes(reorder(sample_3$Agency_Name,sample_3$mean_amount),sample_3$mean_amount), stat = 'identity',
                           color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(text = element_text(size=7),
    axis.text.x = element_text(angle = 90, hjust = 1)) + labs(
    title = "Average Amount per Transaction by Agency Name",
    x = "Agency Name",
    y = "Value")
gg2 + coord_flip()
```


![png](/assets/img/creditcard/output_34_0.png)



```python
emp_bene <- time_by_agency %>% filter(Agency_Name == "EMPLOYEES BENEFITS DEPARTMENT") %>% group_by(Vendor,Merchant_Category)
emp_bene
```


<table>
<thead><tr><th scope=col>Year_Month</th><th scope=col>Agency_Number</th><th scope=col>Agency_Name</th><th scope=col>Cardholder_Last_Name</th><th scope=col>Cardholder_First_Initial</th><th scope=col>Description</th><th scope=col>Amount</th><th scope=col>Vendor</th><th scope=col>Transaction_Date</th><th scope=col>Posted_Date</th><th scope=col>Merchant_Category</th><th scope=col>Time</th><th scope=col>month_yr</th></tr></thead>
<tbody>
	<tr><td>201307                       </td><td>81500                        </td><td>EMPLOYEES BENEFITS DEPARTMENT</td><td>81500                        </td><td>8                            </td><td>GENERAL PURCHASE             </td><td>343148.50                    </td><td>PAYMENT ADJUSTMENT           </td><td>2013-06-13                   </td><td>2013-07-03                   </td><td>OTHER FEES                   </td><td>20 days                      </td><td>2013-06                      </td></tr>
	<tr><td>201307                       </td><td>81500                        </td><td>EMPLOYEES BENEFITS DEPARTMENT</td><td>81500                        </td><td>8                            </td><td>GENERAL PURCHASE             </td><td>    90.72                    </td><td>CBR MAN CK                   </td><td>2013-07-08                   </td><td>2013-07-09                   </td><td>OTHER FEES                   </td><td> 1 days                      </td><td>2013-07                      </td></tr>
</tbody>
</table>



***Conclusion***: This feature shows an average amount of spending per transaction and the chart above shows that Employees Benefits Department has extremely high average spending per transaction, which would need to be investigated further. Additionally, we could use this feature to make comparsons for upcoming monthly expenses by agency.

In this case, the agency only made 2 transactions, which are both in 2013 while the data is up to June, 2014. The agency has not used the card for almost a year and the amounts of these transactions are odd. 

### 4: The average amount spent per day every month <a class="anchor" id="feature4"></a>

This feature helps to show how much an agency spends per day each month. If there is a increase of the average expenses, it indicates that something has changed and should be investigated. 


```python
daily_by_agency <- time_by_agency %>%
          group_by(Agency_Name, month_yr) %>%
          summarise(daily_amount = sum(Amount)/30)
```


```python
# Example of using the 
sample_4 <- daily_by_agency%>% filter(Agency_Name == "ATTORNEY GENERAL") 
```


```python
gg4 <- ggplot() + geom_bar(aes(sample_4$month_yr,sample_4$daily_amount), 
                           stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + labs(
    title = "Average amount spent per day every month",
    x = "Time",
    y = "Daily Expense") 

gg4
```


![png](/assets/img/creditcard/output_40_0.png)


***Conclusion***: As shown in the graph, there is a sudden increase that indicates a change in normal behaviour for this agency. If the feature is monitored, it can help predict if anomalies are present.

### 5: Total number of transactions with the same vendor during the past 30 days/per month <a class="anchor" id="feature5"></a>

This feature shows how often an agency makes transaction with vendors. It takes the transactions over the past 30 days and compares how many transactions are made per vendor.


```python
count_by_agency_vendor <- time_by_agency %>% group_by(Agency_Name, Vendor, month_yr) %>%
    summarise(count_amt = n()) %>%
    arrange(desc(count_amt)) %>% ungroup() 
```


```python
# 2014-06 is the lastest month in the dataset
sample_5 <- count_by_agency_vendor%>% filter(Agency_Name == "ATTORNEY GENERAL", month_yr == "2014-06" ) 
nrow(sample_5)
```


82



```python
# Top 10 total number of transactions with the same merchant in the past 30 days
sample_5 <- sample_5[1:10,]

gg5 <- ggplot() + geom_bar(aes(reorder(sample_5$Vendor,sample_5$count_amt),sample_5$count_amt), stat = 'identity',
                          color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(text = element_text(size=8),
        axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Top 10 total number of transactions with the same merchant",
        x = "Vendor",
        y = "Total Transactions")
gg5 +coord_flip()
```


![png](/assets/img/creditcard/output_45_0.png)


***Conclusion*** : As shown in the graph, this agency has the highest number of transactions with Staples. If this figure was tracked on a moving average basis, it can help determine if there is a change in the frequency of transaction to help predict anomalies if there is a change.

### 6: Average Amount per day spent on the same vendor during the last 30 days <a class="anchor" id="feature6"></a>

This feature looks at the transactions an agency makes over the last 30 days and tracks the average amount spent per day. It looks at a single month for a single agency, but can be expanded to track the amounts across all agencies.


```python
mean_by_agency_vendor <- time_by_agency %>% group_by(Agency_Name, Vendor, month_yr) %>%
    summarise(mean_vendor = mean(Amount)) %>%
    arrange(desc(mean_vendor)) %>% ungroup() 
```


```python
# 2014-06 is the lastest month in the dataset
sample_6 <- mean_by_agency_vendor%>% filter(Agency_Name == "ATTORNEY GENERAL", month_yr == "2014-06" ) 
nrow(sample_6)
```


82



```python
# Top 10 total number of transactions with the same merchant in the past 30 days
sample_6 <- sample_6[1:10,]

gg6 <- ggplot() + geom_bar(aes(reorder(sample_6$Vendor,sample_6$mean_vendor),sample_6$mean_vendor),
                           stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(text = element_text(size=8),
        axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Top 10 Average Amount Spent on the Same Vendor During a 30 Day Period",
        x = "Vendor",
        y = "Average Amount")
gg6 +coord_flip()
```


![png](/assets/img/creditcard/output_50_0.png)


***Insights*** :The feature shows that the agency, ATTORNEY GENERAL, spent over $800 on Thomsom West, almost 4 times than the second ranked average amount spending in June 2014. This amount may possibly not be abnormal if its similar for other months, but would require further investigation. 

This feature helps to identify the average amount spent per month with different vendor and would help to keep track if there is any irregular amount spent on these vendors. If there is a sudden increase, it can help predict if anomalies are present. 

### 7: Total number of transactions with the same merchant category <a class="anchor" id="feature7"></a>
This feature counts the number of transactions an agency has per merchant category. It looks at a single agency at a time and sums the amount within each category.


```python
count_by_agency_merchant <- ccard %>% group_by(Agency_Name, Merchant_Category) %>%
    summarise(count_amount = n()) %>%
    arrange(desc(count_amount)) %>% ungroup() 
```


```python
sample_7 <- count_by_agency_merchant%>% filter(Agency_Name == "ARDMORE HIGHER EDUCATION CENTER")
nrow(sample_7)
```


18



```python
# Visualize the top 20 most spending merchant category for one agency
gg7 <- ggplot() + geom_bar(aes(reorder(sample_7$Merchant_Category,sample_7$count_amount),sample_7$count_amount),
                           stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(text = element_text(size=8),
      axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Total Number of Transactions with the Same Category",
        x = "Category",
        y = "Value")
gg7 + coord_flip()
```


![png](/assets/img/creditcard/output_55_0.png)


***Conclusion***: The above graph shows the amount of transactions an agency has within a certain category. If monitored over time, it can show if there is a sudden increase for a category, indicating a possible issue. This feature helps in predicting anomalies by showing what an agency typically spends on by category and notices if there is change that should be investigated. 

### 8: The Average Amount Spent on the Same Merchant  Category<a class="anchor" id="feature8"></a>
This feature looks at the average amount that is spent by an agency for a merchant category. It looks at the amount that is spent to compare changes that occur over time.


```python
mean_by_agency_merchant <- ccard %>% group_by(Agency_Name, Merchant_Category) %>%
    summarise(mean_amount = mean(Amount)) %>%
    arrange(desc(mean_amount)) %>% ungroup() 
```


```python
sample_8 <- mean_by_agency_merchant%>% filter(Agency_Name == "ARDMORE HIGHER EDUCATION CENTER")
```


```python
# Visualize the top 20 most spending merchant category for one agency
sample_8 <- sample_8[1:20,]

gg8 <- ggplot() + geom_bar(aes(reorder(sample_8$Merchant_Category,sample_8$mean_amount),sample_8$mean_amount),
                           stat = 'identity', color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Top 20 Most Spending Merchant Category",
        x = "Category",
        y = "Value")+
  theme(plot.title = element_text(size = 8),
        text = element_text(size=7))

gg8 + coord_flip()
```

    Warning message:
    "Removed 2 rows containing missing values (position_stack)."


![png](/assets/img/creditcard/output_60_1.png)


***Conclusion***: This feature helps to indicate what is a typical charge for an agency based on historical data. If future charges suddenly are higher than average, it helps to predict fraud by highlighting transactions that are much higher than what is expected. Through showing transactions that are higher, it can indicate issues which should be investigated.

### 9: Percentage of expenses by merchant category for an agency <a class="anchor" id="feature9"></a>
This feature takes transactions from a single agency and finds the percent of expenses by a category. It first makes a subset to find the sum of all transactions to determine what the percent transactions there are by category.


```python
# Change Agency_Name to explore different organizations
sample_9 <- ccard%>%filter(Agency_Name == "UNIVERSITY OF OKLAHOMA")
nrow(sample_9)
```


76143



```python
sum_by_agency_merchant <- sample_9 %>% group_by(Agency_Name, Merchant_Category) %>%
    summarise(total = sum(Amount)) %>%
    arrange(desc(total))%>% ungroup() 
```


```python
percent_by_agency_merchant <- sum_by_agency_merchant %>%
    mutate(row = rep(1:nrow(sum_by_agency_merchant)),
          percent = total/sum(total)*100) %>%
          arrange(desc(percent)) %>%
    select(Agency_Name,Merchant_Category,total, percent)
```


```python
# Visualize the top 20 highest percentage of expense by merchant category for an agency
percent_by_agency_merchant <- percent_by_agency_merchant[1:20,]

gg9 <- ggplot() + geom_bar(aes(reorder(percent_by_agency_merchant$Merchant_Category,
                                       percent_by_agency_merchant$percent),
                                       percent_by_agency_merchant$percent),
                           stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 1, hjust = 1)) + 
  labs(title = "Percentage of expense by merchant category for an agency",
        x = "Category",
        y = "Percentage")+
  theme(plot.title = element_text(size = 8),
        text = element_text(size=8))

gg9 + coord_flip()
```


![png](/assets/img/creditcard/output_66_0.png)


***Conclusion***: This feature highlights what an agency typically spends their money on and gives a percentage by category. Through monitoring it over time, it can predict if there is unusual activity if the percentage quickly changes.

### 10: Percentage of Expense by Vendor for an Agency <a class="anchor" id="feature10"></a>
This feature takes a sample from a single agency and groups their expenses by vendors. It then divides this number by the total transactions to find the percentage of dollars it spends at that vendor.


```python
# Change Agency_Name to explore different organizations
sample_10 <- ccard%>%filter(Agency_Name == "ATTORNEY GENERAL")
nrow(sample_10)
```


1495



```python
sum_by_agency_vendor <- sample_10 %>% group_by(Agency_Name, Vendor) %>%
    summarise(total = sum(Amount)) %>%
    arrange(desc(total))%>% ungroup() 
```


```python
percent_by_agency_vendor <- sum_by_agency_vendor %>%
    mutate(row = rep(1:nrow(sum_by_agency_vendor)),
          percent = total/sum(total)*100) %>%
          arrange(desc(percent)) %>%
    select(Agency_Name,Vendor,total, percent)
```


```python
# Visualize the top 20 highest percentage of expense by vendor for an agency
percent_by_agency_vendor <- percent_by_agency_vendor[1:20,]

gg10 <- ggplot() + geom_bar(aes(reorder(percent_by_agency_vendor$Vendor,
                                        percent_by_agency_vendor$percent),
                                percent_by_agency_vendor$percent),
                            stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Percentage of Expenses by Vendor for An Agency",
        x = "Vendor",
        y = "Percentage")+
  theme(plot.title = element_text(size = 8),
        text = element_text(size=7))

gg10 + coord_flip()
```


![png](/assets/img/creditcard/output_72_0.png)


***Conclusion***: The above graph creates a representation of where this agency spends their money. If this figure was tracked over time, it can help to predict if anything unusual is happening by comparing the change in percentages. Although new vendors may alter this list, it provides an idea of where their money is typically spent which can be further categorized for increased accuracy.

### 11: Total number of transactions with the same vendor during the past 3 months<a class="anchor" id="feature11"></a>
This feature takes the transaction from the latest 3 months of the dataset and then counts the number of transactions each agency has with all of their vendors. It then filters down the dataset to a single agency to compare their transactions with vendors. 


```python
total_by_agency_vendor <- time_by_agency %>% group_by(Agency_Name, Vendor) %>%
    # lastest month in the dataset is 2014-06
    filter(month_yr >= "2014-04")%>%
    summarise(total_trans = n()) %>%
    arrange(desc(total_trans)) %>% ungroup() 
```


```python
# Change Agency_Name to explore different organizations
sample_11 <- total_by_agency_vendor%>%filter(Agency_Name == "ATTORNEY GENERAL")
nrow(sample_11)
```


215



```python
# Visualize the top 20 most spending merchant category for one agency
sample_11 <- sample_11[1:20,]

gg11 <- ggplot() + geom_bar(aes(reorder(sample_11$Vendor,sample_11$total_trans),sample_11$total_trans),
                            stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Total Number of Transactions by Vendor over The Past 3 Months",
        x = "Vendor",
        y = "Total Transactions")+
  theme(plot.title = element_text(size = 8),
        text = element_text(size=7))

gg11 + coord_flip()
```


![png](/assets/img/creditcard/output_77_0.png)


***Conclusion:*** This feature helps to identify who an agency typically makes transactions with. These numbers can then be tracked over time to predict if any changes may indicate the presence of anomalies. 

### 12: Maximum Amount Spent by Vendor over a 30 Day Period<a class="anchor" id="feature12"></a>

This feature helps to identify any unsually large each month to vendors. It groups the amounts by agency and vendor to apply a filter based on month. It then aggregrates all transactions to find the max before filtering it by agency.


```python
max_by_agency_vendor <- time_by_agency %>% group_by(Agency_Name, Vendor) %>%
    # lastest month in the dataset is 2014-06
    filter(month_yr == "2014-06")%>%
    summarise(max_trans = max(Amount)) %>%
    arrange(desc(max_trans)) %>% ungroup() 
```


```python
# Change Agency_Name to explore different organizations
sample_12 <- max_by_agency_vendor%>%filter(Agency_Name == "ATTORNEY GENERAL")
nrow(sample_12)
```


82



```python
# Total Number of Transactions by Vendor over The Past 3 Months
sample_12 <- sample_12[1:20,]

gg12 <- ggplot() + geom_bar(aes(reorder(sample_12$Vendor,sample_12$max_trans),sample_12$max_trans),
                            stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Maximun Amount Spent by Vendor in a 30 Day Period",
        x = "Vendor",
        y = "Amount")+
  theme(plot.title = element_text(size = 8),
        text = element_text(size=7))

gg12 + coord_flip()
```


![png](/assets/img/creditcard/output_82_0.png)


***Conclusion:*** Through tracking this feature, any changes in spending can be noticed. If these amounts differ greatly over time, it can help to predict if there is suspicious behaviour which should be investigated further.

### 13: Maximum Amount Spent by Merchant Category in a 30 Day Period<a class="anchor" id="feature13"></a>
This feature is similar to feature 12, but it groups by merchant category instead of vendor to provide a broader idea of what an agency is spending money on. It filters the transactions by month before summing them to find the max for a 30 day period.


```python
max_by_agency_mer <- time_by_agency %>% group_by(Agency_Name, Merchant_Category) %>%
    # lastest month in the dataset is 2014-06
    filter(month_yr == "2014-06")%>%
    summarise(max_trans_mer = max(Amount)) %>%
    arrange(desc(max_trans_mer)) %>% ungroup() 
```


```python
# Change Agency_Name to explore different organizations
sample_13 <- max_by_agency_mer%>%filter(Agency_Name == "ATTORNEY GENERAL")
nrow(sample_13)
```


31



```python
# Total Number of Transactions by Merchant Category over The Past 3 Months
sample_13<- sample_13[1:20,]

gg13 <- ggplot() + geom_bar(aes(reorder(sample_13$Merchant_Category,sample_13$max_trans_mer),sample_13$max_trans_mer),
                            stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Maximum Amount Spent by Merchant Category in a 30 Day Period",
        x = "Merchant Category",
        y = "Amount")+
  theme(plot.title = element_text(size = 8),
        text = element_text(size=7))

gg13 + coord_flip()
```


![png](/assets/img/creditcard/output_87_0.png)


***Conclusion:*** When used with feature 12, this feature helps to detect if there are any large changes month-to-month for the transactions of an agency. By looking at the broader view of merchant category, it can help to predict if there are changes in spending pattern, regardless if vendors change.

### 14: Known Monthly Vendors for Agency (3+ transactions over past 3 months) <a class="anchor" id="feature14"></a>
This feature filters transactions by agency and then groups them by vendor and time. It then takes only transactions from the past 3 months before grouping by vendor and filtering the count to be 3 or larger.


```python
sample14 = filter(ccard, Agency_Name == "UNIVERSITY OF OKLAHOMA")
```


```python
monthly_vendors <- sample14 %>% group_by(Vendor, Year_Month) %>%
    summarise(total = sum(Amount)) %>%
    arrange(order(Vendor, -Year_Month))%>% ungroup()
```


```python
last_3_months = filter(monthly_vendors, Year_Month >= 2014-04)
```


```python
known_vendors = last_3_months %>%
                    group_by(Vendor) %>%
                    filter(n() >= 3)
```


```python
distinct_known_vendors =  known_vendors %>% distinct(Vendor)
```


```python
distinct_known_vendors[1:15,]
```


<table>
<thead><tr><th scope=col>Vendor</th></tr></thead>
<tbody>
	<tr><td>003 CENTURYLINK MY ACCOUN</td></tr>
	<tr><td>045 OBI               086</td></tr>
	<tr><td>1000BULBS.COM            </td></tr>
	<tr><td>183 BUILD-OKLAHOMA       </td></tr>
	<tr><td>2XL CORP/CARE-GYMWIPES   </td></tr>
	<tr><td>37S BASECAMP 1932438     </td></tr>
	<tr><td>37S BASECAMP 2038751     </td></tr>
	<tr><td>37S BASECAMP 2052769     </td></tr>
	<tr><td>37S BASECAMP 2309927     </td></tr>
	<tr><td>37S HIGHRISE 1999665     </td></tr>
	<tr><td>3D ROBOTICS INC          </td></tr>
	<tr><td>500 NORMAN TRANSCRIPT    </td></tr>
	<tr><td><span style=white-space:pre-wrap>A &amp; D SUPPLY OKC         </span></td></tr>
	<tr><td><span style=white-space:pre-wrap>A &amp; N CORPORATION        </span></td></tr>
	<tr><td>A AND D SUPPLY OF OKC    </td></tr>
</tbody>
</table>



***Conclusion***: This feature essentially creates a whitelist for transactions for a certain agency. As these transactions with these vendors occur at least monthly on average, they are most likely valid. This can help with predicting fraud by limiting the amount of data that is analyzed, provided that it mets other criteria provided by other features. 

### 15: Average of Total Number of Transactions for Known Vendors by Month<a class="anchor" id="feature15"></a>
This feature creates a sample set of data that filters by agency and transaction over the past 3 months to limit the amount of data being manipulated. It then groups by vendor and filters out vendors that have less than an average of 1 transaction a month. Next, it groups by agency and vendor, counts their number of transactions and then divides the count by 3 to find the average monthly number of transactions.


```python
# Change Agency_Name to explore different organizations and pick any month as the count_ven has the same values per month
sample_15 <- time_by_agency%>%filter(Agency_Name == "DEPARTMENT OF TRANSPORTATION", month_yr >= "2014-04")
nrow(sample_15)
```


3863



```python
known_vendors = sample_15 %>%
                    group_by(Vendor) %>%
                    filter(n() >= 3)
nrow(known_vendors)
```


2988



```python
freq_monthly_vendor <- known_vendors%>% group_by(Agency_Name, Vendor) %>% 
                    # n()/3 to get number transactions per month as monthly vendor contains data of the lastest 3 months
                    summarise(count_ven = n()/3, count_ven = round(count_ven))%>%
                    arrange(desc(count_ven))%>% ungroup()
```


```python
head(freq_monthly_vendor)
```


<table>
<thead><tr><th scope=col>Agency_Name</th><th scope=col>Vendor</th><th scope=col>count_ven</th></tr></thead>
<tbody>
	<tr><td>DEPARTMENT OF TRANSPORTATION</td><td>WW GRAINGER                 </td><td>50                          </td></tr>
	<tr><td>DEPARTMENT OF TRANSPORTATION</td><td>OK NATURAL GAS/TNB          </td><td>49                          </td></tr>
	<tr><td>DEPARTMENT OF TRANSPORTATION</td><td>FASTENAL COMPANY01          </td><td>38                          </td></tr>
	<tr><td>DEPARTMENT OF TRANSPORTATION</td><td>OG&amp;E/USPAYMENTSBILLPAY  </td><td>38                          </td></tr>
	<tr><td>DEPARTMENT OF TRANSPORTATION</td><td>STAPLES                     </td><td>38                          </td></tr>
	<tr><td>DEPARTMENT OF TRANSPORTATION</td><td>OREILLY AUTO  00002659      </td><td>32                          </td></tr>
</tbody>
</table>




```python
# Top 10 Highest Number of Transactions with Known Vendors by Month
freq_monthly_vendor<- freq_monthly_vendor[1:10,]

gg15 <- ggplot() + geom_bar(aes(reorder(freq_monthly_vendor$Vendor,
                                        freq_monthly_vendor$count_ven),
                                freq_monthly_vendor$count_ven),
                            stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Top 10 Highest Number of Transactions with Known Vendors by Month",
        x = "Vendor",
        y = "Total Transactions")+
  theme(plot.title = element_text(size = 8),
        text = element_text(size=7))

gg15 + coord_flip()
```


![png](/assets/img/creditcard/output_102_0.png)


***Conclusion***: This feature helps to find vendors that have a sudden increase to the number of transactions they have per month. By using a rolling average over 3 months, it evens out the peaks to find a better representation of their number of transactions. These averages can then by compared to previous figures to help predict anomalies if large changes are noticed.

### 16: Rare vendor identify (1-2 transactions) over past 3 months<a class="anchor" id="feature16"></a>
This feature is similar to feature 14, but finds rare vendors instead of frequent ones. It creates a sample for transactions by a single agency before grouping them by vendor and date. It then filters for the last 3 months before grouping again to find vendors with 1 or 2 transactions.


```python
rare_vendor <- ccard %>% group_by(Agency_Name, Vendor, Year_Month) %>%
    summarise(num_trans = n()) %>%
    filter(num_trans <= 2)%>%
    filter(Year_Month >= 201404)%>%
    arrange(order(Vendor, -Year_Month))%>% ungroup()
```


```python
sample_16 <- rare_vendor%>%filter(Agency_Name == "UNIVERSITY OF OKLAHOMA")
```


```python
distinct_new_vendors =  rare_vendor %>% distinct(Vendor)

distinct_new_vendors[1:15,]
```


<table>
<thead><tr><th scope=col>Vendor</th></tr></thead>
<tbody>
	<tr><td>PAYPAL  BLUETOAD         </td></tr>
	<tr><td>OREILLY AUTO  00003129   </td></tr>
	<tr><td>AMERICAN AI 0017446958078</td></tr>
	<tr><td>JOURNYHSE   SIMPSON      </td></tr>
	<tr><td>BEST WESTERN INN AND CONF</td></tr>
	<tr><td>FAIRFIELD INN&amp;SUITES MUSK</td></tr>
	<tr><td><span style=white-space:pre-wrap>STANLEY SUPPLY &amp; SVCS    </span></td></tr>
	<tr><td>SQ  THE GOVERNOR'S CLUB 3</td></tr>
	<tr><td>STAPLES DIRECT           </td></tr>
	<tr><td>DMI  DELL K-12/GOVT      </td></tr>
	<tr><td>ENTERPRISE RENT-A-CAR    </td></tr>
	<tr><td>JOURNYHSE   KITCHEN      </td></tr>
	<tr><td>MCLAIN-CHITWOOD OFFICE PR</td></tr>
	<tr><td><span style=white-space:pre-wrap>NATIONAL COWBOY &amp; WEST   </span></td></tr>
	<tr><td>BCI ALLEGIANCE, LLC      </td></tr>
</tbody>
</table>



***Conclusion:*** This feature helps to find transactions with vendors that are more infrequent. By creating a list, it provides a lens on what transactions should be more carefully considered. If unusually large transactions are placed with these merchants, it can help predict if unusual activity is present which should be investigated further.

### 17: Maximum amount spent on rare vendor<a class="anchor" id="feature17"></a>
This feature groups by agency and vendor before finding the maximum charge for them. It then finds the unique vendors by filtering the number of transactions to be 1 or 2. Next, it creates a filter for a single agency to find the maximum value of the transactions by that vendor.


```python
max_rare_vendor <- time_by_agency %>% group_by(Agency_Name, Vendor) %>%
    summarise(num_trans = n(),
             max = max(Amount)) %>%
    filter(num_trans <= 2)%>%
    arrange(order(Vendor))%>% ungroup()
```


```python
sample_17 <- max_rare_vendor%>%filter(Agency_Name == "UNIVERSITY OF OKLAHOMA")%>%arrange(desc(max))
```


```python
sample_17<- sample_17[1:20,]

gg17 <- ggplot() + geom_bar(aes(reorder(sample_17$Vendor,sample_17$max),sample_17$max),
                            stat = 'identity',color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Maximum Amount Spent on Rare Vendor",
        x = "Merchant Category",
        y = "Amount")+
  theme(plot.title = element_text(size = 8),
        text = element_text(size=7))

gg17 + coord_flip()
```


![png](/assets/img/creditcard/output_112_0.png)


***Conclusion:*** This feature helps to see the values of unique vendors for an agency. By determining the maximum amount of the sales, it helps to identify big purchases which may need further investigation. Through highlighting these figures, it helps to predict anomalies by determing what transactions are unusual.

### 18: Number of Transactions by Unique Users for an Agency<a class="anchor" id="feature18"></a>
This feature groups transactions by date, cardholder name and agency name to count the number of transactions each person made in a day. It then filters by a certain agency, a certain period of time and also by the number of transactions if desired. The results ar ethen visualized in a list or by a chart.


```python
count_daily_trans <- time_by_agency %>% 
        group_by(Transaction_Date,Cardholder_Last_Name, Cardholder_First_Initial,Agency_Name) %>%
        summarize(count_trans = n())%>%
        arrange((Transaction_Date),desc(count_trans)) %>% ungroup() 
```


```python
# Change Agency_Name to explore different organizations
sample_18 <- count_daily_trans%>%filter(Agency_Name == "UNIVERSITY OF OKLAHOMA",
                                       Transaction_Date >= "2014-06-01",
                                       count_trans >= 2)
```


```python
gg18 <- ggplot() + geom_bar(aes(sample_18$Transaction_Date,sample_18$count_trans), stat = 'identity',
                           color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Number of daily transactions per mechant by cardholder",
        x = "Transaction Date",
        y = "Number of Transactions")+
  theme(plot.title = element_text(size = 10),
        text = element_text(size=10))

gg18 #+ coord_flip()
```


![png](/assets/img/creditcard/output_117_0.png)


***Conclusion***: This feature helps to see how many transactions are made by a single users on a certain date. Through comparing it against historical information, it can see if the number of transactions is exceedingly high and may indicate fraud. These figures can help predict fraud by noticing the change in the number of transactions a user makes and flags any large changes which may indicate suspicious behavior.

### 19: Month over Month Expense Changes<a class="anchor" id="feature19"></a>
This feature helps calculates the monthly expenses for an agency every month. It then compares them against the previous month and creates a month over month percentage change. These figures are then tracked and compared against the previous month.


```python
# Total monthly expenses by agency
monthly_expenses <- time_by_agency %>% 
        group_by(Agency_Name, month_yr) %>%
        summarize(monthly_cost = sum(Amount))
```


```python
# Calculate MoM 
monthly_report <- monthly_expenses %>%
      mutate(MoM = (monthly_cost - lag(monthly_cost)) / lag(monthly_cost))
monthly_report <- monthly_report %>%
    mutate(MoM = round(MoM * 100, 1))
```


```python
# Change Agency_Name to explore different organizations
sample_19 <- monthly_report%>%filter(Agency_Name == "ATTORNEY GENERAL")
```


```python
gg19 <- ggplot() + geom_bar(aes(sample_19$month_yr,sample_19$MoM), stat = 'identity',
                           color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Month over Month Expense Change in Percentages",
        x = "Time",
        y = "Month over month changes")+
  theme(plot.title = element_text(size = 10),
        text = element_text(size=10))

gg19 #+ coord_flip()
```

    Warning message:
    "Removed 1 rows containing missing values (position_stack)."


![png](/assets/img/creditcard/output_123_1.png)


***Conclusion***: This feature helps to visualize how the charges an agency has change over time. If there are sudden increases or decreases, it may mean there is suspicious behaviour which should be monitored. Through tracking these percent changes, it helps to predict anomalies by highlighting how the transactions change over time.

### 20: Total Number of Transaction per Day for an agency <a class="anchor" id="feature20"></a>
This feature groups by agency and transaction date to count how many transactions are made per day. It then filters by agency and time by specifying a month to examine.


```python
total_by_agency <- time_by_agency %>% group_by(Agency_Name, Transaction_Date) %>%
    summarise(total_trans = n()) %>%ungroup() 
```


```python
# Change Agency_Name to explore different organizations
sample_20 <- total_by_agency%>%filter(Agency_Name == "ATTORNEY GENERAL",                                       
                                      Transaction_Date >= "2014-06-01")
```


```python
gg20 <- ggplot() + geom_bar(aes(sample_20$Transaction_Date,sample_20$total_trans), stat = 'identity',
                           color = "blue", fill = "steelblue") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Total number of transaction per day for an agency",
        x = "Date",
        y = "Total Transaction")+
  theme(plot.title = element_text(size = 10),
        text = element_text(size=10))

gg20 #+ coord_flip()
```


![png](/assets/img/creditcard/output_128_0.png)


***Conclusion***: This feature helps to visualize how many transactions an agency has over time. Through comparing this chart and its figures against other days, it creates a pattern of spending which can be monitored. Also, it can help predict fraud by catching very high amounts of transactions in a day which are above the average which indicates suspicious behavior.
