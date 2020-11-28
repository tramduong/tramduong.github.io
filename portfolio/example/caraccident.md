---
layout: page
title: Seattle Car Accident Data
description: >
  Analyzing car accident data with MySQL for data exploration and Tableau for visuals
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

</style>


* This unordered list will be replaced by the table of contents
{:toc}


<img src="/assets/img/caraccident/caraccident_cover.jpg"  alt="Car Accident Project" class="banner">


### Overview

<p class="justify">
For this project, I analyzed car accident data using MySQL for data analysis and Tableau for visuals. I unfortunately was in a major car accident in 2018 and while I recovered over time, it made wonder if anything could be done to help prevent further collisions. To investigate this goal, I found a data set collected by the Seattle Department of Transportation made from records of accidents from the Seattle Police Department, I began my investigation.
</p>

### Data

<p class="justify">
In the initial data set, there were over 200 thousand records with 40 different variables, covering a period from 2003 to 2019. However, earlier records were not as complete as later ones and not all the variables were informative as they were often redundant or missing for most entries. Thus, I used Python to remove any records from before 2013 and selected 25 variables to work, narrowing the scope of the data.
<br><br>
Additionally, there was a categorical variable which recorded the severity of the accident. While it would be invaluable for the analysis, it used subcategories for any collisions, such as  injuries and serious injures having codes of 2a and 2b. Knowing this would cause problems later, I changed the coding system to consist of only integers ranging from 0 with little to no damage and 4 which resulted in fatalities.
</p>

<figure>
  <img src="/assets/img/caraccident/caraccident_frequency.jpg"  alt="Car Accident Frequency" style="center;">
  <figcaption>Frequency of Accidents by Month</figcaption>
</figure>

<p class="justify">
After importing the data into MySQL Workbench, I noticed 2 more anomalies: any record with a severity value of 0 was missing information and data from 2019 only covered part of the year. Instead of making changes in Python and importing the data again, I found it easier to create a new table dropping any records from 2019 or with a severity code of 0. This left me with almost 63 thousand records to analyze.
</p>

### Analysis

<p class="justify">
From exploring the remaining variables, I found that many of them did not have a large effect on the severity or number of accidents. Some factors, such as road condition, weather, light levels, or inattention, did not have a significant impact on the data, but two variables did: DUI and speeding. Where these variables were present, the severity of accidents and fatalities greatly increased. For instance, DUI’s resulted in over 4 times as many fatalities and speeding resulted in almost 5 times more.
<br><br>
Additionally, there was geographic information contained in the data set, including cross street and coordinates. Using the street data, I first generated a list of where the most collisions occurred. However, this information did not prove to be very useful as it was difficult to find hotspots based solely on street name. Instead, I used Tableau to create a heatmap based on the latitude and longitude. This style was much easier to read as darker colors represented trouble spots that should have greater focus than others.
</p>

<figure>
  <img src="/assets/img/caraccident/caraccident_conditions.jpg"  alt="Car Accident Conditions" width="50%">
  <figcaption>Car Accident Severity by Condition Type</figcaption>
</figure>
<figure>
  <img src="/assets/img/caraccident/caraccident_heatmap.jpg"  alt="Car Accident Heatmap">
  <figcaption>Car Accident Heatmap for Seattle</figcaption>
</figure>

### Outcome

<p class="justify">
Overall, the project provided insights into the nature of accidents, including how bad they were and location. It was not a surprise to see that most of the collisions were in busy areas, such as Downtown Seattle or the University District. Furthermore, comparing how DUIs and speeding affected accidents was interesting as well. As there are high fines for both, it was intriguing to see how they were justified due to the increased severity they introduced. From this project, a variety of stakeholders, such as the DOT or police, could analyze where accidents were more frequent, providing greater focus on these trouble spots.
<br><br>
If I were to do the project again, I would like to expand my data. As many of the accidents centered around freeway on-ramps, including information from the highway patrol may provide greater insight into the nature of accidents. Additionally, if traffic information could be included, it may help to explain why certain areas had much higher rates of collisions. Lastly, it would be interesting to compare Seattle’s accident data against information from other cities to see if various trends are similar or unique to the region.
</p>
<br><br>

____

### Supporting Figures
<br>

<figure>
  <img src="/assets/img/caraccident/caraccident_code1.jpg"  alt="Car Accident Code Snippet 1">
  <figcaption>Creating new table</figcaption>
</figure>

<figure>
  <img src="/assets/img/caraccident/caraccident_code2.jpg"  alt="Car Accident Code Snippet 2">
  <figcaption>Finding accident percentages per severity level depending on condition</figcaption>
</figure>

<figure>
  <img src="/assets/img/caraccident/caraccident_code3.jpg"  alt="Car Accident Code Snippet 3">
  <figcaption>Aggregating accidents by location</figcaption>
</figure>
