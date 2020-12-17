---
layout: page
title: Recommending Songs on Spotify through Python
description: >
  This project used a logistic regression model to create a recommendation model for songs using SciKit-Learn, Pandas and SciPy through Python.
hide_description: true
sitemap: false
---

<style>

.banner {
  box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
  center;
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

</style>

* This unordered list will be replaced by the table of contents
{:toc}

<img src="/assets/img/spotify/spotify_cover.jpg"  alt="Spotify Project" class="banner">

### Overview

<p style="text-align: justify;">
In this project, I used a logistic regression model to create a recommendation model for songs using SciKit-Learn, Pandas and SciPy through Python. I have always had a great interest in music and have used Spotify for years, but always wondered how their algorithms for recommendations were always superior to their competition. With this idea in mind, I set out to investigate how they suggested music to users to gain a better understanding of their process.
</p>

### Data

<p style="text-align: justify;">
Thankfully, Spotify is extremely transparent with their song data and grants access to developers for parts of it. After searching Kaggle, I found a relevant dataset that another user had scraped and parsed to begin my analysis. The data set consisted of just over 2,000 instances and consisted of 17 variables with 14 being relevant.
<br><br>
As the information was compiled directly from Spotify, it was already clean, so little preprocessing was needed. Instead, I focused on exploring the data, noticing irregularities in variables and the general shape of it to determine my best approach. While some of the variables, such as danceability, seemed subjective and required greater explanation, others were less complicated, such as time signature. Most of the variables were float data ranging from 0 to 1, with a few integers, such as duration in milliseconds. These facts made the summary statistics relatively straightforward and only a few outliers appeared.
</p>

### Analysis

<div class="row">
    <div class="column40">
      <figure>
        <img src="/assets/img/spotify/spotify_valence.png" style ="padding-right: 15px;">
      </figure>
    </div>
    <div class="column60">
      <p style="text-align: justify;">Although the data was straightforward, evaluating it was more complicated. Beginning with a correlation matrix, I observed the variables to get an estimate on which ones should be focused on, but few were strongly correlated. However, I was able to determine a few to work with and looked at their distributions. Some were extremely straightforward, such as valence, but others, like instrumentalness, were not Gaussian and required normalization.</p>
    </div>
</div>

<div class="row">
    <div class="column50">
      <p style="text-align: justify;">Next, I used dbscan from sklearn to perform outlier detection, but most values fit inside the normal distribution. Speechiness, or how much speech without music, had the most outliers, but was extremely low with only 46 out of 2,017 values or 2.28%. I hypothesized that these values could come from podcasts and removed them as they affected the distribution and trend lines without providing a significant impact on the data.
      </p>
    </div>
    <div class="column50">
      <figure>
        <img src="/assets/img/spotify/spotify_speechiness.png" style ="padding-left: 15px;">
      </figure>
    </div>
</div>

<p style="text-align: justify;">
To perform the analysis, I used RFE, or recursive feature elimination, from sklearn to automatically tune the model, finding the best result based on cross validation. Once optimized, the logistic regression model reached 67.6% accuracy on the train data and 63% on the test data. While the results were not exceedingly excellent, the model did outperform the naïve approach making it a success, especially considering how complex musical tastes are.
</p>

<pre>
  <code>
Optimization terminated successfully.          
      Current function value: 0.675697          
      Iterations 5                              
Results: Logit
<center>======================================================</center>
Model:              Logit            Pseudo R-squared: 0.025
Dependent Variable: target            AIC:              2187.7993  
Date:               2019-08-20 22:29 BIC:              2209.3427  
No. Observations:   1613             Log-Likelihood:   -1089.9    
Df Model:           3                LL-Null:          -1117.9    
Df Residuals:       1609             LLR p-value:      4.3995e-12
Converged:          1.0000           Scale:            1.0000     
No. Iterations:     5.0000                                        
<center>-----------------------------------------------------
Coef.   Std.Err.    z   P>|z|  [0.025   0.975]
-----------------------------------------------------</center>
 danceability      -0.7192   0.2301 -3.1252 0.0018 -1.1703 -0.2682
 instrumentalness   1.1280   0.1919  5.8789 0.0000  0.7520  1.5041
 speechiness        2.4381   0.6087  4.0055 0.0001  1.2451  3.6310
 valence            0.4408   0.2316  1.9032 0.0570  0.0131  0.8947
 <center>=====================================================</center>
  </code>
</pre>
<figure>
  <figcaption>Final Logistic Model</figcaption>
</figure>

### Outcome

<p style="text-align: justify;">
Overall, completing the project gave me a greater respect for recommendation algorithms. Although we use them daily, it is often with little thought into how they affect our lives for the better, whether it is with recommendations for movies on Netflix or shopping on Amazon. I also gained a respect into how difficult it is to classify nebulous and hard to define values into actionable data, such as song characteristics, and showcased Spotify’s expertise on the subject.
<br><br>
If I were to do this project again, I would make two major changes. First, I would have liked to use my own Spotify data instead of pre-existing data from other users. Although I created a developer account and pulled information from their API, I did not have the skills with Node.js for preexisting solutions and was not familiar enough with dictionaries to scrape it with Python at the time. Second, as this was a project for my Master degree, I was required to use a multi-linear or logistic regression model which was not the ideal method. If I were to use a KNN or gradient boosted random forest approach with machine learning, I predict that the accuracy would greatly rise due to how relations to variables are formed. From making these two changes, I would have a more individualistic project that I could personally compare results to and their accuracy could improve when using the more complex models.
</p>
<br><br>

____

### Supporting Figures
<br>

<figure>
  <img src="/assets/img/spotify/spotify_corr.png"  alt="Correlation Matrix">
  <figcaption>Correlation Matrix for Variables using Colorblind Scale</figcaption>
</figure>

<figure>
  <img src="/assets/img/spotify/spotify_roc.png"  alt="ROC Curve">
  <figcaption>Performance of Model through ROC Curve</figcaption>
</figure>
