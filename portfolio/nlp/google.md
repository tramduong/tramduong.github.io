---
layout: page
title: Natural Language Processing for Google News in a 30-day period
description: >
  This page contains the nlp for airline industry
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


This project aims to track <b>top mentioned topics and trending</b> for Google during Covid-19 time, develop recommendations for the company through <b>topic classification</b>, and define application for other industries.<br>
For a more detailed explanation, check out [My Github](https://github.com/tramduong/Data-Science-Portfolio/tree/master/Google%20News)
{:.note}

<br>


<img src="/assets/img/nlp/google.jpg"  alt="Portfolio Banner" class="banner">


<br>
### Data Information
___
<div class="row">
<div class="column50">
  Raw data from Webhose:<br>
   + Thread title: Google <br>
   + Type of sites: news and blogs<br>
   + Country: US <br>
   + Laguage: English<br>
   + Dates: last 30 days since 13 JUNE 2020<br>
   + Size: 27,798 feeds<br>
 </div>
 <div class="column50">
   The Final dataset:<br>
   + Thread title: Google<br>
   + Type of sites: news and blogs<br>
   + Country: US<br>
   + Laguage: English<br>
   + Dates: last 30 days since 13 JUNE 2020<br>
   + Size: 18116 feeds<br>
   </div>
 </div>
###  Simhash Text Deduplication
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  Use Simhash and Word2Vec models to deduplicate Webhose feeds based on titles .<br><br>
  Deduplication should be carried out on the entire dataset, so that the output does not contain duplicate titles/articles. <br><br>
    <i><b>Libraries:</b> webhoseio, Simhash, json, gensim, numpy, logging</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/Deduplicated/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Google%20News/doc/Deduplicated.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Topic Modeling with Gensim and pyLDAvis
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  In this project, I used <b>Gensim</b> packages to build a LDA model. Then I applied the model to a great visualization for topic modeling using <b>pyLDAvis,</b><br>
    <i><b>Libraries:</b> request, json, re, pandas, nltk, string, datetime, iexfinance, pyLDAvis, gensim, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/TD_Gensim/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Google%20News/doc/TD_Gensim.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Topic Classification using Taxonomy & Word2Vec
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  In this project, I created a taxonomy based on LDA model and mannual data exploration for Google feeds during the time period.<br><br>
  Then I classified the Webhose article titles against the developed taxonomy using word2vec similarity. <br>
    <i><b>Libraries:</b> pandas, sumy, gensim, operator, scipy , numpy, json.</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/Topic Classification/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Google%20News/doc/Topic%20Classification.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### LDA model to identify topic distribution and keywords
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  In this project, I implemented an LDA training and topic modeling on the  deduplicated Webhose feeds dataset.<br><br>
  I use LDA from Scikit-Learn and modified the values of min_df and max_df, max_features and max_iter (sklearn) to achieve best results. <br>
    <i><b>Libraries:</b> json, sklearn, nltk, re, pandas, random.</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/Word_tokenize_LDA/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Google%20News/doc/Word_tokenize_LDA.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>
