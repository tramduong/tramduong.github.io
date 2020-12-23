---
layout: page
title: Topics and Sentiment Tracking of the Airline Industry Since COVID-19
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


The reputation of the US airline industry has been greatly damaged due to Covid-19. Tracking the topics and sentiment during these times can help airline companies when best to proceed with a relaunch strategy.<br>
For a more detailed explanation, check out [My Github](https://github.com/tramduong/Data-Science-Portfolio/tree/master/Airlines%20Covid-19)
{:.note}

<br>


<img src="/assets/img/nlp/airplane.jpg"  alt="Portfolio Banner" class="banner">

<br>

### Data Information
___
<div class="row">
<div class="column50">
  Raw data from Webhose:<br>
   + Thread title: airline <br>
   + Type of sites: news and blogs<br>
   + Country: US <br>
   + Laguage: English<br>
   + Dates: last 30 days since 27 JULY 2020<br>
   + Size: 20,015 feeds<br>
 </div>
 <div class="column50">
   The Final dataset:<br>
   + Thread title: airline<br>
   + Type of sites: news and blogs<br>
   + Country: US<br>
   + Laguage: English<br>
   + Dates: last 30 days since 27 JULY 2020<br>
   + Size: 13,341 feeds<br>
   </div>
 </div>
### Latent semantic analysis
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  Latent Semantic Analysis(LSA)is an efficient way of analysing the text and finding the hidden topics by understanding the context of the text.<br><br>
  This approach helps to find the hidden topics represented by the US news for Airlines industry during COVID-19 time, such as top 5 articles and their similarity scores/ <br>
    <i><b>Libraries:</b> json, re, nltk, time, gensim, operator, scipy, sklearn, numpy</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/LSA/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Airlines%20Covid-19/doc/LSA%20.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Named-entity recognition (NER)  
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  Named entity recognition (NER) helps you easily identify the key elements in a text, like names of people, places, brands, monetary values, and more. <br><br>
  This approach helps to extract the main entities in the news, such as which airline has been mentioned mostly in the news during the pandemic.
    <i><b>Libraries:</b>  json, spacy, pandas, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/NER/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Airlines%20Covid-19/doc/NER.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Simhash and Word2Vec Models
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  Use Simhash and Word2Vec models for semantic deduplication to deduplicate Webhose feeds based on titles .<br><br>
  Deduplication should be carried out on the entire dataset, so that the output does not contain duplicate titles/articles. <br><br>
    <i><b>Libraries:</b> webhoseio, Simhash, json, gensim, numpy, logging</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/Simhash&Word2Vec/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Sentiment Analysis using NLTK Sentiment Intensity Analyzer
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  Sentiment analysis is a common NLP task, which involves classifying texts or parts of texts into a pre-defined sentiment.<br><br>
  In this project, I use NLTK Sentiment Intensity Analyzer to define weekly summaries sentiment analysis into positive and negative scores. <br><br>
    <i><b>Libraries:</b> json, pandas, spacy, sumy, re, time, sklearn, pyod, seaborn, SentimentIntensityAnalyzer, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/WeeklySummaries_SentimentAnalysis/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Airlines%20Covid-19/doc/WeeklySummaries_SentimentAnalysis.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>
