---
layout: page
title:  Popular Techniques in Natural Language Processing
description: >
  This page contains the supervised algorithms for loan default
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


This page contains the list of different popular techniques in Natural Language Processing. Understanding human language is considered a difficult task due to its complexity. Thus,  from these break-down tasks, we can come to different areas of NLP easily when needed.<br>
For a more detailed explanation, check out [My Github](https://github.com/tramduong/Text-and-Natural-Language-Analytics/tree/main/Short-NLP-Projects)
{:.note}

<br>


<img src="/assets/img/nlp/nlp.jpg"  alt="Portfolio Banner" class="banner">


<br>

### Introduction  
___

Humans communication through some form of language either by text or speech. Natural language processing (NLP) is all about making computers to learn, process and manipulate natural languages. NLP is a branch of artificial intelligence (AI) that focus on analyzing, understanding and generating the languages that humans use naturally in order to interface with computers in both written and spoken contexts using natural human languages instead of computer languages. <br>
In this page, I will look at some of the common practices used in natural language processing tasks.<br>

### Basic Text Processing
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  This project aims to use the following techniques to understand the main topic and extract important information of a random article:<br>
    - Regular expression patterns and functionality<br>
    - Strip HTML tags, images, code scripts<br>
    - Tokenize words and sentences.<br>
    - Lemmatize stemmed word tokens.<br>
    - Assign Part-of-Speech (POS) tags.<br>    
    <i><b>Libraries:</b> urllib, beautiful soup, regular expression, and NTLK.</i>
      <div class="row">
      <div class="column50">
      <a href="/portfolio/projects/nlp/Basic Text Processing/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
      </div>
      <div class="column50">
      <a href="https://github.com/tramduong/Text-and-Natural-Language-Analytics/blob/main/Short-NLP-Projects/Basic%20Text%20Processing.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
      </div>
    </div>
</li></ul></p>

### Key Phrase Extraction and Text Summarization
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  This project aims to:<br>
    - Extract subject-verb-object relations from an article body<br>
    - Extract keywords from an article body<br>
    - Produce an extractive summary of an article.<br>
    <i><b>Libraries:</b> urllib, BeautifulSoup, spacy, numpy, sumy  </i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/Keyphraseextract/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Text-and-Natural-Language-Analytics/blob/main/Short-NLP-Projects/Keyphraseextract.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Named Entity Recognition (NER)
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  This project aims to take preconditioned text and apply transformations for:<br>
    - Tagging named entities<br>
    - Entity Recognition<br>
    - Entity Disambiguation<br>
    <i><b>Libraries:</b> urllib, BeautifulSoup, spacy, pyspark.</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/NamedEntityRecognition/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Text-and-Natural-Language-Analytics/blob/main/Short-NLP-Projects/NamedEntityRecognition.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>

</li></ul></p>

### Word2Vec and Pyspark Similarity
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  This project processes and tokenizes Webhose article bodies and train a Word2Vec model  using Spark MLLib library. Demonstrate a search query implementation and retrieved article titles.<br>
    <i><b>Libraries:</b> Pyspark, spacy, en_core_web_sm, numpy</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/Pyspacktoken/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Text-and-Natural-Language-Analytics/blob/main/Short-NLP-Projects/Pyspacktoken.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Sematic Similarity Score Project
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  This project uses Word2Vec pre-trained model and your Webhose dataset, to identify 100 most similar titles to any one chosen title. The main method used here is sematic similarity<br>
    <i><b>Libraries:</b> gensim, scipy, json, random, pandas, numpy</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/nlp/SemanticSimilarity/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Text-and-Natural-Language-Analytics/blob/main/Short-NLP-Projects/SemanticSimilarity.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>
