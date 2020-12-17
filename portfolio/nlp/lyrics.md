---
layout: page
title: Lyrics Text Mining and Sentiment Analysis
description: >
  This page contains the unsupervised algorithms for hospital charges
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


This report is a guidebook with selected Q&A aims at finding out the insights and differences between Pop and Metal lyrics over time.<br>
For a more detailed explanation, check out [My Github](https://github.com/tramduong/Data-Science-Portfolio/tree/master/PopVsMetal)
{:.note}

<br>


<img src="/assets/img/nlp/lyrics.jpg"  alt="Portfolio Banner" class="banner">


<br>

### Project Description
___

+ Project title: Pop vs Metal: What are the differences in their lyrical changes and their sentiment?

+ Project summary: This report is a guidebook with selected Q&A aims at finding out the insights and differences between Pop and Metal

First,I processed the raw textual data 'dt_lyrics' saved in $ data $ file by cleaning data, removing stopwords and creating a tidy version of texts which is saved in $ output $ file as 'processed_data'. Then I load the processed data directly from output folder.

**Used packages:** tm, tidytext, tidyverse, DT, wordcloud, scales, gridExtra, ngram, igraph, ggraph, rsconnect, syuzhet, ggwordcloud.

### Project Outcomes
___

+ Three questions will be answered in this report are:
  1. What are the music trends of Pop and Metal in general? Is pop really more positive than metal?
  2. Who are the hardest working artists in these genres? Did they follow the trends? Or did they write their music with their own style?
  3. How did the sentiments in Pop and Metal change over the decades and how do they compare to each other?

### Project Links
___

The final code deliverables and explanation are conducted below.<br>
<div class="row">
  <div class="column50">
    <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/PopVsMetal/doc/SongLyrics.pdf" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" class="button_smaller">      
    </a>
  </div>
  <div class="column50">
    <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/PopVsMetal" target="_blank"><img src="/assets/img/github_button.png" alt="View on GitHub" class="button_smaller">
    </a>
  </div>
</div>
