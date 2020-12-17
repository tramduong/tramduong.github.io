---
layout: page
title: Algorithm implementation and evaluation - Collaborative Filtering
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


In this project, I will implement, evaluate and compare algorithms for Collaborative Filtering.<br>
For a more detailed explanation, check out [My Github](https://github.com/tramduong/Data-Science-Portfolio/tree/master/Collaborative%20Filtering)
{:.note}

<br>


<img src="/assets/img/ml/Collaborative.jpg"  alt="Portfolio Banner" class="banner">


<br>

### Project Description
___

*Collaborative filtering* refers to the process of making automatic predictions (*filtering*) about the interests of a user by collecting preferences or taste information from many users (*collaborating*).

For this project, I will focus on some **specific algorithms** from the *Collaborative filtering* literature. Then I will study the algorithms carefully and implement them, from scratch.

### Project Modeling
___

In this project, I implement matrix factorization by focusing on alternating least square algorithms and KNN as post-processing, in which I try to evaluate how regularization will impact the prediction results.<br>

I will use RMSE results for both train and test dataset as the basis of our evaluation. The objective of this project is to produce a prediction of users' movies preference based on the ratings given respective users. I compare the performance between the following models:

+ **Model 1:** Alternating Least Squares (ALS) with Temporal Dynamics (TD) Regularization and K-nearest Neighbors (KNN) Post-processing
+ **Model 2:** Alternating Least Squares (ALS) with K-nearest Neighbors (KNN) Post-processing

### Project Outcomes
___

The final code deliverables and explanation are conducted below.<br>
<div class="row">
  <div class="column50">
    <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Collaborative%20Filtering/doc/Main.Rmd" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" class="button_smaller">
    </a>
  </div>
  <div class="column50">
    <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/Collaborative%20Filtering" target="_blank"><img src="/assets/img/github_button.png" alt="View on GitHub" class="button_smaller">
    </a>
  </div>
</div>
