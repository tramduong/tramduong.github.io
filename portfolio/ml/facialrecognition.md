---
layout: page
title: Face Image Emotion Recognition
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


This project creates an advanced classification engine for faction emotion recognition.<br>
For a more detailed explanation, check out [My Github](https://github.com/tramduong/Data-Science-Portfolio/tree/master/Image%20Recognition)
{:.note}

<br>


<img src="/assets/img/ml/facialrecog.jpg"  alt="Portfolio Banner" class="banner">


<br>

### Project Description
___

For this project, the data contains **a set of 2,500 facial images with 22 different emotions**. Here is an alternative [data link](https://www.dropbox.com/s/kvi949ea1rey1d8/train_set.zip?dl=0). These two links lead to **identical** data.

The project focuses on creating an mobile AI program that **accurately recognizes the emotion from facial images**.

The portability of this AI program (holding storage and memory cost) and the computational efficiency (test running time cost) are of great concern to your client. This translates to a balance between the complexity of variable/features/models used and the predictive performance.


### Project Modeling
___

+ Feature selection for advanced model using PCA.
+ Gradient Boosting Machine - Baseline model
+ CNN, KNN, Random Forest with cross validation
+ SVM - Advanced model

### Project Outcomes
___

+ In this project, I created a classification engine for facial emotion recognition. I developed advanced classification models to compare their accuracy and efficiency to the client's original baseline model.
+ I conducted PCA for our feature selection.
+ Boosted Decision Stumps(gbm), which has a model accuracy of 44.4%.
+ The advanced model I tried was KNN, Support Vector Machine(SVM), CNN, and Random Forest.
+ Among the 4 advanced models, SVM has the highest accuracy, which is around 52.8%, so I chose SVM as our final advanced model.

### Project Links
___

The final code deliverables and explanation are conducted below.<br>
<div class="row">
  <div class="column50">
    <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Image%20Recognition/doc/Main.pdf" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" class="button_smaller">
    </a>
  </div>
  <div class="column50">
    <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/Image%20Recognition" target="_blank"><img src="/assets/img/github_button.png" alt="View on GitHub" class="button_smaller">
    </a>
  </div>
</div>
