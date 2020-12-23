---
layout: page
title:  Supervised Learning - Loan Default Prediction
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


This page contains the different algorithms for supervised learning with the loan default dataset. This approach aims to predict mortgage probability of default data and provide insights for these predictions using different advanced algorithms.<br>
For a more detailed explanation, check out [My Github](https://github.com/tramduong/Data-Science-Portfolio/tree/master/Loan%20Default)
{:.note}

<br>


<img src="/assets/img/ml/loan.jpg"  alt="Portfolio Banner" class="banner">


<br>

### Industry Background
___

Mortgage fraud is getting worse as more people lie about their income to qualify for loans. Home values are high, the housing market is competitive, and more buyers want to get in. As a result, an increasing number of buyers are lying and cheating. <br>

Mortgage fraud risk jumped more than 12 percent year over year at the end of the second quarter, according to CoreLogic, which measures six fraud indicators: identity, income, occupancy, property, transaction and undisclosed real estate debt. One in every 109 mortgage applications is estimated to have indications of fraud. <br>

Therefore, I would like to use different supervised algorithms to predict a **probability of default (PD)** – The probability that a debtor will not fulfil its obligations. <br>

### Logistic regression, Random Forest, and Gradient Boosting Model
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  In this project, I applied <b>logistic regression, random forest, and gradient boosting algorithms</b> to predict the mortgage fraud probability. <br>
  The models are built using the 80% training data and validated by the 20% test data. <br>
  The model comparison results in <b>Gradient Boosting</b> as highest prediction scores. <br>
  I also use <b>confusion matrix, ROC curve, gain tables</b> for model evaluation part. <br>
    <i><b>Libraries:</b> pandas, numpy, sklearn, seaborn, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/supervised/Loandefault_GradientBoosting_RF/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Loan%20Default/Doc/Loandefault_GradientBoosting_RF.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Random Forest and Sampling Techniques  
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  The dataset is <b>highly unbalanced</b> which can result in a serious bias towards the majority class, reducing the classification performance and increasing the number of false negatives. Thus, to reduce the issue, the most commonly used techniques are data resampling either under-sampling the majority of the class, or oversampling the minority class, or a mix of both.<br><br>
  In this project, I used several method of <b>under_sampling and over_sampling</b> to improve classification performance. Additionally,  I used H2O Grid-search to find the <b>optimal hyper-parameters</b> for the random forest model and comparing with resampling methods. <br>  
    <i><b>Libraries:</b> pandas, h2o, imblearn, sklearn, collections, under_sampling, over_sampling, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/supervised/SupervisedRF&SamplingTechniques/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Loan%20Default/Doc/SupervisedRF%26SamplingTechniques.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Gradient Boosting (GBM) and Deep Learning
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  GBM is a boosting method, which builds on weak classifiers. The idea is to add a classifier at a time, so that the next classifier is trained to improve the already trained ensemble.<br><br>
  H2O Deep Learning is based on a multi-layer feedforward artificial neural network that contains a number of hidden layers consisting of neurons. In this project, I tried different parameters to optimize the model result. <br>
  I also use <b>AUC, PR curve, gain tables, r2 and RMSE</b> for model evaluation part. <br><br>
  <b>Key takeaways</b>:<br>
  For this dataset, GBM models aim to find optimal linear combination of trees by training (assume final model is the weighted sum of predictions of individual trees) in relation to given train data, and performed much better than RF models. This leads me to believe that the dataset prefer more extra tunning (GBM) and less classifier approach (RF). <br>
    <i><b>Libraries:</b> pandas, numpy, datetime, H2O, sklearn, seaborn, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/supervised/GBM&DEEPLEARNING/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Loan%20Default/Doc/GBM%26DEEPLEARNING.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>

</li></ul></p>

### Generalized Linear Model and AutoML
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  GLM estimates regression models for outcomes following exponential distributions. In order to create less complex (parsimonious) model as our data has large number of features, I used some of the regularization techniques to address over-fitting and feature selection, L1 Regularization and L2 Regularization. <br><br>
  AutoML tends to automate the maximum number of steps in an ML pipeline - with a minimun amount of human effort - without compromising the model's performance.H2O AutoML can be used for automating the machine learning workflow, which includes automatic training and tuning of many models within a user-specified time-limit.<br><br>
    <i><b>Libraries:</b>  pandas, numpy, datetime, H2O, sklearn, seaborn, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/supervised/GLM&AutoML/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Loan%20Default/Doc/GLM%26AutoML.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>
