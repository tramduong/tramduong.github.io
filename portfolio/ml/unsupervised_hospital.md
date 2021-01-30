---
layout: page
title: Unsupervised Learning - Fraud Detection with Hospital data
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


This page contains the different unsupervised learning algorithms for a hospital charges dataset. This approach aims to analyze a hospital charge data to determine outliers, identify anomalous activities and provide insights of these behaviors using different advanced algorithms.<br>
For a more detailed explanation, check out [My Github](https://github.com/tramduong/Data-Science-Portfolio/tree/master/Hospital%20Charges%20Fraud)
{:.note}

<br>


<img src="/assets/img/ml/hospital.jpg"  alt="Portfolio Banner" class="banner">


<br>

### Kmeans with Elbows and Silhouette Scores
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  In this project, I used K-mean algorithm for classification model. Instead of relying on one method solely, I decided to use a hybrid approach by choosing off the silhouettes and the k-means elbow, thus providing a better indication of what cluster to use. <br>
  Then I generate a average calculation approach among all features to define anomalous cluster groups. <br>
    <i><b>Libraries:</b> pandas, scipy, sklearn, StandardScaler, seaborn, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/unsupervised/K-means_Clustering/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/K-means_Clustering.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### PCA and KNN Algorithms
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  The <b>PCA-Based Anomaly Detection</b> module solves the problem by analyzing available features to determine what constitutes a "normal" class, and applying distance metrics to identify cases that represent anomalies.<br><br>
  The <b>KNN-based anomaly detection</b> methods relies on neighbors search to decide whether a data point is an outlier. An isolated data point has a large distance to other observations and it can be seen as an outlier through KNN. <br><br>
  By aggregating multiple models, the chance of overfitting is greatly reduced and the prediction accuracy will be improved. In this project, I used the four methods from the <b?Pyod module</b> to aggregate the outcome (Average, Maximum of Maximum (MOM),Average of Maximum (AOM), Maximum of Average (MOA)). <br><br>
    <i><b>Libraries:</b> pandas, numpy, scipy, sklearn, pyod, seaborn, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/unsupervised/PCA_KNN/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/PCA_KNN.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>

### Autoencoder and Isolation Forest
___

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  In this project, I will use <b>autoencoder and isolation forest</b> algorithm to detect outliers. I will try 3 different models, the average and the maximun of maximun methods for both of the approaches.<br><br>
  <b>Autoencoder techniques</b> can perform non-linear transformations with their non-linear activation function and multiple layers. It is more efficient to train several layers with an autoencoder, rather than training one huge transformation with PCA. <br><br>
  <b>Isolation Forest</b> is an unsupervised learning that calculates an anomaly score and separates into binary based on an anomaly threshold. Isolation forest is <b>an advanced outlier detection</b> that delects anomalies based on the concept of isolation instead of distance or density measurement. It is different from other methods like KNN or PCA in anomalies detection and is knowns as <b>an effective method</b> at reducing frauds.<br><br>
    <i><b>Libraries:</b> pandas, scipy, sklearn, pyod, seaborn, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/unsupervised/Autoencoder_IsolationForest/"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>
