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


This page contains the different algorithms for unsupervised learning with a hospital charges dataset. This approach aims to analyze a hospital charge data to determine outliers, identify anomalous activities and provide insights of these behaviors using different advanced algorithms.<br>
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
  Analyzing a hospital charge data to determine outliers, identify anomalous activities and provide insights of these behaviors using different advanced algorithms.<br>
    <i><b>Libraries:</b> pandas, scipy, sklearn, pyod, seaborn, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/unsupervised/K-means_Clustering/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
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
  Analyzing a hospital charge data to determine outliers, identify anomalous activities and provide insights of these behaviors using different advanced algorithms.<br>
    <i><b>Libraries:</b> pandas, scipy, sklearn, pyod, seaborn, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/unsupervised/pca_knn/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
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
  Analyzing a hospital charge data to determine outliers, identify anomalous activities and provide insights of these behaviors using different advanced algorithms.<br>
    <i><b>Libraries:</b> pandas, scipy, sklearn, pyod, seaborn, matplotlib</i>
        <div class="row">
        <div class="column50">
        <a href="/portfolio/projects/unsupervised/Autoencoder_IsolationForest/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" class="button_smallest"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" class="button_smallest"></a>
        </div>
      </div>
</li></ul></p>
