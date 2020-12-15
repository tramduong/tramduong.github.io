---
layout: page
title: Portfolio
description: >
  Here is a collection of data science projects I have done, ranging from course work to personal endeavors.
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


The pages contains the portfolio of data science projects completed by me for academic, self learning, and hobby purposes. Presented in the form of Python Notebooks, and R markdown files.<br>
For a more detailed explanation, check out [Tram Duong's Github](https://github.com/tramduong/Data-Science-Portfolio)
{:.note}

<br>


<img src="/assets/img/portfolio.jpg"  alt="Portfolio Banner" class="banner">




<br>
### Machine Learning
___
#### Unsupervised Learning: Fraud Detection with Hospital data

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  Analyzing a hospital charge data to determine outliers, identify anomalous activities and provide insights of these behaviors using different advanced algorithms.<br>
    <li style="padding-left: 20px; list-style-type: none;">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/PCA_KNN.ipynb" target="_blank">
        &#9679; Kmeans with Elbows and Silhouette Scores
      </a>
    </li>
</li></ul></p>

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  Analyzing a hospital charge data to determine outliers, identify anomalous activities and provide insights of these behaviors using different advanced algorithms.<br>
    <li style="padding-left: 20px; list-style-type: none;">
        &#9679; Kmeans with Elbows and Silhouette Scores
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/K-means_Clustering/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/K-means_Clustering.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; PCA and KNN Algorithms                   
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/pca_knn/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/PCA_KNN.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; Autoencoder and Isolation Forest
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/Autoencoder_IsolationForest/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
    </li>
  <i><b>Libraries:</b> pandas, scipy, sklearn, pyod, seaborn, matplotlib</i>
</li></ul></p>

#### Supervised Learning: Loan Default Prediction

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  Analyzing a hospital charge data to determine outliers, identify anomalous activities and provide insights of these behaviors using different advanced algorithms.<br>
    <li style="padding-left: 20px; list-style-type: none;">
        &#9679; Logistic regression, Random Forest, and Gradient Boosting Model
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/K-means_Clustering/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Loan%20Default/Doc/Loandefault_GradientBoosting_RF.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; Random Forest and Sampling Techniques                 
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/pca_knn/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Loan%20Default/Doc/SupervisedRF%26SamplingTechniques.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; Gradient Boosting and Deep Learning
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/Autoencoder_IsolationForest/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Loan%20Default/Doc/GBM%26DEEPLEARNING.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; Generalized Linear Model and AutoML
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/Autoencoder_IsolationForest/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Loan%20Default/Doc/GLM%26AutoML.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
    </li>
  <i><b>Libraries:</b> pandas, numpy, sklearn, h2o, seaborn, matplotlib.</i>
</li></ul></p>


#### Machine Learning Model Interpretation

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
   SHAP Values (Shapley Additive Explanations) break down a prediction to show the impact of each feature. In other words, each SHAP value measures how much each feature in our model contributes, either positively or negatively, to each prediction.<br>
   In this project, I use Shap Values to interpret feature impact to the model output. <br>
  <div class="row">
    <div class="column50">
      <a href="/portfolio/example/ML_ShapValues/">
        <img src="/assets/img/project_button.png" alt="View Project" class="button_smaller">
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Loan%20Default/Doc/ML_ShapValues.ipynb" target="_blank">
        <img src="/assets/img/github_button.png" class="button_smaller" alt="View on GitHub">
      </a>
    </div>
  </div>
  </li></ul></p>


#### Face Image Emotion Recognition

<p style="display: inline;">
  <img src="/assets/icons/r.png" width="40">
  <ul><li style="list-style-type: none;">
  For this project, the data contains a set of 2,500 facial images with 22 different emotions. The project focuses on creating an <b>mobile AI program</b> that accurately recognizes the emotion from facial images.<br><br>
  The portability of this AI program (holding storage and memory cost) and the computational efficiency (test running time cost) are of great concern to your client. This translates to a balance between the complexity of variable/features/models used and the predictive performance.<br><br>
  Create an advanced classification engine for faction emotion recognition, using PCA, GBM, CNN, KNN, RF with cross-validation, SVM. Resulting with <b>44% accuracy</b>.<br>
  <div class="row">
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Image%20Recognition/doc/Main.pdf" target="_blank">
          &#9679; Project Report
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/Image%20Recognition" target="_blank">
          &#9679; Github Page
      </a>
    </div>
  </div>
</li></ul></p>


#### Algorithm implementation and evaluation

<p style="display: inline;">
  <img src="/assets/icons/r.png" width="40">
  <ul><li style="list-style-type: none;">
  In this project, I implement matrix factorization by focusing on alternating least square algorithms and KNN as post-processing, in which I try to evaluate how regularization will impact the prediction results.<br>
  Collaborative Filtering using ALS, KNN, TD: Focusing on alternating least square algorithms and KNN as post-processing, in which I try to evaluate how regularization will impact the prediction results.<br>
  <div class="row">
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Collaborative%20Filtering/doc/Main.Rmd" target="_blank">
          &#9679; Project Report
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/Collaborative%20Filtering" target="_blank">
          &#9679; Github Page
      </a>
    </div>
  </div>
</li></ul></p>



#### Shinny App: Eating Healthy in Fast Food Restaurant Chains in NYC

<p style="display: inline;">
  <img src="/assets/icons/r.png" width="40">
  <ul><li style="list-style-type: none;">
  A shinny app provides users with nutrition information for fast food chain restaurants in NYC. Users can choose restaurants based on location and compare menu items.<br>
  <div class="row">
    <div class="column50">
      <a href="https://ttd2111.shinyapps.io/NYCRestaurants/" target="_blank">
          &#9679; Shinny App
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/RestaurantApp" target="_blank">
          &#9679; Github Page
      </a>
    </div>
  </div>
</li></ul></p>

### testing

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  style 2<br>
    <li style="padding-left: 20px; list-style-type: none;">
        &#9679; Kmeans with Elbows and Silhouette Scores
        &nbsp;&nbsp;&nbsp;
        <a href="/portfolio/example/K-means_Clustering/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 100px;"></a>
        &nbsp;&nbsp;&nbsp;
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/PCA_KNN.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
      <br>
        &#9679; Kmeans with Elbows
        &nbsp;&nbsp;&nbsp;
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/PCA_KNN.ipynb" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 100px;"></a>
        &nbsp;&nbsp;&nbsp;
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/PCA_KNN.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
    </li>
    <br>
  <i><b>Libraries:</b> pandas, scipy, sklearn, pyod, seaborn, matplotlib</i>
</li></ul></p>

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  style 1<br>
  <li style="padding-left: 20px; list-style-type: none;">
      &#9679; Kmeans with Elbows and Silhouette Scores
      &nbsp;&nbsp;&nbsp;
  <div class="row">
    <div class="column50">
      <a href="/portfolio/example/Autoencoder_IsolationForest/">
        <img src="/assets/img/project_button.png" alt="View Project" class="button_smallest">
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank">
        <img src="/assets/img/github_button.png" class="button_smallest" alt="View on GitHub">
      </a>
    </div>
  </div>
  </li>
  </li></ul></p>

### Machine Learning
___
#### Unsupervised Learning: Fraud Detection with Hospital data

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  Analyzing a hospital charge data to determine outliers, identify anomalous activities and provide insights of these behaviors using different advanced algorithms.
  <div class="row">
    <div class="column50">
      <a href="/portfolio/example/Autoencoder_IsolationForest/">
        <img src="/assets/img/project_button.png" alt="View Project" class="button_smaller">
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank">
        <img src="/assets/img/github_button.png" class="button_smaller" alt="View on GitHub">
      </a>
    </div>
  </div>
</li></ul></p>

### Natural Language Processing
___
#### Airline Industry Since COVID-19:

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  The reputation of the US airline industry has been greatly damaged due to Covid-19. Tracking the topics and sentiment during these times can help airline companies when best to proceed with a relaunch strategy.<br>

  <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/Airlines%20Covid-19" target="_blank">
    &nbsp; <i><b>Project Details</b></i>
  </a>
    <li style="padding-left: 20px; list-style-type: none;">
        &#9679; Latent semantic analysis
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/LSA /" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Airlines%20Covid-19/doc/LSA%20.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; Named-entity recognition (NER)                   
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/NER/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Airlines%20Covid-19/doc/NER.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; Simhash and Word2Vec Models
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/Simhash&Word2Vec/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Airlines%20Covid-19/doc/Simhash%26Word2Vec.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; Sentiment Analysis using NLTK Sentiment Intensity Analyzer
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/WeeklySummaries_SentimentAnalysis/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Airlines%20Covid-19/doc/WeeklySummaries_SentimentAnalysis.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
    </li>
  <i><b>Libraries:</b> json, nltk, sklearn, scikit spacy, re, sumy, time, pandas, matplotlib</i>
</li></ul></p>


####  Natural Language Processing for Google News in a 30-day period

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75">
  <ul><li style="list-style-type: none;">
  This project aims to track <b>top mentioned topics and trending</b> for Google during Covid-19 time, develop recommendations for the company through <b>topic classification</b>, and define application for other industries. <br>
  <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/Google%20News" target="_blank">
    &nbsp; <i><b>Project Details</b></i>
  </a>
    <li style="padding-left: 20px; list-style-type: none;">
        &#9679; Simhash Text Deduplication
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/Deduplicated/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Google%20News/doc/Deduplicated.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; Topic Modeling with Gensim and pyLDAvis                 
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/TD_Gensim/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Google%20News/doc/TD_Gensim.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; Topic Classification using Taxonomy & Word2Vec
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/Topic Classification/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Google%20News/doc/Topic%20Classification.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
        &#9679; LDA model to identify topic distribution and keywords
        &nbsp;&nbsp;&nbsp;
        <div class="row">
        <div class="column50">
        <a href="/portfolio/example/Word_tokenize_LDA/" target="_blank"><img src="/assets/img/project_button.png" alt="View Project" style="width: 110px;"></a>
        </div>
        <div class="column50">
        <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Google%20News/doc/Word_tokenize_LDA.ipynb" target="_blank"><img src="/assets/img/github_button.png" alt="View on Github" style="width: 110px;"></a>
        </div>
      </div>
    </li>
  <i><b>Libraries:</b> NLTK, gensim, spacy, json, sklearn, scikit, sumy_</i>
</li></ul></p>

#### Lyrics Text Mining

<p style="display: inline;">
  <img src="/assets/icons/r.png" width="40">
  <ul><li style="list-style-type: none;">
  This report is a guidebook with selected Q&A aims at finding out the insights and differences between Pop and Metal lyrics over time.
  <div class="row">
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/PopVsMetal/doc/SongLyrics.pdf" target="_blank">
          &#9679; Project Report
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/PopVsMetal" target="_blank">
          &#9679; Github Page
      </a>
    </div>
  </div>
  <i><b>Libraries:</b>dplyr, tm, tidytext, shiny, wordcloud, ggraph.</i>
</li></ul></p>

### Data Analysis and Visualizations

#### Lorem ipsum dolor sit amet

<p style="display: inline;">
  <img src="/assets/icons/python.png" width="75"> &nbsp;&nbsp;&nbsp;
  <img src="/assets/icons/mysql.png" width="50">
  <ul><li style="list-style-type: none;">
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
  <div class="row">
    <div class="column50">
      <a href="/portfolio/miniprojects/rps_classification/">
        <img src="/assets/img/project_button.png" alt="View Project" class="button">
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank">
        <img src="/assets/img/github_button.png" class="button" alt="View on GitHub">
      </a>
    </div>
  </div>
</li></ul></p>


#### Consectetur adipiscing elit

<p style="display: inline;">
  <img src="/assets/icons/postgresql.png" width="75"> &nbsp;&nbsp;&nbsp;
  <img src="/assets/icons/tableau.png" width="100"> &nbsp;&nbsp;&nbsp;
  <img src="/assets/icons/r.png" width="50">
  <ul><li style="list-style-type: none;">
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
  <div class="row">
    <div class="column50">
      <a href="/portfolio/miniprojects/rps_classification/">
        <img src="/assets/img/project_button.png" alt="View Project" class="button">
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank">
        <img src="/assets/img/github_button.png" class="button" alt="View on GitHub">
      </a>
    </div>
  </div>
</li></ul></p>


#### Consectetur adipiscing elit

<p style="display: inline;">
  <img src="/assets/icons/postgresql.png" width="75"> &nbsp;&nbsp;&nbsp;
  <img src="/assets/icons/tableau.png" width="100"> &nbsp;&nbsp;&nbsp;
  <img src="/assets/icons/r.png" width="50">
  <ul><li style="list-style-type: none;">
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
  <div class="row">
    <div class="column50">
      <a href="/portfolio/miniprojects/rps_classification/">
        <img src="/assets/img/project_button.png" alt="View Project" class="button">
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank">
        <img src="/assets/img/github_button.png" class="button" alt="View on GitHub">
      </a>
    </div>
  </div>
</li></ul></p>



#### Consectetur adipiscing elit

<p style="display: inline;">
  <img src="/assets/icons/postgresql.png" width="75"> &nbsp;&nbsp;&nbsp;
  <img src="/assets/icons/tableau.png" width="100"> &nbsp;&nbsp;&nbsp;
  <img src="/assets/icons/r.png" width="50">
  <ul><li style="list-style-type: none;">
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
  <div class="row">
    <div class="column50">
      <a href="/portfolio/miniprojects/rps_classification/">
        <img src="/assets/img/project_button.png" alt="View Project" class="button">
      </a>
    </div>
    <div class="column50">
      <a href="https://github.com/tramduong/Data-Science-Portfolio/blob/master/Hospital%20Charges%20Fraud/Unspervised/Autoencoder_IsolationForest.ipynb" target="_blank">
        <img src="/assets/img/github_button.png" class="button" alt="View on GitHub">
      </a>
    </div>
  </div>
</li></ul></p>
