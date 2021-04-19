---
layout: page
title: NYC Fast Food Restaurant Nutrition App
description: >
  This page contains the code and the description of the app.
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


This project creates a shiny app that provides users with nutrition information for fast food chain restaurants in NYC. Users can choose restaurants based on location and compare menu items.<br>

For a more detailed explanation, check out [My Github](https://github.com/tramduong/Data-Science-Portfolio/tree/master/RestaurantApp)
{:.note}

<br>


<img src="/assets/img/ml/nyshiny.jpg"  alt="Portfolio Banner" class="banner">


<br>

### Shiny App
___

The app provides users with nutrition information for fast food chain restaurants in NYC. Users can choose restaurants based on location and compare menu items.<br>
<div class="row">
  <div class="column50">
    <a href="https://ttd2111.shinyapps.io/NYCRestaurants/" target="_blank"><img src="/assets/img/shiny_button.png" alt="View Project" class="button_smaller">
    </a>
  </div>
  <div class="column50">
    <a href="https://github.com/tramduong/Data-Science-Portfolio/tree/master/RestaurantApp" target="_blank"><img src="/assets/img/github_button.png" alt="View on GitHub" class="button_smaller">
    </a>
  </div>
</div>

### Project Description
___

Do you know what are the nutrients of menu items from national restaurant chains? Nowadays, customers care more and more about the nutritional value of the foods they are eating. Although some restaurants already include calories information and other nutrition information in their menus, many customers want to see more details and compare similar items in different restaurants.

<b>Goal:</b> Inspired by how nutritional values affect human health, this project aims to develop an App using R shiny to visualize the most common nutrients and the menu information of the top national restaurant chains, following with these restaurants location in New York City. This app does not only give users insight into the nutritional values of top restaurants, but also provides a useful tool for finding and comparing nearby restaurants.


### User Guide
___

+ ***Map***: This part contains a map of NYC. The user can click a location on the map and view the restaurants in that area.
+ ***Comparison***: Choose the restaurants, food types, and nutritional facts you want to explore/compare. Click a food item to see a breakdown of its contents.
+ ***Statistics Analysis***: This part contains some interactive graphs and bar charts that help users to better understand of all main nutritional factors and provide the list of low or high content of each specific nutrient per each restaurant.
+ ***Data Search***: Search the menu data or restaurant location data used to develop our app.
