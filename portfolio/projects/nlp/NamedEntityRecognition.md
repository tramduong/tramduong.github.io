
# Named Entity Recognition

This project aims to take preconditioned text and apply transformations for:
+ Tagging named entities
+ Entity Recognition
+ Entity Disambiguation

**Extract text from an article**


```python
# Use urllib or requests package to read this CNBC article through its URL link
import urllib
html = urllib.request.urlopen('https://www.cnbc.com/2020/06/27/us-coronavirus-cases-surge-by-more-than-45000-as-states-roll-back-reopenings.html').read()
```


```python
# Use BeautifulSoup (Links to an external site.) or another HTML parsing package to extract text from the article.
from bs4 import BeautifulSoup
from bs4.element import Comment

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

cnbc_url=text_from_html(html)
```

**Extract Company/Organization and Geo entities**

---




```python
import spacy as sp
```


```python
# Load pre-existing spacy model
nlp = sp.load("en_core_web_sm")
```


```python
#Count number of entities
article = nlp(cnbc_url)
from collections import Counter
labels = [x.label_ for x in article.ents]
Counter(labels)
```




    Counter({'CARDINAL': 26,
             'DATE': 53,
             'GPE': 46,
             'MONEY': 1,
             'NORP': 3,
             'ORG': 35,
             'PERCENT': 15,
             'PERSON': 17,
             'TIME': 4})




```python
# Visualization of entities from the article
from spacy import displacy
options = {"ents":["ORG","GPE"]}
displacy.render(article, style ="ent", options = options, jupyter = True)
```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">× LOG IN SIGN UP Keep Me Logged In Skip Navigation SIGN IN Pro Watchlist Make It Select USA INTL Markets Pre-Markets U.S. Markets Currencies Cryptocurrency Futures &amp; Commodities Bonds Funds &amp; ETFs 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Watchlist Business Economy Finance Health &amp; Science Media
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 Real Estate Energy Transportation Industrials Retail Wealth Small Business Investing Invest In You Personal Finance Financial Advisors Trading Nation Options Action ETF Street Buffett Archive Earnings Trader Talk Tech Cybersecurity Enterprise Internet Media Mobile Social Media Venture Capital Tech Guide Politics 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    White House
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 Policy Defense Congress 2020 Elections 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CNBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 TV Live TV Live Audio Latest Video Top Video CEO Interviews Business Day Shows Primetime Shows CNBC World Digital Originals Full Episodes Menu SEARCH QUOTES Markets Pre-Markets U.S. Markets Currencies Cryptocurrency Futures &amp; Commodities Bonds Funds &amp; ETFs 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Watchlist Business Economy Finance Health &amp; Science Media
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 Real Estate Energy Transportation Industrials Retail Wealth Small Business Investing Invest In You Personal Finance Financial Advisors Trading Nation Options Action ETF Street Buffett Archive Earnings Trader Talk Tech Cybersecurity Enterprise Internet Media Mobile Social Media Venture Capital Tech Guide Politics 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    White House
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 Policy Defense Congress 2020 Elections 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CNBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 TV Live TV Live Audio Latest Video Top Video CEO Interviews Business Day Shows Primetime Shows CNBC World Digital Originals Full Episodes Menu Health and 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Science
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    U.S.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 coronavirus cases surge by more than 45,000 in one day, total surpasses 2.5 million Published Sat, Jun 27 2020 12:05 PM EDT 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Updated Sun
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, Jun 28 2020 11:59 AM EDT Noah Higgins-Dunn @higginsdunn Key Points There were 45,255 additional Covid-19 cases reported across the nation on Friday, bringing the total to more than 2.5 million cases, according to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Johns Hopkins University
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 data. As of Friday, the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    U.S.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 seven-day average of new cases increased more than 41% compared with a week ago. Cases are growing by 5% or more based on a seven-day average in 34 states across the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    U.S.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, including 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Arizona
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Texas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    California
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Florida
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 and 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Nevada
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
. Some states, like 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Texas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 and 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Florida
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, have had to re-close some businesses while others, like 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Arizona
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, have put any further plans on pause. Arrows pointing outwards The 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    U.S.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 reported more than 45,000 cases of coronavirus on Friday, a record breaking increase, as some of the hardest-hit states begin to pause or roll back their reopening plans, according to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Johns Hopkins University
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 data. There were 45,255 additional Covid-19 cases reported across the nation on Friday, bringing the total to more than 2.46 million cases, according to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Johns Hopkins
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 data. As of Friday, the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    U.S.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 seven-day average of new cases increased more than 41% compared with a week ago. It was still above 40% on Saturday. The number of people infected with the coronavirus surpassed 2.5 million — the most infections for any country across the globe — with more than 42,000 new infections reported Saturday alone, according to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Johns Hopkins
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
. At least 125,559 people have died from the virus in the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    U.S.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 Cases are growing by 5% or more based on a seven-day average in 38 states across the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    U.S.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, including 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Arizona
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Texas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    California
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Florida
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 and 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Nevada
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
. &quot;There are more cases. There are more hospitalizations in some of those places and soon you'll be seeing more deaths,&quot; 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    White House
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 health advisor Dr. Anthony Fauci said in an interview with 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CNBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
's Meg Tirrell on Friday that was aired by 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the Milken Institute
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
. &quot;Even though the deaths are coming down as a country, that doesn't mean that you're not going to start seeing them coming up now,&quot; he said. Deaths caused by Covid-19 lag behind other data points such as hospitalizations, which lag confirmed infections as the disease can take weeks to fully develop in a person. Hospitalizations due to Covid-19 were growing in 16 states as of Saturday, according to a 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CNBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 analysis of 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Covid Tracking Project
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 data. On Friday, the 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    White House
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 coronavirus task force warned younger people , who they say are driving new infections, that they may not be as vulnerable to serious disease but could infect someone who is. &quot;If there's one message that comes through today I hope it is saying to younger Americans in these states, and in these counties in particular, that they are a big part of the numbers that we are seeing in new cases,&quot; Vice President Mike Pence said at a press briefing. Dr. Deborah Birx, the 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    White House
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 coronavirus response coordinator, added that the coronavirus poses a greater risk to those with underlying health conditions, such as diabetes and significant obesity, which span every age group. While the rise in daily case numbers could reflect increased testing in certain locations, some states are reporting higher positivity rates. The positivity rate indicates the percentage of tests that come back positive in a specific region. Epidemiologists say this number can indicate how broadly the virus is spreading throughout a community. 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    California
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
's positivity rate has increased to more than 5% over the last two weeks as the state reports record increases in daily new cases. 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Texas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
' positivity rate exceeded 10% on Wednesday, which is a level that raises a &quot; warning flag ,&quot; according to Gov. Greg Abbott. 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Arizona
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
's rate is now averaging above 11%, according to the state's 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    department of health
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
. On Saturday, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Washington
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 Gov. Jay Inslee announced that the state would halt some of its counties from moving into &quot;Phase 4&quot; of the state's reopening plan as the coronavirus shows signs of accelerating. The state is taking a phrased approach to reopening, allowing some counties to reopen before others. &quot;Phase 4&quot; of the state's reopening plan would resume recreational activities and would allow for gatherings of more than 50 people, according to the state's plan. A county in phase 4 would also be allowed to reopen its nightclubs, concert venues and large sporting events, according to the state's plan. Eight of the state's 39 counties were prepared to move into phase 4, according to the order. 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Texas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 An employee sweeps inside a closed bar in 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Austin
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Texas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, June 26, 2020. Sergio Flores | AFP | Getty Images 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Texas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 Gov. Greg Abbott on Thursday ordered all licensed hospitals in hard-hit 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Bexar
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Dallas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, Harris and Travis counties to postpone elective procedures in order to protect hospital capacity for Covid-19 patients. Those counties include the state's largest cities — 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Houston
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    San Antonio
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Dallas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 and 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Austin
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
. As of Friday, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Texas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 reported a 57% increase in hospitalizations, based on a seven-day average, compared with one week ago. It's average number of daily cases grew by nearly 70%, according to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    John Hopkins
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 data. Abbott said he would roll back some of the state's reopening plan on Friday, closing the state's bars and reducing the capacity for indoor dining, among other modifications and closures . &quot;As I said from the start, if the positivity rate rose above 10%, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the State of Texas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 would take further action to mitigate the spread of COVID-19,&quot; Abbott said in a press release. 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Arizona
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Arizona
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 Gov. Doug Ducey said on Thursday that the state's hospitals are seeing additional stress and are &quot;likely to hit surge capacity very soon.&quot; The state's health department reported more than 3,500 new cases on Saturday, nearing the record daily peak set on Tuesday. As of Friday, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Arizona
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 reported a 36% increase in hospitalizations, based on a seven-day average, compared with one week ago. It's average number of daily cases grew by more than 42%, according to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    John Hopkins
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 data. &quot;Covid-19 is widespread in 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Arizona
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
. It's in all 15 of our counties. It's growing, and it's growing fast across all age groups and demographics. Anyone can get this virus, and anyone can spread this virus,&quot; he said at a press conference. If the state's reopening were a traffic light, Ducey said that it would be in the &quot;yellow&quot; or &quot;yield&quot; position. 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Florida
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Florida
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 has banned drinking at bars again in an effort to slow the spread of the coronavirus. The state reported more than 9,600 new Covid-19 cases on Saturday , the second day of record breaking new cases, according to the state's health department. Gov. Ron DeSantis said on Thursday that the state doesn't have plans for continuing its step-by-step reopening plan. He added that the state &quot;never anticipated&quot; continuing to move forward at this point. &quot;We are where we are. I didn't say we were going to go on to the next phase,&quot; DeSantis said at a news briefing. On Thursday, the city of 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Miami
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 issued a mandate requiring facial coverings be worn in public at all times until further notice. Those who defy the order could be fined $50 and face court appearances for repeated offenses. &quot;All options have to be on the table. When we see our hospitalizations go up, our 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ICU
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 beds go up, our ventilators are going up. Still with sufficient capacity, but going up. It's worrisome,&quot; 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Miami
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 Mayor Francis Suarez told 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CNN
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 on Friday when asked whether he would consider instituting another stay-at-home order. 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Nevada Nevada
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 reported a record jump of nearly 1,100 new Covid-19 cases on Saturday, according to the state's health department. As of Friday, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Nevada
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 reported a near 17% increase in hospitalizations, based on a seven-day average, compared with one week ago. It's average number of daily cases grew by more than 49%, according to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Johns Hopkins
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 data. On Wednesday, Gov. Steve Sisolak ordered that all people in the state wear a face covering when in pubic after four weeks of climbing case numbers. The amount of patients requiring intensive-care unit beds and ventilators has continued to hold steady, according to the order. &quot;Clearly for many, the excitement and enthusiasm for escaping our confinement and finally being able to enjoy dinner out with our families, buy new clothes or get a haircut, overshadowed the good judgement we practiced in the previous months,&quot; Sisolak said in a statement. — 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CNBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
's 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Nate Rattner
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 contributed to this report. VIDEO 3:55 03:55 Dr. Fauci's blunt assessment of 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    U.S.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 coronavirus spikes News Videos Related Tags Nate Rattner Meg Tirrell Health care industry Politics 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    U.S.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 Economy Subscribe to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CNBC PRO Licensing &amp; Reprints
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 CNBC Councils Supply Chain Values Advertise With Us Join the CNBC Panel Digital Products News Releases Closed Captioning Corrections About CNBC Internships Site Map Podcasts AdChoices Careers Help Contact News Tips Got a confidential news tip? We want to hear from you. Get In Touch CNBC Newsletters Sign up for free newsletters and get more 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CNBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 delivered to your inbox Sign Up Now Get this delivered to your inbox, and more info about our products and services. Privacy Policy | Do Not Sell My Personal Information | Terms of Service © 2020 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    CNBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 LLC. All Rights Reserved.  A Division of NBCUniversal Data is a real-time snapshot *Data is delayed at least 15 minutes. 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Global Business and Financial News
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, Stock Quotes, and 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Market Data
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 and Analysis. Market Data Terms of Use and 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Disclaimers Data
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 also provided by </div></span>



```python
# Extract Company/Organization and Geo entities from the article chosen
for x in article.ents:
    if x.label_ == 'ORG' or x.label_ == 'GPE':
        print(x.label_ + ' - ' + x.text)
```

    ORG - Watchlist Business Economy Finance Health & Science Media
    ORG - White House
    ORG - CNBC
    ORG - Watchlist Business Economy Finance Health & Science Media
    ORG - White House
    ORG - CNBC
    ORG - Science
    GPE - U.S.
    ORG - Updated Sun
    ORG - Johns Hopkins University
    GPE - U.S.
    GPE - U.S.
    GPE - Arizona
    GPE - Texas
    GPE - California
    GPE - Florida
    GPE - Nevada
    GPE - Texas
    GPE - Florida
    GPE - Arizona
    GPE - U.S.
    ORG - Johns Hopkins University
    ORG - Johns Hopkins
    GPE - U.S.
    ORG - Johns Hopkins
    GPE - U.S.
    GPE - U.S.
    GPE - Arizona
    GPE - Texas
    GPE - California
    GPE - Florida
    GPE - Nevada
    ORG - White House
    ORG - CNBC
    ORG - the Milken Institute
    ORG - CNBC
    ORG - Covid Tracking Project
    ORG - White House
    ORG - White House
    GPE - California
    GPE - Texas
    GPE - Arizona
    ORG - department of health
    GPE - Washington
    GPE - Texas
    GPE - Austin
    GPE - Texas
    GPE - Texas
    ORG - Bexar
    GPE - Dallas
    GPE - Houston
    GPE - San Antonio
    GPE - Dallas
    GPE - Austin
    GPE - Texas
    ORG - John Hopkins
    GPE - the State of Texas
    GPE - Arizona
    GPE - Arizona
    GPE - Arizona
    ORG - John Hopkins
    GPE - Arizona
    GPE - Florida
    GPE - Florida
    GPE - Miami
    ORG - ICU
    GPE - Miami
    ORG - CNN
    ORG - Nevada Nevada
    GPE - Nevada
    ORG - Johns Hopkins
    ORG - CNBC
    ORG - Nate Rattner
    GPE - U.S.
    GPE - U.S.
    ORG - CNBC PRO Licensing & Reprints
    ORG - CNBC
    ORG - CNBC
    ORG - Global Business and Financial News
    ORG - Market Data
    ORG - Disclaimers Data
    

**Spark Entities Matched**


```python
!pip install pyspark
```

    Requirement already satisfied: pyspark in /usr/local/lib/python3.6/dist-packages (3.0.0)
    Requirement already satisfied: py4j==0.10.9 in /usr/local/lib/python3.6/dist-packages (from pyspark) (0.10.9)
    


```python
from pyspark.conf import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext() 
config = sc.getConf() 
sqlContext = SQLContext(sc)

print("Using Apache Spark Version", sc.version)
```

    Using Apache Spark Version 3.0.0
    


```python
df = sqlContext.read.option("header", "true").option("delimiter", ",") \
                    .option("inferSchema", "true") \
                    .csv("/content/drive/My Drive/data/cb_odm_092419.csv")
```


```python
#Create the list of Company or Organization entities in the picked article
org_list = []
for x in article.ents:
    if x.label_ == 'ORG'and x.text not in org_list:
        org_list.append(x.text)
org_list
```




    ['Watchlist Business Economy Finance Health & Science Media',
     'White House',
     'CNBC',
     'Science',
     'Updated Sun',
     'Johns Hopkins University',
     'Johns Hopkins',
     'the Milken Institute',
     'Covid Tracking Project',
     'department of health',
     'Bexar',
     'John Hopkins',
     'ICU',
     'CNN',
     'Nevada Nevada',
     'Nate Rattner',
     'CNBC PRO Licensing & Reprints',
     'Global Business and Financial News',
     'Market Data',
     'Disclaimers Data']




```python
# Find matches of Company or Organization entities identified in Step 3 using rlik
for x in org_list:
  match_df = df[df['name'].rlike(x)]
  if match_df.count() > 0:
    match_df['crunchbase_uuid','name','homepage_domain','stock_symbol'].show(10, truncate=False)
```

    +------------------------------------+---------------------------------------------------+-------------------------+------------+
    |crunchbase_uuid                     |name                                               |homepage_domain          |stock_symbol|
    +------------------------------------+---------------------------------------------------+-------------------------+------------+
    |8c3d547e-2b14-2787-a593-c656d253e2fb|White House Business Solutions Pvt Ltd.,           |whitehouseit.com         |:           |
    |65fb148f-9051-a982-4697-185c675cc21b|The White House                                    |whitehouse.gov           |:           |
    |c5ff601c-17bf-af99-3cfe-068b92c86110|White House Black Market                           |whitehouseblackmarket.com|:           |
    |ac9c0e82-2903-0923-2993-001e9b009add|White House Brothers                               |whitehousebrothers.com   |:           |
    |48d207fb-221a-6048-1e83-219f35a473a8|Flat White Houseboat                               |whitehouseboats.com.au   |:           |
    |35fc96f0-562c-5fa3-a567-3d6d40bba44b|White House Office of Science and Technology Policy|whitehouse.gov           |:           |
    |6f8d9641-5cf8-f03c-d2c4-d31457c3c35b|White House Historical Association                 |whitehousehistory.org    |:           |
    |c541416a-1f4a-4711-57b7-1339cac5419b|Council of Economic Affairs, The White House       |whitehouse.gov           |:           |
    |0f1f5ffe-9435-7382-ddbf-270615f43695|The White House                                    |whitehouse.gov           |:           |
    |2afb6019-00fc-84b6-e550-db629ca8b4bd|The White House                                    |whitehouse.gov           |:           |
    +------------------------------------+---------------------------------------------------+-------------------------+------------+
    only showing top 10 rows
    
    
