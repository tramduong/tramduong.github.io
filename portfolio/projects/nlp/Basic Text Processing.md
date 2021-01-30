
# Basic Text Processing

This project aims to:
+ Explore regular expression patterns and functionality
+ Strip HTML tags, images, code scripts
+ Tokenize words and sentences
+ Lemmatize stem word tokens
+ Assign Part-of-Speech (POS) tags

# Use urllib or requests package to read html 


```python
from urllib import request
```


```python
#1. Use urllib or requests package to read html 
html = request.urlopen('https://www.cnbc.com/2020/12/16/coronavirus-stimulus-update-congress-may-offer-900-billion-relief-plan.html').read()
# Range used to limit amount of output
html[:2000]
```




    b'<!DOCTYPE html><html lang="en" prefix="og=https://ogp.me/ns#" itemType="https://schema.org/WebPage"><head><script src="//fm.cnbc.com/applications/cnbc.com/resources/newrelic/agent.js" defer=""></script><link rel="preload" as="script" href="https://sb.scorecardresearch.com/beacon.js"/><title itemProp="name">Coronavirus stimulus update: Congress working on $900 billion relief plan</title><meta name="viewport" content="initial-scale=1.0, width=device-width"/><meta http-equiv="X-UA-Compatible" content="IE=Edge"/><meta property="AssetType" content="cnbcnewsstory"/><meta property="pageNodeId" content="106812089"/><meta itemProp="description" name="description" content="The coronavirus relief deal would reportedly include direct payments but not liability protections or state and local relief."/><link itemProp="url" rel="canonical" href="https://www.cnbc.com/2020/12/16/coronavirus-stimulus-update-congress-may-offer-900-billion-relief-plan.html"/><link rel="icon" type="image/png" href="/favicon.ico"/><meta property="og:type" content="article"/><meta property="og:title" content="Congress closes in on a $900 billion Covid relief deal as Americans await aid"/><meta property="og:description" content="The coronavirus relief deal would reportedly include direct payments but not liability protections or state and local relief."/><meta property="og:url" content="https://www.cnbc.com/2020/12/16/coronavirus-stimulus-update-congress-may-offer-900-billion-relief-plan.html"/><meta property="og:site_name" content="CNBC"/><meta http-equiv="pics-label" content="(pics-1.1 &quot;http://www.icra.org/ratingsv02.html&quot; l gen true for &quot;http://www.msnbc.msn.com&quot; r (nz 1vz 1lz 1oz 1cz 1) &quot;http://www.rsac.org/ratingsv01.html&quot; l gen true for &quot;http://www.msnbc.msn.com&quot; r (l 0n 0s 0v 0))"/><meta itemProp="dateCreated" content="2020-12-16T14:11:08+0000"/><meta itemProp="dateModified" content="2020-12-16T17:57:09+0000"/><meta itemProp="lastReviewed" content="2020-12-16T'



# Use BeautifulSoup or another HTML parsing package to extract text from the article


```python
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
```


```python
# Extract text from the article.
text = text_from_html(html)
```

#  Use re (regular expression) package


```python
import re
#Find all matches of $ amounts in the article
amount = re.findall('\$\d*\.?\d+?', text)

print('The $ amounts in the article: \n'+ str(amount))
first_index = 0
amount_indexes = []

for i in range(text.count('$')):
    new_list = text[first_index:]
    match_indexes = new_list.index('$') + 1
    amount_indexes.append(first_index + new_list.index('$'))
    first_index += match_indexes
    
print('Matched positions of the $ amounts: \n' + str(amount_indexes))

```

    The $ amounts in the article: 
    ['$900', '$900', '$900', '$900', '$300', '$900', '$900', '$600', '$1', '$900', '$600', '$600', '$900']
    Matched positions of the $ amounts: 
    [1722, 1921, 2394, 2494, 3684, 3893, 5030, 5337, 5604, 6000, 6215, 6234, 7056]
    


```python
#substitute all numbers with # and print output
subs_output = re.sub(r'[0-9]','#',text)
print(subs_output[:2000])
```

    Skip Navigation Markets Pre-Markets U.S. Markets Currencies Cryptocurrency Futures & Commodities Bonds Funds & ETFs Business Economy Finance Health & Science Media Real Estate Energy Transportation Industrials Retail Wealth Life Small Business Investing Invest In You Personal Finance Fintech Financial Advisors Trading Nation Options Action ETF Street Buffett Archive Earnings Trader Talk Tech Cybersecurity Enterprise Internet Media Mobile Social Media Venture Capital Tech Guide Politics White House Policy Defense Congress #### Elections CNBC TV Live TV Live Audio Business Day Shows The News with Shepard Smith Entertainment Shows Full Episodes Latest Video Top Video CEO Interviews CNBC Documentaries CNBC Podcasts CNBC World Digital Originals Live TV Schedule Watchlist PRO PRO News PRO Live Subscribe Sign In Menu Make It Select USA INTL Search quotes, news & videos SIGN IN Markets Pre-Markets U.S. Markets Currencies Cryptocurrency Futures & Commodities Bonds Funds & ETFs Business Economy Finance Health & Science Media Real Estate Energy Transportation Industrials Retail Wealth Life Small Business Investing Invest In You Personal Finance Fintech Financial Advisors Trading Nation Options Action ETF Street Buffett Archive Earnings Trader Talk Tech Cybersecurity Enterprise Internet Media Mobile Social Media Venture Capital Tech Guide Politics White House Policy Defense Congress #### Elections CNBC TV Live TV Live Audio Business Day Shows The News with Shepard Smith Entertainment Shows Full Episodes Latest Video Top Video CEO Interviews CNBC Documentaries CNBC Podcasts CNBC World Digital Originals Live TV Schedule Watchlist PRO PRO News PRO Live Subscribe Sign In Menu Politics Congress closes in on a $### billion Covid relief deal as Americans await aid Published Wed, Dec ## #### #:## AM EST Updated Wed, Dec ## #### ##:## PM EST Jacob Pramuk @jacobpramuk Key Points Congress is getting close to a $### billion coronavirus relief deal and could announce it as soon as Wednesday
    


```python
#Count (using regular expressions) ”Netflix” and “Disney” mentions 
print('Total number mentions of Democrats: '+str(len(re.findall('Democrats', text, flags=0))))
print('Total number mentions of Congress: '+str(len(re.findall('Congress', text, flags=0))))
```

    Total number mentions of Democrats: 4
    Total number mentions of Congress: 11
    

#  Use NTLK and/or Spacy (Links to an external site.) tokenization features


```python
import nltk
from nltk import word_tokenize, sent_tokenize, ngrams, pos_tag, RegexpParser
from collections import Counter
```


```python
#Tokenize sentences
sentences = sent_tokenize(text)
for sentence in sentences[:15]:
    print(sentence)
```

    Skip Navigation Markets Pre-Markets U.S. Markets Currencies Cryptocurrency Futures & Commodities Bonds Funds & ETFs Business Economy Finance Health & Science Media Real Estate Energy Transportation Industrials Retail Wealth Life Small Business Investing Invest In You Personal Finance Fintech Financial Advisors Trading Nation Options Action ETF Street Buffett Archive Earnings Trader Talk Tech Cybersecurity Enterprise Internet Media Mobile Social Media Venture Capital Tech Guide Politics White House Policy Defense Congress 2020 Elections CNBC TV Live TV Live Audio Business Day Shows The News with Shepard Smith Entertainment Shows Full Episodes Latest Video Top Video CEO Interviews CNBC Documentaries CNBC Podcasts CNBC World Digital Originals Live TV Schedule Watchlist PRO PRO News PRO Live Subscribe Sign In Menu Make It Select USA INTL Search quotes, news & videos SIGN IN Markets Pre-Markets U.S. Markets Currencies Cryptocurrency Futures & Commodities Bonds Funds & ETFs Business Economy Finance Health & Science Media Real Estate Energy Transportation Industrials Retail Wealth Life Small Business Investing Invest In You Personal Finance Fintech Financial Advisors Trading Nation Options Action ETF Street Buffett Archive Earnings Trader Talk Tech Cybersecurity Enterprise Internet Media Mobile Social Media Venture Capital Tech Guide Politics White House Policy Defense Congress 2020 Elections CNBC TV Live TV Live Audio Business Day Shows The News with Shepard Smith Entertainment Shows Full Episodes Latest Video Top Video CEO Interviews CNBC Documentaries CNBC Podcasts CNBC World Digital Originals Live TV Schedule Watchlist PRO PRO News PRO Live Subscribe Sign In Menu Politics Congress closes in on a $900 billion Covid relief deal as Americans await aid Published Wed, Dec 16 2020 9:11 AM EST Updated Wed, Dec 16 2020 12:57 PM EST Jacob Pramuk @jacobpramuk Key Points Congress is getting close to a $900 billion coronavirus relief deal and could announce it as soon as Wednesday.
    Top Republicans and Democrats negotiated a government funding and pandemic aid agreement into Tuesday night, and Senate leaders Mitch McConnell and Chuck Schumer said they hoped to reach an accord "soon."
    The government will shut down on Saturday and 12 million people will lose unemployment benefits the day after Christmas if Congress fails to act.
    VIDEO 3:25 03:25 Congress closes in on a $900 billion Covid relief deal — Here's the latest News Videos Congressional leaders closed in on a $900 billion coronavirus relief deal Wednesday as millions of struggling Americans wait for help.
    The developing aid agreement would not include liability protections for businesses or aid to state and local government, CNBC confirmed.
    Disagreements over those two issues have blocked lawmakers from crafting a year-end rescue package.
    "We made major headway toward hammering out a targeted pandemic relief package that would be able to pass both chambers with bipartisan majorities," Senate Majority Leader Mitch McConnell , R-Ky., said on the Senate floor Wednesday after a night of talks among the top four congressional leaders.
    Speaking after McConnell, Senate Minority Leader Chuck Schumer, D-N.Y., said "we are close to an agreement" but noted "it's not a done deal yet."
    He added that Democrats "would like to have gone considerably further" to offer relief and will press for more aid after President-elect Joe Biden takes office on Jan. 20.
    The measure would contain a direct payment to Americans in some amount, as well as enhanced federal unemployment insurance, NBC News reported.
    In addition, Republican Sen. Steve Daines of Montana told CNBC that the deal would have roughly $300 billion in small business aid including Paycheck Protection Program loans, money for Covid-19 vaccine distribution and testing and relief for hospitals.
    "I'm cautiously optimistic we're going to see this $900 billion package released today, and this will likely get passed before we go home this weekend," Daines told CNBC's "Squawk Box" on Wednesday morning.
    VIDEO 3:21 03:21 Sen. Daines: Covid relief deal could be announced Wednesday morning Squawk Box Congress has rushed to find consensus on legislation to fund the government and rescue a health-care system and economy buckling under the pandemic.
    If lawmakers fail to act, the government will shut down on Saturday, 12 million people will lose unemployment benefits the day after Christmas and millions more could face the threat of eviction.
    Congress finally neared an emergency relief agreement after Tuesday night negotiations among McConnell, Schumer, House Speaker Nancy Pelosi, D-Calif., and House Minority Leader Kevin McCarthy, R-Calif. Republicans and Democrats had failed for months to make progress toward a bill that could get through a divided Congress.
    


```python
# Tokenize word from text
words = word_tokenize(text)
for word in words[:10]:
    print(word)
```

    Skip
    Navigation
    Markets
    Pre-Markets
    U.S.
    Markets
    Currencies
    Cryptocurrency
    Futures
    &
    


```python
# Remove all English stop words
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 
filtered_words = [] 

for w in words: 
    if w not in stop_words: 
        filtered_words.append(w) 

print('Removing all English stop words: \n',filtered_words,'\n')
```

    Removing all English stop words: 
     ['Skip', 'Navigation', 'Markets', 'Pre-Markets', 'U.S.', 'Markets', 'Currencies', 'Cryptocurrency', 'Futures', '&', 'Commodities', 'Bonds', 'Funds', '&', 'ETFs', 'Business', 'Economy', 'Finance', 'Health', '&', 'Science', 'Media', 'Real', 'Estate', 'Energy', 'Transportation', 'Industrials', 'Retail', 'Wealth', 'Life', 'Small', 'Business', 'Investing', 'Invest', 'In', 'You', 'Personal', 'Finance', 'Fintech', 'Financial', 'Advisors', 'Trading', 'Nation', 'Options', 'Action', 'ETF', 'Street', 'Buffett', 'Archive', 'Earnings', 'Trader', 'Talk', 'Tech', 'Cybersecurity', 'Enterprise', 'Internet', 'Media', 'Mobile', 'Social', 'Media', 'Venture', 'Capital', 'Tech', 'Guide', 'Politics', 'White', 'House', 'Policy', 'Defense', 'Congress', '2020', 'Elections', 'CNBC', 'TV', 'Live', 'TV', 'Live', 'Audio', 'Business', 'Day', 'Shows', 'The', 'News', 'Shepard', 'Smith', 'Entertainment', 'Shows', 'Full', 'Episodes', 'Latest', 'Video', 'Top', 'Video', 'CEO', 'Interviews', 'CNBC', 'Documentaries', 'CNBC', 'Podcasts', 'CNBC', 'World', 'Digital', 'Originals', 'Live', 'TV', 'Schedule', 'Watchlist', 'PRO', 'PRO', 'News', 'PRO', 'Live', 'Subscribe', 'Sign', 'In', 'Menu', 'Make', 'It', 'Select', 'USA', 'INTL', 'Search', 'quotes', ',', 'news', '&', 'videos', 'SIGN', 'IN', 'Markets', 'Pre-Markets', 'U.S.', 'Markets', 'Currencies', 'Cryptocurrency', 'Futures', '&', 'Commodities', 'Bonds', 'Funds', '&', 'ETFs', 'Business', 'Economy', 'Finance', 'Health', '&', 'Science', 'Media', 'Real', 'Estate', 'Energy', 'Transportation', 'Industrials', 'Retail', 'Wealth', 'Life', 'Small', 'Business', 'Investing', 'Invest', 'In', 'You', 'Personal', 'Finance', 'Fintech', 'Financial', 'Advisors', 'Trading', 'Nation', 'Options', 'Action', 'ETF', 'Street', 'Buffett', 'Archive', 'Earnings', 'Trader', 'Talk', 'Tech', 'Cybersecurity', 'Enterprise', 'Internet', 'Media', 'Mobile', 'Social', 'Media', 'Venture', 'Capital', 'Tech', 'Guide', 'Politics', 'White', 'House', 'Policy', 'Defense', 'Congress', '2020', 'Elections', 'CNBC', 'TV', 'Live', 'TV', 'Live', 'Audio', 'Business', 'Day', 'Shows', 'The', 'News', 'Shepard', 'Smith', 'Entertainment', 'Shows', 'Full', 'Episodes', 'Latest', 'Video', 'Top', 'Video', 'CEO', 'Interviews', 'CNBC', 'Documentaries', 'CNBC', 'Podcasts', 'CNBC', 'World', 'Digital', 'Originals', 'Live', 'TV', 'Schedule', 'Watchlist', 'PRO', 'PRO', 'News', 'PRO', 'Live', 'Subscribe', 'Sign', 'In', 'Menu', 'Politics', 'Congress', 'closes', '$', '900', 'billion', 'Covid', 'relief', 'deal', 'Americans', 'await', 'aid', 'Published', 'Wed', ',', 'Dec', '16', '2020', '9:11', 'AM', 'EST', 'Updated', 'Wed', ',', 'Dec', '16', '2020', '12:57', 'PM', 'EST', 'Jacob', 'Pramuk', '@', 'jacobpramuk', 'Key', 'Points', 'Congress', 'getting', 'close', '$', '900', 'billion', 'coronavirus', 'relief', 'deal', 'could', 'announce', 'soon', 'Wednesday', '.', 'Top', 'Republicans', 'Democrats', 'negotiated', 'government', 'funding', 'pandemic', 'aid', 'agreement', 'Tuesday', 'night', ',', 'Senate', 'leaders', 'Mitch', 'McConnell', 'Chuck', 'Schumer', 'said', 'hoped', 'reach', 'accord', '``', 'soon', '.', "''", 'The', 'government', 'shut', 'Saturday', '12', 'million', 'people', 'lose', 'unemployment', 'benefits', 'day', 'Christmas', 'Congress', 'fails', 'act', '.', 'VIDEO', '3:25', '03:25', 'Congress', 'closes', '$', '900', 'billion', 'Covid', 'relief', 'deal', '—', 'Here', "'s", 'latest', 'News', 'Videos', 'Congressional', 'leaders', 'closed', '$', '900', 'billion', 'coronavirus', 'relief', 'deal', 'Wednesday', 'millions', 'struggling', 'Americans', 'wait', 'help', '.', 'The', 'developing', 'aid', 'agreement', 'would', 'include', 'liability', 'protections', 'businesses', 'aid', 'state', 'local', 'government', ',', 'CNBC', 'confirmed', '.', 'Disagreements', 'two', 'issues', 'blocked', 'lawmakers', 'crafting', 'year-end', 'rescue', 'package', '.', '``', 'We', 'made', 'major', 'headway', 'toward', 'hammering', 'targeted', 'pandemic', 'relief', 'package', 'would', 'able', 'pass', 'chambers', 'bipartisan', 'majorities', ',', "''", 'Senate', 'Majority', 'Leader', 'Mitch', 'McConnell', ',', 'R-Ky.', ',', 'said', 'Senate', 'floor', 'Wednesday', 'night', 'talks', 'among', 'top', 'four', 'congressional', 'leaders', '.', 'Speaking', 'McConnell', ',', 'Senate', 'Minority', 'Leader', 'Chuck', 'Schumer', ',', 'D-N.Y.', ',', 'said', '``', 'close', 'agreement', "''", 'noted', '``', "'s", 'done', 'deal', 'yet', '.', "''", 'He', 'added', 'Democrats', '``', 'would', 'like', 'gone', 'considerably', "''", 'offer', 'relief', 'press', 'aid', 'President-elect', 'Joe', 'Biden', 'takes', 'office', 'Jan.', '20', '.', 'The', 'measure', 'would', 'contain', 'direct', 'payment', 'Americans', 'amount', ',', 'well', 'enhanced', 'federal', 'unemployment', 'insurance', ',', 'NBC', 'News', 'reported', '.', 'In', 'addition', ',', 'Republican', 'Sen.', 'Steve', 'Daines', 'Montana', 'told', 'CNBC', 'deal', 'would', 'roughly', '$', '300', 'billion', 'small', 'business', 'aid', 'including', 'Paycheck', 'Protection', 'Program', 'loans', ',', 'money', 'Covid-19', 'vaccine', 'distribution', 'testing', 'relief', 'hospitals', '.', '``', 'I', "'m", 'cautiously', 'optimistic', "'re", 'going', 'see', '$', '900', 'billion', 'package', 'released', 'today', ',', 'likely', 'get', 'passed', 'go', 'home', 'weekend', ',', "''", 'Daines', 'told', 'CNBC', "'s", '``', 'Squawk', 'Box', "''", 'Wednesday', 'morning', '.', 'VIDEO', '3:21', '03:21', 'Sen.', 'Daines', ':', 'Covid', 'relief', 'deal', 'could', 'announced', 'Wednesday', 'morning', 'Squawk', 'Box', 'Congress', 'rushed', 'find', 'consensus', 'legislation', 'fund', 'government', 'rescue', 'health-care', 'system', 'economy', 'buckling', 'pandemic', '.', 'If', 'lawmakers', 'fail', 'act', ',', 'government', 'shut', 'Saturday', ',', '12', 'million', 'people', 'lose', 'unemployment', 'benefits', 'day', 'Christmas', 'millions', 'could', 'face', 'threat', 'eviction', '.', 'Congress', 'finally', 'neared', 'emergency', 'relief', 'agreement', 'Tuesday', 'night', 'negotiations', 'among', 'McConnell', ',', 'Schumer', ',', 'House', 'Speaker', 'Nancy', 'Pelosi', ',', 'D-Calif.', ',', 'House', 'Minority', 'Leader', 'Kevin', 'McCarthy', ',', 'R-Calif.', 'Republicans', 'Democrats', 'failed', 'months', 'make', 'progress', 'toward', 'bill', 'could', 'get', 'divided', 'Congress', '.', 'But', 'appeared', 'move', 'close', 'deal', 'talks', '.', 'After', 'discussions', ',', 'McConnell', 'Schumer', 'said', 'hoped', 'agreement', '``', 'soon', '.', "''", 'Politico', 'first', 'reported', 'congressional', 'leaders', 'near', '$', '900', 'billion', 'deal', '.', 'Parts', 'bill', 'appear', 'reflect', 'bipartisan', 'plan', 'released', 'rank-and-file', 'lawmakers', 'week', '.', 'However', ',', 'proposal', 'include', 'direct', 'payments', '.', 'The', 'measure', 'developed', 'congressional', 'leaders', 'contain', 'stimulus', 'checks', 'individuals', 'could', 'come', '$', '600', 'per', 'person', ',', 'according', 'NBC', '.', 'Congressional', 'progressives', 'urged', 'party', 'leaders', 'include', 'direct', 'payments', 'legislation', '.', 'Sens', '.', 'Bernie', 'Sanders', ',', 'I-Vt.', ',', 'Josh', 'Hawley', ',', 'R-Mo.', ',', 'also', 'threatened', 'delay', 'passage', 'bill', 'include', 'second', '$', '1,200', 'deposit', 'Americans', '.', 'At', 'stage', ',', 'Senate', 'would', 'likely', 'need', 'unanimous', 'support', 'pass', 'bill', 'quickly', 'enough', 'meet', 'midnight', 'Friday', 'deadline', '.', 'It', 'remains', 'seen', 'potentially', 'smaller', 'stimulus', 'check', ',', 'exclusion', 'state', 'local', 'government', 'relief', ',', 'would', 'affect', 'support', 'legislation', 'week', '.', 'As', 'agreement', 'developed', 'Wednesday', ',', 'Sanders', 'told', 'MSNBC', '$', '900', 'billion', '``', 'much', 'smaller', 'amount', 'country', 'needs', 'moment', 'economic', 'desperation', '.', "''", 'However', ',', 'called', '``', 'good', 'news', "''", 'understands', ',', 'bill', 'would', 'send', 'working-families', 'Americans', '$', '600', 'per', 'adult', '$', '600', 'per', 'child', '.', 'Hawley', ',', 'meanwhile', ',', 'told', 'reporters', 'direct', 'payment', 'provision', '``', 'progress', "''", '``', 'I', 'would', 'like', '.', "''", 'Lawmakers', 'send', 'help', 'soon', 'enough', 'millions', 'Americans', '.', 'The', 'economy', 'taken', 'hit', 'face', 'unchecked', 'coronavirus', 'outbreak', 'killed', '300,000', 'people', 'U.S.', 'As', 'millions', 'still', 'gained', 'back', 'jobs', 'lost', 'start', 'pandemic', ',', 'long', 'lines', 'formed', 'food', 'banks', 'around', 'country', '.', 'Many', 'Americans', 'remain', 'homes', 'due', 'eviction', 'moratoriums', 'lack', 'money', 'pay', 'rent', 'owe', '.', 'In', 'addition', ',', 'distribution', 'Covid', 'vaccinations', '—', 'started', 'week', 'gave', 'Americans', 'glimmer', 'hope', 'crisis', 'could', 'ease', 'coming', 'months', '—', 'rely', 'additional', 'federal', 'funding', '.', 'Of', 'course', ',', 'many', 'Washington', 'feel', '$', '900', 'billion', 'plan', 'go', 'nearly', 'far', 'enough', 'lift', 'families', 'merely', 'scraping', 'pandemic', '.', 'Biden', ',', 'like', 'Schumer', ',', 'signaled', 'Democrats', 'push', 'relief', 'new', 'year', '.', 'Speaking', 'event', 'introduced', 'Transportation', 'Secretary', 'nominee', 'Pete', 'Buttigieg', 'Wednesday', ',', 'Biden', 'called', 'developing', 'proposal', '``', 'encouraging', '.', "''", '``', 'But', "'s", 'payment', ',', 'important', 'payment', "'s", 'going', 'done', 'beginning', 'end', 'January', ',', 'beginning', 'February', ',', "'s", 'important', 'get', 'done', 'I', 'compliment', 'bipartisan', 'group', 'getting', 'done', ',', "''", 'said', '.', '—', 'CNBC', "'s", 'Ylan', 'Mui', 'Christina', 'Wilkie', 'contributed', 'report', 'Subscribe', 'CNBC', 'YouTube', '.', 'Related', 'Tags', 'Coronavirus', ':', 'Business', 'Charles', 'Schumer', 'Mitch', 'McConnell', 'Steve', 'Daines', 'White', 'House', 'Subscribe', 'CNBC', 'PRO', 'Licensing', '&', 'Reprints', 'CNBC', 'Councils', 'Supply', 'Chain', 'Values', 'CNBC', 'Peacock', 'Advertise', 'With', 'Us', 'Join', 'CNBC', 'Panel', 'Digital', 'Products', 'News', 'Releases', 'Closed', 'Captioning', 'Corrections', 'About', 'CNBC', 'Internships', 'Site', 'Map', 'Podcasts', 'Ad', 'Choices', 'Careers', 'Help', 'Contact', 'News', 'Tips', 'Got', 'confidential', 'news', 'tip', '?', 'We', 'want', 'hear', '.', 'Get', 'In', 'Touch', 'CNBC', 'Newsletters', 'Sign', 'free', 'newsletters', 'get', 'CNBC', 'delivered', 'inbox', 'Sign', 'Up', 'Now', 'Get', 'delivered', 'inbox', ',', 'info', 'products', 'services', '.', 'Privacy', 'Policy', '|', 'Do', 'Not', 'Sell', 'My', 'Personal', 'Information', '|', 'CA', 'Notice', '|', 'Terms', 'Service', '©', '2020', 'CNBC', 'LLC', '.', 'All', 'Rights', 'Reserved', '.', 'A', 'Division', 'NBCUniversal', 'Data', 'real-time', 'snapshot', '*Data', 'delayed', 'least', '15', 'minutes', '.', 'Global', 'Business', 'Financial', 'News', ',', 'Stock', 'Quotes', ',', 'Market', 'Data', 'Analysis', '.', 'Market', 'Data', 'Terms', 'Use', 'Disclaimers', 'Data', 'also', 'provided'] 
    
    

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\tramh\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
#List and count n-grams for any given input n
ngram_list = []
ngram_count = {}
def ngram_listAndCount(tokens,n_grams):
    # List of n_grams 
    for i in ngrams(tokens,n_grams):
        ngram_list.append(i)
    #Count of n_grams
    for i in ngram_list:
        if i in ngram_count:
            ngram_count[i] = ngram_count[i] + 1
        else:
            ngram_count[i] = 1
    return(ngram_count)             
```


```python
# Example of list and count n-grams for any given input n
ngram_listAndCount(filtered_words, 4)
```




    {('Skip', 'Navigation', 'Markets', 'Pre-Markets'): 1,
     ('Navigation', 'Markets', 'Pre-Markets', 'U.S.'): 1,
     ('Markets', 'Pre-Markets', 'U.S.', 'Markets'): 2,
     ('Pre-Markets', 'U.S.', 'Markets', 'Currencies'): 2,
     ('U.S.', 'Markets', 'Currencies', 'Cryptocurrency'): 2,
     ('Markets', 'Currencies', 'Cryptocurrency', 'Futures'): 2,
     ('Currencies', 'Cryptocurrency', 'Futures', '&'): 2,
     ('Cryptocurrency', 'Futures', '&', 'Commodities'): 2,
     ('Futures', '&', 'Commodities', 'Bonds'): 2,
     ('&', 'Commodities', 'Bonds', 'Funds'): 2,
     ('Commodities', 'Bonds', 'Funds', '&'): 2,
     ('Bonds', 'Funds', '&', 'ETFs'): 2,
     ('Funds', '&', 'ETFs', 'Business'): 2,
     ('&', 'ETFs', 'Business', 'Economy'): 2,
     ('ETFs', 'Business', 'Economy', 'Finance'): 2,
     ('Business', 'Economy', 'Finance', 'Health'): 2,
     ('Economy', 'Finance', 'Health', '&'): 2,
     ('Finance', 'Health', '&', 'Science'): 2,
     ('Health', '&', 'Science', 'Media'): 2,
     ('&', 'Science', 'Media', 'Real'): 2,
     ('Science', 'Media', 'Real', 'Estate'): 2,
     ('Media', 'Real', 'Estate', 'Energy'): 2,
     ('Real', 'Estate', 'Energy', 'Transportation'): 2,
     ('Estate', 'Energy', 'Transportation', 'Industrials'): 2,
     ('Energy', 'Transportation', 'Industrials', 'Retail'): 2,
     ('Transportation', 'Industrials', 'Retail', 'Wealth'): 2,
     ('Industrials', 'Retail', 'Wealth', 'Life'): 2,
     ('Retail', 'Wealth', 'Life', 'Small'): 2,
     ('Wealth', 'Life', 'Small', 'Business'): 2,
     ('Life', 'Small', 'Business', 'Investing'): 2,
     ('Small', 'Business', 'Investing', 'Invest'): 2,
     ('Business', 'Investing', 'Invest', 'In'): 2,
     ('Investing', 'Invest', 'In', 'You'): 2,
     ('Invest', 'In', 'You', 'Personal'): 2,
     ('In', 'You', 'Personal', 'Finance'): 2,
     ('You', 'Personal', 'Finance', 'Fintech'): 2,
     ('Personal', 'Finance', 'Fintech', 'Financial'): 2,
     ('Finance', 'Fintech', 'Financial', 'Advisors'): 2,
     ('Fintech', 'Financial', 'Advisors', 'Trading'): 2,
     ('Financial', 'Advisors', 'Trading', 'Nation'): 2,
     ('Advisors', 'Trading', 'Nation', 'Options'): 2,
     ('Trading', 'Nation', 'Options', 'Action'): 2,
     ('Nation', 'Options', 'Action', 'ETF'): 2,
     ('Options', 'Action', 'ETF', 'Street'): 2,
     ('Action', 'ETF', 'Street', 'Buffett'): 2,
     ('ETF', 'Street', 'Buffett', 'Archive'): 2,
     ('Street', 'Buffett', 'Archive', 'Earnings'): 2,
     ('Buffett', 'Archive', 'Earnings', 'Trader'): 2,
     ('Archive', 'Earnings', 'Trader', 'Talk'): 2,
     ('Earnings', 'Trader', 'Talk', 'Tech'): 2,
     ('Trader', 'Talk', 'Tech', 'Cybersecurity'): 2,
     ('Talk', 'Tech', 'Cybersecurity', 'Enterprise'): 2,
     ('Tech', 'Cybersecurity', 'Enterprise', 'Internet'): 2,
     ('Cybersecurity', 'Enterprise', 'Internet', 'Media'): 2,
     ('Enterprise', 'Internet', 'Media', 'Mobile'): 2,
     ('Internet', 'Media', 'Mobile', 'Social'): 2,
     ('Media', 'Mobile', 'Social', 'Media'): 2,
     ('Mobile', 'Social', 'Media', 'Venture'): 2,
     ('Social', 'Media', 'Venture', 'Capital'): 2,
     ('Media', 'Venture', 'Capital', 'Tech'): 2,
     ('Venture', 'Capital', 'Tech', 'Guide'): 2,
     ('Capital', 'Tech', 'Guide', 'Politics'): 2,
     ('Tech', 'Guide', 'Politics', 'White'): 2,
     ('Guide', 'Politics', 'White', 'House'): 2,
     ('Politics', 'White', 'House', 'Policy'): 2,
     ('White', 'House', 'Policy', 'Defense'): 2,
     ('House', 'Policy', 'Defense', 'Congress'): 2,
     ('Policy', 'Defense', 'Congress', '2020'): 2,
     ('Defense', 'Congress', '2020', 'Elections'): 2,
     ('Congress', '2020', 'Elections', 'CNBC'): 2,
     ('2020', 'Elections', 'CNBC', 'TV'): 2,
     ('Elections', 'CNBC', 'TV', 'Live'): 2,
     ('CNBC', 'TV', 'Live', 'TV'): 2,
     ('TV', 'Live', 'TV', 'Live'): 2,
     ('Live', 'TV', 'Live', 'Audio'): 2,
     ('TV', 'Live', 'Audio', 'Business'): 2,
     ('Live', 'Audio', 'Business', 'Day'): 2,
     ('Audio', 'Business', 'Day', 'Shows'): 2,
     ('Business', 'Day', 'Shows', 'The'): 2,
     ('Day', 'Shows', 'The', 'News'): 2,
     ('Shows', 'The', 'News', 'Shepard'): 2,
     ('The', 'News', 'Shepard', 'Smith'): 2,
     ('News', 'Shepard', 'Smith', 'Entertainment'): 2,
     ('Shepard', 'Smith', 'Entertainment', 'Shows'): 2,
     ('Smith', 'Entertainment', 'Shows', 'Full'): 2,
     ('Entertainment', 'Shows', 'Full', 'Episodes'): 2,
     ('Shows', 'Full', 'Episodes', 'Latest'): 2,
     ('Full', 'Episodes', 'Latest', 'Video'): 2,
     ('Episodes', 'Latest', 'Video', 'Top'): 2,
     ('Latest', 'Video', 'Top', 'Video'): 2,
     ('Video', 'Top', 'Video', 'CEO'): 2,
     ('Top', 'Video', 'CEO', 'Interviews'): 2,
     ('Video', 'CEO', 'Interviews', 'CNBC'): 2,
     ('CEO', 'Interviews', 'CNBC', 'Documentaries'): 2,
     ('Interviews', 'CNBC', 'Documentaries', 'CNBC'): 2,
     ('CNBC', 'Documentaries', 'CNBC', 'Podcasts'): 2,
     ('Documentaries', 'CNBC', 'Podcasts', 'CNBC'): 2,
     ('CNBC', 'Podcasts', 'CNBC', 'World'): 2,
     ('Podcasts', 'CNBC', 'World', 'Digital'): 2,
     ('CNBC', 'World', 'Digital', 'Originals'): 2,
     ('World', 'Digital', 'Originals', 'Live'): 2,
     ('Digital', 'Originals', 'Live', 'TV'): 2,
     ('Originals', 'Live', 'TV', 'Schedule'): 2,
     ('Live', 'TV', 'Schedule', 'Watchlist'): 2,
     ('TV', 'Schedule', 'Watchlist', 'PRO'): 2,
     ('Schedule', 'Watchlist', 'PRO', 'PRO'): 2,
     ('Watchlist', 'PRO', 'PRO', 'News'): 2,
     ('PRO', 'PRO', 'News', 'PRO'): 2,
     ('PRO', 'News', 'PRO', 'Live'): 2,
     ('News', 'PRO', 'Live', 'Subscribe'): 2,
     ('PRO', 'Live', 'Subscribe', 'Sign'): 2,
     ('Live', 'Subscribe', 'Sign', 'In'): 2,
     ('Subscribe', 'Sign', 'In', 'Menu'): 2,
     ('Sign', 'In', 'Menu', 'Make'): 1,
     ('In', 'Menu', 'Make', 'It'): 1,
     ('Menu', 'Make', 'It', 'Select'): 1,
     ('Make', 'It', 'Select', 'USA'): 1,
     ('It', 'Select', 'USA', 'INTL'): 1,
     ('Select', 'USA', 'INTL', 'Search'): 1,
     ('USA', 'INTL', 'Search', 'quotes'): 1,
     ('INTL', 'Search', 'quotes', ','): 1,
     ('Search', 'quotes', ',', 'news'): 1,
     ('quotes', ',', 'news', '&'): 1,
     (',', 'news', '&', 'videos'): 1,
     ('news', '&', 'videos', 'SIGN'): 1,
     ('&', 'videos', 'SIGN', 'IN'): 1,
     ('videos', 'SIGN', 'IN', 'Markets'): 1,
     ('SIGN', 'IN', 'Markets', 'Pre-Markets'): 1,
     ('IN', 'Markets', 'Pre-Markets', 'U.S.'): 1,
     ('Sign', 'In', 'Menu', 'Politics'): 1,
     ('In', 'Menu', 'Politics', 'Congress'): 1,
     ('Menu', 'Politics', 'Congress', 'closes'): 1,
     ('Politics', 'Congress', 'closes', '$'): 1,
     ('Congress', 'closes', '$', '900'): 2,
     ('closes', '$', '900', 'billion'): 2,
     ('$', '900', 'billion', 'Covid'): 2,
     ('900', 'billion', 'Covid', 'relief'): 2,
     ('billion', 'Covid', 'relief', 'deal'): 2,
     ('Covid', 'relief', 'deal', 'Americans'): 1,
     ('relief', 'deal', 'Americans', 'await'): 1,
     ('deal', 'Americans', 'await', 'aid'): 1,
     ('Americans', 'await', 'aid', 'Published'): 1,
     ('await', 'aid', 'Published', 'Wed'): 1,
     ('aid', 'Published', 'Wed', ','): 1,
     ('Published', 'Wed', ',', 'Dec'): 1,
     ('Wed', ',', 'Dec', '16'): 2,
     (',', 'Dec', '16', '2020'): 2,
     ('Dec', '16', '2020', '9:11'): 1,
     ('16', '2020', '9:11', 'AM'): 1,
     ('2020', '9:11', 'AM', 'EST'): 1,
     ('9:11', 'AM', 'EST', 'Updated'): 1,
     ('AM', 'EST', 'Updated', 'Wed'): 1,
     ('EST', 'Updated', 'Wed', ','): 1,
     ('Updated', 'Wed', ',', 'Dec'): 1,
     ('Dec', '16', '2020', '12:57'): 1,
     ('16', '2020', '12:57', 'PM'): 1,
     ('2020', '12:57', 'PM', 'EST'): 1,
     ('12:57', 'PM', 'EST', 'Jacob'): 1,
     ('PM', 'EST', 'Jacob', 'Pramuk'): 1,
     ('EST', 'Jacob', 'Pramuk', '@'): 1,
     ('Jacob', 'Pramuk', '@', 'jacobpramuk'): 1,
     ('Pramuk', '@', 'jacobpramuk', 'Key'): 1,
     ('@', 'jacobpramuk', 'Key', 'Points'): 1,
     ('jacobpramuk', 'Key', 'Points', 'Congress'): 1,
     ('Key', 'Points', 'Congress', 'getting'): 1,
     ('Points', 'Congress', 'getting', 'close'): 1,
     ('Congress', 'getting', 'close', '$'): 1,
     ('getting', 'close', '$', '900'): 1,
     ('close', '$', '900', 'billion'): 1,
     ('$', '900', 'billion', 'coronavirus'): 2,
     ('900', 'billion', 'coronavirus', 'relief'): 2,
     ('billion', 'coronavirus', 'relief', 'deal'): 2,
     ('coronavirus', 'relief', 'deal', 'could'): 1,
     ('relief', 'deal', 'could', 'announce'): 1,
     ('deal', 'could', 'announce', 'soon'): 1,
     ('could', 'announce', 'soon', 'Wednesday'): 1,
     ('announce', 'soon', 'Wednesday', '.'): 1,
     ('soon', 'Wednesday', '.', 'Top'): 1,
     ('Wednesday', '.', 'Top', 'Republicans'): 1,
     ('.', 'Top', 'Republicans', 'Democrats'): 1,
     ('Top', 'Republicans', 'Democrats', 'negotiated'): 1,
     ('Republicans', 'Democrats', 'negotiated', 'government'): 1,
     ('Democrats', 'negotiated', 'government', 'funding'): 1,
     ('negotiated', 'government', 'funding', 'pandemic'): 1,
     ('government', 'funding', 'pandemic', 'aid'): 1,
     ('funding', 'pandemic', 'aid', 'agreement'): 1,
     ('pandemic', 'aid', 'agreement', 'Tuesday'): 1,
     ('aid', 'agreement', 'Tuesday', 'night'): 1,
     ('agreement', 'Tuesday', 'night', ','): 1,
     ('Tuesday', 'night', ',', 'Senate'): 1,
     ('night', ',', 'Senate', 'leaders'): 1,
     (',', 'Senate', 'leaders', 'Mitch'): 1,
     ('Senate', 'leaders', 'Mitch', 'McConnell'): 1,
     ('leaders', 'Mitch', 'McConnell', 'Chuck'): 1,
     ('Mitch', 'McConnell', 'Chuck', 'Schumer'): 1,
     ('McConnell', 'Chuck', 'Schumer', 'said'): 1,
     ('Chuck', 'Schumer', 'said', 'hoped'): 1,
     ('Schumer', 'said', 'hoped', 'reach'): 1,
     ('said', 'hoped', 'reach', 'accord'): 1,
     ('hoped', 'reach', 'accord', '``'): 1,
     ('reach', 'accord', '``', 'soon'): 1,
     ('accord', '``', 'soon', '.'): 1,
     ('``', 'soon', '.', "''"): 2,
     ('soon', '.', "''", 'The'): 1,
     ('.', "''", 'The', 'government'): 1,
     ("''", 'The', 'government', 'shut'): 1,
     ('The', 'government', 'shut', 'Saturday'): 1,
     ('government', 'shut', 'Saturday', '12'): 1,
     ('shut', 'Saturday', '12', 'million'): 1,
     ('Saturday', '12', 'million', 'people'): 1,
     ('12', 'million', 'people', 'lose'): 2,
     ('million', 'people', 'lose', 'unemployment'): 2,
     ('people', 'lose', 'unemployment', 'benefits'): 2,
     ('lose', 'unemployment', 'benefits', 'day'): 2,
     ('unemployment', 'benefits', 'day', 'Christmas'): 2,
     ('benefits', 'day', 'Christmas', 'Congress'): 1,
     ('day', 'Christmas', 'Congress', 'fails'): 1,
     ('Christmas', 'Congress', 'fails', 'act'): 1,
     ('Congress', 'fails', 'act', '.'): 1,
     ('fails', 'act', '.', 'VIDEO'): 1,
     ('act', '.', 'VIDEO', '3:25'): 1,
     ('.', 'VIDEO', '3:25', '03:25'): 1,
     ('VIDEO', '3:25', '03:25', 'Congress'): 1,
     ('3:25', '03:25', 'Congress', 'closes'): 1,
     ('03:25', 'Congress', 'closes', '$'): 1,
     ('Covid', 'relief', 'deal', '—'): 1,
     ('relief', 'deal', '—', 'Here'): 1,
     ('deal', '—', 'Here', "'s"): 1,
     ('—', 'Here', "'s", 'latest'): 1,
     ('Here', "'s", 'latest', 'News'): 1,
     ("'s", 'latest', 'News', 'Videos'): 1,
     ('latest', 'News', 'Videos', 'Congressional'): 1,
     ('News', 'Videos', 'Congressional', 'leaders'): 1,
     ('Videos', 'Congressional', 'leaders', 'closed'): 1,
     ('Congressional', 'leaders', 'closed', '$'): 1,
     ('leaders', 'closed', '$', '900'): 1,
     ('closed', '$', '900', 'billion'): 1,
     ('coronavirus', 'relief', 'deal', 'Wednesday'): 1,
     ('relief', 'deal', 'Wednesday', 'millions'): 1,
     ('deal', 'Wednesday', 'millions', 'struggling'): 1,
     ('Wednesday', 'millions', 'struggling', 'Americans'): 1,
     ('millions', 'struggling', 'Americans', 'wait'): 1,
     ('struggling', 'Americans', 'wait', 'help'): 1,
     ('Americans', 'wait', 'help', '.'): 1,
     ('wait', 'help', '.', 'The'): 1,
     ('help', '.', 'The', 'developing'): 1,
     ('.', 'The', 'developing', 'aid'): 1,
     ('The', 'developing', 'aid', 'agreement'): 1,
     ('developing', 'aid', 'agreement', 'would'): 1,
     ('aid', 'agreement', 'would', 'include'): 1,
     ('agreement', 'would', 'include', 'liability'): 1,
     ('would', 'include', 'liability', 'protections'): 1,
     ('include', 'liability', 'protections', 'businesses'): 1,
     ('liability', 'protections', 'businesses', 'aid'): 1,
     ('protections', 'businesses', 'aid', 'state'): 1,
     ('businesses', 'aid', 'state', 'local'): 1,
     ('aid', 'state', 'local', 'government'): 1,
     ('state', 'local', 'government', ','): 1,
     ('local', 'government', ',', 'CNBC'): 1,
     ('government', ',', 'CNBC', 'confirmed'): 1,
     (',', 'CNBC', 'confirmed', '.'): 1,
     ('CNBC', 'confirmed', '.', 'Disagreements'): 1,
     ('confirmed', '.', 'Disagreements', 'two'): 1,
     ('.', 'Disagreements', 'two', 'issues'): 1,
     ('Disagreements', 'two', 'issues', 'blocked'): 1,
     ('two', 'issues', 'blocked', 'lawmakers'): 1,
     ('issues', 'blocked', 'lawmakers', 'crafting'): 1,
     ('blocked', 'lawmakers', 'crafting', 'year-end'): 1,
     ('lawmakers', 'crafting', 'year-end', 'rescue'): 1,
     ('crafting', 'year-end', 'rescue', 'package'): 1,
     ('year-end', 'rescue', 'package', '.'): 1,
     ('rescue', 'package', '.', '``'): 1,
     ('package', '.', '``', 'We'): 1,
     ('.', '``', 'We', 'made'): 1,
     ('``', 'We', 'made', 'major'): 1,
     ('We', 'made', 'major', 'headway'): 1,
     ('made', 'major', 'headway', 'toward'): 1,
     ('major', 'headway', 'toward', 'hammering'): 1,
     ('headway', 'toward', 'hammering', 'targeted'): 1,
     ('toward', 'hammering', 'targeted', 'pandemic'): 1,
     ('hammering', 'targeted', 'pandemic', 'relief'): 1,
     ('targeted', 'pandemic', 'relief', 'package'): 1,
     ('pandemic', 'relief', 'package', 'would'): 1,
     ('relief', 'package', 'would', 'able'): 1,
     ('package', 'would', 'able', 'pass'): 1,
     ('would', 'able', 'pass', 'chambers'): 1,
     ('able', 'pass', 'chambers', 'bipartisan'): 1,
     ('pass', 'chambers', 'bipartisan', 'majorities'): 1,
     ('chambers', 'bipartisan', 'majorities', ','): 1,
     ('bipartisan', 'majorities', ',', "''"): 1,
     ('majorities', ',', "''", 'Senate'): 1,
     (',', "''", 'Senate', 'Majority'): 1,
     ("''", 'Senate', 'Majority', 'Leader'): 1,
     ('Senate', 'Majority', 'Leader', 'Mitch'): 1,
     ('Majority', 'Leader', 'Mitch', 'McConnell'): 1,
     ('Leader', 'Mitch', 'McConnell', ','): 1,
     ('Mitch', 'McConnell', ',', 'R-Ky.'): 1,
     ('McConnell', ',', 'R-Ky.', ','): 1,
     (',', 'R-Ky.', ',', 'said'): 1,
     ('R-Ky.', ',', 'said', 'Senate'): 1,
     (',', 'said', 'Senate', 'floor'): 1,
     ('said', 'Senate', 'floor', 'Wednesday'): 1,
     ('Senate', 'floor', 'Wednesday', 'night'): 1,
     ('floor', 'Wednesday', 'night', 'talks'): 1,
     ('Wednesday', 'night', 'talks', 'among'): 1,
     ('night', 'talks', 'among', 'top'): 1,
     ('talks', 'among', 'top', 'four'): 1,
     ('among', 'top', 'four', 'congressional'): 1,
     ('top', 'four', 'congressional', 'leaders'): 1,
     ('four', 'congressional', 'leaders', '.'): 1,
     ('congressional', 'leaders', '.', 'Speaking'): 1,
     ('leaders', '.', 'Speaking', 'McConnell'): 1,
     ('.', 'Speaking', 'McConnell', ','): 1,
     ('Speaking', 'McConnell', ',', 'Senate'): 1,
     ('McConnell', ',', 'Senate', 'Minority'): 1,
     (',', 'Senate', 'Minority', 'Leader'): 1,
     ('Senate', 'Minority', 'Leader', 'Chuck'): 1,
     ('Minority', 'Leader', 'Chuck', 'Schumer'): 1,
     ('Leader', 'Chuck', 'Schumer', ','): 1,
     ('Chuck', 'Schumer', ',', 'D-N.Y.'): 1,
     ('Schumer', ',', 'D-N.Y.', ','): 1,
     (',', 'D-N.Y.', ',', 'said'): 1,
     ('D-N.Y.', ',', 'said', '``'): 1,
     (',', 'said', '``', 'close'): 1,
     ('said', '``', 'close', 'agreement'): 1,
     ('``', 'close', 'agreement', "''"): 1,
     ('close', 'agreement', "''", 'noted'): 1,
     ('agreement', "''", 'noted', '``'): 1,
     ("''", 'noted', '``', "'s"): 1,
     ('noted', '``', "'s", 'done'): 1,
     ('``', "'s", 'done', 'deal'): 1,
     ("'s", 'done', 'deal', 'yet'): 1,
     ('done', 'deal', 'yet', '.'): 1,
     ('deal', 'yet', '.', "''"): 1,
     ('yet', '.', "''", 'He'): 1,
     ('.', "''", 'He', 'added'): 1,
     ("''", 'He', 'added', 'Democrats'): 1,
     ('He', 'added', 'Democrats', '``'): 1,
     ('added', 'Democrats', '``', 'would'): 1,
     ('Democrats', '``', 'would', 'like'): 1,
     ('``', 'would', 'like', 'gone'): 1,
     ('would', 'like', 'gone', 'considerably'): 1,
     ('like', 'gone', 'considerably', "''"): 1,
     ('gone', 'considerably', "''", 'offer'): 1,
     ('considerably', "''", 'offer', 'relief'): 1,
     ("''", 'offer', 'relief', 'press'): 1,
     ('offer', 'relief', 'press', 'aid'): 1,
     ('relief', 'press', 'aid', 'President-elect'): 1,
     ('press', 'aid', 'President-elect', 'Joe'): 1,
     ('aid', 'President-elect', 'Joe', 'Biden'): 1,
     ('President-elect', 'Joe', 'Biden', 'takes'): 1,
     ('Joe', 'Biden', 'takes', 'office'): 1,
     ('Biden', 'takes', 'office', 'Jan.'): 1,
     ('takes', 'office', 'Jan.', '20'): 1,
     ('office', 'Jan.', '20', '.'): 1,
     ('Jan.', '20', '.', 'The'): 1,
     ('20', '.', 'The', 'measure'): 1,
     ('.', 'The', 'measure', 'would'): 1,
     ('The', 'measure', 'would', 'contain'): 1,
     ('measure', 'would', 'contain', 'direct'): 1,
     ('would', 'contain', 'direct', 'payment'): 1,
     ('contain', 'direct', 'payment', 'Americans'): 1,
     ('direct', 'payment', 'Americans', 'amount'): 1,
     ('payment', 'Americans', 'amount', ','): 1,
     ('Americans', 'amount', ',', 'well'): 1,
     ('amount', ',', 'well', 'enhanced'): 1,
     (',', 'well', 'enhanced', 'federal'): 1,
     ('well', 'enhanced', 'federal', 'unemployment'): 1,
     ('enhanced', 'federal', 'unemployment', 'insurance'): 1,
     ('federal', 'unemployment', 'insurance', ','): 1,
     ('unemployment', 'insurance', ',', 'NBC'): 1,
     ('insurance', ',', 'NBC', 'News'): 1,
     (',', 'NBC', 'News', 'reported'): 1,
     ('NBC', 'News', 'reported', '.'): 1,
     ('News', 'reported', '.', 'In'): 1,
     ('reported', '.', 'In', 'addition'): 1,
     ('.', 'In', 'addition', ','): 2,
     ('In', 'addition', ',', 'Republican'): 1,
     ('addition', ',', 'Republican', 'Sen.'): 1,
     (',', 'Republican', 'Sen.', 'Steve'): 1,
     ('Republican', 'Sen.', 'Steve', 'Daines'): 1,
     ('Sen.', 'Steve', 'Daines', 'Montana'): 1,
     ('Steve', 'Daines', 'Montana', 'told'): 1,
     ('Daines', 'Montana', 'told', 'CNBC'): 1,
     ('Montana', 'told', 'CNBC', 'deal'): 1,
     ('told', 'CNBC', 'deal', 'would'): 1,
     ('CNBC', 'deal', 'would', 'roughly'): 1,
     ('deal', 'would', 'roughly', '$'): 1,
     ('would', 'roughly', '$', '300'): 1,
     ('roughly', '$', '300', 'billion'): 1,
     ('$', '300', 'billion', 'small'): 1,
     ('300', 'billion', 'small', 'business'): 1,
     ('billion', 'small', 'business', 'aid'): 1,
     ('small', 'business', 'aid', 'including'): 1,
     ('business', 'aid', 'including', 'Paycheck'): 1,
     ('aid', 'including', 'Paycheck', 'Protection'): 1,
     ('including', 'Paycheck', 'Protection', 'Program'): 1,
     ('Paycheck', 'Protection', 'Program', 'loans'): 1,
     ('Protection', 'Program', 'loans', ','): 1,
     ('Program', 'loans', ',', 'money'): 1,
     ('loans', ',', 'money', 'Covid-19'): 1,
     (',', 'money', 'Covid-19', 'vaccine'): 1,
     ('money', 'Covid-19', 'vaccine', 'distribution'): 1,
     ('Covid-19', 'vaccine', 'distribution', 'testing'): 1,
     ('vaccine', 'distribution', 'testing', 'relief'): 1,
     ('distribution', 'testing', 'relief', 'hospitals'): 1,
     ('testing', 'relief', 'hospitals', '.'): 1,
     ('relief', 'hospitals', '.', '``'): 1,
     ('hospitals', '.', '``', 'I'): 1,
     ('.', '``', 'I', "'m"): 1,
     ('``', 'I', "'m", 'cautiously'): 1,
     ('I', "'m", 'cautiously', 'optimistic'): 1,
     ("'m", 'cautiously', 'optimistic', "'re"): 1,
     ('cautiously', 'optimistic', "'re", 'going'): 1,
     ('optimistic', "'re", 'going', 'see'): 1,
     ("'re", 'going', 'see', '$'): 1,
     ('going', 'see', '$', '900'): 1,
     ('see', '$', '900', 'billion'): 1,
     ('$', '900', 'billion', 'package'): 1,
     ('900', 'billion', 'package', 'released'): 1,
     ('billion', 'package', 'released', 'today'): 1,
     ('package', 'released', 'today', ','): 1,
     ('released', 'today', ',', 'likely'): 1,
     ('today', ',', 'likely', 'get'): 1,
     (',', 'likely', 'get', 'passed'): 1,
     ('likely', 'get', 'passed', 'go'): 1,
     ('get', 'passed', 'go', 'home'): 1,
     ('passed', 'go', 'home', 'weekend'): 1,
     ('go', 'home', 'weekend', ','): 1,
     ('home', 'weekend', ',', "''"): 1,
     ('weekend', ',', "''", 'Daines'): 1,
     (',', "''", 'Daines', 'told'): 1,
     ("''", 'Daines', 'told', 'CNBC'): 1,
     ('Daines', 'told', 'CNBC', "'s"): 1,
     ('told', 'CNBC', "'s", '``'): 1,
     ('CNBC', "'s", '``', 'Squawk'): 1,
     ("'s", '``', 'Squawk', 'Box'): 1,
     ('``', 'Squawk', 'Box', "''"): 1,
     ('Squawk', 'Box', "''", 'Wednesday'): 1,
     ('Box', "''", 'Wednesday', 'morning'): 1,
     ("''", 'Wednesday', 'morning', '.'): 1,
     ('Wednesday', 'morning', '.', 'VIDEO'): 1,
     ('morning', '.', 'VIDEO', '3:21'): 1,
     ('.', 'VIDEO', '3:21', '03:21'): 1,
     ('VIDEO', '3:21', '03:21', 'Sen.'): 1,
     ('3:21', '03:21', 'Sen.', 'Daines'): 1,
     ('03:21', 'Sen.', 'Daines', ':'): 1,
     ('Sen.', 'Daines', ':', 'Covid'): 1,
     ('Daines', ':', 'Covid', 'relief'): 1,
     (':', 'Covid', 'relief', 'deal'): 1,
     ('Covid', 'relief', 'deal', 'could'): 1,
     ('relief', 'deal', 'could', 'announced'): 1,
     ('deal', 'could', 'announced', 'Wednesday'): 1,
     ('could', 'announced', 'Wednesday', 'morning'): 1,
     ('announced', 'Wednesday', 'morning', 'Squawk'): 1,
     ('Wednesday', 'morning', 'Squawk', 'Box'): 1,
     ('morning', 'Squawk', 'Box', 'Congress'): 1,
     ('Squawk', 'Box', 'Congress', 'rushed'): 1,
     ('Box', 'Congress', 'rushed', 'find'): 1,
     ('Congress', 'rushed', 'find', 'consensus'): 1,
     ('rushed', 'find', 'consensus', 'legislation'): 1,
     ('find', 'consensus', 'legislation', 'fund'): 1,
     ('consensus', 'legislation', 'fund', 'government'): 1,
     ('legislation', 'fund', 'government', 'rescue'): 1,
     ('fund', 'government', 'rescue', 'health-care'): 1,
     ('government', 'rescue', 'health-care', 'system'): 1,
     ('rescue', 'health-care', 'system', 'economy'): 1,
     ('health-care', 'system', 'economy', 'buckling'): 1,
     ('system', 'economy', 'buckling', 'pandemic'): 1,
     ('economy', 'buckling', 'pandemic', '.'): 1,
     ('buckling', 'pandemic', '.', 'If'): 1,
     ('pandemic', '.', 'If', 'lawmakers'): 1,
     ('.', 'If', 'lawmakers', 'fail'): 1,
     ('If', 'lawmakers', 'fail', 'act'): 1,
     ('lawmakers', 'fail', 'act', ','): 1,
     ('fail', 'act', ',', 'government'): 1,
     ('act', ',', 'government', 'shut'): 1,
     (',', 'government', 'shut', 'Saturday'): 1,
     ('government', 'shut', 'Saturday', ','): 1,
     ('shut', 'Saturday', ',', '12'): 1,
     ('Saturday', ',', '12', 'million'): 1,
     (',', '12', 'million', 'people'): 1,
     ('benefits', 'day', 'Christmas', 'millions'): 1,
     ('day', 'Christmas', 'millions', 'could'): 1,
     ('Christmas', 'millions', 'could', 'face'): 1,
     ('millions', 'could', 'face', 'threat'): 1,
     ('could', 'face', 'threat', 'eviction'): 1,
     ('face', 'threat', 'eviction', '.'): 1,
     ('threat', 'eviction', '.', 'Congress'): 1,
     ('eviction', '.', 'Congress', 'finally'): 1,
     ('.', 'Congress', 'finally', 'neared'): 1,
     ('Congress', 'finally', 'neared', 'emergency'): 1,
     ('finally', 'neared', 'emergency', 'relief'): 1,
     ('neared', 'emergency', 'relief', 'agreement'): 1,
     ('emergency', 'relief', 'agreement', 'Tuesday'): 1,
     ('relief', 'agreement', 'Tuesday', 'night'): 1,
     ('agreement', 'Tuesday', 'night', 'negotiations'): 1,
     ('Tuesday', 'night', 'negotiations', 'among'): 1,
     ('night', 'negotiations', 'among', 'McConnell'): 1,
     ('negotiations', 'among', 'McConnell', ','): 1,
     ('among', 'McConnell', ',', 'Schumer'): 1,
     ('McConnell', ',', 'Schumer', ','): 1,
     (',', 'Schumer', ',', 'House'): 1,
     ('Schumer', ',', 'House', 'Speaker'): 1,
     (',', 'House', 'Speaker', 'Nancy'): 1,
     ('House', 'Speaker', 'Nancy', 'Pelosi'): 1,
     ('Speaker', 'Nancy', 'Pelosi', ','): 1,
     ('Nancy', 'Pelosi', ',', 'D-Calif.'): 1,
     ('Pelosi', ',', 'D-Calif.', ','): 1,
     (',', 'D-Calif.', ',', 'House'): 1,
     ('D-Calif.', ',', 'House', 'Minority'): 1,
     (',', 'House', 'Minority', 'Leader'): 1,
     ('House', 'Minority', 'Leader', 'Kevin'): 1,
     ('Minority', 'Leader', 'Kevin', 'McCarthy'): 1,
     ('Leader', 'Kevin', 'McCarthy', ','): 1,
     ('Kevin', 'McCarthy', ',', 'R-Calif.'): 1,
     ('McCarthy', ',', 'R-Calif.', 'Republicans'): 1,
     (',', 'R-Calif.', 'Republicans', 'Democrats'): 1,
     ('R-Calif.', 'Republicans', 'Democrats', 'failed'): 1,
     ('Republicans', 'Democrats', 'failed', 'months'): 1,
     ('Democrats', 'failed', 'months', 'make'): 1,
     ('failed', 'months', 'make', 'progress'): 1,
     ('months', 'make', 'progress', 'toward'): 1,
     ('make', 'progress', 'toward', 'bill'): 1,
     ('progress', 'toward', 'bill', 'could'): 1,
     ('toward', 'bill', 'could', 'get'): 1,
     ('bill', 'could', 'get', 'divided'): 1,
     ('could', 'get', 'divided', 'Congress'): 1,
     ('get', 'divided', 'Congress', '.'): 1,
     ('divided', 'Congress', '.', 'But'): 1,
     ('Congress', '.', 'But', 'appeared'): 1,
     ('.', 'But', 'appeared', 'move'): 1,
     ('But', 'appeared', 'move', 'close'): 1,
     ('appeared', 'move', 'close', 'deal'): 1,
     ('move', 'close', 'deal', 'talks'): 1,
     ('close', 'deal', 'talks', '.'): 1,
     ('deal', 'talks', '.', 'After'): 1,
     ('talks', '.', 'After', 'discussions'): 1,
     ('.', 'After', 'discussions', ','): 1,
     ('After', 'discussions', ',', 'McConnell'): 1,
     ('discussions', ',', 'McConnell', 'Schumer'): 1,
     (',', 'McConnell', 'Schumer', 'said'): 1,
     ('McConnell', 'Schumer', 'said', 'hoped'): 1,
     ('Schumer', 'said', 'hoped', 'agreement'): 1,
     ('said', 'hoped', 'agreement', '``'): 1,
     ('hoped', 'agreement', '``', 'soon'): 1,
     ('agreement', '``', 'soon', '.'): 1,
     ('soon', '.', "''", 'Politico'): 1,
     ('.', "''", 'Politico', 'first'): 1,
     ("''", 'Politico', 'first', 'reported'): 1,
     ('Politico', 'first', 'reported', 'congressional'): 1,
     ('first', 'reported', 'congressional', 'leaders'): 1,
     ('reported', 'congressional', 'leaders', 'near'): 1,
     ('congressional', 'leaders', 'near', '$'): 1,
     ('leaders', 'near', '$', '900'): 1,
     ('near', '$', '900', 'billion'): 1,
     ('$', '900', 'billion', 'deal'): 1,
     ('900', 'billion', 'deal', '.'): 1,
     ('billion', 'deal', '.', 'Parts'): 1,
     ('deal', '.', 'Parts', 'bill'): 1,
     ('.', 'Parts', 'bill', 'appear'): 1,
     ('Parts', 'bill', 'appear', 'reflect'): 1,
     ('bill', 'appear', 'reflect', 'bipartisan'): 1,
     ('appear', 'reflect', 'bipartisan', 'plan'): 1,
     ('reflect', 'bipartisan', 'plan', 'released'): 1,
     ('bipartisan', 'plan', 'released', 'rank-and-file'): 1,
     ('plan', 'released', 'rank-and-file', 'lawmakers'): 1,
     ('released', 'rank-and-file', 'lawmakers', 'week'): 1,
     ('rank-and-file', 'lawmakers', 'week', '.'): 1,
     ('lawmakers', 'week', '.', 'However'): 1,
     ('week', '.', 'However', ','): 1,
     ('.', 'However', ',', 'proposal'): 1,
     ('However', ',', 'proposal', 'include'): 1,
     (',', 'proposal', 'include', 'direct'): 1,
     ('proposal', 'include', 'direct', 'payments'): 1,
     ('include', 'direct', 'payments', '.'): 1,
     ('direct', 'payments', '.', 'The'): 1,
     ('payments', '.', 'The', 'measure'): 1,
     ('.', 'The', 'measure', 'developed'): 1,
     ('The', 'measure', 'developed', 'congressional'): 1,
     ('measure', 'developed', 'congressional', 'leaders'): 1,
     ('developed', 'congressional', 'leaders', 'contain'): 1,
     ('congressional', 'leaders', 'contain', 'stimulus'): 1,
     ('leaders', 'contain', 'stimulus', 'checks'): 1,
     ('contain', 'stimulus', 'checks', 'individuals'): 1,
     ('stimulus', 'checks', 'individuals', 'could'): 1,
     ('checks', 'individuals', 'could', 'come'): 1,
     ('individuals', 'could', 'come', '$'): 1,
     ('could', 'come', '$', '600'): 1,
     ('come', '$', '600', 'per'): 1,
     ('$', '600', 'per', 'person'): 1,
     ('600', 'per', 'person', ','): 1,
     ('per', 'person', ',', 'according'): 1,
     ('person', ',', 'according', 'NBC'): 1,
     (',', 'according', 'NBC', '.'): 1,
     ('according', 'NBC', '.', 'Congressional'): 1,
     ('NBC', '.', 'Congressional', 'progressives'): 1,
     ('.', 'Congressional', 'progressives', 'urged'): 1,
     ('Congressional', 'progressives', 'urged', 'party'): 1,
     ('progressives', 'urged', 'party', 'leaders'): 1,
     ('urged', 'party', 'leaders', 'include'): 1,
     ('party', 'leaders', 'include', 'direct'): 1,
     ('leaders', 'include', 'direct', 'payments'): 1,
     ('include', 'direct', 'payments', 'legislation'): 1,
     ('direct', 'payments', 'legislation', '.'): 1,
     ('payments', 'legislation', '.', 'Sens'): 1,
     ('legislation', '.', 'Sens', '.'): 1,
     ('.', 'Sens', '.', 'Bernie'): 1,
     ('Sens', '.', 'Bernie', 'Sanders'): 1,
     ('.', 'Bernie', 'Sanders', ','): 1,
     ('Bernie', 'Sanders', ',', 'I-Vt.'): 1,
     ('Sanders', ',', 'I-Vt.', ','): 1,
     (',', 'I-Vt.', ',', 'Josh'): 1,
     ('I-Vt.', ',', 'Josh', 'Hawley'): 1,
     (',', 'Josh', 'Hawley', ','): 1,
     ('Josh', 'Hawley', ',', 'R-Mo.'): 1,
     ('Hawley', ',', 'R-Mo.', ','): 1,
     (',', 'R-Mo.', ',', 'also'): 1,
     ('R-Mo.', ',', 'also', 'threatened'): 1,
     (',', 'also', 'threatened', 'delay'): 1,
     ('also', 'threatened', 'delay', 'passage'): 1,
     ('threatened', 'delay', 'passage', 'bill'): 1,
     ('delay', 'passage', 'bill', 'include'): 1,
     ('passage', 'bill', 'include', 'second'): 1,
     ('bill', 'include', 'second', '$'): 1,
     ('include', 'second', '$', '1,200'): 1,
     ('second', '$', '1,200', 'deposit'): 1,
     ('$', '1,200', 'deposit', 'Americans'): 1,
     ('1,200', 'deposit', 'Americans', '.'): 1,
     ('deposit', 'Americans', '.', 'At'): 1,
     ('Americans', '.', 'At', 'stage'): 1,
     ('.', 'At', 'stage', ','): 1,
     ('At', 'stage', ',', 'Senate'): 1,
     ('stage', ',', 'Senate', 'would'): 1,
     (',', 'Senate', 'would', 'likely'): 1,
     ('Senate', 'would', 'likely', 'need'): 1,
     ('would', 'likely', 'need', 'unanimous'): 1,
     ('likely', 'need', 'unanimous', 'support'): 1,
     ('need', 'unanimous', 'support', 'pass'): 1,
     ('unanimous', 'support', 'pass', 'bill'): 1,
     ('support', 'pass', 'bill', 'quickly'): 1,
     ('pass', 'bill', 'quickly', 'enough'): 1,
     ('bill', 'quickly', 'enough', 'meet'): 1,
     ('quickly', 'enough', 'meet', 'midnight'): 1,
     ('enough', 'meet', 'midnight', 'Friday'): 1,
     ('meet', 'midnight', 'Friday', 'deadline'): 1,
     ('midnight', 'Friday', 'deadline', '.'): 1,
     ('Friday', 'deadline', '.', 'It'): 1,
     ('deadline', '.', 'It', 'remains'): 1,
     ('.', 'It', 'remains', 'seen'): 1,
     ('It', 'remains', 'seen', 'potentially'): 1,
     ('remains', 'seen', 'potentially', 'smaller'): 1,
     ('seen', 'potentially', 'smaller', 'stimulus'): 1,
     ('potentially', 'smaller', 'stimulus', 'check'): 1,
     ('smaller', 'stimulus', 'check', ','): 1,
     ('stimulus', 'check', ',', 'exclusion'): 1,
     ('check', ',', 'exclusion', 'state'): 1,
     (',', 'exclusion', 'state', 'local'): 1,
     ('exclusion', 'state', 'local', 'government'): 1,
     ('state', 'local', 'government', 'relief'): 1,
     ('local', 'government', 'relief', ','): 1,
     ('government', 'relief', ',', 'would'): 1,
     ('relief', ',', 'would', 'affect'): 1,
     (',', 'would', 'affect', 'support'): 1,
     ('would', 'affect', 'support', 'legislation'): 1,
     ('affect', 'support', 'legislation', 'week'): 1,
     ('support', 'legislation', 'week', '.'): 1,
     ('legislation', 'week', '.', 'As'): 1,
     ('week', '.', 'As', 'agreement'): 1,
     ('.', 'As', 'agreement', 'developed'): 1,
     ('As', 'agreement', 'developed', 'Wednesday'): 1,
     ('agreement', 'developed', 'Wednesday', ','): 1,
     ('developed', 'Wednesday', ',', 'Sanders'): 1,
     ('Wednesday', ',', 'Sanders', 'told'): 1,
     (',', 'Sanders', 'told', 'MSNBC'): 1,
     ('Sanders', 'told', 'MSNBC', '$'): 1,
     ('told', 'MSNBC', '$', '900'): 1,
     ('MSNBC', '$', '900', 'billion'): 1,
     ('$', '900', 'billion', '``'): 1,
     ('900', 'billion', '``', 'much'): 1,
     ('billion', '``', 'much', 'smaller'): 1,
     ('``', 'much', 'smaller', 'amount'): 1,
     ('much', 'smaller', 'amount', 'country'): 1,
     ('smaller', 'amount', 'country', 'needs'): 1,
     ('amount', 'country', 'needs', 'moment'): 1,
     ('country', 'needs', 'moment', 'economic'): 1,
     ('needs', 'moment', 'economic', 'desperation'): 1,
     ('moment', 'economic', 'desperation', '.'): 1,
     ('economic', 'desperation', '.', "''"): 1,
     ('desperation', '.', "''", 'However'): 1,
     ('.', "''", 'However', ','): 1,
     ("''", 'However', ',', 'called'): 1,
     ('However', ',', 'called', '``'): 1,
     (',', 'called', '``', 'good'): 1,
     ('called', '``', 'good', 'news'): 1,
     ('``', 'good', 'news', "''"): 1,
     ('good', 'news', "''", 'understands'): 1,
     ('news', "''", 'understands', ','): 1,
     ("''", 'understands', ',', 'bill'): 1,
     ('understands', ',', 'bill', 'would'): 1,
     (',', 'bill', 'would', 'send'): 1,
     ('bill', 'would', 'send', 'working-families'): 1,
     ('would', 'send', 'working-families', 'Americans'): 1,
     ('send', 'working-families', 'Americans', '$'): 1,
     ('working-families', 'Americans', '$', '600'): 1,
     ('Americans', '$', '600', 'per'): 1,
     ('$', '600', 'per', 'adult'): 1,
     ('600', 'per', 'adult', '$'): 1,
     ('per', 'adult', '$', '600'): 1,
     ('adult', '$', '600', 'per'): 1,
     ('$', '600', 'per', 'child'): 1,
     ('600', 'per', 'child', '.'): 1,
     ('per', 'child', '.', 'Hawley'): 1,
     ('child', '.', 'Hawley', ','): 1,
     ('.', 'Hawley', ',', 'meanwhile'): 1,
     ('Hawley', ',', 'meanwhile', ','): 1,
     (',', 'meanwhile', ',', 'told'): 1,
     ('meanwhile', ',', 'told', 'reporters'): 1,
     (',', 'told', 'reporters', 'direct'): 1,
     ('told', 'reporters', 'direct', 'payment'): 1,
     ('reporters', 'direct', 'payment', 'provision'): 1,
     ('direct', 'payment', 'provision', '``'): 1,
     ('payment', 'provision', '``', 'progress'): 1,
     ('provision', '``', 'progress', "''"): 1,
     ('``', 'progress', "''", '``'): 1,
     ('progress', "''", '``', 'I'): 1,
     ("''", '``', 'I', 'would'): 1,
     ('``', 'I', 'would', 'like'): 1,
     ('I', 'would', 'like', '.'): 1,
     ('would', 'like', '.', "''"): 1,
     ('like', '.', "''", 'Lawmakers'): 1,
     ('.', "''", 'Lawmakers', 'send'): 1,
     ("''", 'Lawmakers', 'send', 'help'): 1,
     ('Lawmakers', 'send', 'help', 'soon'): 1,
     ('send', 'help', 'soon', 'enough'): 1,
     ('help', 'soon', 'enough', 'millions'): 1,
     ('soon', 'enough', 'millions', 'Americans'): 1,
     ('enough', 'millions', 'Americans', '.'): 1,
     ('millions', 'Americans', '.', 'The'): 1,
     ('Americans', '.', 'The', 'economy'): 1,
     ('.', 'The', 'economy', 'taken'): 1,
     ('The', 'economy', 'taken', 'hit'): 1,
     ('economy', 'taken', 'hit', 'face'): 1,
     ('taken', 'hit', 'face', 'unchecked'): 1,
     ('hit', 'face', 'unchecked', 'coronavirus'): 1,
     ('face', 'unchecked', 'coronavirus', 'outbreak'): 1,
     ('unchecked', 'coronavirus', 'outbreak', 'killed'): 1,
     ('coronavirus', 'outbreak', 'killed', '300,000'): 1,
     ('outbreak', 'killed', '300,000', 'people'): 1,
     ('killed', '300,000', 'people', 'U.S.'): 1,
     ('300,000', 'people', 'U.S.', 'As'): 1,
     ('people', 'U.S.', 'As', 'millions'): 1,
     ('U.S.', 'As', 'millions', 'still'): 1,
     ('As', 'millions', 'still', 'gained'): 1,
     ('millions', 'still', 'gained', 'back'): 1,
     ('still', 'gained', 'back', 'jobs'): 1,
     ('gained', 'back', 'jobs', 'lost'): 1,
     ('back', 'jobs', 'lost', 'start'): 1,
     ('jobs', 'lost', 'start', 'pandemic'): 1,
     ('lost', 'start', 'pandemic', ','): 1,
     ('start', 'pandemic', ',', 'long'): 1,
     ('pandemic', ',', 'long', 'lines'): 1,
     (',', 'long', 'lines', 'formed'): 1,
     ('long', 'lines', 'formed', 'food'): 1,
     ('lines', 'formed', 'food', 'banks'): 1,
     ('formed', 'food', 'banks', 'around'): 1,
     ('food', 'banks', 'around', 'country'): 1,
     ('banks', 'around', 'country', '.'): 1,
     ('around', 'country', '.', 'Many'): 1,
     ('country', '.', 'Many', 'Americans'): 1,
     ('.', 'Many', 'Americans', 'remain'): 1,
     ('Many', 'Americans', 'remain', 'homes'): 1,
     ('Americans', 'remain', 'homes', 'due'): 1,
     ('remain', 'homes', 'due', 'eviction'): 1,
     ('homes', 'due', 'eviction', 'moratoriums'): 1,
     ('due', 'eviction', 'moratoriums', 'lack'): 1,
     ('eviction', 'moratoriums', 'lack', 'money'): 1,
     ('moratoriums', 'lack', 'money', 'pay'): 1,
     ('lack', 'money', 'pay', 'rent'): 1,
     ('money', 'pay', 'rent', 'owe'): 1,
     ('pay', 'rent', 'owe', '.'): 1,
     ('rent', 'owe', '.', 'In'): 1,
     ('owe', '.', 'In', 'addition'): 1,
     ('In', 'addition', ',', 'distribution'): 1,
     ('addition', ',', 'distribution', 'Covid'): 1,
     (',', 'distribution', 'Covid', 'vaccinations'): 1,
     ('distribution', 'Covid', 'vaccinations', '—'): 1,
     ('Covid', 'vaccinations', '—', 'started'): 1,
     ('vaccinations', '—', 'started', 'week'): 1,
     ('—', 'started', 'week', 'gave'): 1,
     ('started', 'week', 'gave', 'Americans'): 1,
     ('week', 'gave', 'Americans', 'glimmer'): 1,
     ('gave', 'Americans', 'glimmer', 'hope'): 1,
     ('Americans', 'glimmer', 'hope', 'crisis'): 1,
     ('glimmer', 'hope', 'crisis', 'could'): 1,
     ('hope', 'crisis', 'could', 'ease'): 1,
     ('crisis', 'could', 'ease', 'coming'): 1,
     ('could', 'ease', 'coming', 'months'): 1,
     ('ease', 'coming', 'months', '—'): 1,
     ('coming', 'months', '—', 'rely'): 1,
     ('months', '—', 'rely', 'additional'): 1,
     ('—', 'rely', 'additional', 'federal'): 1,
     ('rely', 'additional', 'federal', 'funding'): 1,
     ('additional', 'federal', 'funding', '.'): 1,
     ('federal', 'funding', '.', 'Of'): 1,
     ('funding', '.', 'Of', 'course'): 1,
     ('.', 'Of', 'course', ','): 1,
     ('Of', 'course', ',', 'many'): 1,
     ('course', ',', 'many', 'Washington'): 1,
     (',', 'many', 'Washington', 'feel'): 1,
     ('many', 'Washington', 'feel', '$'): 1,
     ('Washington', 'feel', '$', '900'): 1,
     ('feel', '$', '900', 'billion'): 1,
     ('$', '900', 'billion', 'plan'): 1,
     ('900', 'billion', 'plan', 'go'): 1,
     ('billion', 'plan', 'go', 'nearly'): 1,
     ('plan', 'go', 'nearly', 'far'): 1,
     ('go', 'nearly', 'far', 'enough'): 1,
     ('nearly', 'far', 'enough', 'lift'): 1,
     ('far', 'enough', 'lift', 'families'): 1,
     ('enough', 'lift', 'families', 'merely'): 1,
     ('lift', 'families', 'merely', 'scraping'): 1,
     ('families', 'merely', 'scraping', 'pandemic'): 1,
     ('merely', 'scraping', 'pandemic', '.'): 1,
     ('scraping', 'pandemic', '.', 'Biden'): 1,
     ('pandemic', '.', 'Biden', ','): 1,
     ('.', 'Biden', ',', 'like'): 1,
     ('Biden', ',', 'like', 'Schumer'): 1,
     (',', 'like', 'Schumer', ','): 1,
     ('like', 'Schumer', ',', 'signaled'): 1,
     ('Schumer', ',', 'signaled', 'Democrats'): 1,
     (',', 'signaled', 'Democrats', 'push'): 1,
     ('signaled', 'Democrats', 'push', 'relief'): 1,
     ('Democrats', 'push', 'relief', 'new'): 1,
     ('push', 'relief', 'new', 'year'): 1,
     ('relief', 'new', 'year', '.'): 1,
     ('new', 'year', '.', 'Speaking'): 1,
     ('year', '.', 'Speaking', 'event'): 1,
     ('.', 'Speaking', 'event', 'introduced'): 1,
     ('Speaking', 'event', 'introduced', 'Transportation'): 1,
     ('event', 'introduced', 'Transportation', 'Secretary'): 1,
     ('introduced', 'Transportation', 'Secretary', 'nominee'): 1,
     ('Transportation', 'Secretary', 'nominee', 'Pete'): 1,
     ('Secretary', 'nominee', 'Pete', 'Buttigieg'): 1,
     ('nominee', 'Pete', 'Buttigieg', 'Wednesday'): 1,
     ('Pete', 'Buttigieg', 'Wednesday', ','): 1,
     ('Buttigieg', 'Wednesday', ',', 'Biden'): 1,
     ('Wednesday', ',', 'Biden', 'called'): 1,
     (',', 'Biden', 'called', 'developing'): 1,
     ('Biden', 'called', 'developing', 'proposal'): 1,
     ('called', 'developing', 'proposal', '``'): 1,
     ('developing', 'proposal', '``', 'encouraging'): 1,
     ('proposal', '``', 'encouraging', '.'): 1,
     ('``', 'encouraging', '.', "''"): 1,
     ('encouraging', '.', "''", '``'): 1,
     ('.', "''", '``', 'But'): 1,
     ("''", '``', 'But', "'s"): 1,
     ('``', 'But', "'s", 'payment'): 1,
     ('But', "'s", 'payment', ','): 1,
     ("'s", 'payment', ',', 'important'): 1,
     ('payment', ',', 'important', 'payment'): 1,
     (',', 'important', 'payment', "'s"): 1,
     ('important', 'payment', "'s", 'going'): 1,
     ('payment', "'s", 'going', 'done'): 1,
     ("'s", 'going', 'done', 'beginning'): 1,
     ('going', 'done', 'beginning', 'end'): 1,
     ('done', 'beginning', 'end', 'January'): 1,
     ('beginning', 'end', 'January', ','): 1,
     ('end', 'January', ',', 'beginning'): 1,
     ('January', ',', 'beginning', 'February'): 1,
     (',', 'beginning', 'February', ','): 1,
     ('beginning', 'February', ',', "'s"): 1,
     ('February', ',', "'s", 'important'): 1,
     (',', "'s", 'important', 'get'): 1,
     ("'s", 'important', 'get', 'done'): 1,
     ('important', 'get', 'done', 'I'): 1,
     ('get', 'done', 'I', 'compliment'): 1,
     ('done', 'I', 'compliment', 'bipartisan'): 1,
     ('I', 'compliment', 'bipartisan', 'group'): 1,
     ('compliment', 'bipartisan', 'group', 'getting'): 1,
     ('bipartisan', 'group', 'getting', 'done'): 1,
     ('group', 'getting', 'done', ','): 1,
     ('getting', 'done', ',', "''"): 1,
     ('done', ',', "''", 'said'): 1,
     (',', "''", 'said', '.'): 1,
     ("''", 'said', '.', '—'): 1,
     ('said', '.', '—', 'CNBC'): 1,
     ('.', '—', 'CNBC', "'s"): 1,
     ('—', 'CNBC', "'s", 'Ylan'): 1,
     ('CNBC', "'s", 'Ylan', 'Mui'): 1,
     ("'s", 'Ylan', 'Mui', 'Christina'): 1,
     ('Ylan', 'Mui', 'Christina', 'Wilkie'): 1,
     ('Mui', 'Christina', 'Wilkie', 'contributed'): 1,
     ('Christina', 'Wilkie', 'contributed', 'report'): 1,
     ('Wilkie', 'contributed', 'report', 'Subscribe'): 1,
     ('contributed', 'report', 'Subscribe', 'CNBC'): 1,
     ('report', 'Subscribe', 'CNBC', 'YouTube'): 1,
     ('Subscribe', 'CNBC', 'YouTube', '.'): 1,
     ('CNBC', 'YouTube', '.', 'Related'): 1,
     ('YouTube', '.', 'Related', 'Tags'): 1,
     ('.', 'Related', 'Tags', 'Coronavirus'): 1,
     ('Related', 'Tags', 'Coronavirus', ':'): 1,
     ('Tags', 'Coronavirus', ':', 'Business'): 1,
     ('Coronavirus', ':', 'Business', 'Charles'): 1,
     (':', 'Business', 'Charles', 'Schumer'): 1,
     ('Business', 'Charles', 'Schumer', 'Mitch'): 1,
     ('Charles', 'Schumer', 'Mitch', 'McConnell'): 1,
     ('Schumer', 'Mitch', 'McConnell', 'Steve'): 1,
     ('Mitch', 'McConnell', 'Steve', 'Daines'): 1,
     ('McConnell', 'Steve', 'Daines', 'White'): 1,
     ('Steve', 'Daines', 'White', 'House'): 1,
     ('Daines', 'White', 'House', 'Subscribe'): 1,
     ('White', 'House', 'Subscribe', 'CNBC'): 1,
     ('House', 'Subscribe', 'CNBC', 'PRO'): 1,
     ('Subscribe', 'CNBC', 'PRO', 'Licensing'): 1,
     ('CNBC', 'PRO', 'Licensing', '&'): 1,
     ('PRO', 'Licensing', '&', 'Reprints'): 1,
     ('Licensing', '&', 'Reprints', 'CNBC'): 1,
     ('&', 'Reprints', 'CNBC', 'Councils'): 1,
     ('Reprints', 'CNBC', 'Councils', 'Supply'): 1,
     ('CNBC', 'Councils', 'Supply', 'Chain'): 1,
     ('Councils', 'Supply', 'Chain', 'Values'): 1,
     ('Supply', 'Chain', 'Values', 'CNBC'): 1,
     ('Chain', 'Values', 'CNBC', 'Peacock'): 1,
     ('Values', 'CNBC', 'Peacock', 'Advertise'): 1,
     ('CNBC', 'Peacock', 'Advertise', 'With'): 1,
     ('Peacock', 'Advertise', 'With', 'Us'): 1,
     ('Advertise', 'With', 'Us', 'Join'): 1,
     ('With', 'Us', 'Join', 'CNBC'): 1,
     ('Us', 'Join', 'CNBC', 'Panel'): 1,
     ('Join', 'CNBC', 'Panel', 'Digital'): 1,
     ('CNBC', 'Panel', 'Digital', 'Products'): 1,
     ('Panel', 'Digital', 'Products', 'News'): 1,
     ('Digital', 'Products', 'News', 'Releases'): 1,
     ('Products', 'News', 'Releases', 'Closed'): 1,
     ('News', 'Releases', 'Closed', 'Captioning'): 1,
     ('Releases', 'Closed', 'Captioning', 'Corrections'): 1,
     ('Closed', 'Captioning', 'Corrections', 'About'): 1,
     ('Captioning', 'Corrections', 'About', 'CNBC'): 1,
     ('Corrections', 'About', 'CNBC', 'Internships'): 1,
     ('About', 'CNBC', 'Internships', 'Site'): 1,
     ('CNBC', 'Internships', 'Site', 'Map'): 1,
     ('Internships', 'Site', 'Map', 'Podcasts'): 1,
     ('Site', 'Map', 'Podcasts', 'Ad'): 1,
     ('Map', 'Podcasts', 'Ad', 'Choices'): 1,
     ('Podcasts', 'Ad', 'Choices', 'Careers'): 1,
     ('Ad', 'Choices', 'Careers', 'Help'): 1,
     ('Choices', 'Careers', 'Help', 'Contact'): 1,
     ('Careers', 'Help', 'Contact', 'News'): 1,
     ('Help', 'Contact', 'News', 'Tips'): 1,
     ('Contact', 'News', 'Tips', 'Got'): 1,
     ('News', 'Tips', 'Got', 'confidential'): 1,
     ('Tips', 'Got', 'confidential', 'news'): 1,
     ('Got', 'confidential', 'news', 'tip'): 1,
     ('confidential', 'news', 'tip', '?'): 1,
     ('news', 'tip', '?', 'We'): 1,
     ('tip', '?', 'We', 'want'): 1,
     ('?', 'We', 'want', 'hear'): 1,
     ('We', 'want', 'hear', '.'): 1,
     ('want', 'hear', '.', 'Get'): 1,
     ('hear', '.', 'Get', 'In'): 1,
     ('.', 'Get', 'In', 'Touch'): 1,
     ('Get', 'In', 'Touch', 'CNBC'): 1,
     ('In', 'Touch', 'CNBC', 'Newsletters'): 1,
     ('Touch', 'CNBC', 'Newsletters', 'Sign'): 1,
     ('CNBC', 'Newsletters', 'Sign', 'free'): 1,
     ('Newsletters', 'Sign', 'free', 'newsletters'): 1,
     ('Sign', 'free', 'newsletters', 'get'): 1,
     ('free', 'newsletters', 'get', 'CNBC'): 1,
     ('newsletters', 'get', 'CNBC', 'delivered'): 1,
     ('get', 'CNBC', 'delivered', 'inbox'): 1,
     ('CNBC', 'delivered', 'inbox', 'Sign'): 1,
     ('delivered', 'inbox', 'Sign', 'Up'): 1,
     ('inbox', 'Sign', 'Up', 'Now'): 1,
     ('Sign', 'Up', 'Now', 'Get'): 1,
     ('Up', 'Now', 'Get', 'delivered'): 1,
     ('Now', 'Get', 'delivered', 'inbox'): 1,
     ('Get', 'delivered', 'inbox', ','): 1,
     ('delivered', 'inbox', ',', 'info'): 1,
     ('inbox', ',', 'info', 'products'): 1,
     (',', 'info', 'products', 'services'): 1,
     ('info', 'products', 'services', '.'): 1,
     ('products', 'services', '.', 'Privacy'): 1,
     ('services', '.', 'Privacy', 'Policy'): 1,
     ('.', 'Privacy', 'Policy', '|'): 1,
     ('Privacy', 'Policy', '|', 'Do'): 1,
     ('Policy', '|', 'Do', 'Not'): 1,
     ('|', 'Do', 'Not', 'Sell'): 1,
     ('Do', 'Not', 'Sell', 'My'): 1,
     ('Not', 'Sell', 'My', 'Personal'): 1,
     ('Sell', 'My', 'Personal', 'Information'): 1,
     ('My', 'Personal', 'Information', '|'): 1,
     ('Personal', 'Information', '|', 'CA'): 1,
     ('Information', '|', 'CA', 'Notice'): 1,
     ('|', 'CA', 'Notice', '|'): 1,
     ('CA', 'Notice', '|', 'Terms'): 1,
     ('Notice', '|', 'Terms', 'Service'): 1,
     ('|', 'Terms', 'Service', '©'): 1,
     ('Terms', 'Service', '©', '2020'): 1,
     ('Service', '©', '2020', 'CNBC'): 1,
     ...}




```python
#Lemmatize and deduplicate unigrams into a vocabulary of terms.
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\tramh\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    


```python
# using Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
deduplicate_lem=[]
#Lemmatize and deduplicate unigrams into a vocabulary of terms.
for w in words: 
    if wordnet_lemmatizer.lemmatize(w) not in deduplicate_lem:
        deduplicate_lem.append(wordnet_lemmatizer.lemmatize(w))
```


```python
# Example of lemmatize and deduplicate unigrams into a vocabulary of terms.
print(deduplicate_lem[:200])
print(words[:200])
```

    ['Skip', 'Navigation', 'Markets', 'Pre-Markets', 'U.S.', 'Currencies', 'Cryptocurrency', 'Futures', '&', 'Commodities', 'Bonds', 'Funds', 'ETFs', 'Business', 'Economy', 'Finance', 'Health', 'Science', 'Media', 'Real', 'Estate', 'Energy', 'Transportation', 'Industrials', 'Retail', 'Wealth', 'Life', 'Small', 'Investing', 'Invest', 'In', 'You', 'Personal', 'Fintech', 'Financial', 'Advisors', 'Trading', 'Nation', 'Options', 'Action', 'ETF', 'Street', 'Buffett', 'Archive', 'Earnings', 'Trader', 'Talk', 'Tech', 'Cybersecurity', 'Enterprise', 'Internet', 'Mobile', 'Social', 'Venture', 'Capital', 'Guide', 'Politics', 'White', 'House', 'Policy', 'Defense', 'Congress', '2020', 'Elections', 'CNBC', 'TV', 'Live', 'Audio', 'Day', 'Shows', 'The', 'News', 'with', 'Shepard', 'Smith', 'Entertainment', 'Full', 'Episodes', 'Latest', 'Video', 'Top', 'CEO', 'Interviews', 'Documentaries', 'Podcasts', 'World', 'Digital', 'Originals', 'Schedule', 'Watchlist', 'PRO', 'Subscribe', 'Sign', 'Menu', 'Make', 'It', 'Select', 'USA', 'INTL', 'Search', 'quote', ',', 'news', 'video', 'SIGN', 'IN', 'close', 'in', 'on', 'a', '$', '900', 'billion', 'Covid', 'relief', 'deal', 'Americans', 'await', 'aid', 'Published', 'Wed', 'Dec', '16', '9:11', 'AM', 'EST', 'Updated', '12:57', 'PM', 'Jacob', 'Pramuk', '@', 'jacobpramuk', 'Key', 'Points', 'is', 'getting', 'to', 'coronavirus', 'and', 'could', 'announce', 'it', 'soon', 'Wednesday', '.', 'Republicans', 'Democrats', 'negotiated', 'government', 'funding', 'pandemic', 'agreement', 'into', 'Tuesday', 'night', 'Senate', 'leader', 'Mitch', 'McConnell', 'Chuck', 'Schumer', 'said', 'they', 'hoped', 'reach', 'an', 'accord', '``', "''", 'will', 'shut', 'down', 'Saturday', '12', 'million', 'people', 'lose', 'unemployment', 'benefit', 'the', 'day', 'after', 'Christmas', 'if', 'fails', 'act', 'VIDEO', '3:25', '03:25', '—', 'Here', "'s", 'latest', 'Videos', 'Congressional', 'closed', 'of', 'struggling', 'wait']
    ['Skip', 'Navigation', 'Markets', 'Pre-Markets', 'U.S.', 'Markets', 'Currencies', 'Cryptocurrency', 'Futures', '&', 'Commodities', 'Bonds', 'Funds', '&', 'ETFs', 'Business', 'Economy', 'Finance', 'Health', '&', 'Science', 'Media', 'Real', 'Estate', 'Energy', 'Transportation', 'Industrials', 'Retail', 'Wealth', 'Life', 'Small', 'Business', 'Investing', 'Invest', 'In', 'You', 'Personal', 'Finance', 'Fintech', 'Financial', 'Advisors', 'Trading', 'Nation', 'Options', 'Action', 'ETF', 'Street', 'Buffett', 'Archive', 'Earnings', 'Trader', 'Talk', 'Tech', 'Cybersecurity', 'Enterprise', 'Internet', 'Media', 'Mobile', 'Social', 'Media', 'Venture', 'Capital', 'Tech', 'Guide', 'Politics', 'White', 'House', 'Policy', 'Defense', 'Congress', '2020', 'Elections', 'CNBC', 'TV', 'Live', 'TV', 'Live', 'Audio', 'Business', 'Day', 'Shows', 'The', 'News', 'with', 'Shepard', 'Smith', 'Entertainment', 'Shows', 'Full', 'Episodes', 'Latest', 'Video', 'Top', 'Video', 'CEO', 'Interviews', 'CNBC', 'Documentaries', 'CNBC', 'Podcasts', 'CNBC', 'World', 'Digital', 'Originals', 'Live', 'TV', 'Schedule', 'Watchlist', 'PRO', 'PRO', 'News', 'PRO', 'Live', 'Subscribe', 'Sign', 'In', 'Menu', 'Make', 'It', 'Select', 'USA', 'INTL', 'Search', 'quotes', ',', 'news', '&', 'videos', 'SIGN', 'IN', 'Markets', 'Pre-Markets', 'U.S.', 'Markets', 'Currencies', 'Cryptocurrency', 'Futures', '&', 'Commodities', 'Bonds', 'Funds', '&', 'ETFs', 'Business', 'Economy', 'Finance', 'Health', '&', 'Science', 'Media', 'Real', 'Estate', 'Energy', 'Transportation', 'Industrials', 'Retail', 'Wealth', 'Life', 'Small', 'Business', 'Investing', 'Invest', 'In', 'You', 'Personal', 'Finance', 'Fintech', 'Financial', 'Advisors', 'Trading', 'Nation', 'Options', 'Action', 'ETF', 'Street', 'Buffett', 'Archive', 'Earnings', 'Trader', 'Talk', 'Tech', 'Cybersecurity', 'Enterprise', 'Internet', 'Media', 'Mobile', 'Social', 'Media', 'Venture', 'Capital', 'Tech', 'Guide', 'Politics', 'White', 'House', 'Policy', 'Defense', 'Congress', '2020', 'Elections']
    


```python
from nltk.util import ngrams
#Create the list of first 5 sentences
sentences_5 = sentences[:5]

#Function to return ngrams 
def grams_output(sent, n):
    words = word_tokenize(sent)
    grams_return = ngrams(words, n)
    return(list(grams_return))
```


```python
#Print bigrams and trigrams in the first 5 sentences
for i in range(5):
    print("Bigrams and trigrams" + "for sentence " + str(i+1))
    print(grams_output(sentences_5[i], 2))
    print(grams_output(sentences_5[i], 3))
    print('\n')
```

    Bigrams and trigramsfor sentence 1
    [('Skip', 'Navigation'), ('Navigation', 'Markets'), ('Markets', 'Pre-Markets'), ('Pre-Markets', 'U.S.'), ('U.S.', 'Markets'), ('Markets', 'Currencies'), ('Currencies', 'Cryptocurrency'), ('Cryptocurrency', 'Futures'), ('Futures', '&'), ('&', 'Commodities'), ('Commodities', 'Bonds'), ('Bonds', 'Funds'), ('Funds', '&'), ('&', 'ETFs'), ('ETFs', 'Business'), ('Business', 'Economy'), ('Economy', 'Finance'), ('Finance', 'Health'), ('Health', '&'), ('&', 'Science'), ('Science', 'Media'), ('Media', 'Real'), ('Real', 'Estate'), ('Estate', 'Energy'), ('Energy', 'Transportation'), ('Transportation', 'Industrials'), ('Industrials', 'Retail'), ('Retail', 'Wealth'), ('Wealth', 'Life'), ('Life', 'Small'), ('Small', 'Business'), ('Business', 'Investing'), ('Investing', 'Invest'), ('Invest', 'In'), ('In', 'You'), ('You', 'Personal'), ('Personal', 'Finance'), ('Finance', 'Fintech'), ('Fintech', 'Financial'), ('Financial', 'Advisors'), ('Advisors', 'Trading'), ('Trading', 'Nation'), ('Nation', 'Options'), ('Options', 'Action'), ('Action', 'ETF'), ('ETF', 'Street'), ('Street', 'Buffett'), ('Buffett', 'Archive'), ('Archive', 'Earnings'), ('Earnings', 'Trader'), ('Trader', 'Talk'), ('Talk', 'Tech'), ('Tech', 'Cybersecurity'), ('Cybersecurity', 'Enterprise'), ('Enterprise', 'Internet'), ('Internet', 'Media'), ('Media', 'Mobile'), ('Mobile', 'Social'), ('Social', 'Media'), ('Media', 'Venture'), ('Venture', 'Capital'), ('Capital', 'Tech'), ('Tech', 'Guide'), ('Guide', 'Politics'), ('Politics', 'White'), ('White', 'House'), ('House', 'Policy'), ('Policy', 'Defense'), ('Defense', 'Congress'), ('Congress', '2020'), ('2020', 'Elections'), ('Elections', 'CNBC'), ('CNBC', 'TV'), ('TV', 'Live'), ('Live', 'TV'), ('TV', 'Live'), ('Live', 'Audio'), ('Audio', 'Business'), ('Business', 'Day'), ('Day', 'Shows'), ('Shows', 'The'), ('The', 'News'), ('News', 'with'), ('with', 'Shepard'), ('Shepard', 'Smith'), ('Smith', 'Entertainment'), ('Entertainment', 'Shows'), ('Shows', 'Full'), ('Full', 'Episodes'), ('Episodes', 'Latest'), ('Latest', 'Video'), ('Video', 'Top'), ('Top', 'Video'), ('Video', 'CEO'), ('CEO', 'Interviews'), ('Interviews', 'CNBC'), ('CNBC', 'Documentaries'), ('Documentaries', 'CNBC'), ('CNBC', 'Podcasts'), ('Podcasts', 'CNBC'), ('CNBC', 'World'), ('World', 'Digital'), ('Digital', 'Originals'), ('Originals', 'Live'), ('Live', 'TV'), ('TV', 'Schedule'), ('Schedule', 'Watchlist'), ('Watchlist', 'PRO'), ('PRO', 'PRO'), ('PRO', 'News'), ('News', 'PRO'), ('PRO', 'Live'), ('Live', 'Subscribe'), ('Subscribe', 'Sign'), ('Sign', 'In'), ('In', 'Menu'), ('Menu', 'Make'), ('Make', 'It'), ('It', 'Select'), ('Select', 'USA'), ('USA', 'INTL'), ('INTL', 'Search'), ('Search', 'quotes'), ('quotes', ','), (',', 'news'), ('news', '&'), ('&', 'videos'), ('videos', 'SIGN'), ('SIGN', 'IN'), ('IN', 'Markets'), ('Markets', 'Pre-Markets'), ('Pre-Markets', 'U.S.'), ('U.S.', 'Markets'), ('Markets', 'Currencies'), ('Currencies', 'Cryptocurrency'), ('Cryptocurrency', 'Futures'), ('Futures', '&'), ('&', 'Commodities'), ('Commodities', 'Bonds'), ('Bonds', 'Funds'), ('Funds', '&'), ('&', 'ETFs'), ('ETFs', 'Business'), ('Business', 'Economy'), ('Economy', 'Finance'), ('Finance', 'Health'), ('Health', '&'), ('&', 'Science'), ('Science', 'Media'), ('Media', 'Real'), ('Real', 'Estate'), ('Estate', 'Energy'), ('Energy', 'Transportation'), ('Transportation', 'Industrials'), ('Industrials', 'Retail'), ('Retail', 'Wealth'), ('Wealth', 'Life'), ('Life', 'Small'), ('Small', 'Business'), ('Business', 'Investing'), ('Investing', 'Invest'), ('Invest', 'In'), ('In', 'You'), ('You', 'Personal'), ('Personal', 'Finance'), ('Finance', 'Fintech'), ('Fintech', 'Financial'), ('Financial', 'Advisors'), ('Advisors', 'Trading'), ('Trading', 'Nation'), ('Nation', 'Options'), ('Options', 'Action'), ('Action', 'ETF'), ('ETF', 'Street'), ('Street', 'Buffett'), ('Buffett', 'Archive'), ('Archive', 'Earnings'), ('Earnings', 'Trader'), ('Trader', 'Talk'), ('Talk', 'Tech'), ('Tech', 'Cybersecurity'), ('Cybersecurity', 'Enterprise'), ('Enterprise', 'Internet'), ('Internet', 'Media'), ('Media', 'Mobile'), ('Mobile', 'Social'), ('Social', 'Media'), ('Media', 'Venture'), ('Venture', 'Capital'), ('Capital', 'Tech'), ('Tech', 'Guide'), ('Guide', 'Politics'), ('Politics', 'White'), ('White', 'House'), ('House', 'Policy'), ('Policy', 'Defense'), ('Defense', 'Congress'), ('Congress', '2020'), ('2020', 'Elections'), ('Elections', 'CNBC'), ('CNBC', 'TV'), ('TV', 'Live'), ('Live', 'TV'), ('TV', 'Live'), ('Live', 'Audio'), ('Audio', 'Business'), ('Business', 'Day'), ('Day', 'Shows'), ('Shows', 'The'), ('The', 'News'), ('News', 'with'), ('with', 'Shepard'), ('Shepard', 'Smith'), ('Smith', 'Entertainment'), ('Entertainment', 'Shows'), ('Shows', 'Full'), ('Full', 'Episodes'), ('Episodes', 'Latest'), ('Latest', 'Video'), ('Video', 'Top'), ('Top', 'Video'), ('Video', 'CEO'), ('CEO', 'Interviews'), ('Interviews', 'CNBC'), ('CNBC', 'Documentaries'), ('Documentaries', 'CNBC'), ('CNBC', 'Podcasts'), ('Podcasts', 'CNBC'), ('CNBC', 'World'), ('World', 'Digital'), ('Digital', 'Originals'), ('Originals', 'Live'), ('Live', 'TV'), ('TV', 'Schedule'), ('Schedule', 'Watchlist'), ('Watchlist', 'PRO'), ('PRO', 'PRO'), ('PRO', 'News'), ('News', 'PRO'), ('PRO', 'Live'), ('Live', 'Subscribe'), ('Subscribe', 'Sign'), ('Sign', 'In'), ('In', 'Menu'), ('Menu', 'Politics'), ('Politics', 'Congress'), ('Congress', 'closes'), ('closes', 'in'), ('in', 'on'), ('on', 'a'), ('a', '$'), ('$', '900'), ('900', 'billion'), ('billion', 'Covid'), ('Covid', 'relief'), ('relief', 'deal'), ('deal', 'as'), ('as', 'Americans'), ('Americans', 'await'), ('await', 'aid'), ('aid', 'Published'), ('Published', 'Wed'), ('Wed', ','), (',', 'Dec'), ('Dec', '16'), ('16', '2020'), ('2020', '9:11'), ('9:11', 'AM'), ('AM', 'EST'), ('EST', 'Updated'), ('Updated', 'Wed'), ('Wed', ','), (',', 'Dec'), ('Dec', '16'), ('16', '2020'), ('2020', '12:57'), ('12:57', 'PM'), ('PM', 'EST'), ('EST', 'Jacob'), ('Jacob', 'Pramuk'), ('Pramuk', '@'), ('@', 'jacobpramuk'), ('jacobpramuk', 'Key'), ('Key', 'Points'), ('Points', 'Congress'), ('Congress', 'is'), ('is', 'getting'), ('getting', 'close'), ('close', 'to'), ('to', 'a'), ('a', '$'), ('$', '900'), ('900', 'billion'), ('billion', 'coronavirus'), ('coronavirus', 'relief'), ('relief', 'deal'), ('deal', 'and'), ('and', 'could'), ('could', 'announce'), ('announce', 'it'), ('it', 'as'), ('as', 'soon'), ('soon', 'as'), ('as', 'Wednesday'), ('Wednesday', '.')]
    [('Skip', 'Navigation', 'Markets'), ('Navigation', 'Markets', 'Pre-Markets'), ('Markets', 'Pre-Markets', 'U.S.'), ('Pre-Markets', 'U.S.', 'Markets'), ('U.S.', 'Markets', 'Currencies'), ('Markets', 'Currencies', 'Cryptocurrency'), ('Currencies', 'Cryptocurrency', 'Futures'), ('Cryptocurrency', 'Futures', '&'), ('Futures', '&', 'Commodities'), ('&', 'Commodities', 'Bonds'), ('Commodities', 'Bonds', 'Funds'), ('Bonds', 'Funds', '&'), ('Funds', '&', 'ETFs'), ('&', 'ETFs', 'Business'), ('ETFs', 'Business', 'Economy'), ('Business', 'Economy', 'Finance'), ('Economy', 'Finance', 'Health'), ('Finance', 'Health', '&'), ('Health', '&', 'Science'), ('&', 'Science', 'Media'), ('Science', 'Media', 'Real'), ('Media', 'Real', 'Estate'), ('Real', 'Estate', 'Energy'), ('Estate', 'Energy', 'Transportation'), ('Energy', 'Transportation', 'Industrials'), ('Transportation', 'Industrials', 'Retail'), ('Industrials', 'Retail', 'Wealth'), ('Retail', 'Wealth', 'Life'), ('Wealth', 'Life', 'Small'), ('Life', 'Small', 'Business'), ('Small', 'Business', 'Investing'), ('Business', 'Investing', 'Invest'), ('Investing', 'Invest', 'In'), ('Invest', 'In', 'You'), ('In', 'You', 'Personal'), ('You', 'Personal', 'Finance'), ('Personal', 'Finance', 'Fintech'), ('Finance', 'Fintech', 'Financial'), ('Fintech', 'Financial', 'Advisors'), ('Financial', 'Advisors', 'Trading'), ('Advisors', 'Trading', 'Nation'), ('Trading', 'Nation', 'Options'), ('Nation', 'Options', 'Action'), ('Options', 'Action', 'ETF'), ('Action', 'ETF', 'Street'), ('ETF', 'Street', 'Buffett'), ('Street', 'Buffett', 'Archive'), ('Buffett', 'Archive', 'Earnings'), ('Archive', 'Earnings', 'Trader'), ('Earnings', 'Trader', 'Talk'), ('Trader', 'Talk', 'Tech'), ('Talk', 'Tech', 'Cybersecurity'), ('Tech', 'Cybersecurity', 'Enterprise'), ('Cybersecurity', 'Enterprise', 'Internet'), ('Enterprise', 'Internet', 'Media'), ('Internet', 'Media', 'Mobile'), ('Media', 'Mobile', 'Social'), ('Mobile', 'Social', 'Media'), ('Social', 'Media', 'Venture'), ('Media', 'Venture', 'Capital'), ('Venture', 'Capital', 'Tech'), ('Capital', 'Tech', 'Guide'), ('Tech', 'Guide', 'Politics'), ('Guide', 'Politics', 'White'), ('Politics', 'White', 'House'), ('White', 'House', 'Policy'), ('House', 'Policy', 'Defense'), ('Policy', 'Defense', 'Congress'), ('Defense', 'Congress', '2020'), ('Congress', '2020', 'Elections'), ('2020', 'Elections', 'CNBC'), ('Elections', 'CNBC', 'TV'), ('CNBC', 'TV', 'Live'), ('TV', 'Live', 'TV'), ('Live', 'TV', 'Live'), ('TV', 'Live', 'Audio'), ('Live', 'Audio', 'Business'), ('Audio', 'Business', 'Day'), ('Business', 'Day', 'Shows'), ('Day', 'Shows', 'The'), ('Shows', 'The', 'News'), ('The', 'News', 'with'), ('News', 'with', 'Shepard'), ('with', 'Shepard', 'Smith'), ('Shepard', 'Smith', 'Entertainment'), ('Smith', 'Entertainment', 'Shows'), ('Entertainment', 'Shows', 'Full'), ('Shows', 'Full', 'Episodes'), ('Full', 'Episodes', 'Latest'), ('Episodes', 'Latest', 'Video'), ('Latest', 'Video', 'Top'), ('Video', 'Top', 'Video'), ('Top', 'Video', 'CEO'), ('Video', 'CEO', 'Interviews'), ('CEO', 'Interviews', 'CNBC'), ('Interviews', 'CNBC', 'Documentaries'), ('CNBC', 'Documentaries', 'CNBC'), ('Documentaries', 'CNBC', 'Podcasts'), ('CNBC', 'Podcasts', 'CNBC'), ('Podcasts', 'CNBC', 'World'), ('CNBC', 'World', 'Digital'), ('World', 'Digital', 'Originals'), ('Digital', 'Originals', 'Live'), ('Originals', 'Live', 'TV'), ('Live', 'TV', 'Schedule'), ('TV', 'Schedule', 'Watchlist'), ('Schedule', 'Watchlist', 'PRO'), ('Watchlist', 'PRO', 'PRO'), ('PRO', 'PRO', 'News'), ('PRO', 'News', 'PRO'), ('News', 'PRO', 'Live'), ('PRO', 'Live', 'Subscribe'), ('Live', 'Subscribe', 'Sign'), ('Subscribe', 'Sign', 'In'), ('Sign', 'In', 'Menu'), ('In', 'Menu', 'Make'), ('Menu', 'Make', 'It'), ('Make', 'It', 'Select'), ('It', 'Select', 'USA'), ('Select', 'USA', 'INTL'), ('USA', 'INTL', 'Search'), ('INTL', 'Search', 'quotes'), ('Search', 'quotes', ','), ('quotes', ',', 'news'), (',', 'news', '&'), ('news', '&', 'videos'), ('&', 'videos', 'SIGN'), ('videos', 'SIGN', 'IN'), ('SIGN', 'IN', 'Markets'), ('IN', 'Markets', 'Pre-Markets'), ('Markets', 'Pre-Markets', 'U.S.'), ('Pre-Markets', 'U.S.', 'Markets'), ('U.S.', 'Markets', 'Currencies'), ('Markets', 'Currencies', 'Cryptocurrency'), ('Currencies', 'Cryptocurrency', 'Futures'), ('Cryptocurrency', 'Futures', '&'), ('Futures', '&', 'Commodities'), ('&', 'Commodities', 'Bonds'), ('Commodities', 'Bonds', 'Funds'), ('Bonds', 'Funds', '&'), ('Funds', '&', 'ETFs'), ('&', 'ETFs', 'Business'), ('ETFs', 'Business', 'Economy'), ('Business', 'Economy', 'Finance'), ('Economy', 'Finance', 'Health'), ('Finance', 'Health', '&'), ('Health', '&', 'Science'), ('&', 'Science', 'Media'), ('Science', 'Media', 'Real'), ('Media', 'Real', 'Estate'), ('Real', 'Estate', 'Energy'), ('Estate', 'Energy', 'Transportation'), ('Energy', 'Transportation', 'Industrials'), ('Transportation', 'Industrials', 'Retail'), ('Industrials', 'Retail', 'Wealth'), ('Retail', 'Wealth', 'Life'), ('Wealth', 'Life', 'Small'), ('Life', 'Small', 'Business'), ('Small', 'Business', 'Investing'), ('Business', 'Investing', 'Invest'), ('Investing', 'Invest', 'In'), ('Invest', 'In', 'You'), ('In', 'You', 'Personal'), ('You', 'Personal', 'Finance'), ('Personal', 'Finance', 'Fintech'), ('Finance', 'Fintech', 'Financial'), ('Fintech', 'Financial', 'Advisors'), ('Financial', 'Advisors', 'Trading'), ('Advisors', 'Trading', 'Nation'), ('Trading', 'Nation', 'Options'), ('Nation', 'Options', 'Action'), ('Options', 'Action', 'ETF'), ('Action', 'ETF', 'Street'), ('ETF', 'Street', 'Buffett'), ('Street', 'Buffett', 'Archive'), ('Buffett', 'Archive', 'Earnings'), ('Archive', 'Earnings', 'Trader'), ('Earnings', 'Trader', 'Talk'), ('Trader', 'Talk', 'Tech'), ('Talk', 'Tech', 'Cybersecurity'), ('Tech', 'Cybersecurity', 'Enterprise'), ('Cybersecurity', 'Enterprise', 'Internet'), ('Enterprise', 'Internet', 'Media'), ('Internet', 'Media', 'Mobile'), ('Media', 'Mobile', 'Social'), ('Mobile', 'Social', 'Media'), ('Social', 'Media', 'Venture'), ('Media', 'Venture', 'Capital'), ('Venture', 'Capital', 'Tech'), ('Capital', 'Tech', 'Guide'), ('Tech', 'Guide', 'Politics'), ('Guide', 'Politics', 'White'), ('Politics', 'White', 'House'), ('White', 'House', 'Policy'), ('House', 'Policy', 'Defense'), ('Policy', 'Defense', 'Congress'), ('Defense', 'Congress', '2020'), ('Congress', '2020', 'Elections'), ('2020', 'Elections', 'CNBC'), ('Elections', 'CNBC', 'TV'), ('CNBC', 'TV', 'Live'), ('TV', 'Live', 'TV'), ('Live', 'TV', 'Live'), ('TV', 'Live', 'Audio'), ('Live', 'Audio', 'Business'), ('Audio', 'Business', 'Day'), ('Business', 'Day', 'Shows'), ('Day', 'Shows', 'The'), ('Shows', 'The', 'News'), ('The', 'News', 'with'), ('News', 'with', 'Shepard'), ('with', 'Shepard', 'Smith'), ('Shepard', 'Smith', 'Entertainment'), ('Smith', 'Entertainment', 'Shows'), ('Entertainment', 'Shows', 'Full'), ('Shows', 'Full', 'Episodes'), ('Full', 'Episodes', 'Latest'), ('Episodes', 'Latest', 'Video'), ('Latest', 'Video', 'Top'), ('Video', 'Top', 'Video'), ('Top', 'Video', 'CEO'), ('Video', 'CEO', 'Interviews'), ('CEO', 'Interviews', 'CNBC'), ('Interviews', 'CNBC', 'Documentaries'), ('CNBC', 'Documentaries', 'CNBC'), ('Documentaries', 'CNBC', 'Podcasts'), ('CNBC', 'Podcasts', 'CNBC'), ('Podcasts', 'CNBC', 'World'), ('CNBC', 'World', 'Digital'), ('World', 'Digital', 'Originals'), ('Digital', 'Originals', 'Live'), ('Originals', 'Live', 'TV'), ('Live', 'TV', 'Schedule'), ('TV', 'Schedule', 'Watchlist'), ('Schedule', 'Watchlist', 'PRO'), ('Watchlist', 'PRO', 'PRO'), ('PRO', 'PRO', 'News'), ('PRO', 'News', 'PRO'), ('News', 'PRO', 'Live'), ('PRO', 'Live', 'Subscribe'), ('Live', 'Subscribe', 'Sign'), ('Subscribe', 'Sign', 'In'), ('Sign', 'In', 'Menu'), ('In', 'Menu', 'Politics'), ('Menu', 'Politics', 'Congress'), ('Politics', 'Congress', 'closes'), ('Congress', 'closes', 'in'), ('closes', 'in', 'on'), ('in', 'on', 'a'), ('on', 'a', '$'), ('a', '$', '900'), ('$', '900', 'billion'), ('900', 'billion', 'Covid'), ('billion', 'Covid', 'relief'), ('Covid', 'relief', 'deal'), ('relief', 'deal', 'as'), ('deal', 'as', 'Americans'), ('as', 'Americans', 'await'), ('Americans', 'await', 'aid'), ('await', 'aid', 'Published'), ('aid', 'Published', 'Wed'), ('Published', 'Wed', ','), ('Wed', ',', 'Dec'), (',', 'Dec', '16'), ('Dec', '16', '2020'), ('16', '2020', '9:11'), ('2020', '9:11', 'AM'), ('9:11', 'AM', 'EST'), ('AM', 'EST', 'Updated'), ('EST', 'Updated', 'Wed'), ('Updated', 'Wed', ','), ('Wed', ',', 'Dec'), (',', 'Dec', '16'), ('Dec', '16', '2020'), ('16', '2020', '12:57'), ('2020', '12:57', 'PM'), ('12:57', 'PM', 'EST'), ('PM', 'EST', 'Jacob'), ('EST', 'Jacob', 'Pramuk'), ('Jacob', 'Pramuk', '@'), ('Pramuk', '@', 'jacobpramuk'), ('@', 'jacobpramuk', 'Key'), ('jacobpramuk', 'Key', 'Points'), ('Key', 'Points', 'Congress'), ('Points', 'Congress', 'is'), ('Congress', 'is', 'getting'), ('is', 'getting', 'close'), ('getting', 'close', 'to'), ('close', 'to', 'a'), ('to', 'a', '$'), ('a', '$', '900'), ('$', '900', 'billion'), ('900', 'billion', 'coronavirus'), ('billion', 'coronavirus', 'relief'), ('coronavirus', 'relief', 'deal'), ('relief', 'deal', 'and'), ('deal', 'and', 'could'), ('and', 'could', 'announce'), ('could', 'announce', 'it'), ('announce', 'it', 'as'), ('it', 'as', 'soon'), ('as', 'soon', 'as'), ('soon', 'as', 'Wednesday'), ('as', 'Wednesday', '.')]
    
    
    Bigrams and trigramsfor sentence 2
    [('Top', 'Republicans'), ('Republicans', 'and'), ('and', 'Democrats'), ('Democrats', 'negotiated'), ('negotiated', 'a'), ('a', 'government'), ('government', 'funding'), ('funding', 'and'), ('and', 'pandemic'), ('pandemic', 'aid'), ('aid', 'agreement'), ('agreement', 'into'), ('into', 'Tuesday'), ('Tuesday', 'night'), ('night', ','), (',', 'and'), ('and', 'Senate'), ('Senate', 'leaders'), ('leaders', 'Mitch'), ('Mitch', 'McConnell'), ('McConnell', 'and'), ('and', 'Chuck'), ('Chuck', 'Schumer'), ('Schumer', 'said'), ('said', 'they'), ('they', 'hoped'), ('hoped', 'to'), ('to', 'reach'), ('reach', 'an'), ('an', 'accord'), ('accord', '``'), ('``', 'soon'), ('soon', '.'), ('.', "''")]
    [('Top', 'Republicans', 'and'), ('Republicans', 'and', 'Democrats'), ('and', 'Democrats', 'negotiated'), ('Democrats', 'negotiated', 'a'), ('negotiated', 'a', 'government'), ('a', 'government', 'funding'), ('government', 'funding', 'and'), ('funding', 'and', 'pandemic'), ('and', 'pandemic', 'aid'), ('pandemic', 'aid', 'agreement'), ('aid', 'agreement', 'into'), ('agreement', 'into', 'Tuesday'), ('into', 'Tuesday', 'night'), ('Tuesday', 'night', ','), ('night', ',', 'and'), (',', 'and', 'Senate'), ('and', 'Senate', 'leaders'), ('Senate', 'leaders', 'Mitch'), ('leaders', 'Mitch', 'McConnell'), ('Mitch', 'McConnell', 'and'), ('McConnell', 'and', 'Chuck'), ('and', 'Chuck', 'Schumer'), ('Chuck', 'Schumer', 'said'), ('Schumer', 'said', 'they'), ('said', 'they', 'hoped'), ('they', 'hoped', 'to'), ('hoped', 'to', 'reach'), ('to', 'reach', 'an'), ('reach', 'an', 'accord'), ('an', 'accord', '``'), ('accord', '``', 'soon'), ('``', 'soon', '.'), ('soon', '.', "''")]
    
    
    Bigrams and trigramsfor sentence 3
    [('The', 'government'), ('government', 'will'), ('will', 'shut'), ('shut', 'down'), ('down', 'on'), ('on', 'Saturday'), ('Saturday', 'and'), ('and', '12'), ('12', 'million'), ('million', 'people'), ('people', 'will'), ('will', 'lose'), ('lose', 'unemployment'), ('unemployment', 'benefits'), ('benefits', 'the'), ('the', 'day'), ('day', 'after'), ('after', 'Christmas'), ('Christmas', 'if'), ('if', 'Congress'), ('Congress', 'fails'), ('fails', 'to'), ('to', 'act'), ('act', '.')]
    [('The', 'government', 'will'), ('government', 'will', 'shut'), ('will', 'shut', 'down'), ('shut', 'down', 'on'), ('down', 'on', 'Saturday'), ('on', 'Saturday', 'and'), ('Saturday', 'and', '12'), ('and', '12', 'million'), ('12', 'million', 'people'), ('million', 'people', 'will'), ('people', 'will', 'lose'), ('will', 'lose', 'unemployment'), ('lose', 'unemployment', 'benefits'), ('unemployment', 'benefits', 'the'), ('benefits', 'the', 'day'), ('the', 'day', 'after'), ('day', 'after', 'Christmas'), ('after', 'Christmas', 'if'), ('Christmas', 'if', 'Congress'), ('if', 'Congress', 'fails'), ('Congress', 'fails', 'to'), ('fails', 'to', 'act'), ('to', 'act', '.')]
    
    
    Bigrams and trigramsfor sentence 4
    [('VIDEO', '3:25'), ('3:25', '03:25'), ('03:25', 'Congress'), ('Congress', 'closes'), ('closes', 'in'), ('in', 'on'), ('on', 'a'), ('a', '$'), ('$', '900'), ('900', 'billion'), ('billion', 'Covid'), ('Covid', 'relief'), ('relief', 'deal'), ('deal', '—'), ('—', 'Here'), ('Here', "'s"), ("'s", 'the'), ('the', 'latest'), ('latest', 'News'), ('News', 'Videos'), ('Videos', 'Congressional'), ('Congressional', 'leaders'), ('leaders', 'closed'), ('closed', 'in'), ('in', 'on'), ('on', 'a'), ('a', '$'), ('$', '900'), ('900', 'billion'), ('billion', 'coronavirus'), ('coronavirus', 'relief'), ('relief', 'deal'), ('deal', 'Wednesday'), ('Wednesday', 'as'), ('as', 'millions'), ('millions', 'of'), ('of', 'struggling'), ('struggling', 'Americans'), ('Americans', 'wait'), ('wait', 'for'), ('for', 'help'), ('help', '.')]
    [('VIDEO', '3:25', '03:25'), ('3:25', '03:25', 'Congress'), ('03:25', 'Congress', 'closes'), ('Congress', 'closes', 'in'), ('closes', 'in', 'on'), ('in', 'on', 'a'), ('on', 'a', '$'), ('a', '$', '900'), ('$', '900', 'billion'), ('900', 'billion', 'Covid'), ('billion', 'Covid', 'relief'), ('Covid', 'relief', 'deal'), ('relief', 'deal', '—'), ('deal', '—', 'Here'), ('—', 'Here', "'s"), ('Here', "'s", 'the'), ("'s", 'the', 'latest'), ('the', 'latest', 'News'), ('latest', 'News', 'Videos'), ('News', 'Videos', 'Congressional'), ('Videos', 'Congressional', 'leaders'), ('Congressional', 'leaders', 'closed'), ('leaders', 'closed', 'in'), ('closed', 'in', 'on'), ('in', 'on', 'a'), ('on', 'a', '$'), ('a', '$', '900'), ('$', '900', 'billion'), ('900', 'billion', 'coronavirus'), ('billion', 'coronavirus', 'relief'), ('coronavirus', 'relief', 'deal'), ('relief', 'deal', 'Wednesday'), ('deal', 'Wednesday', 'as'), ('Wednesday', 'as', 'millions'), ('as', 'millions', 'of'), ('millions', 'of', 'struggling'), ('of', 'struggling', 'Americans'), ('struggling', 'Americans', 'wait'), ('Americans', 'wait', 'for'), ('wait', 'for', 'help'), ('for', 'help', '.')]
    
    
    Bigrams and trigramsfor sentence 5
    [('The', 'developing'), ('developing', 'aid'), ('aid', 'agreement'), ('agreement', 'would'), ('would', 'not'), ('not', 'include'), ('include', 'liability'), ('liability', 'protections'), ('protections', 'for'), ('for', 'businesses'), ('businesses', 'or'), ('or', 'aid'), ('aid', 'to'), ('to', 'state'), ('state', 'and'), ('and', 'local'), ('local', 'government'), ('government', ','), (',', 'CNBC'), ('CNBC', 'confirmed'), ('confirmed', '.')]
    [('The', 'developing', 'aid'), ('developing', 'aid', 'agreement'), ('aid', 'agreement', 'would'), ('agreement', 'would', 'not'), ('would', 'not', 'include'), ('not', 'include', 'liability'), ('include', 'liability', 'protections'), ('liability', 'protections', 'for'), ('protections', 'for', 'businesses'), ('for', 'businesses', 'or'), ('businesses', 'or', 'aid'), ('or', 'aid', 'to'), ('aid', 'to', 'state'), ('to', 'state', 'and'), ('state', 'and', 'local'), ('and', 'local', 'government'), ('local', 'government', ','), ('government', ',', 'CNBC'), (',', 'CNBC', 'confirmed'), ('CNBC', 'confirmed', '.')]
    
    
    


```python
nltk.download('averaged_perceptron_tagger')
# Print POS tags in the first 5 sentences
for i in range(5):
    tokens = word_tokenize(sentences_5[i])
    print("POS tags" + "for sentence " + str(i+1))
    sentence_pos = pos_tag(tokens)
    print(sentence_pos)
```

    POS tagsfor sentence 1
    [('Skip', 'JJ'), ('Navigation', 'NNP'), ('Markets', 'NNP'), ('Pre-Markets', 'NNP'), ('U.S.', 'NNP'), ('Markets', 'NNP'), ('Currencies', 'NNP'), ('Cryptocurrency', 'NNP'), ('Futures', 'NNP'), ('&', 'CC'), ('Commodities', 'NNP'), ('Bonds', 'NNP'), ('Funds', 'NNP'), ('&', 'CC'), ('ETFs', 'NNP'), ('Business', 'NNP'), ('Economy', 'NNP'), ('Finance', 'NNP'), ('Health', 'NNP'), ('&', 'CC'), ('Science', 'NNP'), ('Media', 'NNP'), ('Real', 'NNP'), ('Estate', 'NNP'), ('Energy', 'NNP'), ('Transportation', 'NNP'), ('Industrials', 'NNP'), ('Retail', 'NNP'), ('Wealth', 'NNP'), ('Life', 'NNP'), ('Small', 'NNP'), ('Business', 'NNP'), ('Investing', 'NNP'), ('Invest', 'NNP'), ('In', 'IN'), ('You', 'PRP'), ('Personal', 'NNP'), ('Finance', 'NNP'), ('Fintech', 'NNP'), ('Financial', 'NNP'), ('Advisors', 'NNPS'), ('Trading', 'NNP'), ('Nation', 'NN'), ('Options', 'NNP'), ('Action', 'NNP'), ('ETF', 'NNP'), ('Street', 'NNP'), ('Buffett', 'NNP'), ('Archive', 'NNP'), ('Earnings', 'NNP'), ('Trader', 'NNP'), ('Talk', 'NNP'), ('Tech', 'NNP'), ('Cybersecurity', 'NNP'), ('Enterprise', 'NNP'), ('Internet', 'NNP'), ('Media', 'NNP'), ('Mobile', 'NNP'), ('Social', 'NNP'), ('Media', 'NNP'), ('Venture', 'NNP'), ('Capital', 'NNP'), ('Tech', 'NNP'), ('Guide', 'NNP'), ('Politics', 'NNP'), ('White', 'NNP'), ('House', 'NNP'), ('Policy', 'NN'), ('Defense', 'NNP'), ('Congress', 'NNP'), ('2020', 'CD'), ('Elections', 'NNP'), ('CNBC', 'NNP'), ('TV', 'NNP'), ('Live', 'NNP'), ('TV', 'NN'), ('Live', 'NNP'), ('Audio', 'NNP'), ('Business', 'NNP'), ('Day', 'NNP'), ('Shows', 'VBZ'), ('The', 'DT'), ('News', 'NNP'), ('with', 'IN'), ('Shepard', 'NNP'), ('Smith', 'NNP'), ('Entertainment', 'NNP'), ('Shows', 'NNP'), ('Full', 'NNP'), ('Episodes', 'NNP'), ('Latest', 'NNP'), ('Video', 'NNP'), ('Top', 'NNP'), ('Video', 'NNP'), ('CEO', 'NNP'), ('Interviews', 'NNP'), ('CNBC', 'NNP'), ('Documentaries', 'NNP'), ('CNBC', 'NNP'), ('Podcasts', 'NNP'), ('CNBC', 'NNP'), ('World', 'NNP'), ('Digital', 'NNP'), ('Originals', 'NNP'), ('Live', 'NNP'), ('TV', 'NN'), ('Schedule', 'NNP'), ('Watchlist', 'NNP'), ('PRO', 'NNP'), ('PRO', 'NNP'), ('News', 'NNP'), ('PRO', 'NNP'), ('Live', 'NNP'), ('Subscribe', 'NNP'), ('Sign', 'NNP'), ('In', 'IN'), ('Menu', 'NNP'), ('Make', 'NNP'), ('It', 'PRP'), ('Select', 'NNP'), ('USA', 'NNP'), ('INTL', 'NNP'), ('Search', 'NNP'), ('quotes', 'VBZ'), (',', ','), ('news', 'NN'), ('&', 'CC'), ('videos', 'NN'), ('SIGN', 'NN'), ('IN', 'NNP'), ('Markets', 'NNP'), ('Pre-Markets', 'NNP'), ('U.S.', 'NNP'), ('Markets', 'NNP'), ('Currencies', 'NNP'), ('Cryptocurrency', 'NNP'), ('Futures', 'NNP'), ('&', 'CC'), ('Commodities', 'NNP'), ('Bonds', 'NNP'), ('Funds', 'NNP'), ('&', 'CC'), ('ETFs', 'NNP'), ('Business', 'NNP'), ('Economy', 'NNP'), ('Finance', 'NNP'), ('Health', 'NNP'), ('&', 'CC'), ('Science', 'NNP'), ('Media', 'NNP'), ('Real', 'NNP'), ('Estate', 'NNP'), ('Energy', 'NNP'), ('Transportation', 'NNP'), ('Industrials', 'NNP'), ('Retail', 'NNP'), ('Wealth', 'NNP'), ('Life', 'NNP'), ('Small', 'NNP'), ('Business', 'NNP'), ('Investing', 'NNP'), ('Invest', 'NNP'), ('In', 'IN'), ('You', 'PRP'), ('Personal', 'NNP'), ('Finance', 'NNP'), ('Fintech', 'NNP'), ('Financial', 'NNP'), ('Advisors', 'NNPS'), ('Trading', 'NNP'), ('Nation', 'NN'), ('Options', 'NNP'), ('Action', 'NNP'), ('ETF', 'NNP'), ('Street', 'NNP'), ('Buffett', 'NNP'), ('Archive', 'NNP'), ('Earnings', 'NNP'), ('Trader', 'NNP'), ('Talk', 'NNP'), ('Tech', 'NNP'), ('Cybersecurity', 'NNP'), ('Enterprise', 'NNP'), ('Internet', 'NNP'), ('Media', 'NNP'), ('Mobile', 'NNP'), ('Social', 'NNP'), ('Media', 'NNP'), ('Venture', 'NNP'), ('Capital', 'NNP'), ('Tech', 'NNP'), ('Guide', 'NNP'), ('Politics', 'NNP'), ('White', 'NNP'), ('House', 'NNP'), ('Policy', 'NN'), ('Defense', 'NNP'), ('Congress', 'NNP'), ('2020', 'CD'), ('Elections', 'NNP'), ('CNBC', 'NNP'), ('TV', 'NNP'), ('Live', 'NNP'), ('TV', 'NN'), ('Live', 'NNP'), ('Audio', 'NNP'), ('Business', 'NNP'), ('Day', 'NNP'), ('Shows', 'VBZ'), ('The', 'DT'), ('News', 'NNP'), ('with', 'IN'), ('Shepard', 'NNP'), ('Smith', 'NNP'), ('Entertainment', 'NNP'), ('Shows', 'NNP'), ('Full', 'NNP'), ('Episodes', 'NNP'), ('Latest', 'NNP'), ('Video', 'NNP'), ('Top', 'NNP'), ('Video', 'NNP'), ('CEO', 'NNP'), ('Interviews', 'NNP'), ('CNBC', 'NNP'), ('Documentaries', 'NNP'), ('CNBC', 'NNP'), ('Podcasts', 'NNP'), ('CNBC', 'NNP'), ('World', 'NNP'), ('Digital', 'NNP'), ('Originals', 'NNP'), ('Live', 'NNP'), ('TV', 'NN'), ('Schedule', 'NNP'), ('Watchlist', 'NNP'), ('PRO', 'NNP'), ('PRO', 'NNP'), ('News', 'NNP'), ('PRO', 'NNP'), ('Live', 'NNP'), ('Subscribe', 'NNP'), ('Sign', 'NNP'), ('In', 'IN'), ('Menu', 'NNP'), ('Politics', 'NNP'), ('Congress', 'NNP'), ('closes', 'VBZ'), ('in', 'IN'), ('on', 'IN'), ('a', 'DT'), ('$', '$'), ('900', 'CD'), ('billion', 'CD'), ('Covid', 'NNP'), ('relief', 'NN'), ('deal', 'NN'), ('as', 'IN'), ('Americans', 'NNPS'), ('await', 'VBP'), ('aid', 'NN'), ('Published', 'VBN'), ('Wed', 'NNP'), (',', ','), ('Dec', 'NNP'), ('16', 'CD'), ('2020', 'CD'), ('9:11', 'CD'), ('AM', 'NNP'), ('EST', 'NNP'), ('Updated', 'NNP'), ('Wed', 'NNP'), (',', ','), ('Dec', 'NNP'), ('16', 'CD'), ('2020', 'CD'), ('12:57', 'CD'), ('PM', 'NNP'), ('EST', 'NNP'), ('Jacob', 'NNP'), ('Pramuk', 'NNP'), ('@', 'NNP'), ('jacobpramuk', 'NN'), ('Key', 'NNP'), ('Points', 'NNP'), ('Congress', 'NNP'), ('is', 'VBZ'), ('getting', 'VBG'), ('close', 'RB'), ('to', 'TO'), ('a', 'DT'), ('$', '$'), ('900', 'CD'), ('billion', 'CD'), ('coronavirus', 'NN'), ('relief', 'NN'), ('deal', 'NN'), ('and', 'CC'), ('could', 'MD'), ('announce', 'VB'), ('it', 'PRP'), ('as', 'RB'), ('soon', 'RB'), ('as', 'IN'), ('Wednesday', 'NNP'), ('.', '.')]
    POS tagsfor sentence 2
    [('Top', 'JJ'), ('Republicans', 'NNPS'), ('and', 'CC'), ('Democrats', 'NNPS'), ('negotiated', 'VBD'), ('a', 'DT'), ('government', 'NN'), ('funding', 'NN'), ('and', 'CC'), ('pandemic', 'JJ'), ('aid', 'NN'), ('agreement', 'NN'), ('into', 'IN'), ('Tuesday', 'NNP'), ('night', 'NN'), (',', ','), ('and', 'CC'), ('Senate', 'NNP'), ('leaders', 'NNS'), ('Mitch', 'NNP'), ('McConnell', 'NNP'), ('and', 'CC'), ('Chuck', 'NNP'), ('Schumer', 'NNP'), ('said', 'VBD'), ('they', 'PRP'), ('hoped', 'VBD'), ('to', 'TO'), ('reach', 'VB'), ('an', 'DT'), ('accord', 'NN'), ('``', '``'), ('soon', 'RB'), ('.', '.'), ("''", "''")]
    POS tagsfor sentence 3
    [('The', 'DT'), ('government', 'NN'), ('will', 'MD'), ('shut', 'VB'), ('down', 'RP'), ('on', 'IN'), ('Saturday', 'NNP'), ('and', 'CC'), ('12', 'CD'), ('million', 'CD'), ('people', 'NNS'), ('will', 'MD'), ('lose', 'VB'), ('unemployment', 'NN'), ('benefits', 'NNS'), ('the', 'DT'), ('day', 'NN'), ('after', 'IN'), ('Christmas', 'NNP'), ('if', 'IN'), ('Congress', 'NNP'), ('fails', 'VBZ'), ('to', 'TO'), ('act', 'VB'), ('.', '.')]
    POS tagsfor sentence 4
    [('VIDEO', 'NNP'), ('3:25', 'CD'), ('03:25', 'CD'), ('Congress', 'NNP'), ('closes', 'NNS'), ('in', 'IN'), ('on', 'IN'), ('a', 'DT'), ('$', '$'), ('900', 'CD'), ('billion', 'CD'), ('Covid', 'NNP'), ('relief', 'NN'), ('deal', 'NN'), ('—', 'NNP'), ('Here', 'RB'), ("'s", 'VBZ'), ('the', 'DT'), ('latest', 'JJS'), ('News', 'NNP'), ('Videos', 'NNP'), ('Congressional', 'NNP'), ('leaders', 'NNS'), ('closed', 'VBD'), ('in', 'IN'), ('on', 'IN'), ('a', 'DT'), ('$', '$'), ('900', 'CD'), ('billion', 'CD'), ('coronavirus', 'NN'), ('relief', 'NN'), ('deal', 'NN'), ('Wednesday', 'NNP'), ('as', 'IN'), ('millions', 'NNS'), ('of', 'IN'), ('struggling', 'VBG'), ('Americans', 'NNPS'), ('wait', 'VBP'), ('for', 'IN'), ('help', 'NN'), ('.', '.')]
    POS tagsfor sentence 5
    [('The', 'DT'), ('developing', 'NN'), ('aid', 'NN'), ('agreement', 'NN'), ('would', 'MD'), ('not', 'RB'), ('include', 'VB'), ('liability', 'NN'), ('protections', 'NNS'), ('for', 'IN'), ('businesses', 'NNS'), ('or', 'CC'), ('aid', 'NN'), ('to', 'TO'), ('state', 'NN'), ('and', 'CC'), ('local', 'JJ'), ('government', 'NN'), (',', ','), ('CNBC', 'NNP'), ('confirmed', 'VBD'), ('.', '.')]
    

    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     C:\Users\tramh\AppData\Roaming\nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    
