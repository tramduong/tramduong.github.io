
## Extractive Summarization


```python
import json

import spacy
from spacy.util import minibatch, compounding
from spacy.pipeline import SentenceSegmenter
from spacy.lang.en.stop_words import STOP_WORDS

from sumy.parsers.plaintext import PlaintextParser
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import re
import time
import pandas as pd
```


```python
# Defining a LexRank class 
class TextSummary(object): 
    def __init__(self, feeds_str, num_sents):
        self.summary = str()
        parser = PlaintextParser.from_string(feeds_str, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        sentences = summarizer(parser.document, num_sents)  # Summarize the document with 5 sentences
        for sentence in sentences:
            self.summary += (sentence.__unicode__())
    def output(self):
        return self.summary
```


```python
#Parsing json files 
def parse_json_file(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    parsed_json = [json.loads(line) for line in lines]
    return parsed_json

```


```python
DATA_FILE=r"C:/Users/tramh/github/Data-Science-Portfolio/Airlines Covid-19/data/Airlines_dedup.json"
```


```python
# Load in deduped dataset - clean
mydata = parse_json_file(DATA_FILE)
```


```python
# Add new cleaned date field
for feed in mydata:
    feed['date_clean'] = feed['published'][:10]
```


```python
# Unique ID for each article
ids = [feed['id'] for feed in mydata]

# Extract dates and feeds and put into a dictionary
dates = [feed['date_clean'] for feed in mydata]
feeds = [feed['text'] for feed in mydata]
feed_dict = {ids[i]: [dates[i], feeds[i]] for i in range(len(feeds))}

# Clean the dates field
dates_clean = [date[:10] for date in dates]
unique_dates = set(dates_clean)

# Create a list of indices for each article
for i in range(len(mydata)):
    mydata[i]['id'] = i
```


```python
# Summarizing each individual feed to five sentences
for feed in mydata:
    try:
        feed['summary'] = TextSummary(feed['text'], num_sents=5).output()
    except:
        continue

text_summaries = [feed['summary'] for feed in mydata]
```


```python
#List of feeds by week

W1 = ['2020-06-28', '2020-06-29', '2020-06-30', '2020-07-01', '2020-07-02', '2020-07-03', '2020-07-04']
W2 = ['2020-07-05', '2020-07-06', '2020-07-07', '2020-07-08', '2020-07-09', '2020-07-10', '2020-07-11']
W3 = ['2020-07-11', '2020-07-12', '2020-07-13', '2020-07-14', '2020-07-15', '2020-07-16', '2020-07-17']
W3 = ['2020-07-18', '2020-07-19', '2020-07-20', '2020-07-21', '2020-07-22', '2020-07-23', '2020-07-24']

W1_feeds = []
W2_feeds = []
W3_feeds = []
W4_feeds = []

for feed in mydata:
    for day in unique_dates:
        if feed['date_clean'] == day:
            if day in W1:
                W1_feeds.append(feed['summary'])
            elif day in W2:
                W2_feeds.append(feed['summary'])
            elif day in W3:
                W3_feeds.append(feed['summary'])
            else:
                W4_feeds.append(feed['summary'])
```


```python
# Extractive Summarization by week

weekly_feeds = [W1_feeds, W2_feeds, W3_feeds, W4_feeds]
weekly_summary = []
for week_feed in weekly_feeds:
    tic = time.perf_counter()
    print('Moving on to another week...')
    text = ' '.join([str(elem) for elem in week_feed]) 
    week_summary = TextSummary(text, 3).output()
    weekly_summary.append(week_summary)
    toc = time.perf_counter()
    print('Week summaries completed in {} mins'.format((toc-tic)/60))
    print('======================================================')

```

    Moving on to another week...
    Week summaries completed in 2.236081195 mins
    ======================================================
    Moving on to another week...
    Week summaries completed in 1.7656558033333332 mins
    ======================================================
    Moving on to another week...
    Week summaries completed in 2.2741670950000006 mins
    ======================================================
    Moving on to another week...
    Week summaries completed in 25.22879964666667 mins
    ======================================================
    


```python
for summary in weekly_summary:
    print(summary)
    print()
```

    Sunday, 28 June 2020 White House does not commit to temperature checks in meeting with U.S. airlines Top U.S. airline executives met on Friday with Vice President Mike Pence and other senior administration officials but did not come away with any commitments from the White House on mandating temperature checks for airline passengers.Airlines want the U.S. government to administer temperature checks to all passengers in a bid to reassure the public.from Yahoo News - Latest News & Headlines https://ift.tt/2ZcJNwa - “As more people continue to travel, customers may notice that flights are booked to capacity starting July 1,” the airline said.The change in policy puts the world’s largest airline in line with United and Spirit airlines.“Our customers trust us to make every aspect of their journey safe.We won’t let them down,” said Alison Taylor, the airline’s chief customer officer.“We will continue to refine and update our practices based on the latest information from health authorities and our own Travel Health Advisory Panel.” (Photo: Accura Media Group) United Airlines Holdings Inc ( UAL - Free Report ) plans to resume services to China, following a temporary suspension of services to the nation amid coronavirus concerns.Effective Jul 8, the carrier will re-start its China services with twice-weekly flights between San Francisco and Shanghai's Pudong International Airport via Seoul's Incheon International Airport.From July onward, it will operate flights once a week from Seattle and Detroit, also via Incheon.Key Picks Some better-ranked stocks in the broader Transportation sector are Scorpio Tankers Inc. ( STNG - Free Report ) and Frontline Ltd ( FRO - Free Report ) , both sporting a Zacks Rank #1 (Strong Buy).See these 7 breakthrough stocks now>> by Gary Leff on June 29, 2020 American Airlines CEO Doug Parker, speaking to a group of employees last week in a recording reviewed b View From The Wing , described the pandemic and crisis in travel as having a “silving lining” which is the ability to re-imagine the world’s largest airline from scratch.So now we get to add back and we get to add back only those things that make sense, and as we do that you can make sure you’re learning.We’re not just taking 30% of every group, we’re going through and thinking about which of these groups can we consolidate, where do we find ways to make sure we’re working smarter and faster?A good thing about so little traffic is we can do more things for the traffic that’s there, and we can use these as tests for technology and figure out ways to have a more personal customer experience for example.We need to use this to where when we are back to 100% of flying we’re not doing the exact same stuff.The policy is similar to that of United Airlines Holdings, Inc. ( UAL - Free Report ) , which had never blocked out seats or put a cap on the capacity of its flights.At the same time, A4A said that its member carriers will refund tickets to a passenger who fails the screening test at the airport.The airline industry has been hit hard by the coronavirus pandemic and now as travel resumes, blocking seats means incurring more losses.At the same time, the airline industry has requested the Transportation Security Administration (TSA) to start administering temperature checks on passengers as they go through security checkpoints and anyone found to have an elevated temperature — as defined by the Centers for Disease Control and Prevention (CDC) guidelines — will not be allowed to fly and be made eligible for a ticket refund.See 8 breakthrough stocks now>> Given the ongoing coronavirus pandemic, the design and marketing consultant showed up at the airport in El Paso wearing swim goggles and two masks layered over one another, ready to board the first of her two American Airlines flights home to Michigan.To her dismay, as she made her way back to her seat in the 20th row, she realized it would be virtually impossible to maintain social distancing on the plane: Passengers filled every row including the middle seats; as far as she could see, only one place appeared to be unoccupied.American Airlines had offered travelers up to $250 to switch to a different flight that “has more space” ahead of departure, but Macatangay had an appointment she couldn’t miss.American Airlines notifies passengers when their flight is likely to be crowded—and sometimes offers monetary incentives to take a different flight when the plane is too full.Kass McQuillen, a former contestant on CBS reality series “Survivor,” wrote on Twitter that she’d flown on four “overbooked” American Airlines flights in the past two weeks and “was offered vouchers worth more than my ticket to give up my seat.” The American Airlines spokesperson says “no flight was oversold or sold out” during the pandemic, but that if it were over the 85% capacity limit “it would solicit for volunteers in advance.” Still, with the airline selling tickets beyond that restricted threshold, it may encounter a problem if every passenger shows up wanting to fly.And even if the planes didn’t take off with a full load, if more than 85% of seats were filled upon boarding, it’s unclear whether the restrictions made a difference in preventing the spread of COVID-19 germs at all.Laughable since I’ve been on four flights in the past two weeks that were overbooked and was offered vouchers worth more than my ticket to give up my seat.@CC - you do understand that "people with money" who are used to a "privileged life" typically DON'T FLY COMMERCIAL!If you can afford it you ALWAYS fly private.Not only do you not have to deal with the BS and crowds in a typical airport (don't know if you have ever flown from an FBO but it is a totally different experience) but you only fly with people you actually want to be around.Oh yeah it is can be as comfortable (if not more so) than any commercial seat, you leave and arrive around your schedule (not the airlines) and you have a lot more flexibility of flying into smaller, closer in airports.Obviously you haven't experience private jets or you wouldn't make a comment like this.
    
    Share If you are flying on a U.S. commercial airline, you often have two options: the major carriers (US Air, American, United, Delta and their affiliates) or Southwest.You will find that the people who work for the airlines are attentive to your needs, and they will go the extra mile for you because they want to keep your business.United entered bankruptcy in 2002 and emerged in 2006.But imagine, if you will, that you and a group of people are put in a room and asked to design a just airline.You do not know if you will be young or old, you do not know your gender identity or sexuality, you do not know if you will be able-bodied, you do not know your race, ethnicity, religious views, native language.That amounts to almost half of the company’s total US-based frontline workforce.“Our primary goal throughout this crisis has been to ensure United – and the jobs it supports – are here when customers are flying again,” United Airlines said in a statement to employees.United Airlines is receiving $5bn of those funds and won’t make any cutbacks until 1 October, it said.“It is the first of the major US airlines to come out and clearly say ‘we are going to be a much smaller airline after all of this is over’.” “Now we’ve got to see what the other airlines will do but we’re expecting the industry as a whole is going to shrink and it’s probably going to be about a third smaller than it was coming into this crisis,” she added.Last week, American Airlines said it could have 20,000 more front-line workers than it needs to operate, but that not all of them would be furloughed in October.Thomas Pallini/Business Insider The big four US airlines are split down the middle in how they're handling social distancing onboard their aircraft.Delta and Southwest are blocking middle seats and limiting bookings while American Airlines and United Airlines are only offering free flight changes.Visit Business Insider's homepage for more stories .The big four US airlines can be neatly divided into two camps when it comes to their social distancing policies: those that block middle seats and those that don't .See the rest of the story at Business Insider See Also: Private jet CEO reveals why his company acquired a competitor after taking $20 million PPP funds and furloughing employees at the start of the pandemic United Airlines will warn 36,000 workers of possible layoffs — more than a third of its entire workforce A Delta flight was forced to make an emergency landing after hail made the plane's nose collapse SEE ALSO: I flew on the 3rd and 4th-largest US airlines to see how they're handling the pandemic and the difference was night and day DON'T MISS: A UK charter airline is selling $10,000 seats for flights from London to Barbados on a massive Airbus private jet that requires a COVID-19 test – see inside On the other, a surge of airline layoffs and furloughs would occur just before November elections, meaning voters would go to the polls with an industry-wide economic disaster on their mind.“And that will be included in the last jobs report before this election.” Treasury Secretary Steven Mnuchin, interviewed Thursday on CNBC, stopped short of making a specific request to Congress but reiterated the need to keep the industry from going under, saying, “they’re going to need more help.” [ Transport workers still seek enforceable COVID-19 rules ] The $2 trillion COVID-19 spending bill approved in March included $25 billion in grants to passenger airlines and $25 billion in loans to passenger airlines.But with all bets off after Sept 30, two labor unions — the Air Line Pilots Association and the Association of Flight Attendants-CWA — are calling for Congress to consider a second round of airline aid.‘Depressed’ environment “The reality is that United simply cannot continue at our current payroll level past Oct. 1 in an environment where travel demand is so depressed,” the airline, which collected $5 billion in payroll support through that March spending bill, told employees.“We’re going to be hearing from a lot more airlines about what they are predicting in terms of the numbers of layoffs for the fall,” she said.
    
    Are you new?Little Known Facts About Spirit Airlines Contact Number.Little Known Facts About Spirit Airlines Contact Number.We discover this tactic most regarding.Where by a flight is cancelled, we stimulate our customers to discuss their alternatives with us, such as booking on another date.”Your new travel date must happen in a yr of the original travel date and you simply’ll really have to pay back any difference in fares.In its place, Once you cancel you’ll Possess a credit history While using the airline, which can be employed over a potential JetBlue flight.“We’re serving to customers with impending vacation plans rebook gratis, to change their desired destination or date of travel, or get a voucher and maintain their ticket open, up until 30th April 2021.Des suggests: It truly is about to damage the airlines with the global reduction in traveling over the subsequent handful of monthsAll passengers with MilleMiglia award ticket, who make your mind up to not travel, can request by 31 August a change of read more reservation or choose to reclaim their miles and get a refund of your airport taxes.What occurs if we get rid of our work and cannot journey?“Virgin Atlantic really wants to offer you just as much versatility as you can for our customers impacted with the fast accelerating Covid-19 circumstance,” Associates with the airline reported in a press release to The Washington Publish.Earlier, Spirit experienced released a statement for the media Having said that: “Considering that late January, we’ve been presenting flexible journey choices to our Friends who arrive at out with worries about the COVID-19 coronavirus.The waiver is valid for primary tickets that have been purchased concerning March 1st and April fifteenth, 2020 Airline face a long path to recovery By - July 20, 2020 SINGAPORE, 20 July 2020: A major industry poll of professionals from across the global aviation industry reached a consensus that widespread recovery of the industry could be up to three years away.That was the consensus from a major industry poll conducted as part of FlightPlan: Charting a Course into the Future, an online broadcast by Inmarsat and the Airline Passenger Experience Association (APEX).More than half of respondents (60%) expect a recovery period between 18 months to three years.Half of the respondents (45%) believe that in terms of passenger experience, the crisis will only cause a short-term reduction in investment, and almost a third (32%) believe there will be an overall increase in investment.Contactless catering was highlighted by 57 per cent as being important during the recovery period, and almost half (44%) expect to see empty middle seats as a standard feature of the passenger journey in the coming months despite contrary guidance given by IATA in May.By: WicklowNews 4 days ago Wicklow Sinn Féin TD John Brady has called on the government to step in and prevent airlines punishing people for taking the decision not to holiday abroad this year due to the Covid-19 pandemic.Deputy Brady said: “I have had angry and disappointed families from right across Wicklow contacting me after they have had to cancel family holidays.Many of these holidays have been paid for over the last year.” “Thousands of people have already missed their summer holidays abroad due to the ongoing pandemic.“The Irish Travel Agents Association estimates that over €800,000 a day is being lost by people who had foreign holidays booked in advance of this pandemic, but now cannot go.“People want to do the right thing and protect their families and wider communities from Covid-19.They are rightly taking a cautious approach, but they are being financially punished for this, as some airlines are refusing to provide refunds or flight vouchers, while others are charging exorbitant rescheduling fees.“I’ve been contacted by hundreds of people about this issue, with one individual losing €3,000, another losing €6,700 on a holiday that they had to cancel.“The airline sector is facing a huge challenge, and the state does need to address this in a proportionate and fair way.But the sector’s recovery cannot be based on taking money from families who make the decision not to holiday abroad this year, in line with advice from public health experts.“Airlines should be offering refunds, flight credit or free rescheduling, while travel insurance companies should be compensating those who have already lost money due to the restrictions on travel.“If airlines and travel insurance companies refuse to change tack, the government must take action, they cannot sit on the side-lines and allow this rip-off to continue.” Sign up to our weekly newsletter Coventry 15.8.99 copyright © 1999 Chris Chennell Discount carrier Flair Airlines will soon have flights going into and out of Saskatoon and Regina.In a release Thursday, the airline said it will begin service at the Regina airport on Aug. 24 and at the Saskatoon airport Aug. 27.Right now with everything going on with COVID, there are not a lot of flights currently operating out of the airport, so we’re limited for what’s available for seat capacity, so it’s great to get some of that capacity into the market as people do travel a little bit,” said CJ Dushinksi.“You know the markets some of the airlines would traditionally fly into in the United States or elsewhere have been closed, so our Canadian airlines are looking for other opportunities domestically to pick up capacity.” Dushinski says at its worst, the Saskatoon airport was down to about 20 passengers a day back in April.“What Flair is doing is they’re adding service twice a week (to Toronto and Vancouver).” Bogusz said because Flair is a discount airline, he anticipates that Regina travellers will be looking forward to checking out the airline instead of “getting in the car and trying to drive to one of these places.” “We’re welcoming (Flair) with open arms and we’re certainly going to let the community know that we have another airline serving our market,” he said.
    
    When I started using commercial airliners years ago, people would dress up in their best business suits.Also, on the planes, one encounters smelly people who don’t know what deodorant is or refuse to use it, fat slobs who take up more than one seat, people with tattoos, who look like they are from a motorcycle gang, passengers who persist in talking loudly on their cell phones, when it is prohibited to do so, and passengers who like to kick the seat in front of them.Essentially, there is very little difference onboard a commercial aircraft, or riding the NYC subway.In summary, if anyone can avoid flying commercial, they should do so.Reply Just too bad that animals are allowed to fly as humans.Reply These are not ladies they are animals.Reply Not many people could fly private.Reply Yes, or the dregs of society that wait outside in the pouring rain to get into Target on Flatbush Ave or Marshalls in Gerritsen.What is so important in the store to stand out in the rain like a worthless dog?Are they giving out gold nuggets?Anyway, these are the same people you could be sitting next to on a cheap crap airline like Spirit.Reply Spirit Airlines is probably one of the lousier carriers in the US, but that does not justify the actions of these poople… Reply Black Lives Matter, does that pertain to animals as well?I hope they go to jail, see if they like it better there, oh wait, they are black hmmm, maybe is was just a misunderstanding…;no jail time-alls good Reply But what does this mean for you?For starters, the miles you earn from the best airline credit cards will likely become more valuable.What we would expect from a partnership like this would be similar to what we’ve seen American Airlines do with Alaska Airlines.They will likely include the ability to redeem miles from either program for award flights on both airlines.Could it even be possible to use JetBlue points for international American Airlines flights??)is YOU.You?
    
    

### Sentiment Analysis using NLTK Sentiment Intensity Analyzer


```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
```


```python
i=0

sent_analyzer = SentimentIntensityAnalyzer()
for sentence in weekly_summary:
    scores = sent_analyzer.polarity_scores(sentence)
    print("Week " + str(i+1) + " Sentiment Analysis:")
    i+=1
    for k in scores:
        print('{0}: {1}, '.format(k, scores[k]), end='')
    print()
```

    Week 1 Sentiment Analysis:
    neg: 0.032, neu: 0.879, pos: 0.089, compound: 0.996, 
    Week 2 Sentiment Analysis:
    neg: 0.046, neu: 0.89, pos: 0.064, compound: 0.7424, 
    Week 3 Sentiment Analysis:
    neg: 0.06, neu: 0.87, pos: 0.07, compound: 0.829, 
    Week 4 Sentiment Analysis:
    neg: 0.031, neu: 0.827, pos: 0.142, compound: 0.9901, 
    


```python
score_list = []
sent_list = []
sent_analyzer = SentimentIntensityAnalyzer()
for sentence in weekly_summary:
    scores = sent_analyzer.polarity_scores(sentence)
    for k in scores:
        sent_list.append(k)
        score_list.append(scores[k])
```


```python
week_list = ['Week 1','Week 1','Week 1','Week 1',
             'Week 2', 'Week 2','Week 2','Week 2',
             'Week 3','Week 3','Week 3','Week 3',
             'Week 4','Week 4','Week 4','Week 4']
```


```python
sent_df = pd.DataFrame(list(zip(week_list, sent_list, score_list)),columns =['Week','Sentiment', 'Score'])
```


```python
neg_df = sent_df[sent_df['Sentiment'] == 'neg']
```


```python
import matplotlib.pyplot as plt
neg_df.plot(kind= 'bar',x='Week', y= 'Score', title = 'Negative Sentiment per Week')
plt.show()
```


![png](/assets/img/sentimentanalysis/output_19_0.png)



```python
pos_df = sent_df[sent_df['Sentiment'] == 'pos']
```


```python
import matplotlib.pyplot as plt
pos_df.plot(kind= 'bar',x='Week', y= 'Score', title = 'Positive Sentiment per Week')
plt.show()
```


![png](/assets/img/sentimentanalysis/output_21_0.png)



```python

```
