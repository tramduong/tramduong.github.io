
# Topic Classification

### Load Word2Vec model and Other Functions


```python
import gensim, operator
from scipy import spatial
import numpy as np
from gensim.models import KeyedVectors

model_path = '/github/'
```


```python
def load_wordvec_model(modelName, modelFile, flagBin):
    print('Loading ' + modelName + ' model...')
    model = KeyedVectors.load_word2vec_format(model_path + modelFile, binary=flagBin)
    print('Finished loading ' + modelName + ' model...')
    return model

model_word2vec = load_wordvec_model('Word2Vec', 'GoogleNews-vectors-negative300.bin.gz', True)
#model_fasttext = load_wordvec_model('FastText', 'fastText_wiki_en.vec', False)
```

    Loading Word2Vec model...
    Finished loading Word2Vec model...
    


```python
def vec_similarity(input1, input2, vectors):
    term_vectors = [np.zeros(300), np.zeros(300)]
    terms = [input1, input2]
        
    for index, term in enumerate(terms):
        for i, t in enumerate(term.split(' ')):
            try:
                term_vectors[index] += vectors[t]
            except:
                term_vectors[index] += 0
        
    result = (1 - spatial.distance.cosine(term_vectors[0], term_vectors[1]))
    if result is 'nan':
        result = 0
        
    return result
```


```python
# function checks whether the input words are present in the vocabulary for the model
def vocab_check(vectors, words):
    
    output = list()
    for word in words:
        if word in vectors.vocab:
            output.append(word.strip())
            
    return output
```


```python
# function calculates similarity between two strings using a particular word vector model
def calc_similarity(input1, input2, vectors):
    s1words = set(vocab_check(vectors, input1.split()))
    s2words = set(vocab_check(vectors, input2.split()))
    
    try:
        output = vectors.n_similarity(s1words, s2words)
    except:
        output = 0
    return output
```

### Data Exploration with Sematic Similartity


```python
# Load json data into list of dictionaries
import json
google_json=open("/github/google_deduplicated.json").readlines()
```


```python
# Prints the number of newsfeeds (JSON objects) in the collection
newsfeeds_read = []
for line in google_json:
    newsfeeds_read.append(json.loads(line))
```


```python
# Create randome title list 
import random
title_list = [x['title'] for x in newsfeeds_read]
article_title = random.choice(title_list)
```


```python
# Create similartity score list 
sim_list=[]

for i in title_list:
    try:
        sim = calc_similarity(article_title, i, model_word2vec)
        sim_list.append(sim)
    except:
        #sim_list.append(0, 'ERROR ZERO DIV '+i)
        sim_list.append(0)
```


```python
# Finidng most similar titles in a descending order of similarity scores
import pandas as pd
df = pd.DataFrame(list(zip(title_list, sim_list)),columns =['Title', 'Similarity'])
most_similar = df.sort_values(['Similarity'], ascending=0)
```


```python
most_similar[:600]
```

### Extractive Text Summarization  


```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

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
feed_text = []

for feed in google_json:
    a = json.loads(feed)
    feed_text.append(a['text'])
```

### LDA topics (Assignment 8)
       {0: ['cloud', 'technology', 'team', 'digital', 'health', 'network', 'platform', 'design', 'tool', 'develop'],
        1: ['page', 'https', 'website', 'web', 'site', 'chrome', 'browser', 'link', 'file', 'user'],
        2: ['trump', 'podcast', 'president', 'tech', 'law', 'privacy', 'government', 'order', 'tweet', 'claim'], 
        3: ['police', 'black', 'coronavirus', 'city', 'health', 'case', 'officer', 'floyd', 'protest', 'pm'],
        4: ['million', 'india', 'per', 'pay', 'digital', 'increase', 'billion', 'stock', 'businesses', 'revenue'],
        5: ['game', 'good', 'really', 'lot', 'school', 'nt', 'things', 'students', 'something', 'every'],
        6: ['android', 'phone', 'apps', 'apple', 'de', 'game', 'device', 'pixel', 'store', 'devices'],
        7: ['log', 'smart', 'tv', 'amazon', 'music', 'voice', 'stream', 'assistant', 'youtube', 'never']}

### Manual Data Exploration


```python
title_list
```


```python
# BUILD YOUR OWN TAXONOMY BASED ON LDA and MANUAL DATA EXPLORATION
topic_taxonomy = {
    "Business" : 
    {
        "Business Competition" : "competition rivalry consumer customer market share conflict fight",
        "Business Operations" : "reopen operations integration program recruiting sales performance investors",
        "Business Expansion" : "growth market arena barrier conglomerate takeover buyout buy venture pilot partnership partner",
        "Business Tech" : "innovate blockchain cloud enterprise applications public robots automation IoT AI research",
        "Business Law" : "sue countersue law lawyer illegal espionage settlement contract breach nda disclosure trade secrets",
        "Business Performance" : "stock market revenue dividend nasdaq ticker dive increase invest investors"
    },
    
    "Products" :
    {
        "Product Failure" : "recall defect defective failure fail poor issue problem bug",
        "Product Trends" : " innovate innovative virtual platform cloud technology market team digital health network platform design tool",
        "Product Release" : "new release unveil announce offer launch expand develop"
    },
    
    "Incident" :
    {
        "Disease Outbreak": "Covid-19 health quarantine corona virus coronavirus impact cases recover survivors essential update",
        "Violent Incident": "police city case officer shooting death murder killed robbery crime",
        "Protests" : "black lives matter protest march george floyd racism occupy shooting breonna taylor minority rights",
        "Security Breach": "security hack breach hacker password release data confidential information"
    },
    
    "Technology" :
    {
        "Mobile Tech" :  "android ios apple iphone samsung huawei google pixel device 5g 4g network tower mobile phone cellphone smartphone",
        "Gaming Tech" : "stadia play store microsoft xbox playstation sony nintendo switch vr virtual reality game gaming",
        "Streaming Tech" : "video youtube netflix hulu disney+ starz hbo amazon prime twitch stream on demand zoom tiktok instagram",
        "Payment Tech" : "credit card contactless venmo paypal google pay apple pay square cash"
    },
    
    "Government" :
    {
        "President" :  "trump obama president presidential 45th veto election",
        "Gov Regulations" : "embargo law privacy regulation governance ban injunction",
        "Gov Politics" : "congress representatives house senate judicial judge rules ruling bill economy",
        "Gov Investigation" : "probe investigate investigation allege allegation FBI CIA charges accusation",
        "Gov Relations" : "washington europe united nations china beijing india embassy military dispute refugee visa passport"
    },

    "Artificial Intelligence" : 
    {
        "AI Assistant" : "AI virtual log voice match recognition activation amazon alexa siri google home activate personal assistant notes computer",
        "AI Market" : "AI biotech fintech insurance vehicles healthcare detection self driving global cloud",
        "AI Home" : "AI enabled smart TV music stream lights camera home security google amazon alexa echo mini nest network"
    }
}
```


```python
# function takes an input string, runs similarity for each item in topic_taxonomy, sorts and returns top 3 results
def classify_topics(input, vectors):
    feed_score = dict()
    for key, value in topic_taxonomy.items():
        max_value_score = dict()
        for label, keywords in value.items():
            max_value_score[label] = 0
            topic = (key + ' ' + keywords).strip()
            max_value_score[label] += float(calc_similarity(input, topic, vectors))
            
        sorted_max_score = sorted(max_value_score.items(), key=operator.itemgetter(1), reverse=True)[0]
        feed_score[sorted_max_score[0]] = sorted_max_score[1]
    return sorted(feed_score.items(), key=operator.itemgetter(1), reverse=True)[:1]
```


```python
lst1= []
lst2=[]
for topic in topic_taxonomy.keys():
    lst1.append(topic)
    lst2.append(list(topic_taxonomy[topic].keys()))
```


```python
newlst= list(zip(lst1,lst2))
```


```python
newlst[1][1]
```




    ['Product Failure', 'Product Trends', 'Product Release']




```python
newlst
```




    [('Business',
      ['Business Competition',
       'Business Operations',
       'Business Expansion',
       'Business Tech',
       'Business Law',
       'Business Performance']),
     ('Products', ['Product Failure', 'Product Trends', 'Product Release']),
     ('Incident',
      ['Disease Outbreak', 'Violent Incident', 'Protests', 'Security Breach']),
     ('Technology',
      ['Mobile Tech', 'Gaming Tech', 'Streaming Tech', 'Payment Tech']),
     ('Government',
      ['President',
       'Gov Regulations',
       'Gov Politics',
       'Gov Investigation',
       'Gov Relations']),
     ('Artificial Intelligence', ['AI Assistant', 'AI Market', 'AI Home'])]




```python
for i in title_list:
    output = classify_topics(i,model_word2vec)

    print(i, output)
```


```python
output_list = []
titles=[]

for i in title_list:
    titles.append(i)

for i in title_list:
    output = classify_topics(i,model_word2vec)
    output_list.append(output)
```


```python
labels=[]
values=[]

for x in range(len(output_list)):
    labels.append(output_list[x][0][0])
    

for x in range(len(output_list)):
    values.append(output_list[x][0][1])
```


```python
df = pd.DataFrame(list(zip(titles, labels, values)),columns =['Title', 'Topic','Value'])
```


```python
# Topics and subtopics with 10 closest titles

newlst[1][0]
len(newlst)
for x in range(len(newlst)):
    print("Topic: ", newlst[x][0])
    workinglst = newlst[x]
    for i in range((len(workinglst)+1)):        
        print('\n\nSubtopic: ',workinglst[1][i],'\n')
        working =df[df['Topic']==workinglst[1][i]]
        short_list = working.sort_values(by='Value', ascending=0)[:10]
        for i in range(10):
            print(short_list.iloc[i][2],short_list.iloc[i][0])
    
```

    Topic:  Business
    
    
    Subtopic:  Business Competition 
    
    0.5479925274848938 Reflecting on DuckDuckGo's rise as the privacy-focused search engine and the possibility of increased market share because of EU regulatory pressure (Matt Burgess/WIRED UK)
    0.5377374887466431 Sources: India's antitrust body is looking into allegations that Google is abusing its market position to unfairly promote its mobile payments app Google Pay (Reuters)
    0.5278651714324951 Consumer experience of online grocery must improve, says app start-up
    0.5157110691070557 Google faces India mobile payments competition claim
    0.5131257176399231 Google sues Sonos, escalating wireless speaker battle amid trade panel probe - Midwest Communication
    0.5097876787185669 WhatsApp Pay struggles in India’s mobile payments market dominated by Google, Walmart, Paytm
    0.5092771053314209 Disney+ beats Netflix in consumer satisfaction survey
    0.5074613094329834 India’s telecom wars to heat up? Google eyes 5% stake in struggling Vodafone-Idea – report
    0.5002304315567017 Google downplays value of news in fight over $1bn media fund
    0.5000556707382202 Amazon, Facebook and Google turn to deep network of political allies to battle back antitrust probes
    
    
    Subtopic:  Business Operations 
    
    0.5958307981491089 OnePlus banks on India R&D to drive product innovation across markets | Business Standard News
    0.560510516166687 Covid-19: Dubai to reopen business activities from May 27
    0.559809148311615 Coronavirus: US markets fall amid increase in virus cases and fading hopes for quick economic recovery | Business News | Sky News
    0.5548047423362732 A CEO who recruits executives for Google and Siemens sees the new 'interim economy' likely replacing full-time jobs with gig work
    0.5524231791496277 External factors affecting e-commerce efficiency and sales growth
    0.5508431792259216 Sources: Apple believes Apple Music held a 50%+ weekly US market share of premium hip-hop streams for new releases in 94 out of the last 96 weeks (Tim Ingham/Music Business Worldwide)
    0.5504409074783325 HurixDigital hires ex-Google Executive Sameer Bora as EVP - Operations and Delivery, to focus on the online learning needs of educational institutions
    0.547295868396759 Property Transfers - 12 sales for $4.7 million
    0.5466917753219604 The $20 billion self-driving startup Cruise is adding to its leadership team even as autonomous-vehicle companies are hitting the brakes during the pandemic (GM)
    0.5444566011428833 Macy’s says reopened stores luring back customers
    
    
    Subtopic:  Business Expansion 
    
    0.6505877375602722 Sources: Netflix is in talks with Network18, the media unit of Indian conglomerate Reliance Industries, to establish a multiyear content creation partnership (Reuters)
    0.6167488098144531 KKR, Cinven and Providence launch takeover bid for MásMóvil
    0.6129469871520996 Report: Jeff Bezos is buying a stake in UK digital supply chain startup Beacon
    0.5918633341789246 Decora: UK investment fund buys £10m stake in Lisburn blinds manufacturer
    0.5913022756576538 Austin developer continues buying spree with apartment complex deal - Business - Austin American-Statesman - Austin, TX
    0.5892524719238281 Anthos: Google's bid for Kubernetes differentiation ... and market share
    0.5877708196640015 North West-based entrepreneur secures £8.15m to expand property portfolio
    0.5737400650978088 Amazon reportedly wants Airtel stake for $2 billion ⁠— and a partnership like Jio-Facebook and possibly, Google-Vodafone Idea |
    0.5725589394569397 Google eyes entry into Indian telecom market with 5% stake purchase in Vodafone Idea: Report
    0.5722717642784119 More giant foreign firms plan to shift investment into Vietnam
    Topic:  Products
    
    
    Subtopic:  Product Failure 
    
    0.5547114610671997 mikenov on Twitter: #Covid19Review: #Coronavirus and failure of #USIntelligence Service… covid-19-review.blogspot.com/p/coronavirus-…
    0.5474531650543213 Chrome 85 is finally fixing this bothersome problem
    0.5268161296844482 Common Samsung Galaxy S8 problems and how to fix them
    0.5166292786598206 Google says 70 percent of serious security bugs are memory safety issues
    0.5056346654891968 Mickoski: We have a complete systemic failure in dealing with the crisis
    0.5028896331787109 Ford recalls about 2.5 million vehicles for latch, brake troubles
    0.48348498344421387 Galaxy S10 owners should update their phones now to fix critical issues
    0.48150572180747986 Tinder error 5000? Fix it in 5 easy-to-follow steps
    0.477628231048584 Samsung updates four-year-old, unsupported Galaxy S7 series to fix critical vulnerability
    0.4765341877937317 Why you should avoid self-diagnosing health problems online
    
    
    Subtopic:  Product Trends 
    
    0.7470463514328003 NaaS platform launched in France claiming fast, scalable business access to cloud
    0.7408857345581055 Splunk announces new machine learning enhancements to core platform solutions
    0.7369316816329956 Douglas Alexandra, a digital marketing and brand activation firm announces a new suite of tools that help small businesses manage their online reputation and increase sales
    0.7108955383300781 Safaricom, Google partner to leverage on digital data capability
    0.7096217274665833 Google partners with WWF on digital data platform for sustainable fashion sourcing
    0.7066527605056763 Ecobank, Google partner to offer digital solutions to SMEs
    0.6971423029899597 Sponsored: Leveraging Google Cloud to create readily scalable online games
    0.6866495013237 Ecobank Group and Google collaborating to deliver digital solutions tailored for Ecobank’s Small and Medium-sized Enterprises (SMEs) customers
    0.680660605430603 Phunware launches mobile solution to give small businesses access to Multiscreen-as-a-Service features
    0.679611325263977 GoDaddy buys content creation app Over, plans to integrate features into its product suite
    
    
    Subtopic:  Product Release 
    
    0.6704382300376892 WarnerMedia announces further launch partners for HBO Max
    0.5834186673164368 Megaport unveils European expansion with deployment of NaaS solution in France
    0.5717743635177612 AirAsia’s Redbeat Academy unveils SME development programme
    0.570824921131134 One of Stadia's most anticipated features is launching alongside its ambitious new exclusive, Crayta | GamesRadar+
    0.565302312374115 Omantel to announce new chairman soon
    0.5577467679977417 Realme X50t key specifications, launch date revealed in fresh leak - Technology News
    0.5573707222938538 Pixel Kicks launches new site for StormForce
    0.542980432510376 Cyberpunk 2077 apparently won’t be available on Stadia at launch
    0.5361126065254211 Ford to unveil hotly anticipated new Bronco this summer
    0.5353850722312927 Google launches new search highlighting feature
    Topic:  Incident
    
    
    Subtopic:  Disease Outbreak 
    
    0.7606136798858643 First coronavirus infection case detected in Tanahun
    0.7191710472106934 Covid-19: Sewage may contain the key to stopping new coronavirus outbreaks
    0.6947405338287354 List 3/3 of sports events affected by coronavirus pandemic
    0.6760573983192444 China coronavirus panic: City of 2.8m gripped by 'mysterious' new outbreak - breaking-news-today.org/
    0.6747081279754639 Lockdown 4.0 tops Google’s search trends as people look for containment zones and ways to survive the coronavirus pandemic
    0.6699112057685852 Charity running events become difficult during coronavirus pandemic
    0.6683411598205566 Apple's iOS update is here — and it includes coronavirus contact tracing
    0.6631852388381958 Turkey opens new hospitals as daily coronavirus cases drop
    0.6610223054885864 Amazon workers are tracking coronavirus cases themselves
    0.6586374640464783 The coronavirus pandemic highlights the need for a surveillance debate beyond 'privacy'
    
    
    Subtopic:  Violent Incident 
    
    0.7656497359275818 Police investigate shooting death of 77-year-old Fayetteville man
    0.7360484600067139 Attempted robbery victim injured as police hunt two men and teenager
    0.7228763103485107 Ex-Indiana prisons worker charged with murder in stabbings
    0.7226009964942932 Attempted murder investigation launched after man stabbed in chest
    0.707332968711853 Police seek connection between stolen vehicle and CT murder suspect | BRProud.com | WVLA | WGMB
    0.7067210078239441 Dundonald: Man charged with attempted murder after stabbing
    0.7048490643501282 Police name man accused of stabbing
    0.7035213112831116 Murder probe launched after man stabbed to death following mass brawl
    0.6967706680297852 Police constable accused of abetting wife’s suicide
    0.682953417301178 Police appeal after reported stabbing
    
    
    Subtopic:  Protests 
    
    0.5996947288513184 “house judiciary committee” – Google News: George Floyd: Barr denies police are systemically racist as protests sweep US – live – The Guardian
    0.5871340036392212 Thousands in London decry racial injustice, police violence
    0.5859026312828064 ‘I can’t take the lies anymore!’: Glenn Beck goes on FIERY rant over left’s civil rights HYPOCRISY
    0.583642840385437 Protests against racism and in support of the blac
    0.568368136882782 Here's what tech companies have said they'll do to fight racism in wake of George Floyd protests
    0.561687707901001 George Floyd death: Policeman hugs crying black girl who asked 'are you gonna shoot us?' during protests
    0.5596727132797241 As George Floyd protests continue, Amazon, Google pledge millions to racial justice organizations
    0.555757462978363 Thousands of demonstrators protesting police brutality expected in DC Saturday
    0.5496022701263428 ANC in Nelson Mandela Bay and allies protest against racism
    0.5457563400268555 Support this black business if black lives really matter to you
    Topic:  Technology
    
    
    Subtopic:  Mobile Tech 
    
    0.7558438777923584 Finally, a Google Android TV dongle that doesn’t need a smartphone to work
    0.7085289359092712 HTC Desire 20 Pro smartphone gets Bluetooth and WiFi certifications
    0.7005177736282349 iPhone 后盖玻璃与金属中框之间的塑料
    0.6934922933578491 ZTE's newest dirt-cheap smartphone comes with a removable battery and Android 10
    0.6877042651176453 How to transfer contacts between iPhone and Android devices
    0.6769777536392212 FCC confirms Google Android TV dongle and multi-function remote
    0.6755363345146179 Microsoft Your Phone app now lets you control phone music - BGR India
    0.6741026043891907 The emails apps the iPhone and iPad
    0.6689994931221008 Fortnite 12.61.0_13498347 Apk android
    0.6688302755355835 Lockourier optimized with mobile app
    
    
    Subtopic:  Gaming Tech 
    
    0.637615442276001 NVIDIA GeForce NOW has over 2k games from 200 publishers
    0.6274961233139038 DIY VR headset lets you enjoy PC games thanks to a little Arduino
    0.6273241639137268 nvidia - Video playback freezes after hardware upgrade - Ask Ubuntu
    0.6223659515380859 C'era una volta una console chiamata Terminator 2 - articolo
    0.6177246570587158 Razer Kishi review: A smooth play for mobile gaming - CNET
    0.6146950721740723 Microsoft Edge has a cute game to play when you're offline
    0.602924108505249 How to use your DSLR as a webcam for a PC or Mac - CNET
    0.5992048978805542 Chromebook owners get Android versions of DOOM and DOOM II for free — Here’s how
    0.5987446308135986 Epic Games store to launch on Android, iOS devices - why it makes sense
    0.5984976291656494 Microsoft’s new app will let parents monitor their child’s Xbox gaming sessions
    
    
    Subtopic:  Streaming Tech 
    
    0.6549007892608643 There is now a video of the upcoming Google Android TV streaming device
    0.653354287147522 Youtube TV reports "This video format is not supported." in Firefox - Ask Ubuntu
    0.6488375663757324 youtube vs tiktok: Roast, rage, jealousy, cringe. Who'll have the last laugh in YouTube vs TikTok's online 'class wars'?
    0.6488205194473267 Amazon offerte oggi (fino a -69%): speciale bebè e monopattini (con bonus), MateBook D, iPhone 11 779€, AirPods 129€, TV 4K e altre splendide promo
    0.6389066576957703 7 light-hearted movies on Amazon Prime Video and Netflix to forget about your work stress
    0.6313621997833252 Youtube New Feature: YouTube rolls out new 'Chapters' feature on app to navigate long videos
    0.6280354261398315 Google Camera app update allows Pixel 4 and 4XL users capture videos at 8x zoom and more
    0.6269862055778503 From Hulu to Youtube to Yolamovies, the 10 Best sites for Free Streaming Movies
    0.6262041926383972 Tweet2Video: Twitter video downloader
    0.6257377862930298 Google releases online video trend data in India; here’s all you need to know
    Topic:  Government
    
    
    Subtopic:  President 
    
    0.6246024966239929 Democrats want to restrict political ad targeting ahead of the 2020 election
    0.612710177898407 Attempted hacks of Trump and Biden campaigns reveal race to disrupt election
    0.6073259115219116 Democratic governors to GOP counterparts: oppose deployment
    0.5882061123847961 Ben Salango wins Democratic nomination for governor in West Virginia primary election
    0.5852646827697754 Joe Biden wins Democratic presidential primary in Montana
    0.5788743495941162 Tuesday is election day in Iowa. Here's what to know | Politics and elections | qctimes.com
    0.5624212026596069 US voters face virus, social unrest in primary elections « TheEricErbShow
    0.5556595325469971 Chinese, Iranian hackers targeting Trump, Biden election campaigns, says Google
    0.5523809194564819 Biden looks to clinch nomination as 7 states, DC vote
    0.5472110509872437 Teacher beats West Virginia Senate president in GOP primary
    
    
    Subtopic:  Gov Regulations 
    
    0.6370396614074707 Twitter, WhatsApp sanctions loom in EU privacy crackdown
    0.6069628000259399 California attorney general submits regulations for approval under privacy law
    0.5887899398803711 New Google rule bans discriminatory targeting for housing ads
    0.5748873353004456 Google says use existing EU laws to govern AI
    0.5682357549667358 Court orders dismissal of Trump Muslim travel ban challenges
    0.5577337741851807 Japan enacts law toughening regulations on tech giants
    0.5568976402282715 Tech firms ask govt to frame online harm rules within legal scope - Tech - DAWN.COM
    0.5537392497062683 Google to face $5bn privacy lawsuit as consumer craving for secrecy increases
    0.5529623031616211 Airlines launch legal action against UK quarantine policy
    0.5512072443962097 Trump to order review of law protecting social media firms after Twitter spat: report | TheHill
    
    
    Subtopic:  Gov Politics 
    
    0.6460444927215576 Supreme Court rules in FOIA case long delayed by lawmaker
    0.6400690078735352 “house judiciary committee” – Google News: Democrats unveil broad police reform bill, pledge to transform law enforcement – The Washington Post
    0.6335254311561584 “house judiciary committee” – Google News: House plans dramatic action on police reform – Axios
    0.6305216550827026 “house judiciary committee” – Google News: House GOP leaders vow to keep fighting remote voting during pandemic – CNN
    0.6242994070053101 “house judiciary committee” – Google News: Gov. Laura Kelly backs negotiated COVID-19 bill as legislators meet in special session – Leavenworth Times
    0.618450939655304 Appeals court ruling suggests little legal traction for Trump's anti-Twitter campaign
    0.6105098724365234 Trump sued over social media directive Stimulus watchdog confirmed by Senate Census offices closed due to protests SAT administrator scraps home testing
    0.6064184904098511 Kosovo lawmakers vote in new center-right prime minister
    0.6023396253585815 Kansas to ask Supreme Court to save voter citizenship law
    0.5889809131622314 Lawmakers question federal prisons’ home confinement rules
    Topic:  Artificial Intelligence
    
    
    Subtopic:  AI Assistant 
    
    0.6345881819725037 Viral tweet reminds us how much better voice typing is on Google Pixel vs. iPhone [Video]
    0.6338615417480469 Using the translate feature with screen reading software requires having the synthesizer for the foreign language you request already installed on your screen reader. NVDA users should use the e-Speak NG synthesizer. JAWS users should download and install
    0.6239581108093262 Gmail login security alert messages now embed an ad for Google Chrome
    0.6137943267822266 Google Voice added to Gmail web app, now lets you transfer calls on all platforms
    0.6136289238929749 Is it real new WhatsApp bot works to recognise fake news
    0.6061913967132568 How to accept a Google Calendar invite on your computer or mobile device
    0.598939061164856 Google's chat app may finally get end-to-end encryption
    0.5978620052337646 How to reset my keyboard mapping file - Ask Ubuntu
    0.5945535898208618 Google lets you text ‘Plus Codes’ that reveal your EXACT location – if your actual address is hard to find
    0.594058632850647 Google details how it is using a variety of AI techniques to improve the translation quality for languages that don't have a copious amount of written text (Kyle Wiggers/VentureBeat)
    
    
    Subtopic:  AI Market 
    
    0.6193392276763916 Hack-for-hire firms from India targeting financial services, healthcare amid COVID-19: Google | Banking
    0.6000860929489136 Opinion: Artificial intelligence should be integrated into our workforce
    0.5910974144935608 AI tools could improve fake news detection by analyzing users’ interactions and comments
    0.5891197919845581 Health tech start-up introduces GPS tracking for medical equipment in the community
    0.5867326259613037 Artificial Intelligence (AI) in Insurance Market Global Outlook 2020 : Google, Microsoft Corporation, Amazon Web Services Inc - 3rd Watch News
    0.5712171792984009 News roundup: LabCorp CRO boosts Medable, Propeller Health gains 510(k), EU’s 34 medtech startups, Amazon’s healthcare moves, Google’s Arizona privacy lawsuit
    0.5708098411560059 AI systems are worse at diagnosing disease when training data is skewed by sex
    0.5661823749542236 EU calls for greater regulation of US tech companies
    0.560026228427887 Hack-for-hire firms targeting financial services, healthcare | Communications Today
    0.555166482925415 How Mayo Clinic manages patient data privacy, consent in licensing deals with tech companies
    
    
    Subtopic:  AI Home 
    
    0.6719580888748169 Here's what Google's new Android TV dongle (and remote control) might look like
    0.6633495092391968 Massive Google Android TV leak continues with video teaser of where the interface might be heading
    0.6598238348960876 'Can you Chromecast Apple Music?': How to connect your Google streaming device with Apple's music library
    0.6541474461555481 Altibox offers Google Nest Wi-Fi gadget for wireless TV viewing
    0.6492516994476318 Here’s our best look yet at Google’s new Android TV streaming device
    0.644879937171936 Google creates ‘social distancing’ AR app that puts virtual 2-metre ring around you using camera
    0.642866849899292 Netatmo outdoor security camera has a siren built in, works with Alexa, Google Assistant, Siri - CNET
    0.6426864266395569 Android 11's power menu smart home controls enabled by Google Home app
    0.6419627666473389 Google TV dongle & remote leaked, could launch soon
    0.6395019888877869 How Google tracks internet browsing history even when Incognito mode is on
    
