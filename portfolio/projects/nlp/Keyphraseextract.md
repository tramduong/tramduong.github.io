
# Key Phrase Extraction and Text Summarization

This project aims to:
- Extract subject-verb-object relations from an article body
- Extract keywords from an article body
- Produce an extractive summary of an article.

# Extract article information


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

# Extract and print subject-verb-object (SVO) relations from each sentence


```python
import spacy
nlp = spacy.load('en_core_web_sm')
```


```python
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
```


```python
def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs
```


```python
def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs
```


```python
def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs
```


```python
def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False
```


```python
def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False
```


```python
def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs
```


```python
def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs
```


```python
def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None
```


```python
def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None
```


```python
def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated
```


```python
def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))

    #potentialNewVerb, potentialNewObjs = getObjsFromAttrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs
```


```python
def findSVOs(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    svos.append((sub.lower_, "!" + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_))
    return svos
```


```python
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])

def testSVOs():
    #nlp = English()

    tok = nlp(cnbc_url)
    svos = findSVOs(tok)
    #printDeps(tok)
    print(svos)


```


```python
if __name__ == "__main__":
    testSVOs()
    
```

    [('cases', 'bringing', 'total'), ('others', 'put', 'plans'), ('u.s.', 'reported', 'cases'), ('u.s.', 'reported', 'increase'), ('outwards', 'roll', 'plans'), ('cases', 'bringing', 'total'), ('number', 'surpassed', 'million'), ('you', 'seeing', 'deaths'), ('lag', 'confirmed', 'infections'), ('disease', 'take', 'weeks'), ('force', 'warned', 'people'), ('who', 'driving', 'infections'), ('coronavirus', 'poses', 'risk'), ('conditions', 'span', 'group'), ('rise', 'reflect', 'testing'), ('states', 'reporting', 'rates'), ('rate', 'indicates', 'percentage'), ('state', 'reports', 'increases'), ('level', 'raises', 'flag'), ('state', 'halt', 'some'), ('coronavirus', 'shows', 'signs'), ('state', 'taking', 'approach'), ('phase', 'resume', 'activities'), ('county', 'reopen', 'nightclubs'), ('abbott', 'postpone', 'hospitals'), ('abbott', 'postpone', 'procedures'), ('order', 'protect', 'capacity'), ('counties', 'include', 'cities'), ('texas', 'reported', 'increase'), ('he', 'roll', 'some'), ('state', 'take', 'action'), ('action', 'mitigate', 'spread'), ('hospitals', 'seeing', 'stress'), ('department', 'reported', 'cases'), ('arizona', 'reported', 'increase'), ('anyone', 'spread', 'virus'), ('florida', 'banned', 'drinking'), ('effort', 'slow', 'spread'), ('state', 'reported', 'cases'), ('day', 'breaking', 'cases'), ('plans', 'continuing', 'plan'), ('city', 'issued', 'mandate'), ('mandate', 'requiring', 'coverings'), ('who', 'defy', 'order'), ('suarez', 'told', 'cnn'), ('he', 'instituting', 'order'), ('nevada', 'reported', 'jump'), ('nevada', 'reported', 'increase'), ('people', 'wear', 'face'), ('patients', 'requiring', 'beds'), ('patients', 'requiring', 'ventilators'), ('excitement', 'escaping', 'confinement'), ('excitement', 'enjoy', 'dinner'), ('cnbc', 'join', 'releases'), ('cnbc', 'join', 'corrections'), ('|', '!sell', 'terms'), ('data', 'delayed', 'minutes')]
    

# Apply TextRank for ranking and selecting key phrases, print the result


```python
from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
```


```python
nlp = spacy.load('en_core_web_sm')
```


```python
class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight
```


```python
keyphrase_extractor = TextRank4Keyword()
```


```python
tr4w = TextRank4Keyword()
tr4w.analyze(cnbc_url, candidate_pos = ['NOUN', 'PROPN',"ADP"], window_size=8, lower=False)
tr4w.get_keywords(10)
```

    state - 5.719656280485987
    CNBC - 5.641620450054959
    coronavirus - 4.1543415040903175
    U.S. - 3.6420822213373283
    cases - 3.618643080502066
    Texas - 3.5858016348974235
    Friday - 3.422377277368376
    News - 3.0259776914295595
    states - 2.986466777170625
    counties - 2.768908866191966
    % - 2.632745822285362
    Gov. - 2.4773804920425424
    


```python
# Another TextRank Implementation
def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    import itertools, nltk, string
    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, lambda word__pos__chunk: word__pos__chunk[2] != 'O') if key]

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]
```


```python
def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates
```


```python
def score_keyphrases_by_textrank(text, n_keywords=0.05):
    from itertools import takewhile, tee
    import operator
    import networkx, nltk
    
    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_words(text)
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.items(), key=operator.itemgetter(1), reverse=True)[:n_keywords]}
                  #for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
                  
    #sorted(max_value_score.items(), key=operator.itemgetter(1), reverse=True)[:3]
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
            
    return sorted(keyphrases.items(), key=operator.itemgetter(1), reverse=True)
    #return sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)
```


```python
score_keyphrases_by_textrank(cnbc_url)
```




    [('cnbc', 0.017732349595688614),
     ('state', 0.016488985514366464),
     ('friday', 0.010349578870715125),
     ('data', 0.0097687265803191),
     ('cases', 0.00944310441051565),
     ('texas', 0.009111315443253118),
     ('u.s. cases', 0.008702963357223557),
     ('u.s. coronavirus cases', 0.008653025283789306),
     ('new cases', 0.008646639375986516),
     ('coronavirus', 0.008553149136920802),
     ('news', 0.008543591611741546),
     ('covid-19 cases', 0.008319448753412105),
     ('u.s. coronavirus', 0.008257985720426134),
     ('new covid-19 cases', 0.008163023949427197),
     ('u.s.', 0.007962822303931466),
     ('texas gov', 0.007899001440324769),
     ('new', 0.007850174341457381),
     ('health', 0.007409044735952443),
     ('covid-19', 0.007195793096308561),
     ('daily new cases', 0.006965192167671601),
     ('arizona', 0.0069003279544944006),
     ('arizona arizona gov', 0.006829114448795074),
     ('counties', 0.006716375858089845),
     ('gov', 0.0066866874373964205),
     ('states', 0.006665469951422495),
     ('daily cases', 0.00652270108077871),
     ('saturday', 0.006466661201136969),
     ('order', 0.0064529617201516225),
     ('new infections', 0.006036587028611323),
     ('hospitalizations', 0.006007053140028626),
     ('u.s. markets', 0.00581480695033861),
     ('business', 0.005693076286572363),
     ('sign', 0.005579493745790545),
     ('record', 0.005506531271751153),
     ('people', 0.005344487054026097),
     ('capacity', 0.00529190580681495),
     ('media', 0.00498071420774407),
     ('phase', 0.004917476300098413),
     ('business day', 0.004664872366711205),
     ('record daily', 0.004554414511396461),
     ('thursday', 0.0043581062141046225),
     ('dr.', 0.004312033211451847),
     ('nevada', 0.004267846679835826),
     ('nevada nevada', 0.004267846679835826),
     ('video', 0.004234175454921787),
     ('infections', 0.004222999715765265),
     ('florida', 0.004142505707634992),
     ('florida florida', 0.004142505707634992),
     ('positivity', 0.004056372811193688),
     ('positivity rate', 0.003907846369413694),
     ('abbott', 0.0039053972006901586),
     ('weeks', 0.0038496359182734544),
     ('press', 0.0037799417091365422),
     ('rate', 0.0037593199276336996),
     ('virus', 0.0036947832710180124),
     ('markets', 0.0036667915967457548),
     ('tech', 0.0036381726922285096),
     ('day', 0.0036366684468500466),
     ('daily', 0.003602297751041769),
     ('number', 0.003549494498721493)]



# Apply LexRank to produce an extractive summary of 5 sentences.


```python
#!pip install sumy
#!pip install git+git://github.com/miso-belica/sumy.git 
```


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

        sentences = summarizer(parser.document, num_sents)  # Summarize the document with # sentences
        for sentence in sentences:
            self.summary += (sentence.__unicode__())

    def output(self):
        return self.summary
```


```python
#Apply LexRank to produce an extractive summary of 5 sentences.
text_to_sum = TextSummary(cnbc_url,5)
print(text_to_sum.output())
```

    Cases are growing by 5% or more based on a seven-day average in 34 states across the U.S., including Arizona, Texas, California, Florida and Nevada.At least 125,559 people have died from the virus in the U.S. Cases are growing by 5% or more based on a seven-day average in 38 states across the U.S., including Arizona, Texas, California, Florida and Nevada.There are more hospitalizations in some of those places and soon you'll be seeing more deaths," White House health advisor Dr. Anthony Fauci said in an interview with CNBC's Meg Tirrell on Friday that was aired by the Milken Institute."As I said from the start, if the positivity rate rose above 10%, the State of Texas would take further action to mitigate the spread of COVID-19," Abbott said in a press release.It's in all 15 of our counties.
    
