"""Usage: python3 run_analysis.py <country>"""
from __future__ import unicode_literals
import re, operator, string, sys
import nltk
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
from collections import defaultdict

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


def preprocess(s, lowercase=False):
    b = bytes(s, encoding='ascii')
    s = b.decode('unicode-escape')
    tokens = tokens_re.findall(s)
    tokens = [token[2:] if token.startswith("b'") else token for token in tokens]
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


# Open the tweets CSV file
df = pd.read_csv('data/{country}_tweets.csv'.format(country=sys.argv[1]))

# Build a stopwords list from common stopwords, punctuation, and others that we know about
punctuation = list(string.punctuation)
other_stopwords = ['rt', 'via']
stop = stopwords.words('english') + punctuation + other_stopwords

# Get only the tweet_text column and take a sample. Normally would run on full dataset
tweet_text = df['tweet_text'].sample(50000)

print('Processing {} tweets'.format(tweet_text.count()))
count_all = Counter()

for tweet in tweet_text:
    # preprocess the tweet and tokenize as per previous step
    terms_all = [term for term in preprocess(tweet, lowercase=True)]
    # remove stopwords from the tweet terms
    terms_stopwords_removed = [term for term in terms_all if term not in stop]
    # filter out words deemed to short to be relevant (3 characters or less)
    terms_short_words_removed = [token for token in terms_stopwords_removed if len(token) > 3]
    # filter out terms beginning with hastags or at
    terms_only = [term for term in terms_short_words_removed if not term.startswith(
        ('#', '@', 'http', 'https'))]
    # Update the counter
    count_all.update(terms_only)
print('Finished')

co_matrix = defaultdict(lambda: defaultdict(int))

# f is the file pointer to the JSON data set
for tweet in tweet_text:
    terms_all = [term for term in preprocess(tweet, lowercase=True)]
    terms_stopwords_removed = [term for term in terms_all if term not in stop]
    terms_short_words_removed = [token for token in terms_stopwords_removed if
                                 len(token) > 3]
    terms_only = [term for term in terms_short_words_removed if
                  not term.startswith(
                      ('#', '@', 'http', 'https'))]

    # Build co-occurrence matrix
    for i in range(len(terms_only) - 1):
        for j in range(i + 1, len(terms_only)):
            w1, w2 = sorted([terms_only[i], terms_only[j]])
            if w1 != w2:
                co_matrix[w1][w2] += 1

co_matrix_max = []
# For each term, look for the most common co-occurrent terms
for t1 in co_matrix:
    t1_max_terms = sorted(
        co_matrix[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        co_matrix_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(co_matrix_max, key=operator.itemgetter(1), reverse=True)

# n_docs is the total n. of tweets
p_t = {}
p_t_com = defaultdict(lambda : defaultdict(int))

n_docs = len(df)

for term, n in count_all.items():
    p_t[term] = n / n_docs
    for t2 in co_matrix[term]:
        p_t_com[term][t2] = co_matrix[term][t2] / n_docs

positive_vocab = []
for pos_word in open('data/opinion-lexicon/positive-words.txt', 'rb').readlines()[35:]:
    positive_vocab.append(pos_word.decode('unicode-escape').rstrip())
negative_vocab = []
for neg_word in open('data/opinion-lexicon/negative-words.txt', 'rb').readlines()[35:]:
    negative_vocab.append(neg_word.decode('unicode-escape').rstrip())

import math

pmi = defaultdict(lambda: defaultdict(int))
for t1 in p_t:
    for t2 in co_matrix[t1]:
        denom = p_t[t1] * p_t[t2]
        pmi[t1][t2] = math.log2(p_t_com[t1][t2] / denom)

semantic_orientation = {}
for term, n in p_t.items():
    positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
    negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
    semantic_orientation[term] = positive_assoc - negative_assoc

semantic_sorted = sorted(semantic_orientation.items(),
                         key=operator.itemgetter(1),
                         reverse=True)
top_pos = semantic_sorted[:10]
top_neg = semantic_sorted[-10:]
print(top_pos)
print(top_neg)
for gender_topic in (
    'women',
    'girls',
    'woman',
    'girl',
    'female',
    'men',
    'boys',
    'man',
    'boy',
    'male',
    'gender',
    'equality',
    'inequality',
    'discrimination'
):
    if gender_topic in semantic_orientation.keys():
        print(gender_topic, semantic_orientation[gender_topic])

F = ('women', 'girls', 'woman','girl', 'female')
M = ('men', 'boys', 'man', 'boy', 'male')
SO_Fs = list(semantic_orientation[term] for term in F if term in semantic_orientation.keys())
SO_F_sum = sum(SO_Fs)
F_c_size = len(SO_Fs)
SO_Ms = list(semantic_orientation[term] for term in M if term in semantic_orientation.keys())
SO_M_sum = sum(SO_Ms)
M_c_size = len(SO_Ms)
print((SO_F_sum / F_c_size) - (SO_M_sum - M_c_size))