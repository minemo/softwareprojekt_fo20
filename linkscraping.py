import snscrape.modules.twitter as sntwitter
import math
from operator import itemgetter
import nltk
from functools import reduce
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.corpus import stopwords

def check_sent(word, sentences):
    final = [all([w in x for w in word]) for x in sentences]
    sent_length = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_length))

def get_top_n(dict_element, n): 
    result = dict(sorted(dict_element.items(), key = itemgetter(1), reverse = True)[:n])
    return result

def get_important_words(content: str):
    website = urlopen(content).read()
    soup = BeautifulSoup(website, 'html.parser')
    text = reduce(lambda a, b: a + '\n' + b, [t.strip() for t in soup.text.split('\n') if (t.strip() != '')])
    stop_words = set(stopwords.words('german'))

    total_words = text.split()  
    total_word_length = len(total_words)    #get number of words
    #print(total_word_length)

    total_sentences = nltk.tokenize.sent_tokenize(text)  
    total_sent_len = len(total_sentences)   #get number of sentences
    #print(total_sent_len)

    tf = {}     
    for each_word in total_words:           
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in tf:
                tf[each_word] += 1
            else:
                tf[each_word] = 1
    
    tf.update((x, y/int(total_word_length)) for x, y in tf.items())     #get tf
    #print(tf)

    idf = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in idf:
                idf[each_word] = check_sent(each_word, total_sentences)
            else:
                idf[each_word] = 1
    
    idf.update((x, math.log(int(total_sent_len)/y)) for x, y in idf.items())    #get idf
    #print(idf)

    tf_idf_score = {key: tf[key] * idf.get(key, 0) for key in tf.keys()}    #get tf*idf
    #print(tf_idf_score)
    return get_top_n(tf_idf_score, 5)


def get_image_content(link: str):
    raise NotImplementedError


def extract_content(tweetid: int | str):
    content = [i for i in sntwitter.TwitterTweetScraper(tweetid).get_items()][0]
    isimage = True if content.media and len(content.media) > 0 else False
    if not isimage:
        return [get_important_words(link) for link in content.outlinks]
    else:
        return get_image_content(content.media[0].fullUrl)


#TEST
get_important_words("https://www.gesetze-im-internet.de/gg/art_20.html")