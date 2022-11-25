import snscrape.modules.twitter as sntwitter
import nltk
from functools import reduce
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.corpus import stopwords


def get_important_words(content: str):
    website = urlopen(content).read()
    soup = BeautifulSoup(website, 'html.parser')
    text = reduce(lambda a, b: a + '\n' + b, [t.strip() for t in soup.text.split('\n') if (t.strip() != '')])
    stop_words = set(stopwords.words('german'))

    total_words = text.split()  # get number of words
    total_word_length = len(total_words)
    print(total_word_length)

    total_sentences = nltk.tokenize.sent_tokenize(text)  # get number of sentences
    total_sent_len = len(total_sentences)
    print(total_sent_len)

    # TODO return the most important words
    return []


def get_image_content(link: str):
    raise NotImplementedError


def extract_content(tweetid: int | str):
    content = [i for i in sntwitter.TwitterTweetScraper(tweetid).get_items()][0]
    isimage = True if content.media and len(content.media) > 0 else False
    if not isimage:
        return [get_important_words(link) for link in content.outlinks]
    else:
        return get_image_content(content.media[0].fullUrl)
