import snscrape.modules.twitter as sntwitter
import requests
import nltk
from bs4 import BeautifulSoup
from urllib.request import urlopen
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

def get_important_words(content: str):
    html = urlopen(content).read()                  #get website text
    soup = BeautifulSoup(html, 'html.parser')
    soup_string = soup.text
    
    stop_words = set(stopwords.words('german'))
    
    total_words = soup_string.split()               #get number of words
    total_word_length = len(total_words)
    print(total_word_length)

    total_sentences = nltk.tokenize.sent_tokenize(soup_string)      #get number of sentences
    total_sent_len = len(total_sentences)
    print(total_sent_len)

    #TODO 


def get_image_content(link: str):
    raise NotImplementedError


def extract_content(tweetid: int | str):
    content = [i for i in sntwitter.TwitterTweetScraper(tweetid).get_items()][0]
    print(content)
    isimage = True if len(content.media) > 0 else False
    if not isimage:
        return get_important_words(content)
    else:
        return get_image_content(content.media[0].fullUrl)

get_important_words("https://www.bpb.de/kurz-knapp/zahlen-und-fakten/globalisierung/52727/jaehrliche-aenderung-der-waldbestaende")