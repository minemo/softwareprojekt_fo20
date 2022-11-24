import snscrape.modules.twitter as sntwitter
import requests


def get_important_words(content: str):
    raise NotImplementedError


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
