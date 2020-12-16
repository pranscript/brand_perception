import tweepy
import requests
import simplejson as json

def get_related_tweets(text_query):

	consumer_key = "xxxxxxxxxxxxxxxxxxxxxx"
	consumer_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
	access_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
	access_token_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth) 
	
	inputArray = []
	#search_words1 = '#'+text_query
	search_words = '@'+text_query
	date_since = "2020-01-01"
	
	for tweet in tweepy.Cursor(api.search,q=search_words +" -filter:retweets",lang="en",since=date_since, result_type='recent', timeout=999999, include_rts=False).items(100):
		inputArray.append(tweet.text)
	return inputArray


def get_related_reddit_comments(text_query):
	query=""
	after="30d"
	sub = text_query
	data = getPushshiftData (query, after, sub)
	processedList=[]
	for x in data:
		processedList.append(str(x["body"]))
	return processedList

def getPushshiftData(query, after, sub):
    url = 'https://api.pushshift.io/reddit/search/comment/?title='+str(query)+'&size=100&after='+str(after)+'&subreddit='+str(sub)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']	


