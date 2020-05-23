import pandas as pd
import re


column_tweets = 'text'
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|;]')
REMOVE_URLS = re.compile('http\S+')
emoji_pattern = re.compile("["
	u"\U0001F600-\U0001F64F"  # emoticons
	u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	u"\U0001F680-\U0001F6FF"  # transport & map symbols
	u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	"]+", flags=re.UNICODE)

def read_data(filename):
	data = pd.read_csv(filename, sep=',')
	return data

def remove_emoji(string):
	string = string.replace("ENGLISH TRANSLATION: ", "")
	string = re.sub(REMOVE_URLS,"",string)
	string = re.sub(REPLACE_BY_SPACE_RE," ",string)
	string = re.sub(re.compile('[?]'),"? ",string)
	string = emoji_pattern.sub(r'',string)
	string = re.sub('rt ',"",string)
	return string

'''
This first filter will delete all non-tweets columns, removes emojis, substrings "ENGLISH TRANSLATION" and URLs and finally translate each tweet.
'''
def first_filter(data_name,start_index,how_many_tweets):

#if filter have already started, open the filtered database
	if(start_index > 0): data_name = data_name[0:len(data_name)-4]+"_filtered.csv"

	train = read_data(data_name)
	print("File {} opened Succesfully".format(data_name))
	#removes all non-tweets columns
	train = train.drop([i for i in train.columns if i!=column_tweets],axis=1)

	#import google-cloud translator
	from google.cloud import translate_v2
	tr = translate_v2.Client(target_language='en')
	character_count = 0
	translated_tweets = 0

	#print(train['tweets'][from_tweet])
	#print(tr.translate(train['tweets'][from_tweet])['translatedText'])
	#remove emojis and substring
	for i in range(start_index,len(train[column_tweets])):

		try:
		    #remove emojis
			train[column_tweets][i] = remove_emoji(train[column_tweets][i])
		except:
			print("Error con el tweet {}, tweet -> {}".format(i,train[column_tweets][i]))
			if(start_index > 0):
				train.to_csv(data_name,index=False)
			else:
				train.to_csv(data_name[0:len(data_name)-4]+"_filtered.csv",index=False)
			return [character_count,translated_tweets]


		#Translate tweet, if something happens, database is saved with progress made
		try:
			train[column_tweets][i] = tr.translate(train[column_tweets][i])['translatedText']
		except:
			print("Error al traducir el tweet {}, tweet -> {}".format(i,train[column_tweets][i]))
			if(start_index > 0):
				train.to_csv(data_name,index=False)
			else:
				train.to_csv(data_name[0:len(data_name)-4]+"_filtered.csv",index=False)
			return [character_count,translated_tweets]

		translated_tweets += 1
		character_count += len(train[column_tweets][i])
		if (how_many_tweets == translated_tweets and how_many_tweets > 0 ): break

	if(start_index > 0):
		train.to_csv(data_name,index=False)
	else:
		train.to_csv(data_name[0:len(data_name)-4]+"_filtered.csv",index=False)
	return [character_count,translated_tweets]


'''
Start filtering, just specify from what index tweet start, file name,
and how many tweets want to filtering
- If how_many_tweets is negative, all possible tweets will be filtered
'''
name_file = "Gathered_tweets.csv"
from_tweet = 0
how_many_tweets = -1
filter = first_filter(name_file,from_tweet,how_many_tweets)
print("{} characters translated\n{} translated tweets".format(filter[0],filter[1]))

'''
terrorism --> 17410 tweets_translated ---> [Ready]
AboutIsis --> 59820 (there are 120000 +-) tweets_translated --> 12662283 characters David: 7743211 Juanes: 4919072
IsisFanboy --> 17391 tweets_translated ---> [Ready]

Gathered_tweets --> 130 tweets_translated 61237 characters Juan ---> [Ready]
'''

# Current tweet: 59820


'''
David api has translated 12,900,000 characters
'''
