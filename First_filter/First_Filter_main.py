import pandas as pd
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|;]')

def read_data(filename):
    data = pd.read_csv(filename, sep=',')
    return data

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    string = re.sub(REPLACE_BY_SPACE_RE," ",string)
    string = re.sub(re.compile('[?]'),"? ",string)
    return emoji_pattern.sub(r'', string)

'''
This first filter will delete all non-tweets columns, removes emojis and substrings "ENGLISH TRANSLATION" and finally translate each tweet.
'''
def first_filter(data_name,start_index,how_many_tweets):

    #if filter have already started, open the filtered database
    if(start_index > 0): data_name = data_name[0:len(data_name)-4]+"_filtered.csv"

    train = read_data(data_name)
    print("File {} opened Succesfully".format(data_name))
    #removes all non-tweets columns
    train = train.drop([i for i in train.columns if i!='tweets'],axis=1)

#import google-cloud translator
    from google.cloud import translate_v2
    tr = translate_v2.Client(target_language='en')
    character_count = 0
    translated_tweets = 0

    #print(train['tweets'][from_tweet])
    #print(tr.translate(train['tweets'][from_tweet])['translatedText'])
    #remove emojis and substring
    for i in range(start_index,len(train['tweets'])):

        try:
            #remove emojis
            tweet = remove_emoji(train['tweets'][i])
        except:
            print("Error con el tweet {}, tweet -> {}".format(i,train['tweets'][i]))
            if(start_index > 0):
                train.to_csv(data_name,index=False)
            else:
                train.to_csv(data_name[0:len(data_name)-4]+"_filtered.csv",index=False)
            return [character_count,translated_tweets]

        #remove substring "ENGLISH TRANSLATION"
        if tweet[0:21] == "ENGLISH TRANSLATION: ":
            train['tweets'][i] = tweet[22:]

        #Translate tweet, if something occurs, database is saved with progress made
        try:
            train['tweets'][i] = tr.translate(train['tweets'][i])['translatedText']
        except:
            print("Error al traducir el tweet {}, tweet -> {}".format(i,train['tweets'][i]))
            if(start_index > 0):
                train.to_csv(data_name,index=False)
            else:
                train.to_csv(data_name[0:len(data_name)-4]+"_filtered.csv",index=False)
            return [character_count,translated_tweets]

        translated_tweets += 1
        character_count += len(train['tweets'][i])
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
name_file = "AboutIsis.csv"
from_tweet = "Put here index number of tweet start"
how_many_tweets = -1
filter = first_filter(name_file,from_tweet,how_many_tweets)
print("{} characters translated\n{} translated tweets".format(filter[0],filter[1]))

'''
terrorism --> 17410 tweets_translated ---> [Ready]
AboutIsis --> 23078 tweets_translated --> 7,743,217 characters
IsisFanboy --> 15936 tweets_translated ---> [Ready]
'''

'''
David api has translated 12,900,000 characters
'''
