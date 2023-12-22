# Install the required libraries
# !pip install ibm-watson
# !pip install ibm-cloud-sdk-core

import pandas as pd
import json
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1  import Features, EmotionOptions,SentimentOptions


# Accessing Watson NLU API
api = "<INSERT-WATSON-URL>"
key = "<INSERT-WATSON-API-KEY>"

auth = IAMAuthenticator(apikey = key)
service = NaturalLanguageUnderstandingV1(version='2019-07-12', authenticator= auth)
service.set_service_url(api)
service.set_disable_ssl_verification(True)

# Getting basic scores such as sentiment score & emotions scores (Anger, Disgust, Sadness, Fear, Joy)
def get_nlu_features(post):

    post_length = post.count(' ')

    watson_li = []

    if (post_length >= 3):

        # get_esult() gives a py object
        response = service.analyze(
        text=post,
        language = 'en',
        features=Features(
            sentiment=SentimentOptions(),
            emotion=EmotionOptions(),
        )).get_result()

        x = json.dumps(response, indent=2)
        #obtaining a py dict
        y = json.loads(x)

        #overall post sentiment score
        watson_li.append(y['sentiment']['document']['score'])

        #overall post emotion score
        val_anger = y['emotion']['document']['emotion']['anger']
        val_sadness = y['emotion']['document']['emotion']['sadness']
        val_disgust = y['emotion']['document']['emotion']['disgust']
        val_joy = y['emotion']['document']['emotion']['joy']
        val_fear = y['emotion']['document']['emotion']['fear']

        watson_li.extend([val_anger,val_sadness,val_disgust,val_joy,val_fear])

    else:
        watson_li.extend([0,0,0,0,0,0])


    watson_df = pd.DataFrame(watson_li).transpose()

    return watson_df


data = pd.read_csv('reddit.csv')
data['usertext']=data['usertext'].astype('str')
data['label']=data['label'].astype('int')

watson_features = pd.DataFrame()
for post in data['usertext']:
  post_NLU = get_nlu_features(post)
  watson_features = watson_features.append(post_NLU, ignore_index = True)

watson_features.columns = ['sentiment','anger','sadness','disgust','joy','fear']
watson_features.to_csv('watson_nlu.csv', index = False)  # Set index=False if you don't want to include row indices in your CSV file
