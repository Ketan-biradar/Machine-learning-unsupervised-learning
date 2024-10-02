# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:12:21 2024

@author: ketan
"""

# -- coding: utf-8 --
"""
Created on Sun Sep 15 19:41:11 2024

@author: ketan biradar 
"""
###############################################################################
'''TEXT MINING (NLP) '''
###############################################################################
'''
Problem Statement: -
In the era of widespread internet use, it is necessary for businesses to 
understand what the consumers think of their products. If they can understand 
what the consumers like or dislike about their products, they can improve them 
and thereby increase their profits by keeping their customers happy. For this 
reason, they analyze the reviews of their products on websites such as Amazon 
or Snapdeal by using text mining and sentiment analysis techniques. 
'''
###############################################################################

# 1.Business Problem 
  # Bussiness Objectives:
'''
1) Understanding Consumer Preferences: Identify what consumers like or dislike 
    about products to tailor features and services that better meet customer
    expectations.

2) Improving Product Development: Use insights from reviews to enhance product 
    design, functionality, and quality based on consumer feedback.

3) Enhancing Customer Experience: Identify and address common pain points in 
    customer reviews to improve overall customer satisfaction.

4) Boosting Brand Loyalty: By continuously refining products based on consumer 
    feedback, businesses can foster stronger relationships and brand loyalty.

5) Increasing Sales and Revenue: Satisfied customers are more likely to purchase
    again and recommend the product to others, leading to increased sales.
    '''
   
    # Key Constraints:
'''
1) Data Availability: Access to a large volume of relevant reviews may be limited
    by website restrictions, privacy policies, or the need for API access.

2) Data Quality: Reviews may contain noise, such as irrelevant information, spam, 
    or fake reviews, which can affect the accuracy of the analysis.

3) Language and Tone Variability: Consumer reviews may be written in multiple 
    languages, dialects, or informal language, making it difficult for sentiment 
    analysis models to interpret them correctly.

4) Sarcasm and Ambiguity: Sentiment analysis tools can struggle to detect sarcasm,
    irony, or ambiguous statements, leading to incorrect sentiment classification.

5) Subjectivity of Reviews: Reviews are often subjective and based on personal
    opinions, which can lead to biased interpretations of product performance.
    
    '''
###############################################################################

from bs4 import BeautifulSoup
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

# Setup Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
service = Service("C:/Users/ketan/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe")  # Replace with the path to your chromedriver
driver = webdriver.Chrome(service=service, options=chrome_options)

#load the required web page link to the chrome page
driver.get('https://www.amazon.in/Zebronics-Zeb-Jaguar-Wireless-Precision-Ambidextrous/product-reviews/B098JYT4SY/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews')

# Scroll down to load more reviews (if necessary)

driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

# Get the page source and parse it with BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')
print(soup.prettify())

###############################################################################
''' TASK 1 : Extract reviews of any product from e-commerce website Amazon.'''

''' Extracted reviews of JBL Tune 510BT '''

# finding the required section of web page using inspect and assiging the 
# section of the tag to fetch the data.


# Extracting reviews of the product.
#<div class="a-row a-spacing-top-mini"><span class="a-size-base">the battery on these is literally crazy. i’ve had them for months, use them at least 3 times a week for over an hour and i’ve only had to charge them 3 times. two of those times being on a road trip for 10 hours. they also charge fully in under an hour. connecting to bluetooth is so easy, the volume buttons work perfectly. the sound is amazing. you could have it loud if you want them to be &amp; not hear anything around you. or you turn them down &amp; still hear crystal clear while also being aware of your surroundings. i got these on sale for about $30 i believe and even if i spent $100 on them, it would still be worth the money.</span></div>
reviews2 = soup.find_all('div', {'class': 'a-row a-spacing-top-mini'})
review_data2=[]    
for i in range(0,len(reviews2)):
    review_data2.append(reviews2[i].get_text())
review_data2
review_data2[:]=[d.strip('\n') for d in review_data2]
len(review_data2)
review_data2

reviews1 = soup.find_all('span', {'data-hook': 'review-body'})
review_data1=[]    
for i in range(0,len(reviews1)):
    review_data1.append(reviews1[i].get_text())
review_data1
review_data1[:]=[d.strip('\n') for d in review_data1]
review_data1


###############################################################################
###############################################################################


# Assuming 'df' is your DataFrame and 'ReviewerNames' is the column containing names

data3=review_data2+review_data1
len(data3)

# Saving the Extracted data in the csv file
import pandas as pd
df = pd.DataFrame()
df['Review_data']=data3
df

###############################################################################
#Creating the csv file
df.to_csv("amazon_reviews.csv")

# Close the browser
driver.quit()
###############################################################################

'''Task 2 : Perform sentiment analysis on this extracted data '''

from textblob import TextBlob

sent = "This is interesting"
pol=TextBlob(sent).sentiment.polarity
df=pd.read_csv("C:/Users/ketan/Downloads/amazon_reviews.csv")
df.head()
df['polarity']=df['Review_data'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['polarity'].apply(classify_sentiment)

# Save the sentiment results
df.to_csv('amazon_reviews_sentiment.csv', index=False)

############################################################################

''' Task 2 :Extract reviews for any movie from IMDB and perform sentiment analysis.'''
    

import bs4
from bs4 import BeautifulSoup as bs
import requests
link='https://www.imdb.com/title/tt0111161/reviews/?ref_=tt_urv'
page=requests.get(link)
page
page.content
#### <Response [503]>  we need  API to fetch  
##### give only HTML content
soup=bs(page.content,'html.parser')
print(soup.prettify())


#################################

title=soup.find_all('a',class_='title')
title


review_title=[]
for i in range(0,len(title)):
    review_title.append(title[i].get_text())
    
review_title
review_title[:]=[title.strip('\n') for title in review_title]
review_title
len(review_title)

review=soup.find_all('div',class_='text show-more__control')
review
review_body=[]

for i in range(0,len(review)):
    review_body.append(review[i].get_text())
    
review_body
len(review_body)

import pandas as pd
df=pd.DataFrame()
df['Review_Title']=review_title
df['Review']=review_body
df


#To create csv file
df.to_csv("The_Shawshank_Redemption.csv")

import pandas as pd
from textblob import TextBlob
sent="This is very excellent garden"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv("C:/Users/ketan/Downloads/The_Shawshank_Redemption.csv")
df.head()
df['polarity']=df['Review'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'
df['Sentiment'] = df['polarity'].apply(classify_sentiment)

df
###############################################################################


''' Task 3 : 1.	Choose any other website on the internet and do some research 
            on how to extract text and perform sentiment analysis.
            
  Website : Blog Website (forbesindia)
  https://www.forbesindia.com/blog/technology/how-generative-ai-can-make-search-results-on-websites-more-functional-and-error-free/    

 '''



from bs4 import BeautifulSoup
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

# Setup Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
service = Service("C:/Users/ketan/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe")  # Replace with the path to your chromedriver
driver = webdriver.Chrome(service=service, options=chrome_options)

#load the required web page link to the chrome page
driver.get('https://www.flipkart.com/noise-colorfit-icon-2-1-8-display-bluetooth-calling-ai-voice-assistant-smartwatch/product-reviews/itm8229e27c1df28?pid=SMWGEH7VV8B3H8Y6&lid=LSTSMWGEH7VV8B3H8Y6ZUQIYO&marketplace=FLIPKART')

#### <Response [503]>  we need  API to fetch  
##### give only HTML content
soup = BeautifulSoup(driver.page_source, 'html.parser')
print(soup.prettify())
#################################
reviews2 = soup.find_all('p', {'class': 'z9E0IG'})
review_data2=[]    
for i in range(0,len(reviews2)):
    review_data2.append(reviews2[i].get_text())
review_data2
review_data2[:]=[d.strip('\n') for d in review_data2]
len(review_data2)
review_data2

reviews = soup.find_all('div', {'class': 'ZmyHeo'})
review_data3=[]    
for i in range(0,len(reviews)):
    review_data3.append(reviews[i].get_text())
review_data3
review_data3[:]=[d.strip('\n') for d in review_data3]
len(review_data3)
review_data3

if len(review_data2) < len(review_data3):
    review_data2 += [''] * (len(review_data3) - len(review_data2))
elif len(review_data3) < len(review_data2):
    review_data3 += [''] * (len(review_data2) - len(review_data3))

import pandas as pd
df=pd.DataFrame()
df['Review_Title']=review_data2
df['Review']=review_data3
df


#To create csv file
df.to_csv("flipkart.csv")

# Sentiment Analysis

import pandas as pd
from textblob import TextBlob
sent="This is very excellent garden"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv("C:/Users/ketan/flipkart.csv")
df.head()
df['polarity']=df['Review'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']
driver.quit()

####################################################################################################

# Benefits/impact of the solution 
'''
Enhanced Customer Insights
Understanding Sentiments: By analyzing the sentiments of blog reviews, businesses can gain a clear understanding of customer opinions and feelings about their products or services.

Improved Decision-Making
Product Improvement: Insights from sentiment analysis can guide product development teams in making informed decisions about product features or improvements based on customer feedback.

Enhanced Customer Experience
Targeted Responses: Understanding sentiment allows businesses to respond more effectively to customer concerns or praises. 
    
Competitive Advantage
Market Positioning: By analyzing competitors’ blog reviews, businesses can gain insights into their strengths and weaknesses, enabling them to position themselves more effectively in the market.

Marketing Strategy Optimization
Content Strategy: Businesses can understand which topics or aspects of their products generate positive or negative reactions and adjust their content strategies accordingly.
'''