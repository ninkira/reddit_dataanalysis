import self as self
import torch
from transformers import pipeline

from Pipeline.QuestionAnswering import TransformerTest
from Data.DataCrawling import DataCrawler

if __name__ == '__main__':

    print("Hello World")
    """
        # call methods from package module - works here and in DataCrawling Script - https://stackoverflow.com/questions/18472904/how-to-use-a-python-function-with-keyword-self-in-arguments
        reddit_datacrawler = DataCrawler()
        #reddit_datacrawler.crawl_aita_hot(reddit_datacrawler.reddit_authenfication())
        reddit_datacrawler .crawl_aita_top(reddit_datacrawler.reddit_authenfication())
        transformers_test = TransformerTest()
    
    
    
        print("sentiment analysis")
        classifier = pipeline("sentiment-analysis")
        print(classifier([
            "I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!"
        ]))
        classifier("I've been waiting for a HuggingFace course my whole life.")
    
        question_answerer = pipeline("question-answering")
        print(question_answerer(
            question="Am I the asshole for telling my kids why we're getting a divorce",
            context="lright, this is messy. Me and my ex wife got married at 23, and had 3 kids. The 3 kids are 17, 15, and 14. I recently found out my ex wife has been cheating on me for 3 years. Once I confronted her she broke down, saying that I was never home, always working. Btw I worked like that so she could be a stay at home mom. \n\nI raised my kids to know that cheating is one of the worst things in the world. If you no longer love someone, break up with them, don't cheat. Even if you have issues with the relationship, work it out or leave them. \n\nNow, before we sat them down, their mother begged me to not tell them she cheated. I told her that if they asked, I would not lie. She tried to dance around the whole reason for the divorce, citing \u201cadult issues\u201d. 14 year old asked why we were getting a divorce, and I told him, flat out, \u201cshe cheated on me for 3 years\u201d. \n\nThe mother immediately burst out crying, and all the kids were incredibly angry with her. It\u2019s been 3 months and they still haven\u2019t spoken to her, saying she ruined their family and their lives, and that she\u2019s a \u201ccheater and a liar\u201d. \n\nShe\u2019s been coming after me online saying that I\u2019m a bastard and ruined her relationship with her kids."
        ))
    """
