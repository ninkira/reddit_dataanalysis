import self as self


from Data.DataCrawling import Hot_DataCrawler

if __name__ == '__main__':
    print('Nina')

    # call methods from package module - works here and in DataCrawling Script - https://stackoverflow.com/questions/18472904/how-to-use-a-python-function-with-keyword-self-in-arguments
    hot_reddit_data = Hot_DataCrawler()
    hot_reddit_data.crawl_aita_hot(hot_reddit_data.reddit_authenfication())