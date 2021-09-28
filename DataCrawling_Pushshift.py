import pandas as pd
import requests.auth
import json
import time

from Data.Crawling.DataCrawling_Reddit import RedditCrawler


class CrawlingPushshift:
    """
    Diese Klasse beinhaltet den Crawler von CrawlDirectory, um hiermit große Datenmengen an Postings direkt von CrawlDirectory
    herunterzuladen. Er basiert im wesentlichen auf den den Reddit-API Crawler. Auch hier sind noch Print Statements
    im Code um den zustand des Crawlers zu überprüfen. Skript ist außerdem angelehnt an.
    Author- und Kommentar-Daten werden hier bewusst aus Speicherplatz Gründen ausgelassen.
    https://medium.com/@pasdan/how-to-scrap-reddit-using-pushshift-io-via-python-a3ebcc9b83f4
    """

    def __init__(self):
        self.after_object = ''
        self.after_object_final = ''
        self.after_push_object = 0

    def crawl_data(self):
        """
        Hauptmethode um Daten von CrawlDirectory zu crawlen.
        Quelle für die Request-URLs: https://medium.com/swlh/how-to-scrape-large-amounts-of-reddit-data-using-pushshift-1d33bde9286
        """

        aita_pushshift_df = pd.DataFrame()
        print("CrawlDirectory Crawler starts, after_obj:", self.after_object)
        crawl_iteration = 0

        # auch hier: Aufrufen der CrawlDirectory API mittels While-Loop.
        while True:
            time.sleep(1)
            crawl_iteration += 1
            print("crawl_iteration", crawl_iteration)
            print("after_obj", self.after_object)
            if crawl_iteration >= 700:  # CrawlDirectory hat keine Limits hinsichtlich Downloads, daher hier wesentlich höher.
                print("CrawlDirectory Crawling finished.")
                break
            else:
                if self.after_push_object == '':
                    aita_top_url_base = "https://api.pushshift.io/reddit/search/submission?subreddit=AmItheAsshole&limit=100"
                    data_to_append = self.aita_pushshift_dataframe(aita_top_url_base)
                    aita_pushshift_df = pd.concat([aita_pushshift_df, data_to_append], ignore_index=True)
                    result = aita_pushshift_df.to_json(orient="records")
                    parsed = json.loads(result)
                    filename = "PushshiftCrawling_" + str(crawl_iteration) + ".json"
                    RedditCrawler().save_json_to_file(parsed, "DataFiles/CrawlDirectory/CrawlDirectory", filename)
                    print("Saved ", len(aita_pushshift_df), " items into file", filename)
                else:
                    print("Add aditional data, after_object:", self.after_object)
                    aita_top_url_after = "https://api.pushshift.io/reddit/search/submission?subreddit=AmItheAsshole&limit=100&after=" + str(
                        self.after_push_object) + "h"
                    data_to_append_after = self.aita_pushshift_dataframe(aita_top_url_after)
                    aita_pushshift_df = pd.concat([aita_pushshift_df, data_to_append_after], ignore_index=True)
                    result = aita_pushshift_df.to_json(orient="records")
                    parsed = json.loads(result)
                    filename = "PushshiftCrawling_" + str(crawl_iteration) + ".json"
                    RedditCrawler().save_json_to_file(parsed, "DataFiles/CrawlDirectory/CrawlDirectory", filename)
                    print("Saved ", len(aita_pushshift_df), " items into file", filename)
                # Konvertiere Dataframe zu JSON zum speichern in Datei, vgl. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
        result = aita_pushshift_df.to_json(orient="records")
        parsed = json.loads(result)
        RedditCrawler().save_json_to_file(parsed, "DataFiles/CrawlDirectory/CrawlDirectory", "PushshiftCrawling_699.json")

    def fill_metadata(self, headers):
        """
        Diese Methode fügt zusätzliche Metadaten zu den zuvor gecrawlten Posts hinzu. Hier ist das das
        Moderations-Tag link_flair_text, was nicht über CrawlDirectory verfügbar ist. Die Methode iteriert
        über das CrawlDirectory - Basisdatenset und fügt die Moderations-Tags zu jedem Post hinzu.

        :param headers: Request Header,
        generiert aus Authenfizierungsmethode.
        """
        with open(
                "C:/Users/nceck/PycharmProjects/reddit_dataanalysis/DataFiles/CrawlDirectory/Pushshift/PushshiftCrawling_500.json") as json_file:
            data = json.load(json_file)
        print("items counter", len(data))  # 68415
        headers_request = headers
        pushshift_dataframe = pd.DataFrame()
        verdict = ''
        # Start-Parameter kann dynamisch angepasst werden, je nachdem ob bzw. wann der Crawler während des Durchlaufs crasht. Der Wert entspricht des Index wenn er gecrasht ist.
        for idx, post in enumerate(data, start=28768):
            print("post", post)
            print("idx", idx)
            score = post['score']
            content = post['content']
            print("score", score)

            if score is not None:
                if content != '[removed]' or content != '[deleted]':
                    time.sleep(1)
                    request_url = requests.get("https://oauth.reddit.com/r/AmItheAsshole/comments/" + post['id'],
                                               headers=headers_request)
                    if request_url.status_code == 200:
                        request_json = request_url.json()
                        verdict = request_json[0]['data']['children'][0]['data']['link_flair_text']
                    else:
                        verdict = "No verdict could be fetched"
                        time.sleep(60)
                        headers_request = RedditCrawler().reddit_authenfication()
                        print("New request headers: ", headers_request)
                    pushshift_dataframe = pushshift_dataframe.append({
                        'title': post['title'],
                        'content': content,
                        'verdict': verdict,
                        'score': score,
                        'created': post['created'],
                        'author_fullname': post['author_fullname'],
                        'url': post['url'],
                    }, ignore_index=True)

                    result = pushshift_dataframe.to_json(orient="records")
                    parsed = json.loads(result)
                    filename = "Metadata_PushshiftCrawling_" + str(idx) + ".json"
                    RedditCrawler().save_json_to_file(parsed,
                                                    "C:/Users/nceck/PycharmProjects/reddit_dataanalysis/DataFiles/CrawlDirectory/Pushshift_Metadata",
                                                      filename)
                    print("Saved ", len(pushshift_dataframe), " items into file", filename)

                else:
                    continue
            else:
                continue

        result = pushshift_dataframe.to_json(orient="records")
        parsed = json.loads(result)
        RedditCrawler().save_json_to_file(parsed, "DataFiles/BaseData", "Grounddata_large.json")
        print("Saved ", len(pushshift_dataframe), " items into file Grounddata_large.json")

    def aita_pushshift_dataframe(self, request_url: str, ) -> pd.DataFrame:
        """
        Führt Request an Pushift API aus und sortiert die Daten aus der Request Response. Die Methode wandelt die
        Response in JSON um und sortiert die Daten mithilfe eines Pandas Dataframes. :param request_url: Request URL,
        anhand denen die Daten von CrawlDirectory heruntergeladen werden sollen. :return: Pandas Dataframe mit den
        sortierten Daten.
        """
        dataframe = pd.DataFrame()
        request = requests.get(request_url)
        print("request status code", request.status_code)
        print("request", request)

        if request.status_code != 200:
            print("Request failed: ", request.status_code)
            dataframe = dataframe.append({"Error": "Request failed, data could not be fetched"}, ignore_index=True)
            return dataframe
        else:
            aita_json = request.json()
            self.after_push_object = (self.after_push_object + 24)
            aita_json_list = aita_json['data']
            for idx, post in enumerate(aita_json_list):
                print("index post: ", idx)
                post_author = post['author']

                if 'selftext' in post.keys():
                    print("append new data")
                    # push info into dataframe-
                    dataframe = dataframe.append({
                        'title': post['title'],
                        'content': post['selftext'],
                        'id': post['id'],
                        'score': post['score'],
                        'created': post['created_utc'],
                        'author_fullname': post_author,
                        'url': post['url'],

                    }, ignore_index=True)
                else:
                    continue

            return dataframe


""" Main Methode um den Crawler parallel zu anderen Modulen laufen lassen zu können. """
if __name__ == '__main__':
    datacrawler = RedditCrawler()
    crawling = CrawlingPushshift()
    print("start metadata crawling")

    crawling.fill_metadata(datacrawler.reddit_authenfication())
