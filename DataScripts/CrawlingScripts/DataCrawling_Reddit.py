import pandas as pd
import requests.auth
import json
import os
import time


class RedditCrawler:
    """
    Diese Klasse beinhaltet einen Crawler, welcher direkt auf die Rddit API zugreift und examplerisch aus den Post-Kategorien /hot und /top Posts herunterlädt.
    Außerdem sind hier einzelne Methoden enhalten um z.B. direkt auf Nutzerdaten oder Kommentare zuzugreifen.
    Source ersten beiden Crawls https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps
    Generelle API Referenz:  https://github.com/reddit-archive/reddit/wiki/API
    """

    def __init__(self):
        self.after_object: str = ''
        self.after_object_final = ''
        self.after_push_object = 0

    def reddit_authenfication(self):
        """
        Diese Methode erstellt alle notwendigen Authenfizierungsdaten um Zugang zu der Reddit API zu bekommen.
        Der User-Agent wurde direkt auf der Reddit-Developer Seite erstellt.
        :return: Request Header für Datacrawling aus der Reddit API.
        """
        client_auth = requests.auth.HTTPBasicAuth('FnzYvE0WD851lymNYp7VLA', '8asQ3Nwer6y1g5jzvLRR3JNmIUsRDg')
        post_data = {"grant_type": "password", "username": "_ninkira", "password": "Password@reddit2021!"}
        headers = {"User-Agent": "DataBot/0.1 by _ninkira"}
        response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data,
                                 headers=headers)

        # Ergebnis: {'access_token': '728377922323-6EbD8mJFo3oMaB7OMIyl1W2gr9q8BQ', 'token_type': 'bearer', 'expires_in': 3600, 'scope': '*'}
        response_json = response.json()
        access_token = response_json['access_token']

        #Header für request
        crawling_headers = {"Authorization": "bearer " + access_token, "User-Agent": "DataBot/0.1 by _ninkira"}
        return crawling_headers

    def crawl_userdata(self, header, username):
        """
        Sendet Request für Userdaten, bekommt als Antwort sämtliche öffentlich verfügbare Informationen zurück.
        :param header: Request Header, generiert durch reddit_authenfication()
        :param username: Reddit Username
        """

        # Crawl fremdes Profil
        request_url = "https://oauth.reddit.com/user/"+ username + "/about"
        userdata_request = requests.get(request_url, headers=header)
        userdata_request_json = userdata_request.json()

        # Crawl der eigenen Nutzerdaten
        response1 = requests.get("https://oauth.reddit.com/api/v1/me", headers=header)
        response1.json()

    def crawl_aita_hot(self, header, filename):
        """
        Crawlt Posts aus der Kategorie "Hot" des AITA-Subreddits. Die Request-Antwort enthält somit die aktuell 10 beliebsten Posts aus AITA.
        :param header:
        """
        aita_hot_df = pd.DataFrame()
        crawl_iteration = 0

        while True:
            time.sleep(1)
            crawl_iteration += 1
            print("crawl_iteration", crawl_iteration)
            print("after_obj", self.after_object)
            if crawl_iteration >= 2:
                print("crawling finished")
                break
            else:
                request_url = "https://oauth.reddit.com/r/AmItheAsshole/hot"
                data_to_append = self.aita_dataframe(request_url, header)
                print("data_to_append", data_to_append)
                aita_hot_df = pd.concat([aita_hot_df, data_to_append], ignore_index=True)
                print("anzahl zeilen 1 erste iteration ", len(aita_hot_df))

        result = aita_hot_df.to_json(orient="records")
        parsed = json.loads(result)
        self.save_json_to_file(parsed, filename)

    def crawl_aita_top(self, header):
        """
        Crawlt Posts aus der Top-Kategorie von AITA. Die Funktion macht einen Reuqest an die Reddit-API und bekommt pro Request rund 100 Postings. Die While-Schleife läuft bis keine Postings mehr gecrawlt werden können,
        nach jeder Iteration werden die gecrawlten Posts zwischengespeichert. Die Verarbeitung der Responses erfolgt in aita_dataframe().
        :param header: Request Header, generiert durch reddit_authenfication(); dieser wird im Verlauf des Crawlings überschrieben.
                        Grund: Bei manchen Verläufen bekommt man sonst einen 401 Error, Access Denied, weil das Token der Anwendung auslöuft.
        :var self.after_object: Flag sicher gestellt wird das auch immer "nach hinten" gecrawlt wird und keine Posts doppelt heruntergeladen werden.
        :var crawl_iteration: Anzahl der möglichen Crawling-Durchläufe.
        """

        aita_top_df = pd.DataFrame()
        self.after_object = ''
        print("crawl_aita_top starts; after_obj:", self.after_object)
        crawl_iteration = 0
        request_headers = header

        while True:
            time.sleep(1)
            crawl_iteration += 1
            print("Reddit Crawler - crawl_iteration", crawl_iteration)
            print("after_obj", self.after_object)
            if crawl_iteration >= 100:
                print("crawling finished")
                break
            else:
                if self.after_object == '':
                    print("Reddit Crawler - first crawling object")

                    # Quellen für URL:
                    # https://www.reddit.com/r/redditdev/comments/6anc5z/what_on_earth_is_the_url_to_get_a_simple_json/
                    # https://www.reddit.com/dev/api#listings - for url parameter
                    aita_top_url_base = "https://oauth.reddit.com/r/AmItheAsshole?limit=100"
                    request = requests.get(aita_top_url_base,
                                           headers=request_headers)
                    print("request status code", request.status_code)
                    if request.status_code == 200:
                        request_json = request.json()
                        data_to_append = self.aita_dataframe(request_json, request_headers)
                        print("data_to_append", data_to_append)
                        aita_top_df = pd.concat([aita_top_df, data_to_append], ignore_index=True)
                        print("anzahl zeilen 1 erste iteration ", len(aita_top_df))

                        result = aita_top_df.to_json(orient="records")
                        parsed = json.loads(result)
                        filename = "Crawling_" + str(crawl_iteration) + ".json"
                        self.save_json_to_file(parsed, "../../DataFiles/CrawlDirectory", filename)
                        print("Saved ", len(aita_top_df), " items into file", filename)

                    else:
                        print("Request failed: ", request.status_code)
                        time.sleep(60)
                        request_headers = self.reddit_authenfication()
                        print("new request headers", request_headers)

                else:
                    print("new crawling object for", self.after_object)
                    if self.after_object is not None:
                        aita_top_url_after = "https://oauth.reddit.com/r/AmItheAsshole?limit=100&after=" + self.after_object
                        after_request = requests.get(aita_top_url_after,
                                                     headers=request_headers)
                        print("request status code", after_request.status_code)
                        if after_request.status_code == 200:
                            after_request_json = after_request.json()
                            data_to_append_after = self.aita_dataframe(after_request_json, request_headers)
                            aita_top_df = pd.concat([aita_top_df, data_to_append_after], ignore_index=True)
                            print("anzahl zeilen weitere iterationen", len(aita_top_df))
                            result = aita_top_df.to_json(orient="records")
                            parsed = json.loads(result)
                            filename = "Crawling_" + str(crawl_iteration) + ".json"
                            self.save_json_to_file(parsed, "../../DataFiles/CrawlDirectory", filename)
                            print("Saved ", len(aita_top_df), " items into file", filename)
                        else:
                            print("Request failed: ", after_request.status_code)
                            time.sleep(60)
                            request_headers = self.reddit_authenfication()
                            print("new request headers", request_headers)
                    else:
                        continue
                # convert to json to save in file - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
        result = aita_top_df.to_json(orient="records")
        parsed = json.loads(result)
        self.save_json_to_file(parsed, "DataFiles/BaseData", "BaseDataFinal_2.json")

    def aita_dataframe(self, request_json, headers) -> pd.DataFrame:
        """
        Verarbeitet die Request-Antworten aus aita_top() und aita_hot(). Die Request-Responses werden mit einem Pandas-Dataframe
        sortiert und dann in ein JSON-Format umgewandelt und gespeichert. Dies basiert auf diesem Tutorial: https://www.youtube.com/watch?v=FdjVoOf9HN4&t=1046s

        :param request_json: Request-Response
        :param headers: Initialer
        :return: Pandas-Dateframe mit Auswahl an Metadaten zu dem jeweiligen Post, d.h. z.B. Author, Postinhalt(Content) oder Bewertung.
        """
        request_headers = headers

        dataframe = pd.DataFrame()
        aita_json = request_json
        # Befüllung des after_objects, damit die nächste Iteration weiß wo die aktuelle stehen geblieben ist.
        self.after_object = aita_json['data']['after']
        print("after_obj dataframe method", self.after_object)
        aita_json_list = aita_json['data']['children']


        author_details = {}
        for i, post in enumerate(aita_json_list, start=28742):
            print("index post: ", i)
            post_author = post['data']['author']
            time.sleep(1)
            author_details = self.crawl_author_data(post_author, headers)
            post_url = post['data']['url']
            link_flair_text = post['data']['link_flair_text'] # Urteil bzw. Verdict der Community, hier gespeichert als Moderations-Flag
            time.sleep(1)
            comment_data_final = []
            if post_url == '/r/vaxxhappened/comments/pbe8nj/we_call_upon_reddit_to_take_action_against_the/':
                print("url not working")
            else:

                print("post_url", post_url)
                post_url = post_url.replace("www.reddit.com", "oauth.reddit.com") + "comments"
                commentdata_request = requests.get(post_url,
                                                   headers=request_headers)
                print("commentdata_request", commentdata_request)

                if commentdata_request.status_code != 200:
                    tmp_dict_test = {}
                    tmp_dict_test['error'] = "Request failed."
                    request_headers = self.reddit_authenfication()
                    print("new request comment headers", request_headers)

                    comment_data_final = comment_data_final.append(tmp_dict_test)
                else:
                    comment_data_json = commentdata_request.json()
                    comment_data_final = self.crawl_commentdata(comment_data_json)

            dataframe = dataframe.append({
                'title': post['data']['title'],
                'content': post['data']['selftext'],
                'verdict': link_flair_text,
                'edited': post['data']['edited'],
                'created': post['data']['created_utc'],
                'author_fullname': post_author,
                'author_details': author_details,
                'upvote_ratio': post['data']['upvote_ratio'],
                'downvotes': post['data']['downs'],
                'upvotes': post['data']['ups'],
                'reports_total': post['data']['num_reports'],
                'mod_reports': post['data']['mod_reports'],
                'user_reports': post['data']['user_reports'],
                'url': post['data']['url'],
                'comment_dtails': comment_data_final
            }, ignore_index=True)



        return dataframe

    def crawl_author_data(self, user_name: str, headers):
        """
        Crawlt Daten zum Author des Posts.
        :param user_name: Name des Nutzers, zu dem die Daten gecrawlt werden sollen.
        :param headers: Request Header, aus der Authenfizierungsmethode.
        :return: Dictionary mit den Details zum Author des Posts.
        """
        print("crawl author data")
        author_details = {}
        authordata_request = requests.get("https://oauth.reddit.com/user/" + user_name + "/about",
                                          headers=headers)
        if authordata_request.status_code != 200:
            author_details = {"Error": "Request failed."}
            return author_details
        else:
            if 'data' or 'subreddit' not in authordata_request.json():
                author_details = {"Error": 'No author data found'}
            else:
                author_details = {
                    "over_18": authordata_request.json()['data']['subreddit']['over_18'],
                    "user_is_banned": authordata_request.json()['data']['subreddit']['user_is_banned'],
                    "free_from_reports": authordata_request.json()['data']['subreddit']['free_form_reports'],
                    "total_karma": authordata_request.json()['data']['total_karma'],
                    "comment_karma": authordata_request.json()['data']['comment_karma'],
                    "is_mod": authordata_request.json()['data']['is_mod'],
                    "is_employee": authordata_request.json()['data']['is_employee'],
                }
        return author_details

    def crawl_commentdata(self, request_json):
        """
        Crawlt die zum Post zugehörigen Daten. Auch hier: Sortierung und Bearbeitung der Request Response mit Pandas Dataframe.
        :rtype: Python Liste mit den Daten zu den jeweiligen Kommentaren.
        """
        print("crawl comment data")

        comment_data_final = []
        commentdata_json = request_json
        commentdata_data = commentdata_json[1]['data']['children']

        # https://stackoverflow.com/questions/42204153/generating-a-dynamic-nested-json-object-and-array-python

        for b, comment in enumerate(commentdata_data):
            comment_content = str(comment['data'].get('body'))
            tmp_dict = {}
            if comment['kind'] == 't1':
                tmp_dict['index'] = b
                tmp_dict['author'] = comment['data'].get('author')
                tmp_dict['content'] = comment_content
                tmp_dict['user_reports'] = comment['data'].get('user_reports')
                tmp_dict['mod_reports'] = comment['data'].get('mod_reports')
                tmp_dict['score'] = comment['data'].get('score')
                tmp_dict['comment_ups'] = comment['data'].get('ups')
                comment_data_final.append(tmp_dict)
            else:
                break
        return comment_data_final

    # https://stackoverflow.com/questions/66653796/save-json-file-to-specific-folder-with-python
    def save_json_to_file(self, json_object, directory_path: str, file_title: str):
        complete_path = os.path.join(directory_path, file_title)
        with open(complete_path, "w+") as outfile:
            json.dump(json_object, outfile)

