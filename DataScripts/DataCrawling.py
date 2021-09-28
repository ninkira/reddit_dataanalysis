# Source ersten beiden Crawls https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps
# general doc https://github.com/reddit-archive/reddit/wiki/API
import pandas as pd
import requests.auth
import json
import os
import re, string, timeit
import threading, time


class DataCrawler():
    def __init__(self):
        self.reddit_authenfication_obj = self.reddit_authenfication()
        self.after_object: str = ''


    def reddit_authenfication(self):
        """
        This method provides all the necerssary authentification data to access the Reddit API.
        :return: request header for datacrawling from reddit
        """
        client_auth = requests.auth.HTTPBasicAuth('FnzYvE0WD851lymNYp7VLA', '8asQ3Nwer6y1g5jzvLRR3JNmIUsRDg')
        post_data = {"grant_type": "password", "username": "_ninkira", "password": "Password@reddit2021!"}
        headers = {"User-Agent": "DataBot/0.1 by _ninkira"}
        response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data,
                                 headers=headers)

        # Ergebnis: {'access_token': '728377922323-6EbD8mJFo3oMaB7OMIyl1W2gr9q8BQ', 'token_type': 'bearer', 'expires_in': 3600, 'scope': '*'}
        response_json = response.json()
        access_token = response_json['access_token']

        # header for request
        crawling_headers = {"Authorization": "bearer " + access_token, "User-Agent": "DataBot/0.1 by _ninkira"}
        return crawling_headers

    def crawl_userdata(self, header):
        header = self.reddit_authenfication()
        # Test - Crawl der eigenen Nutzerdaten direkt von der API
        response1 = requests.get("https://oauth.reddit.com/api/v1/me", headers=header)
        response1.json()
        print(response1.json())

        # Test - hier selbes profil aber "von außen"
        userdata_request = requests.get("https://oauth.reddit.com/user/_ninkira/about", headers=header)
        userdata_request_json = userdata_request.json()


    def crawl_aita_hot(self, headers):
        # for threads <oathurl> r= reddit <threadname> <hot> as special functionality

        request_url ="https://oauth.reddit.com/r/AmItheAsshole/hot"
        aita_hot_df = pd.DataFrame()
        data_to_append = self.aita_dataframe(request_url, headers)
        aita_hot_df = pd.concat([aita_hot_df, data_to_append])
        result = aita_hot_df.to_json(orient="records")
        parsed = json.loads(result)
        self.save_json_to_file(parsed, "aita_hot.json")

    def crawl_aita_top(self, headers):

        timeout = time.time() + 60*4
        aita_top_df = pd.DataFrame()
        self.after_object = ''
        print("crawl_aita_top starts; after_obj:", self.after_object)
        crawl_iteration = 0

        while True:
            time.sleep(2)
            crawl_iteration += 1
            print("crawl_iteration", crawl_iteration)
            print("after_obj", self.after_object)
            if crawl_iteration >= 20:
                print("crawling finished")
                break
            else:
                if self.after_object == '':
                    print("first crawling object")
                    # for threads <oathurl> r= reddit <threadname> <hot> as special functionality
                    # https://www.reddit.com/r/redditdev/comments/6anc5z/what_on_earth_is_the_url_to_get_a_simple_json/
                    # https://www.reddit.com/dev/api#listings - for url parameter
                    aita_top_url_base = "https://oauth.reddit.com/r/AmItheAsshole/top?t=all&limit=50"
                    data_to_append = self.aita_dataframe(aita_top_url_base, headers)
                    print("data_to_append", data_to_append)
                    aita_top_df = pd.concat([aita_top_df,data_to_append], ignore_index=True)
                    print("anzahl zeilen 1 erste iteration ",len(aita_top_df))
                else:
                    print("new crawling object for", self.after_object)
                    aita_top_url_after = "https://oauth.reddit.com/r/AmItheAsshole/top?t=all&limit=50&after=" + self.after_object
                    data_to_append_after = self.aita_dataframe(aita_top_url_after, headers)
                    aita_top_df = pd.concat([aita_top_df,data_to_append_after], ignore_index=True)
                    print("anzahl zeilen weitere iterationen", len(aita_top_df))
                # convert to json to save in file - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
        result = aita_top_df.to_json(orient="records")
        parsed = json.loads(result)
        self.save_json_to_file(parsed, "aita_top.json")


    def aita_dataframe(self, request_url: str, headers) -> pd.DataFrame:
        # two subjobject: 'kind' and 'data'

        dataframe = pd.DataFrame()
        request = requests.get(request_url,
                                 headers=headers)
        print("request status code", request.status_code)
        if request.status_code != 200:
            print("something went wrong, status code: ", request.status_code)
            dataframe = dataframe.append({"error": "data could not be fetched"}, ignore_index=True)
            return dataframe
        else:
            aita_json = request.json()
            # Needed if multiple Crawling to get more posts
            self.after_object = aita_json['data']['after']
            print("after_obj dataframe method", self.after_object)
            print("after_obj dataframe method", self.after_object)

            # print("keys:", aita_json['data'].keys())
            # hot_amta['data']['children'] - is a list not a JSON object anymore
            aita_json_list = aita_json['data']['children']

            # dataframe for better visualisation
            author_details = {}
            for i, post in enumerate(aita_json_list):
                print("index post: ", i)
                # again to sub_object: kind and data contains all necerssary information about the posting
                # post is again JSON
                # print(post['data'])
                # selftext, author_fullname, name, downs, upvote_ratio, user_reports, score, edited, likes, view_count, over_18, num_reports
                post_author = post['data']['author']
                time.sleep(1)
                authordata_request = requests.get("https://oauth.reddit.com/user/" + post_author + "/about",
                                                  headers=headers)

                post_url = post['data']['url']
                time.sleep(1)
                post_url = post_url.replace("www.reddit.com", "oauth.reddit.com") + "comments"

                # counts to check what the users think about the og poster
                nta_count = 0
                yta_count = 0
                neutral_count = 0
                total_count = 0
                label = 0
                community_rating = ''
                comment_data_final = []
                commentdata_request = requests.get(post_url,
                                                   headers=headers)
                print("commentdata_requerst", commentdata_request)
                if commentdata_request.status_code != 200:
                    tmp_dict_test = {}
                    tmp_dict_test['error'] = "No commentdata found."

                    comment_data_final = comment_data_final.append(tmp_dict_test)
                else:
                    commentdata_json = commentdata_request.json()
                    commentdata_data = commentdata_json[1]['data']['children']


                    # print("commentdata_json: ", commentdata_data)
                    # https://stackoverflow.com/questions/42204153/generating-a-dynamic-nested-json-object-and-array-python


                    for b, comment in enumerate(commentdata_data):
                        comment_content = str(comment['data'].get('body'))
                        total_count += 1
                        # find char or word in string: https://www.geeksforgeeks.org/python-string-find/
                        # clean up comment_content with regex to remove punctuation and find more variantions of NTA or YTA

                        comment_content_filtered = comment_content.translate(str.maketrans('', '', string.punctuation))
                        # print("comment_content: ", comment_content_filtered)
                        # or comment_content_filtered.find("You're the Asshole") or comment_content_filtered.find('Not the Asshole')
                        if re.match(r'(?:^|\W)YTA(?:$|\W)', comment_content_filtered):
                            yta_count += 1
                            rating = "YTA - You're the asshole"
                        # print('Youre the asshole; yta_count: ', yta_count)
                        elif re.match(r'(?:^|\W)NTA(?:$|\W)', comment_content_filtered):
                            nta_count += 1
                            rating = "NTA - Not the asshole"
                            # print("not the asshole; nta_count: ", nta_count)
                        else:
                            neutral_count += 1
                            # print('Neutral, neutral_count: ', neutral_count)
                            rating = "Neutral"

                        tmp_dict = {}

                        if comment['kind'] == 't1':
                            tmp_dict['index'] = b
                            tmp_dict['author'] = comment['data'].get('author')
                            tmp_dict['content'] = comment_content
                            tmp_dict['rating'] = rating
                            tmp_dict['user_reports'] = comment['data'].get('user_reports')
                            tmp_dict['mod_reports'] = comment['data'].get('mod_reports')
                            tmp_dict['score'] = comment['data'].get('score')
                            tmp_dict['comment_ups'] = comment['data'].get('ups')
                            comment_data_final.append(tmp_dict)
                        else:
                            break

                    # print("comment data final ", comment_data_final)
                    # print("user rating final", total_count, yta_count, nta_count, neutral_count)
                    if yta_count > nta_count and neutral_count:
                        #   print("ASSHOLE - community rating posting")
                        label = 0
                        community_rating = 'Asshole'
                    elif nta_count > yta_count and neutral_count:
                        #  print("ASSHOLE - community rating posting")
                        label = 1
                        community_rating = 'Not an Asshole'
                    elif neutral_count > yta_count and neutral_count:
                        # print("Neutral - community rating posting")
                        label = 2
                        community_rating = 'Neutral'

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

                # push info into dataframe-
                dataframe = dataframe.append({
                    'title': post['data']['title'],
                    'content': post['data']['selftext'],
                    'label': label,
                    'community_ranking': community_rating,
                    'author_fullname': post_author,
                    'author_details': author_details,
                    'upvote_ratio': post['data']['upvote_ratio'],
                    'downvotes': post['data']['downs'],
                    'upvotes': post['data']['ups'],
                    'reports_total': post['data']['num_reports'],
                    'mod_reports': post['data']['mod_reports'],
                    'user_reports': post['data']['user_reports'],
                    'url': post['data']['url'],
                    'comment_rating': {
                        "total_comments": total_count,
                        'NTA': nta_count,
                        'YTA': yta_count,
                        'Neutral': neutral_count
                    },
                    'comment_details': comment_data_final
                }, ignore_index=True)


        # for threads <oathurl> r= reddit <threadname> <hot> as special functionality
        # https://www.reddit.com/r/redditdev/comments/6anc5z/what_on_earth_is_the_url_to_get_a_simple_json/
        # https://www.reddit.com/dev/api#listings - for url parameter
            #print("dataframe ", dataframe)
            return dataframe

    # https://stackoverflow.com/questions/66653796/save-json-file-to-specific-folder-with-python
    def save_json_to_file(self, json_object, file_title: str):
        # json.dumps(parsed, indent=4)
        directory_path = "DataFiles"
        complete_path = os.path.join(directory_path, file_title)
        with open(complete_path, "w+") as outfile:
            json.dump(json_object, outfile)



    # get a reddit thread

    # source https://www.youtube.com/watch?v=FdjVoOf9HN4&t=1046s
    # pandas dataframe to show first visualisation

