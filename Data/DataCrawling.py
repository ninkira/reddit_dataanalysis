# Source ersten beiden Crawls https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps
# general doc https://github.com/reddit-archive/reddit/wiki/API
import pandas as pd
import requests.auth
import json
import re, string, timeit


class Hot_DataCrawler():
    def __init__(self):
        self.reddit_authenfication_obj = self.reddit_authenfication()


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
        response2 = requests.get("https://oauth.reddit.com/r/AmItheAsshole/hot", headers=headers)
        # two subjobject: 'kind' and 'data'
        hot_amta = response2.json()
        print("keys:", hot_amta['data'].keys())
        # hot_amta['data']['children'] - is a list not a JSON object anymore
        amta_hot_subreddit_list = hot_amta['data']['children']

        # dataframe for better visualisation
        atma_hot_df = pd.DataFrame()
        for i, subreddit in enumerate(amta_hot_subreddit_list):
            print("index: ", i)
            if i == 0:
                continue
            elif i == 1:
                continue
            # again to sub_object: kind and data contains all necerssary information about the posting
            # subreddit is again JSON
            print(subreddit['data'])
            # selftext, author_fullname, name, downs, upvote_ratio, user_reports, score, edited, likes, view_count, over_18, num_reports
            post_author = subreddit['data']['author']
            authordata_request = requests.get("https://oauth.reddit.com/user/" + post_author + "/about",
                                              headers=headers)

            post_url = subreddit['data']['url']
            post_url = post_url.replace("www.reddit.com", "oauth.reddit.com") + "comments"

            commentdata_request = requests.get(post_url,
                                               headers=headers)

            commentdata_json = commentdata_request.json()
            commentdata_data = commentdata_json[1]['data']['children']

            # print("commentdata_json: ", commentdata_data)
            # https://stackoverflow.com/questions/42204153/generating-a-dynamic-nested-json-object-and-array-python
            comment_data_final = []

            # counts to check what the users think about the og poster
            nta_count = 0
            yta_count = 0
            neutral_count = 0
            total_count = 0
            label = 0
            community_rating = ''

            for b, comment in enumerate(commentdata_data):
                comment_content = str(comment['data'].get('body'))
                total_count += 1
            # find char or word in string: https://www.geeksforgeeks.org/python-string-find/
                # clean up comment_content with regex to remove punctuation and find more variantions of NTA or YTA

                comment_content_filtered = comment_content.translate(str.maketrans('', '', string.punctuation))
                print("comment_content: ", comment_content_filtered)
                #or comment_content_filtered.find("You're the Asshole") or comment_content_filtered.find('Not the Asshole')
                if re.match(r'(?:^|\W)YTA(?:$|\W)', comment_content_filtered):
                    yta_count += 1
                    rating = "YTA - You're the asshole"
                    print('Youre the asshole; yta_count: ', yta_count)
                elif re.match(r'(?:^|\W)NTA(?:$|\W)', comment_content_filtered):
                    nta_count += 1
                    rating = "NTA - Not the asshole"
                    print("not the asshole; nta_count: ", nta_count)
                else:
                    neutral_count += 1
                    print('Neutral, neutral_count: ', neutral_count)
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

            print("comment data final ", comment_data_final)
            print("user rating final", total_count, yta_count, nta_count, neutral_count)
            if yta_count > nta_count and neutral_count:
                print("ASSHOLE - community rating posting")
                label = 0
                community_rating = 'Asshole'
            elif nta_count > yta_count and neutral_count:
                print("ASSHOLE - community rating posting")
                label = 1
                community_rating = 'Not an Asshole'
            elif neutral_count > yta_count and neutral_count:
                print("Neutral - community rating posting")
                label = 2
                community_rating = 'Neutral'


            # push info into dataframe-
            atma_hot_df = atma_hot_df.append({
                'title': subreddit['data']['title'],
                'content': subreddit['data']['selftext'],
                'label': label,
                'community_ranking': community_rating,
                'author_fullname': post_author,
                'author_details': {
                    "over_18": authordata_request.json()['data']['subreddit']['over_18'],
                    "user_is_banned": authordata_request.json()['data']['subreddit']['user_is_banned'],
                    "free_from_reports": authordata_request.json()['data']['subreddit']['free_form_reports'],
                    "total_karma": authordata_request.json()['data']['total_karma'],
                    "comment_karma": authordata_request.json()['data']['comment_karma'],
                    "is_mod": authordata_request.json()['data']['is_mod'],
                    "is_employee": authordata_request.json()['data']['is_employee'],
                },
                'upvote_ratio': subreddit['data']['upvote_ratio'],
                'downvotes': subreddit['data']['downs'],
                'upvotes': subreddit['data']['ups'],
                'reports_total': subreddit['data']['num_reports'],
                'mod_reports': subreddit['data']['mod_reports'],
                'user_reports': subreddit['data']['user_reports'],
                'url': subreddit['data']['url'],
                'comment_rating':{
                    "total_comments": total_count,
                    'NTA' : nta_count,
                    'YTA' : yta_count,
                    'Neutral' : neutral_count
                },
                'comment_details': comment_data_final
            }, ignore_index=True)

            atma_hot_df
            # convert to json to save in file - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
            result = atma_hot_df.to_json(orient="records")
            parsed = json.loads(result)
            self.save_json_to_file(parsed)

    def save_json_to_file(self, json_object):
        # json.dumps(parsed, indent=4)
        with open("sample.json", "w") as outfile:
            json.dump(json_object, outfile)




    # print("data:", hot_amta['data']['children'][0])
    # get a reddit thread

    # source https://www.youtube.com/watch?v=FdjVoOf9HN4&t=1046s
    # pandas dataframe to show first visualisation

