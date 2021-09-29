import ntpath
import string

import matplotlib.pyplot as plt
import json, os, re

import numpy as np
import pandas as pd


class ImportData():
    # TODO: Exclude, make own file

    def load_datafile(self, filename: str):
        directory_path = "DataFiles"
        complete_path = os.path.join(directory_path, filename)
        with open(complete_path) as json_file:
            data = json.load(json_file)
            # print("Data:", data)
            return data


class RatingAnalysis:
    def __init__(self):
        print("Test")

    def analyse_community_rating(self):
        aita_data = ImportData().load_datafile("aita_top_2.json")
        # print("aita_data: ", aita_data)
        # community rating options
        nta_count = 0  # not the asshole
        nah_count = 0  # no assholes here
        yta_count = 0  # you're the asshole
        esh_count = 0  # everybody sucks here
        neutral_count = 0  # no rating in comment
        total_count_postings = 0
        label = 0
        community_rating = ''
        community_ratings = {}

        nta_count_total = 0  # not the asshole
        nah_count_total = 0  # no assholes here
        yta_count_total = 0  # you're the asshole
        esh_count_total = 0  # everybody sucks here
        neutral_count_total = 0  # no rating in comment
        comment_count = 0

        for post in aita_data:
            # print("post", post)
            total_count_postings += 1
            comments = post['comment_details']
            if comments != None:
                for comment in comments:


                    # print("comment", comment)

                    content = comment['content']
                    upvotes = comment['comment_ups']
                    if upvotes > 500:

                        # find char or word in string: https://www.geeksforgeeks.org/python-string-find/
                        # clean up comment_content with regex to remove punctuation and find more variantions of NTA or YTA
                        comment_content_filtered = content.translate(str.maketrans('', '', string.punctuation))
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
                        elif re.match(r'(?:^|\W)ESH(?:$|\W)', comment_content_filtered):
                            esh_count += 1
                            rating = "ESH - everybody sucks here"
                        # print("everbody is an asshole; esh_count: ", esh_count)
                        elif re.match(r'(?:^|\W)NAH(?:$|\W)', comment_content_filtered):
                            nah_count += 1
                            rating = "NAH - everybody sucks here"
                        # print("no assholes here; nah_count: ", nah_count)
                        else:
                            neutral_count += 1
                            #    print('Neutral, neutral_count: ', neutral_count)
                            rating = "Neutral"
                    else:
                        continue
                # https://stackoverflow.com/questions/15112125/how-to-test-multiple-variables-against-a-single-value
                if yta_count > nta_count and yta_count > neutral_count and yta_count > nah_count and yta_count > esh_count:
                    print("ASSHOLE - community rating posting")
                    label = 0
                    yta_count_total += 1
                    community_rating = 'Asshole'
                elif nta_count > yta_count and nta_count > neutral_count and nta_count > nah_count and nta_count > esh_count:
                    print("Not the Asshole - community rating posting")
                    label = 1
                    nta_count_total += 1
                    community_rating = 'Not an Asshole'
                elif nah_count > yta_count and nah_count > neutral_count and nah_count > nta_count and nah_count > esh_count:
                    print("No Assholes here - community rating posting")
                    label = 2
                    nah_count_total += 1
                    community_rating = 'Not an Asshole'
                elif esh_count > yta_count and esh_count > neutral_count and esh_count > nta_count and esh_count > nah_count:
                    print("Everybody sucks here - community rating posting")
                    label = 3
                    esh_count_total += 1
                    community_rating = 'Everybody sucks here'
                elif neutral_count > yta_count and neutral_count > nta_count and neutral_count > nah_count and neutral_count > esh_count:
                    print("Neutral - community rating posting")
                    label = 4
                    neutral_count_total += 1
                    community_rating = 'Neutral'

                print("post rating: " + community_rating, str(nta_count), str(yta_count), str(esh_count),
                      str(neutral_count), str(nah_count), str(label))
                print("total rating", total_count_postings, nta_count_total, yta_count_total, esh_count_total,
                      neutral_count_total, nah_count_total)
                community_ratings['total_postings'] = total_count_postings
                community_ratings['total_nta'] = nta_count_total
                community_ratings['total_yta'] = yta_count_total
                community_ratings['total_esh'] = esh_count_total
                community_ratings['total_nah'] = nah_count_total
                community_ratings['total_neutral'] = neutral_count_total
                yta_count = 0
                nta_count = 0
                esh_count = 0
                nah_count = 0
                neutral_count = 0
                print("analys finished, next post")


            else:
                print("else")
                print("Community rating final", community_rating)
                return community_ratings

        print("Community rating final", community_rating)
        return community_ratings

    def analyse_official_rating(self):
        print("official rating starts")
        aita_data = ImportData().load_datafile("aita_top_2.json")
        nta_count = 0  # not the asshole
        nah_count = 0  # no assholes here
        yta_count = 0  # you're the asshole
        esh_count = 0  # everybody sucks here
        neutral_count = 0  # no rating in comment
        total_count_postings = 0
        official_rating = {}
        for idx, post in enumerate(aita_data):

            verdict = post["verdict"]

            print("verdict", verdict)
            if verdict == "Not the A-hole":
                nta_count += 1
                print("official nta", nta_count)
            elif verdict == "Asshole":
                yta_count += 1
            elif verdict == "Everyone Sucks":
                esh_count += 1
            elif verdict == "No A-holes here":
                nah_count += 1
            else:
                neutral_count += 1



        print("total rating", total_count_postings, nta_count, yta_count, esh_count,
              neutral_count, nah_count)
        official_rating['total_postings'] = total_count_postings
        official_rating['total_nta'] = nta_count
        official_rating['total_yta'] = yta_count
        official_rating['total_esh'] = esh_count
        official_rating['total_nah'] = nah_count
        official_rating['total_neutral'] = neutral_count
        print("official rating", official_rating)

        return  official_rating



    def visualise_community_results(self):
        # https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html

        labels = 'YTA', 'NTA', 'NAH', 'ESH', "Neutral"
        data = self.analyse_community_rating()
        print(data)
        print("Test data", data['total_yta'], )
        sizes = [data['total_yta'], data['total_nta'], data['total_nah'], data['total_esh'], data['total_neutral']]
        explode = (0, 0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Community rating of posts total')
        plt.show()


    def visiualise_official_results(self):
        print("official results visual")

        labels = 'YTA', 'NTA', 'NAH', 'ESH', "Neutral"
        data = self.analyse_official_rating()
        print(data)
        print("Test data", data['total_yta'], )
        sizes = [data['total_yta'], data['total_nta'], data['total_nah'], data['total_esh'], data['total_neutral']]
        explode = (0, 0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Official rating of posts total')
        plt.show()

    def compare_results(self):
        print("test")
    """


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
      tmp_dict['rating'] = rating

    # counts to check what the users think about the og poster
    nta_count = 0
    yta_count = 0
    neutral_count = 0
    total_count = 0
    label = 0
    community_rating = ''

    """


class GenderAnalysis:
    def __init__(self):
        print("Gender analysis")
#source: https://dvc.org/blog/a-public-reddit-dataset
    def gender_analysis(self,):
        # Import Data
        aita_data = ImportData().load_datafile("aita_top_2.json")

        # first impression: count whenever a person is mentioned in a post, based on community rating
        ahole_contains_mom_counter = 0
        ahole_contains_dad_counter = 0
        ahole_contains_gf_counter = 0
        ahole_contains_bf_counter = 0
        ahole_contains_ex_counter = 0

        nta_contains_mom_counter = 0
        nta_contains_dad_counter = 0
        nta_contains_gf_counter = 0
        nta_contains_bf_counter = 0
        nta_contains_ex_counter = 0
        for idx, post in enumerate(aita_data):
            # text preprocessing
            content = post["content"]
            content_filtered = str(content).translate(str.maketrans('', '', string.punctuation))
            content_filtered = content_filtered.lower()

            verdict = post['verdict']
            print("verdict", verdict)

            if verdict == 'Asshole':


            #print("post_content", content_filtered)
# re.match(r'(?:^|\W)YTA(?:$|\W)', comment_content_filtered):
            #if re.match(r'(?:^|\W)mom(?:$|\W)', content_filtered):
                if content_filtered.__contains__(' mom ') or content_filtered.__contains__(' mother '):
                    print("Text contains Mom")
                    ahole_contains_mom_counter += 1

                if content_filtered.__contains__(' dad ') or content_filtered.__contains__(' father '):
                    print("Text contains Dad")
                    ahole_contains_dad_counter += 1

                #elif re.match(r'(?:^|\W)dad(?:$|\W)', content_filtered) or re.match(r'(?:^|\W)father(?:$|\W)', content_filtered) or  re.match(r'(?:^|\W)papa(?:$|\W)', content_filtered):
                #elif content_filtered.__contains__(' dad ') or content_filtered.__contains__(' father '):
                if content_filtered.__contains__(' gf ') or content_filtered.__contains__(' girlfriend ') or content_filtered.__contains__(' wife '):
                    print("Text contains girlfriend")
                    ahole_contains_gf_counter += 1
                    print("ahole gf counts", ahole_contains_gf_counter)

                if content_filtered.__contains__(' bf ') or content_filtered.__contains__(
                        ' husband ') or content_filtered.__contains__(' boyfriend '):
                    print("Text contains boyfriend")
                    ahole_contains_bf_counter += 1

                if content_filtered.__contains__(
                        ' ex '):
                    print("Text contains ex-gf")
                    ahole_contains_ex_counter += 1

            if verdict == 'Not the A-hole':

                # print("post_content", content_filtered)
                # re.match(r'(?:^|\W)YTA(?:$|\W)', comment_content_filtered):
                # if re.match(r'(?:^|\W)mom(?:$|\W)', content_filtered):
                if content_filtered.__contains__(' mom ') or content_filtered.__contains__(' mother '):
                    print("Text contains Mom")
                    nta_contains_mom_counter += 1

                if content_filtered.__contains__(' dad ') or content_filtered.__contains__(' father '):
                    print("Text contains Dad")
                    nta_contains_dad_counter += 1

                # elif re.match(r'(?:^|\W)dad(?:$|\W)', content_filtered) or re.match(r'(?:^|\W)father(?:$|\W)', content_filtered) or  re.match(r'(?:^|\W)papa(?:$|\W)', content_filtered):
                # elif content_filtered.__contains__(' dad ') or content_filtered.__contains__(' father '):
                if content_filtered.__contains__(' gf ') or content_filtered.__contains__(
                        ' girlfriend ') or content_filtered.__contains__(' wife '):
                    print("Text contains girlfriend")
                    nta_contains_gf_counter += 1
                    print("ahole gf counts", ahole_contains_gf_counter)

                if content_filtered.__contains__(' bf ') or content_filtered.__contains__(
                        ' husband ') or content_filtered.__contains__(' boyfriend '):
                    print("Text contains boyfriend")
                    nta_contains_bf_counter += 1

                if content_filtered.__contains__(
                    ' ex '):
                    print("Text contains ex-gf")
                    nta_contains_ex_counter += 1


        print("a-hole-counters",ahole_contains_bf_counter, ahole_contains_gf_counter, ahole_contains_dad_counter, ahole_contains_mom_counter, ahole_contains_ex_counter)
        print("nta counters", nta_contains_bf_counter, nta_contains_gf_counter, nta_contains_dad_counter,
              nta_contains_mom_counter, nta_contains_ex_counter)
            #elif re.match(r'(?:^|\W)girlfriend(?:$|\W)', content_filtered) or re.match(r'(?:^|\W)wif(?:$|\W)', content_filtered):


        odds_mom = np.log2(ahole_contains_mom_counter / nta_contains_mom_counter)
        odds_dad = np.log2(np.mean(ahole_contains_dad_counter) / np.mean(nta_contains_dad_counter))
        odds_gf = np.log2(np.mean(ahole_contains_gf_counter) / np.mean(nta_contains_gf_counter))
        odds_bf = np.log2(np.mean(ahole_contains_bf_counter) / np.mean(nta_contains_bf_counter))

        print("odds", odds_mom, odds_dad, odds_gf, odds_bf)



        # estimate oddds - based on https://dvc.org/blog/a-public-reddit-dataset
        dataframe = pd.DataFrame()
        for idx, post in enumerate(aita_data):
            # text preprocessing
            content = post["content"]
            verdict = post['verdict']
            content_filtered = str(content).translate(str.maketrans('', '', string.punctuation))
            content_filtered = content_filtered.lower()
            if verdict == "Asshole":
                dataframe = dataframe.append({
                    'text': content_filtered,
                    'is_asshole': 1
                }, ignore_index=True)
            elif verdict == "Not the A-hole":
                dataframe = dataframe.append({
                    'text': content_filtered,
                    'is_asshole': 0
                }, ignore_index=True)

            else:
                continue

        print("dataframe", dataframe['text'])
        dataframe['contains_mom'] = dataframe['text'].str.contains("mom|mother", case=False)
        dataframe['contains_dad'] = dataframe['text'].str.contains("dad|father", case=False)
        dataframe['contains_gf'] = dataframe['text'].str.contains("wife|girlfriend|gf", case=False)
        dataframe['contains_bf'] = dataframe['text'].str.contains("husband|boyfriend|bf", case=False)
        dataframe['contains_ex'] = dataframe['text'].str.contains("ex|ex-bf|ex-gf|ex-boyfriend|ex-girlfriend", case=False)

        yta = dataframe[dataframe['is_asshole'] == 1]
        nta = dataframe[dataframe['is_asshole'] == 0]

        odds_mom = np.log2(np.mean(yta['contains_mom']) / np.mean(nta['contains_mom']))
        odds_dad = np.log2(np.mean(yta['contains_dad']) / np.mean(nta['contains_dad']))
        odds_gf = np.log2(np.mean(yta['contains_gf']) / np.mean(nta['contains_gf']))
        odds_bf = np.log2(np.mean(yta['contains_bf']) / np.mean(nta['contains_bf']))
        odds_ex = np.log2(np.mean(yta['contains_ex']) / np.mean(nta['contains_ex']))

        who = ["Mom", "Dad", "Wife/Girlfriend", "Husband/Boyfriend", "Ex"]
        odds = [odds_mom, odds_dad, odds_gf, odds_bf, odds_ex]

        odds_df = pd.DataFrame(zip(who, odds), columns=["Who", "LogOdds"])
        odds_df['direction'] = odds_df['LogOdds'] > 0
        print("odds df", odds_df)


