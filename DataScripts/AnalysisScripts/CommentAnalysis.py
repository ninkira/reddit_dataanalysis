import string
import matplotlib.pyplot as plt
import json, os, re

from DataScripts.DataProcessing.ProcessingBase import ProcessingBase
""" Diese Klasse wertet die Kommentare aus dem AITA-Top Datenset aus und visualisiert die Ergebnisse in einer Pie Chart. """

class CommunityRating:

    def analyse_community_rating(self):
        aita_data = ProcessingBase().load_datafile("aita_top_2.json")
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