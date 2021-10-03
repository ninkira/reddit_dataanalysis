import string
import matplotlib.pyplot as plt
import json, os, re

import pandas as pd

from DataScripts.DataProcessing.ProcessingBase import ProcessingBase

"""Diese Klasse wertet die Kommentare aus dem AITA-Top Datenset aus und visualisiert die Ergebnisse in einer Pie 
Chart. """


class CommunityRating:

    def analyse_community_rating(self):
        """
        Wertet die Kommentare aus dem Top-Datenset aus. Die Kommentare pro Post werden mit einer Regex je nach Urteil NTA,
        NAH, YTA, ESH und INFO untersucht. Alles, was nicht in diese Kategorien fällt (z.B. ein Kommentar ohne
        explizite Wertung) fällt unter die Kategorie "Neutral".

        :rtype: object
        """
        aita_data = ProcessingBase().load_datafile("DataFiles", "aita_top_final.json")
      
        # Community Bewertung Optionen
        nta_count = 0  # not the asshole
        nah_count = 0  # no assholes here
        yta_count = 0  # you're the asshole
        esh_count = 0  # everybody sucks here
        info_count = 0 # not enough info
        neutral_count = 0  # Keine explizite Wertung im Kommentar
        total_count_postings = 0
        community_rating = ''
        community_ratings = {}



        nta_count_total = 0  # not the asshole
        nah_count_total = 0  # no assholes here
        yta_count_total = 0  # you're the asshole
        esh_count_total = 0  # everybody sucks here
        info_count_total = 0
        neutral_count_total = 0  # no rating in comment
        community_rating_single_post = {}

        for idx, post in enumerate(aita_data):
            # Überspringe ersten Post, da Moderation Bewertung
            total_count_postings += 1
            comments = post['comment_details']
            if comments != None:
                for c_idx, comment in enumerate(comments):
                    content = comment['content']
                    author = comment['author']
                    score = comment['score']
                    # Qualitative Kriterien für Kommentar-Einbeziehung: Herausfiltern von Kommentaren von Bots oder ohne Inhalt, sowie mit einer Grundbewertung von mindestens 10 Punkten.
                    if score > 10 and content != '[deleted]' and content != '[removed]' and author != 'AutoModerator':

                        # Quelle, wie man einzelne Wörter in Strings findet: https://www.geeksforgeeks.org/python-string-find/
                        # Preprocessing: Clean-up eines Kommentar mit regex um Satzzeichen zu entfernen, hilft dabei mehr Varaitionen von NTA bzw. YTA zu finden.
                        comment_content_filtered = content.translate(str.maketrans('', '', string.punctuation))

                        # or comment_content_filtered.find("You're the Asshole") or comment_content_filtered.find('Not the Asshole')
                        if re.match(r'(?:^|\W)YTA(?:$|\W)', comment_content_filtered):
                            yta_count += score
                        elif re.match(r'(?:^|\W)NTA(?:$|\W)', comment_content_filtered):
                            nta_count += score
                        elif re.match(r'(?:^|\W)ESH(?:$|\W)', comment_content_filtered):
                            esh_count += score
                        elif re.match(r'(?:^|\W)NAH(?:$|\W)', comment_content_filtered):
                            nah_count += score
                        elif re.match(r'(?:^|\W)INFO(?:$|\W)', comment_content_filtered):
                            info_count += score
                        else:
                            neutral_count += score # Auskommentieren falls neutrale Kommentare gar nicht gewertet werden sollen sollen
                            # neutral_count += 1 Falls Kommentare nur mit Faktor 1 gewertet werden sollen.
                    else:
                        continue

                if yta_count > nta_count and yta_count > neutral_count and yta_count > nah_count and yta_count > esh_count and yta_count > info_count:
                    yta_count_total += 1
                    community_rating = 'Asshole'
                elif nta_count > yta_count and nta_count > neutral_count and nta_count > nah_count and nta_count > esh_count and nta_count > info_count:
                    nta_count_total += 1
                    community_rating = 'Not an Asshole'
                elif nah_count > yta_count and nah_count > neutral_count and nah_count > nta_count and nah_count > esh_count and nah_count > info_count:
                    nah_count_total += 1
                    community_rating = 'Not an Asshole'
                elif esh_count > yta_count and esh_count > neutral_count and esh_count > nta_count and esh_count > nah_count and esh_count > info_count:
                    esh_count_total += 1
                    community_rating = 'Everybody sucks here'
                elif neutral_count > yta_count and neutral_count > nta_count and neutral_count > nah_count and neutral_count > esh_count and neutral_count > info_count:
                    neutral_count_total += 1
                    community_rating = 'Neutral'
                elif info_count > yta_count and info_count > nta_count and info_count > nah_count and info_count > esh_count and info_count > neutral_count:
                    neutral_count_total += 1
                    community_rating = 'Info'

                community_rating_single_post = {
                    "title": post["title"],
                    "official_rating": post["verdict"],
                    "community_rating": community_rating,
                    "post_nta_score": nta_count,
                    "post_yta_score": yta_count,
                    "post_nah_score": nah_count,
                    "post_esh_score": esh_count,
                    "post_info_score": info_count,
                    "post_neutral_score": neutral_count,
                }
                # Visualisierung der einzelnen Post Bewertungen, kann bei Bedarf auskommentiert werden.
                #self.visualise_solo_results(community_rating_single_post)
                community_ratings['total_postings'] = total_count_postings
                community_ratings['total_nta'] = nta_count_total
                community_ratings['total_yta'] = yta_count_total
                community_ratings['total_esh'] = esh_count_total
                community_ratings['total_nah'] = nah_count_total
                community_ratings['total_info'] = info_count_total
                community_ratings['total_neutral'] = neutral_count_total
                yta_count = 0
                nta_count = 0
                esh_count = 0
                nah_count = 0
                neutral_count = 0
                info_count = 0
                community_rating_single_post = {}
            else:
                return community_ratings
        return community_ratings

    """Visualisierungs-Funktionen, welche die verschiedenen Community Bewertungen für einzelne oder alle Posts 
    darstellen. Basierend auf # https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html. """

    def visualise_community_results(self):

        labels = 'YTA', 'NTA', 'NAH', 'ESH', "Neutral", "Info"
        data = self.analyse_community_rating()
        sizes = [data['total_yta'], data['total_nta'], data['total_nah'], data['total_esh'], data['total_neutral'], data['total_info']]
        explode = (0, 0.1, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Community Bewertung Total')
        plt.show()

    def visualise_solo_results(self, data): # für einzelne Posts
        # https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html

        labels = 'YTA', 'NTA', 'NAH', 'ESH', "Neutral", "Info"
        sizes = [data['post_yta_score'], data['post_nta_score'], data['post_nah_score'], data['post_esh_score'], data['post_neutral_score'],
                 data['post_info_score']]
        explode = (0, 0.1, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        title_str = data["title"] + " (OV: "+ data["official_rating"] + " / CV: " + data["community_rating"] + ")"
        plt.title(title_str)
        plt.suptitle("Rating Analysis")
        plt.show()
