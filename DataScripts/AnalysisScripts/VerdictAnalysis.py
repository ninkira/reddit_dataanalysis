import matplotlib.pyplot as plt
from DataScripts.DataProcessing.ProcessingBase import ProcessingBase
class RatingAnalysis:
    def __init__(self):
        print("VerdictAnalysis")


    def analyse_official_rating(self):
        print("official rating starts")
        aita_data = ProcessingBase().load_datafile("DataFiles", "aita_top_final.json")
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
        plt.title('Offizielle Bewertung Total')
        plt.show()

