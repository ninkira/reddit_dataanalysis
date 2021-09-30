
from DataScripts.DataCrawling import DataCrawler
from DataScripts.CrawlingScripts.DataCrawling_Reddit import RedditCrawler

from DataScripts.AnalysisScripts.CommentAnalysis import CommunityRating
from DataScripts.AnalysisScripts.VerdictAnalysis import RatingAnalysis
from DataScripts.TopicModellingScripts.TopicModelling import TopicModelling
from DataScripts.DataProcessing.ProcessingBase import ProcessingBase

if __name__ == '__main__':
    # Crawling
    # reddit_crawler = RedditCrawler()
    # reddit_crawler.crawl_aita_top(reddit_crawler.reddit_authenfication())


    print("==== Starte Analyse Top Dataset ====")

    # Display Dataset Features
    processing_base = ProcessingBase()
    top_data = processing_base.load_datafile_utf8("DataFiles", "aita_top_8.json")
    # Anzahl Datenpunkte in JSON
    print("Top Datenset Anzahl der Datenpunkte: ", len(top_data))


    # Verdict und Kommentar Untersuchung
    comunity_rating = CommunityRating()
    comunity_rating.analyse_community_rating()
    comunity_rating.visualise_community_results()
    rating_analysis = RatingAnalysis()
    rating_analysis.analyse_official_rating()
    rating_analysis.visiualise_official_results()

    # TopicModelling
    print("=== Topic Modelling Analyse ===")
    topic_modelling = TopicModelling()
    topic_modelling.run_topicmodelling(top_data, "Results/TopicModelling/TopicModelling_Total.html")

