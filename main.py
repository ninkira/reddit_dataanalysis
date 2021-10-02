from DataScripts.CrawlingScripts.DataCrawling_Reddit import RedditCrawler
from MLPipeline.DistilBERT_Pipeline import SequenceClassification
from DataScripts.AnalysisScripts.CommentAnalysis import CommunityRating
from DataScripts.AnalysisScripts.VerdictAnalysis import RatingAnalysis
from DataScripts.TopicModellingScripts.TopicModelling import TopicModelling
from DataScripts.DataProcessing.ProcessingBase import ProcessingBase
from DataScripts.DataProcessing.DatasetCreation.TopicModellingDatasetCreation import TopicModellingDataset
from DataScripts.DataProcessing.DatasetCreation.DataSetCreation import DataSet
if __name__ == '__main__':

    """Das main.py Skript stellt einige der zentralen Methoden des Packages bereit. Hier werden examplerisch einige 
    Visualisierung aufgerufen, z.B. die der Verdict- und Kommentar-Analysen. """
    # Crawling
    # reddit_crawler = RedditCrawler()
    # reddit_crawler.crawl_aita_top(reddit_crawler.reddit_authenfication())


    print("==== Starte Analyse Top Dataset ====")

    # Display Dataset Features
    processing_base = ProcessingBase()
    top_data = processing_base.load_datafile_utf8("DataFiles", "aita_top_final.json")
    # Anzahl Datenpunkte in JSON
    print("Top Datenset Anzahl der Datenpunkte: ", len(top_data))

    base_data = processing_base.load_datafile_utf8("DataFiles", "posts_json_large.json")
    # Anzahl Datenpunkte in JSON
    for batch in base_data:
         print("Top Datenset Anzahl der Datenpunkte: ", len(batch))

    # Verdict und Kommentar Untersuchung
    comunity_rating = CommunityRating()
    comunity_rating.analyse_community_rating()
    comunity_rating.visualise_community_results()
    rating_analysis = RatingAnalysis()
    #rating_analysis.analyse_official_rating()
    #rating_analysis.visiualise_official_results()

    # TopicModelling
    print("=== Topic Modelling Analyse ===")
    print("Erstelle Datenset für Analyse aus Top")
    tm_datasetcreation = TopicModellingDataset()
    #tm_datasetcreation.create_datasets(top_data)
    #top_nta_data = processing_base.load_datafile_utf8("DataFiles/TopicModellingData", "tm_nta_data.json")
    #top_yta_data = processing_base.load_datafile_utf8("DataFiles/TopicModellingData", "tm_yta_data.json")



    #topic_modelling = TopicModelling()
    #topic_modelling.run_topicmodelling(top_data, "Results/TopicModelling/TopicModelling_Total.html") # Gesamtes Top Datenset
    #topic_modelling.run_topicmodelling(top_nta_data,
     #                                  "Results/TopicModelling/TopicModelling_NTA.html")  # NTA Datenset
    #topic_modelling.run_topicmodelling(top_yta_data,
     #                                 "Results/TopicModelling/TopicModelling_YTA.html")  # YTA Datenset

    #print("connect Files")
    #ProcessingBase().connect_files()
    data_set = DataSet()
    #dataset_frame = data_set.create_labels()
    #data_set.sort_dataframe(dataset_frame)
    #data_set.create_dataset()


    sequence_classification = SequenceClassification()
    sequence_classification.run_pipeline()


