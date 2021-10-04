# AITA-Datenanalyse
Datengetriebene Analyse des Subreddits "Am I the Asshole?" in der Hausarbeit "Bewertung von Konflikten auf Reddit:
Eine datengetriebene Untersuchung des Subreddits "Am I the Asshole?" von Nina Eckertz im Kurs "Embdeddings" bei Prof. Dr. Nils Reiter.  

# Repository Aufbau
Das im Repository enthaltene Python Package beinhaltet zwei Module: 
* DataScripts 
* MLPipeline
  
DataScripts enthält in den Sub-Modulen AnalysisScripts, CrwalingScripts, DataProcessing und TopicModellingSripts sämtliche Skripte, um Datenpunkte aus AITA zu crawlen als auch zu analysieren. MLPipeline dagegen enthält die Machine Learning Pipeline auf Basis des DistilBERT-Modells, um AITA-Posts in einer binären Klassifikation zu beurteilen. Zudem werden die verschiedenen Datensätze und Datensets für die Analysen im Ordner DataFiles und die Ergebnisse der in dem Ordner Results bereitgestellt.

Im main.py-Skript können die einzelnen Skripte und ihre Funktionen aufgerufen werden. Im Endzustand werden hier ein Auszug der Analyseergebnisse aufgerufen, z.B. die Konfusion Matrix zu Visualisierung des Trainings der Machine Learning Pipeline. Die oben genannten Skripte und ihre Hauptfunktionen sind hier ebenfalls auskommentiert vorhanden und können bei Bedarf und unter Anpassung der Speicher- und Lade-Pfade für die Dateien wieder ein kommentiert und genutzt werden. Hinter diesem Aufbau von main.py steht der Gedanke, vom Herunterladen über das Vorbereiten, der Anlayse und der Untersuchung mit DistilBERT alle Schritte zu automatisieren und theoretisch in einer Anwendung ausführen zu können.

# Ausführung und Installation

Das Package ist mit dem IDE PyCharm entwickelt worden und darauf auslegt mit PyCharm in einem Virtual Environment ausgeführt zu werden.

Hierfür sind u.a. folgende Packages zu installieren:

* Python                  3.9.0
* en-core-web-sm          3.1.0
* matplotlib              3.4.3
* nlp                     0.4.0
* nltk                    3.6.3
* numpy                   1.21.1
* oauthlib                3.1.1
* pandas                  1.3.1
* pip                     21.2.3
* pyLDAvis                3.2.0
* requests                2.26.0
* requests-oauthlib       1.3.0
* scikit-learn            0.24.2
* sklearn                 0.0
* spacy                   3.1.3
* spacy-legacy            3.0.8
* torch                   1.7.1+cu110
* transformers            4.9.2

Siehe hierzu auch die Import-Statements des auszuführenden Skripts. Die Modelle für das Package ML-Pipeline können unter dem folgenden Link heruntergeladen werden: 
https://drive.google.com/file/d/1nQ8-JDfpFmLLVc7lqRmRdnvYZ--H3hpO/view?usp=sharing

Sollte etwas nicht funktionieren, bitte ich um möglichst schnelle Benachrichtigung.