import nltk

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# spacy
import spacy

nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import TfidfModel
# vis
import pyLDAvis
import pyLDAvis.gensim

from DataScripts.DataProcessing.ProcessingBase import ProcessingBase

"""
    Quellen für das Skript: 
    https://www.youtube.com/watch?v=TKjjlp5_r7o + Notebook https://github.com/wjbmattingly/topic_modeling_textbook/blob/main/03_03_lda_model_demo.ipynb
    https://www.youtube.com/watch?v=UEn3xHNBXJU
    
"""


class TopicModelling:

    def run_topicmodelling(self, data, target_file_path):
        """
        Basis Methode um Topic Modelling zu performen. Die Methode ist in folgende Schritte unterteilt: Vorbereitung der Daten, Preprocessing, Bilden von Bigrams und Trigrams,
        Verarbeitung der Daten mit dem TFIDG Modell und Visualisierung mit pyLDAvis.
        :type data: JSON-Objekt des zu bearbeitenden Datensets; wird in main.py geladen
        :type target_file_path: Zielpfad für Speichern der Visualisierung aus pyLDAvis.
        """
        # prepare data
        data_posts = self.prep_data(data)

        # Preprocessing - Herausfiltern von Stopwords und Transformation der Wörter in Grundform (Lemmatization)
        stop = set(stopwords.words('english'))
        lemmatized_texts = self.lemmatization(data_posts)
        lemmatized_data = list(self.gen_words(lemmatized_texts))
        print("Lemmetized Data Example:", lemmatized_data[0])

        # bigram and trigams
        bigrams_phrases = gensim.models.Phrases(lemmatized_data, min_count=5, threshold=100)
        trigram_phrases = gensim.models.Phrases(bigrams_phrases[lemmatized_data], threshold=100)

        bigram = gensim.models.phrases.Phraser(bigrams_phrases)
        trigram = gensim.models.phrases.Phraser(trigram_phrases)

        trigram_total = (trigram[bigram[lemmatized_data[i]]] for i in range(len(lemmatized_data)))
        trigram_total = list(trigram_total)

        id2word = corpora.Dictionary(trigram_total)
        print("id2word - Unique Tokens Examples:", id2word)
        corpus = [id2word.doc2bow(text) for text in trigram_total]

        # Verarbeitung der Daten mit tfidf
        tfidf = TfidfModel(corpus, id2word=id2word)

        low_value = 0.03
        words = []
        words_missing_in_tfidf = []

        for i in range(0, len(corpus)):
            bow = corpus[i]
            low_value_words = []  # reinitialize to be safe. You can skip this.
            tfidf_ids = [id for id, value in tfidf[bow]]
            bow_ids = [id for id, value in bow]
            low_value_words = [id for id, value in tfidf[bow] if value < low_value]
            words_missing_in_tfidf = [id for id in bow_ids if
                                      id not in tfidf_ids]  # The words with tf-idf socre 0 will be missing

            new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]

            # reassign
            corpus[i] = new_bow
        # Bereitstellung des LDA Model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=20,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha="auto")

        # Visualisierung und Speichern
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
        pyLDAvis.save_html(vis, target_file_path)

    # https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#1introduction

    def prep_data(self, data) -> list:
        """
        Wandelt die JSON Struktur des Datensets um in eine Liste.
        :param data: Zu bearbeitendes Datenset.
        :return: Liste mit Post-Inhalten aus dem Datenset.
        """
        text_list = []
        post_content = ''
        for idx, text in enumerate(data):
            post_content = data[idx]['content']
            text_list.append(post_content)
        return text_list

    def lemmatization(self, texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
        """
        Filtert die Stopwörter aus den Post Texten heraus.
        :param texts: Liste an Post-Texten aus dem Datenset.
        :param allowed_postags: Wörter die nicht Herausgefiltert werden, z.B. Nomen.
        :return: Lemmatized texts, daher die gefilterten Texte.
        """
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Quelle: https://github.com/explosion/spaCy/issues/7453
        texts_out = []
        for text in texts:
            if text != None:
                doc = nlp(text)
                new_text = []
                for token in doc:
                    if token.pos_ in allowed_postags:
                        new_text.append(token.lemma_)

                final = " ".join(new_text)
                texts_out.append(final)
            else:
                continue

        return texts_out

    def gen_words(self, texts):
        """
        Gensim spezifisches Preprocessing.
        :param texts: Liste an Post-Texten aus dem Datenset.
        :return: Verarbeitete Texte.
        """
        final = []
        for text in texts:
            new = gensim.utils.simple_preprocess(text, deacc=True)
            final.append(new)
        return final

    def make_bigram(self, texts, bigram):
        bigram = (bigram[doc] for doc in texts)
        return bigram

    def make_trigram(self, texts, trigram, bigram):
        trigram = (trigram[bigram[text]] for text in enumerate(texts))
        return trigram
