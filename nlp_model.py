import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy import displacy
from collections import Counter
import en_core_web_sm

class NLPModel:
    def __init__(self, model_name):
        self.nlp = spacy.load(model_name)
        self.stop_words = STOP_WORDS

    def process_text(self, text):
        doc = self.nlp(text)
        return doc

    def lemmatize(self, text):
        doc = self.nlp(text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc])
        return lemmatized_text

    def remove_stopwords(self, text):
        doc = self.nlp(text)
        filtered_text = ' '.join([token.text for token in doc if not token.is_stop])
        return filtered_text

    def word_frequency(self, text, n):
        doc = self.nlp(text)
        words = [token.text for token in doc if not token.is_stop and not token.is_punct]
        word_freq = Counter(words)
        common_words = word_freq.most_common(n)
        return common_words

    def visualize_dependency_parsing(self, text):
        doc = self.nlp(text)
        displacy.serve(doc, style='dep')

class TextClassificationModel(NLPModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.label_names = self.nlp.pipe_labels['textcat'].labels

    def predict(self, text):
        doc = self.process_text(text)
        scores = doc.cats
        predicted_label = max(scores, key=scores.get)
        return predicted_label

class EntityRecognitionModel(NLPModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def get_entities(self, text):
        doc = self.process_text(text)
        entities = []
        for ent in doc.ents:
            entities.append({'text': ent.text, 'label': ent.label_})
        return entities

class IntentRecognitionModel(NLPModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def get_intent(self, text):
        doc = self.process_text(text)
        intent = ''
        for token in doc:
            if token.dep_ == 'ROOT':
                intent = token.text
                break
        return intent

class SentimentAnalysisModel(NLPModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def get_sentiment(self, text):
        doc = self.process_text(text)
        sentiment_score = doc.sentiment.polarity
        if sentiment_score > 0:
            sentiment = "Positive"
        elif sentiment_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return sentiment

class NamedEntityRecognitionModel(NLPModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.nlp = en_core_web_sm.load()

    def get_named_entities(self, text):
        doc = self.process_text(text)
        entities = []
        for ent in doc.ents:
            entity_type = "ORG"
            if ent.label_ == entity_type:
                entities.append({'text': ent.text, 'label': ent.label_})
            return entities

# Example usage
if __name__ == '__main__':
    # Specify the name of the NLP model to use
    model_name = "en_core_web_sm"

    # Create an instance of the desired NLP model class (e.g., TextClassificationModel, EntityRecognitionModel, IntentRecognitionModel, SentimentAnalysisModel, or NamedEntityRecognitionModel)
    nlp_model = EntityRecognitionModel(model_name)

    # Use the methods of the NLP model class to process text and extract the
# Example usage
if __name__ == '__main__':
    # Specify the name of the NLP model to use
    model_name = "en_core_web_sm"

    # Create an instance of the desired NLP model class (e.g., TextClassificationModel, EntityRecognitionModel, IntentRecognitionModel, SentimentAnalysisModel, or NamedEntityRecognitionModel)
    nlp_model = EntityRecognitionModel(model_name)

    # Use the methods of the NLP model class to process text and extract the desired information
    text = "Apple is looking at buying U.K. startup for $1 billion"
    entities = nlp_model.get_entities(text)
    print(entities)