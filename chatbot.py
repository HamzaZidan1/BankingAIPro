import streamlit as st
from nlp_model import TextClassificationModel, EntityRecognitionModel, IntentRecognitionModel, SentimentAnalysisModel, NamedEntityRecognitionModel

model_name = "en_core_web_sm"
text_classifier = TextClassificationModel(model_name)
entity_recognizer = EntityRecognitionModel(model_name)
intent_recognizer = IntentRecognitionModel(model_name)
sentiment_analyzer = SentimentAnalysisModel(model_name)
named_entity_recognizer = NamedEntityRecognitionModel(model_name)

def handle_input(user_input):
    # Perform text classification
    predicted_label = text_classifier.predict(user_input)
    st.write("Text classification prediction: {}".format(predicted_label))

    # Perform named entity recognition
    named_entities = entity_recognizer.get_entities(user_input)
    st.write("Named entities: {}".format(named_entities))

    # Perform intent recognition
    intent = intent_recognizer.get_intent(user_input)
    st.write("Intent: {}".format(intent))

    # Perform sentiment analysis
    sentiment = sentiment_analyzer.get_sentiment(user_input)
    st.write("Sentiment: {}".format(sentiment))

    # Perform named entity recognition of specific types
    entity_type = "ORG"
    named_entities = named_entity_recognizer.get_named_entities(user_input, entity_type)
    st.write("Named entities of type {}: {}".format(entity_type, named_entities))

def main():
    st.title("Chatbot Interface")

    # Loop to prompt user for input
    while True:
        # Prompt the user for input
        user_input = st.text_input("Enter some text (or 'quit' to exit): ")

        # Check if the user wants to quit
        if user_input.lower() == "quit":
            st.stop()

        # Handle user input
        handle_input(user_input)

if __name__ == "__main__":
    main()
