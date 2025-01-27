import os
import json
import pandas as pd
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string

def load_source_tweets(file_path):
    """
    Load source tweets from a file.
    Args:
        file_path (str): Path to the source tweets file.
    Returns:
        dict: Dictionary containing tweet IDs and their content.
    """
    source_tweets = {}
    with open(file_path, 'r') as f:
        for line in f:
            tweet_id, tweet_content = line.strip().split('\t')
            source_tweets[tweet_id] = tweet_content
    return source_tweets

def perform_sentiment_analysis(tweets, label, output_dir):
    """
    Perform sentiment analysis and emotion detection on tweets.
    Args:
        tweets (dict): Dictionary containing tweet IDs and their content.
        label (str): Label of the tweets (true, false, etc.).
        output_dir (str): Directory to save the analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Initialize sentiment and emotion analyzers
    sentiment_analyzer = pipeline("sentiment-analysis")
    emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

    for tweet_id, tweet_content in tweets.items():
        sentiment = sentiment_analyzer(tweet_content)[0]
        emotion = emotion_analyzer(tweet_content)[0]

        result = {
            "tweet_id": tweet_id,
            "content": tweet_content,
            "sentiment_score": sentiment['score'],
            "sentiment_label": sentiment['label'],
            "emotion_label": emotion['label'],
            "emotion_score": emotion['score']
        }
        results.append(result)

        print(f"Processed tweet ID: {tweet_id} | Sentiment: {sentiment['label']} | Emotion: {emotion['label']}")

    output_file_path = os.path.join(output_dir, f"{label}_sentiment_emotion_analysis.json")
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Sentiment and emotion analysis results saved to {output_file_path}")

def perform_topic_modeling(tweets, label, output_dir, num_topics=5, num_words=10):
    """
    Perform topic modeling on tweets using LDA.
    Args:
        tweets (dict): Dictionary containing tweet IDs and their content.
        label (str): Label of the tweets (true, false, etc.).
        output_dir (str): Directory to save the analysis results.
        num_topics (int): Number of topics to extract.
        num_words (int): Number of words to show per topic.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess tweets for topic modeling
    processed_tweets = [preprocess_string(tweet_content) for tweet_content in tweets.values()]

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(processed_tweets)
    corpus = [dictionary.doc2bow(text) for text in processed_tweets]

    # Perform LDA topic modeling
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Extract topics
    topics = lda_model.print_topics(num_words=num_words)
    topics_dict = {f"Topic {i+1}": topic for i, topic in topics}

    output_file_path = os.path.join(output_dir, f"{label}_topic_modeling.json")
    with open(output_file_path, 'w') as f:
        json.dump(topics_dict, f, indent=4)

    print(f"Topic modeling results saved to {output_file_path}")

if __name__ == "__main__":
    base_dirs = {
        "Twitter15": "processed_data15",
        "Twitter16": "processed_data16"
    }

    for dataset, base_dir in base_dirs.items():
        output_dir = f"content_analysis_results_{dataset.lower()}"
        os.makedirs(output_dir, exist_ok=True)

        for label in ['true', 'false', 'unverified', 'non-rumor']:
            print(f"Processing {label} tweets for {dataset}...")
            source_tweets_file_path = os.path.join(base_dir, f"{label}_source_tweets.txt")
            tweets = load_source_tweets(source_tweets_file_path)
            label_output_dir = os.path.join(output_dir, label)
            perform_sentiment_analysis(tweets, label, label_output_dir)
            perform_topic_modeling(tweets, label, label_output_dir)
        print(f"Finished processing {dataset}.")
    print("All analyses complete.")