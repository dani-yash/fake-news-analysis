import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# Set paths
base_dir = "./"
content_analysis_results_dir = {
    "Twitter15": "content_analysis_results_twitter15",
    "Twitter16": "content_analysis_results_twitter16"
}
cascade_results_dir = {
    "Regular": "cascade_triggering_analysis_results",
    "Advanced": "cascade_triggering_analysis_results_advanced"
}

# Helper function to load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 1. Topic Modeling Visualization
def visualize_topic_modeling(results_dir):
    for dataset in content_analysis_results_dir:
        for label in ['true', 'false', 'unverified', 'non-rumor']:
            file_path = os.path.join(results_dir[dataset], label, f"{label}_topic_modeling.json")
            topics = load_json(file_path)
            for topic, words in topics.items():
                word_freq = {word.split("*")[1].strip("\""): float(word.split("*")[0]) for word in words.split("+")}
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title(f"{dataset} - {label} - {topic}")
                plt.savefig(f"{dataset}_{label}_{topic}_wordcloud.png")
                plt.close()

# 2. Sentiment and Emotion Distribution
def visualize_sentiment_emotion(results_dir):
    for dataset in content_analysis_results_dir:
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        emotion_counts = {}
        for label in ['true', 'false', 'unverified', 'non-rumor']:
            file_path = os.path.join(results_dir[dataset], label, f"{label}_sentiment_emotion_analysis.json")
            data = load_json(file_path)
            for entry in data:
                sentiment_counts[entry['sentiment_label']] += 1
                if entry['emotion_label'] not in emotion_counts:
                    emotion_counts[entry['emotion_label']] = 0
                emotion_counts[entry['emotion_label']] += 1

        # Plot Sentiment Distribution
        plt.figure(figsize=(8, 6))
        plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['blue', 'orange', 'green'])
        plt.title(f"{dataset} - Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.savefig(f"{dataset}_sentiment_distribution.png")
        plt.close()

        # Plot Emotion Distribution
        plt.figure(figsize=(10, 6))
        plt.bar(emotion_counts.keys(), emotion_counts.values())
        plt.title(f"{dataset} - Emotion Distribution")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.savefig(f"{dataset}_emotion_distribution.png")
        plt.close()

# 3. Cascade Triggering Analysis (Random Model)
def visualize_cascade_triggering_random(results_dir):
    metrics = load_json(os.path.join(results_dir["Regular"], "cascade_triggering_analysis_results_twitter16.json"))
    labels = ["MSE", "MAE", "R²", "Cross-Validation MSE"]
    values = [metrics['mse'], metrics['mae'], metrics['r2'], metrics['cross_val_mse']]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
    plt.title("Random Model - Evaluation Metrics")
    plt.ylabel("Value")
    plt.savefig("random_model_evaluation_metrics.png")
    plt.close()

# 4. Cascade Triggering Analysis (Advanced Model)
def visualize_cascade_triggering_advanced(results_dir):
    metrics = load_json(os.path.join(results_dir["Advanced"], "cascade_triggering_analysis_results_twitter16.json"))
    labels = ["MSE", "MAE", "R²", "Cross-Validation MSE"]
    values = [metrics['mse'], metrics['mae'], metrics['r2'], metrics['cross_val_mse']]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
    plt.title("Advanced Model - Evaluation Metrics")
    plt.ylabel("Value")
    plt.savefig("advanced_model_evaluation_metrics.png")
    plt.close()

# 5. Comparative Analysis
def visualize_comparative_analysis(results_dir):
    regular_metrics = load_json(os.path.join(results_dir["Regular"], "cascade_triggering_analysis_results_twitter16.json"))
    advanced_metrics = load_json(os.path.join(results_dir["Advanced"], "cascade_triggering_analysis_results_twitter16.json"))

    labels = ["MSE", "MAE", "R²", "Cross-Validation MSE"]
    regular_values = [regular_metrics['mse'], regular_metrics['mae'], regular_metrics['r2'], regular_metrics['cross_val_mse']]
    advanced_values = [advanced_metrics['mse'], advanced_metrics['mae'], advanced_metrics['r2'], advanced_metrics['cross_val_mse']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, regular_values, width, label='Regular Model', color='blue')
    rects2 = ax.bar(x + width/2, advanced_values, width, label='Advanced Model', color='green')

    ax.set_xlabel('Metrics')
    ax.set_title('Comparison of Regular and Advanced Models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()

# Run visualizations
visualize_topic_modeling(content_analysis_results_dir)
visualize_sentiment_emotion(content_analysis_results_dir)
visualize_cascade_triggering_random(cascade_results_dir)
visualize_cascade_triggering_advanced(cascade_results_dir)
visualize_comparative_analysis(cascade_results_dir)