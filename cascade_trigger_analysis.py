import os
import json
import pandas as pd
import networkx as nx
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_graph(graph_file_path):
    with open(graph_file_path, 'rb') as f:
        return pickle.load(f)

def extract_graph_features(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    in_degrees = [d for n, d in graph.in_degree()]
    out_degrees = [d for n, d in graph.out_degree()]
    degree_centrality = nx.degree_centrality(graph)
    clustering_coefficient = nx.clustering(graph)

    features = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'mean_in_degree': np.mean(in_degrees),
        'mean_out_degree': np.mean(out_degrees),
        'mean_degree_centrality': np.mean(list(degree_centrality.values())),
        'mean_clustering_coefficient': np.mean(list(clustering_coefficient.values()))
    }
    return features

def combine_features(graph_features, content_features):
    combined_features = graph_features.copy()
    combined_features.update(content_features)
    return combined_features

def analyze_cascade_triggering_ability(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for dataset in ['Twitter15', 'Twitter16']:
        all_features = []
        all_labels = []
        for label in ['true', 'false', 'unverified', 'non-rumor']:
            sentiment_file = os.path.join(base_dir, f"content_analysis_results_{dataset.lower()}", label, f"{label}_sentiment_emotion_analysis.json")
            # Check this path to ensure it's correct
            print(f"Sentiment file path: {sentiment_file}")

            graph_dir = os.path.join(base_dir, f"graphs{dataset[-2:]}", f"{label}_graphs")
            # Check this path to ensure it's correct
            print(f"Graph directory path: {graph_dir}")

            graph_analysis_dir = os.path.join(base_dir, f"analysis_results{dataset[-2:]}", label)
            # Check this path to ensure it's correct
            print(f"Graph analysis directory path: {graph_analysis_dir}")

            sentiment_data = load_json(sentiment_file)
            graph_analysis_files = [f for f in os.listdir(graph_analysis_dir) if f.endswith('_graphs_analysis.json')]

            for entry in sentiment_data:
                tweet_id = entry['tweet_id']
                sentiment_score = entry['sentiment_score']
                sentiment_label = entry['sentiment_label']
                emotion_label = entry['emotion_label']

                graph_file_path = os.path.join(graph_dir, f"{tweet_id}.pkl")
                if dataset == 'Twitter15':
                    graph_analysis_file_path = os.path.join(graph_analysis_dir, f"{tweet_id}_graphs_analysis.json")
                else:
                    graph_analysis_file_path = os.path.join(graph_analysis_dir, f"{label}_{tweet_id}_graphs_analysis.json")
                # Print these paths to manually check their correctness
                print(f"Graph file path: {graph_file_path}")
                print(f"Graph analysis file path: {graph_analysis_file_path}")

                if os.path.exists(graph_file_path) and os.path.exists(graph_analysis_file_path):
                    graph = load_graph(graph_file_path)
                    graph_features = extract_graph_features(graph)
                    graph_analysis_data = load_json(graph_analysis_file_path)

                    content_features = {
                        'sentiment_score': sentiment_score,
                        'sentiment_label': sentiment_label,
                        'emotion_label': emotion_label
                    }

                    combined_features = combine_features(graph_features, content_features)
                    all_features.append(combined_features)
                    all_labels.append({
                        'cascade_size': graph_analysis_data.get('cascade_size', 0),
                        'cascade_depth': graph_analysis_data.get('tree_depth', 0) if isinstance(graph_analysis_data.get('tree_depth', 0), (int, float)) else 0
                    })

        if all_features:
            feature_df = pd.DataFrame(all_features)
            label_df = pd.DataFrame(all_labels)

            # Encode categorical features
            label_encoder = LabelEncoder()
            feature_df['sentiment_label'] = label_encoder.fit_transform(feature_df['sentiment_label'])
            feature_df['emotion_label'] = label_encoder.fit_transform(feature_df['emotion_label'])

            X_train, X_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.2, random_state=42)

            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            cross_val_mse = cross_val_score(model, feature_df, label_df, cv=5, scoring='neg_mean_squared_error')
            cross_val_mse = -cross_val_mse.mean()

            print(f"Mean Squared Error for {dataset}: {mse}")
            print(f"Mean Absolute Error for {dataset}: {mae}")
            print(f"R² Score for {dataset}: {r2}")
            print(f"Cross-Validation MSE for {dataset}: {cross_val_mse}")

            results = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cross_val_mse': cross_val_mse,
                'feature_importances': model.feature_importances_.tolist()
            }

            dataset_output_file = os.path.join(output_dir, f"cascade_triggering_analysis_results_{dataset.lower()}.json")
            with open(dataset_output_file, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Cascade triggering analysis results for {dataset} saved to {dataset_output_file}")

if __name__ == "__main__":
    base_dir = "."
    output_dir = "cascade_triggering_analysis_results"
    
    print("Starting cascade triggering ability analysis...")
    analyze_cascade_triggering_ability(base_dir, output_dir)
    print("Cascade triggering ability analysis complete.")