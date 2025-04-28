# 🚨 Fake News Spread Analysis through Social Media Networks

**Authors:**  
Yash Dani • Eyad Mahmoud • Waleed Samouh • Malik Samouh  
York University, Toronto, Canada  
<yashdani@my.yorku.ca> • <eyad66@my.yorku.ca> • <waleed.samouh@yorku.ca> • <maliktj@my.yorku.ca>

---

## 📖 Table of Contents

1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Dataset](#dataset)  
4. [Installation & Setup](#installation--setup)  
5. [Data Preprocessing](#data-preprocessing)  
6. [Graph Construction](#graph-construction)  
7. [Analysis Workflows](#analysis-workflows)  
   - [1. Content Analysis](#1-content-analysis)  
   - [2. Graph Statistics](#2-graph-statistics)  
   - [3. Cascade Triggering Models](#3-cascade-triggering-models)  
   - [4. Advanced Regression Models](#4-advanced-regression-models)  
8. [Visualization & Reporting](#visualization--reporting)  
9. [Directory Structure](#directory-structure)

---

## 🚀 Project Overview

In an era where **misinformation** can sway elections, fuel social unrest, and endanger public health, understanding how **fake news** propagates across social networks is vital. This project:

- **Quantifies** differences in spread patterns between real vs. fake news.  
- **Builds** interaction graphs from Twitter15 & Twitter16 datasets.  
- **Extracts** text features (sentiment, emotion, topics) and graph features (degree, centrality, delay).  
- **Trains** predictive models to forecast cascade size & depth.  
- **Visualizes** propagation chains, emotional drivers, and model performance.  

---

## ✨ Key Features

- **Data Cleaning & Labeling**  
- **Interaction Graph Construction** using NetworkX  
- **Sentiment & Emotion Analysis** via HuggingFace transformers  
- **Topic Modeling** with Gensim LDA  
- **Graph Metrics**: cascade size, depth, centralities, delays  
- **Predictive Models**: Random Forest, Ridge Regression, MLP  
- **Rich Visualizations**: word-clouds, histograms, network layouts  

---

## 🗂 Dataset

- **Raw Sources:**  
  - `rumor_detection_acl2017/twitter15/`  
  - `rumor_detection_acl2017/twitter16/`  
- **Contents:**  
  - `label.txt`: tweet IDs & labels (`true`, `false`, `unverified`, `non-rumor`)  
  - `source_tweets.txt`: `<tweet_id>\t<text>`  
  - `tree/`: retweet cascades in parent→child format  

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/fake-news-analysis.git
cd fake-news-analysis
python3 -m venv venv
source venv/bin/activate             # macOS/Linux
venv\Scripts\activate                # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧹 Data Preprocessing

Separate raw labels, tweets, and tree files into per-label folders:

```bash
python preprocess_data.py \
  --label-file rumor_detection_acl2017/twitter16/label.txt \
  --source-tweets rumor_detection_acl2017/twitter16/source_tweets.txt \
  --tree-dir rumor_detection_acl2017/twitter16/tree \
  --output-dir processed_data16

python preprocess_data.py \
  --label-file rumor_detection_acl2017/twitter15/label.txt \
  --source-tweets rumor_detection_acl2017/twitter15/source_tweets.txt \
  --tree-dir rumor_detection_acl2017/twitter15/tree \
  --output-dir processed_data15
```

---

## 🌳 Graph Construction

Convert each tweet’s retweet tree into a pickled NetworkX DiGraph:

```bash
python save_graph.py \
  --input-dir processed_data16 \
  --output-dir graphs16

python save_graph.py \
  --input-dir processed_data15 \
  --output-dir graphs15
```

---

## ⚙️ Analysis Workflows

### 1. Content Analysis
- **Sentiment + Emotion** via Transformers  
- **Topic Modeling** (LDA, 5 topics)

```bash
python content_analysis.py
```

Results in content_analysis_results_twitter15/ & ..._twitter16/.

### 2. Graph Statistics
- **Cascade Size, Depth, Degree Distributions** 
- **Propagation Delays, Reaction Times, Centralities**

```bash
python analysis.py --graph-dir graphs16 --output-dir analysis_results16
python convert_txt_to_json.py
```
### 3. Cascade Triggering Models
- **Random Forest Regressor** on combined features 

```bash
python cascade_trigger_analysis.py
```

Outputs in cascade_triggering_analysis_results/.

### 4. Advanced Regression Models
- **Ridge & MLP Regressor** comparison

```bash
python advanced_cascade_trigger_analysis.py
```
Outputs in cascade_triggering_analysis_results_advanced/.

## 📊 Visualization & Reporting

Generate publication-ready figures:

```bash
python visualization_advanced.py
```

- **Word Clouds** for LDA topics
- **Bar Plots** for sentiment, emotion, and model metrics
- **Network layouts** via visualize_graph.py

All images are saved under images/ & presentation/.

---

## 📂 Directory Structure

```bash
fake-news-analysis/
├── rumor_detection_acl2017/      # raw Twitter15 & Twitter16
├── processed_data15/             # per-label tweets & trees
├── processed_data16/             
├── graphs15/                     # pickled networks
├── graphs16/                     
├── content_analysis.py           
├── content_analysis_results_*    
├── analysis.py                   
├── convert_txt_to_json.py        
├── analysis_results15/           
├── analysis_results16/           
├── cascade_trigger_analysis.py   
├── cascade_triggering_analysis_results/    
├── advanced_cascade_trigger_analysis.py
├── cascade_triggering_analysis_results_advanced/
├── visualization_advanced.py     
├── visualize_graph.py            
├── images/                       
└── presentation/                 
```

---

