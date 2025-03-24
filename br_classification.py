########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re
import math
import os
import subprocess

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.naive_bayes import GaussianNB

# Text cleaning & stopwords
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

########## 2. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

########## 3. Configure parameters & Start training ##########

# List of all projects to process
projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']

# Number of repeated experiments
REPEAT = 10

# Output CSV file name for all results
all_results_csv = 'NB_all_results.csv'
all_results = pd.DataFrame(columns=['Project', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])

# Create a DataFrame to store final combined results
final_results_df = pd.DataFrame(columns=['Project', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])

# Process each project
for project in projects:
    print(f"\n{'='*50}")
    print(f"Processing project: {project}")
    print(f"{'='*50}")
    
    path = f'{project}.csv'
    
    # Check if the file exists
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist. Skipping this project.")
        continue
    
    # Output CSV file name for this project
    out_csv_name = f'./{project}_NB.csv'
    
    print(f"Reading data from {path}...")
    pd_all = pd.read_csv(path)
    pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

    # Merge Title and Body into a single column; if Body is NaN, use Title only
    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    # Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })
    
    temp_file = f'{project}_Title+Body.csv'
    pd_tplusb.to_csv(temp_file, index=False, columns=["id", "Number", "sentiment", "text"])
    
    # ========== Read and clean data ==========
    data = pd.read_csv(temp_file).fillna('')
    text_col = 'text'

    # Keep a copy for referencing original data if needed
    original_data = data.copy()

    # Text cleaning
    print(f"Cleaning text data...")
    data[text_col] = data[text_col].apply(remove_html)
    data[text_col] = data[text_col].apply(remove_emoji)
    data[text_col] = data[text_col].apply(remove_stopwords)
    data[text_col] = data[text_col].apply(clean_str)

    # ========== Hyperparameter grid ==========
    # We use logspace for var_smoothing: [1e-12, 1e-11, ..., 1]
    params = {
        'var_smoothing': np.logspace(-12, 0, 13)
    }

    # Lists to store metrics across repeated runs
    accuracies  = []
    precisions  = []
    recalls     = []
    f1_scores   = []
    auc_values  = []

    print(f"Starting {REPEAT} training iterations...")
    for repeated_time in range(REPEAT):
        print(f"  Iteration {repeated_time+1}/{REPEAT}...")
        # --- 4.1 Split into train/test ---
        indices = np.arange(data.shape[0])
        train_index, test_index = train_test_split(
            indices, test_size=0.2, random_state=repeated_time
        )

        train_text = data[text_col].iloc[train_index]
        test_text = data[text_col].iloc[test_index]

        y_train = data['sentiment'].iloc[train_index]
        y_test  = data['sentiment'].iloc[test_index]

        # --- 4.2 TF-IDF vectorization ---
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000  # Adjust as needed
        )
        X_train = tfidf.fit_transform(train_text).toarray()
        X_test = tfidf.transform(test_text).toarray()
       
        # --- 4.3 Naive Bayes model & GridSearch ---
        clf = GaussianNB()
        grid = GridSearchCV(
            clf,
            params,
            cv=5,              # 5-fold CV (can be changed)
            scoring='roc_auc'  # Using roc_auc as the metric for selection
        )
        grid.fit(X_train, y_train)

        # Retrieve the best model
        best_clf = grid.best_estimator_
        best_clf.fit(X_train, y_train)

        # --- 4.4 Make predictions & evaluate ---
        y_pred = best_clf.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Precision (macro)
        prec = precision_score(y_test, y_pred, average='macro')
        precisions.append(prec)

        # Recall (macro)
        rec = recall_score(y_test, y_pred, average='macro')
        recalls.append(rec)

        # F1 Score (macro)
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)

        # AUC
        # If labels are 0/1 only, this works directly.
        # If labels are something else, adjust pos_label accordingly.
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        auc_val = auc(fpr, tpr)
        auc_values.append(auc_val)

    # --- 4.5 Aggregate results ---
    final_accuracy  = np.mean(accuracies)
    final_precision = np.mean(precisions)
    final_recall    = np.mean(recalls)
    final_f1        = np.mean(f1_scores)
    final_auc       = np.mean(auc_values)

    print("\n=== Naive Bayes + TF-IDF Results ===")
    print(f"Project:               {project}")
    print(f"Number of repeats:     {REPEAT}")
    print(f"Average Accuracy:      {final_accuracy:.4f}")
    print(f"Average Precision:     {final_precision:.4f}")
    print(f"Average Recall:        {final_recall:.4f}")
    print(f"Average F1 score:      {final_f1:.4f}")
    print(f"Average AUC:           {final_auc:.4f}")

    # Store project results for the combined report
    project_results = pd.DataFrame([{
        'Project': project,
        'Accuracy': final_accuracy,
        'Precision': final_precision,
        'Recall': final_recall,
        'F1': final_f1,
        'AUC': final_auc
    }])
    
    # Append project results to all_results
    all_results = pd.concat([all_results, project_results], ignore_index=True)
    
    # Clean up the temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)

# Calculate final average across all projects
if all_results.shape[0] > 0:
    # Calculate mean of all metrics across projects
    final_average = pd.DataFrame([{
        'Project': 'AVERAGE',
        'Accuracy': all_results['Accuracy'].mean(),
        'Precision': all_results['Precision'].mean(),
        'Recall': all_results['Recall'].mean(),
        'F1': all_results['F1'].mean(),
        'AUC': all_results['AUC'].mean()
    }])
    
    # Append final average to results
    all_results = pd.concat([all_results, final_average], ignore_index=True)
    
    # Save results to CSV
    all_results.to_csv(all_results_csv, index=False)
    print(f"\n{'='*50}")
    print(f"All project results combined and saved to: {all_results_csv}")
    print(f"{'='*50}")
    print(all_results.to_string(index=False))
else:
    print("\nNo results were generated. Check if project CSV files exist.")