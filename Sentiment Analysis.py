import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from nltk.corpus import stopwords
import spacy
from datasets import Dataset
import pandas as pd
import os
nltk.download('stopwords')

# Get all NLTK stopwords from all languages
nltkLanguages = stopwords.fileids()
nltkStopwords = set()
for lang in nltkLanguages:
    nltkStopwords.update(stopwords.words(lang))

# Load spaCy and get stopwords from all spaCy-supported languages
spacyStopwords = set()
spacyLanguages = [
    "af", "ar", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "ga",
    "gu", "he", "hi", "hr", "hu", "id", "is", "it", "kn", "lt", "lv", "mr", "nb", "nl",
    "pl", "pt", "ro", "ru", "si", "sk", "sl", "sq", "sr", "sv", "ta", "te", "tl", "tr", "uk",
    "ur", "zh"
]

for lang in spacyLanguages:
    try:
        nlp = spacy.blank(lang)
        spacyStopwords.update(nlp.Defaults.stop_words)
    except Exception as e:
        print(f"Error loading stopwords for language '{lang}': {e}")

# Combine all stopwords into a single set
stopwords = nltkStopwords.union(spacyStopwords)

# Load the dataset using Dataset API
hf_dataset = Dataset.from_csv('sentiment_dataset.csv')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Initialize Hugging Face emotion classification model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-English-distilroberta-base", device=0)

# Add "tone" and "impact" columns
def calculate_tone_impact(batch):
    tones = [sia.polarity_scores(text)['compound'] for text in batch['text']]
    impacts = [tone * ((int(likes) // 10) + int(comments)) for tone, likes, comments in zip(tones, batch['likes'], batch['comments'])]
    return {"tone": tones, "impact": impacts}

hf_dataset = hf_dataset.map(calculate_tone_impact, batched=True)

# Calculate frequency column
def calculate_frequency(batch):
    word_frequency = defaultdict(int)
    for text in batch['text']:
        words = text.split()
        for word in words:
            word_frequency[word] += 1
    frequencies = [sum(word_frequency.get(word, 0) for word in text.split()) for text in batch['text']]
    return {"frequency": frequencies}

hf_dataset = hf_dataset.map(calculate_frequency, batched=True)

# Analyze emotions for each text and add as columns
emotions = ["joy", "sadness", "anger", "surprise", "fear"]

def extract_emotions(batch):
    results = emotion_classifier(batch['text'], truncation=True)

    batch_emotion_scores = []
    for text_result in results:
        # Check if text_result is a dictionary or list of dictionaries
        if isinstance(text_result, dict):
            # Process as a single result
            emotion_scores = {emotion['label']: emotion['score'] for emotion in [text_result]}
        elif isinstance(text_result, list):
            # Process as a list of results
            emotion_scores = {emotion['label']: emotion['score'] for emotion in text_result}
        else:
            raise TypeError(f"Unexpected structure for text_result: {type(text_result)}")

        # Extract scores for the predefined emotions
        batch_emotion_scores.append([emotion_scores.get(emotion, 0) for emotion in emotions])

    # Aggregate scores into a dictionary
    return {f"emotion_{emotion}": [float(scores[i]) for scores in batch_emotion_scores] for i, emotion in enumerate(emotions)}

hf_dataset = hf_dataset.map(extract_emotions, batched=True, batch_size=64)

# Calculate impact for each emotion
def calculate_impact_emotions(batch):
    emotion_impacts = {}
    for emotion in emotions:
        emotion_column = f"emotion_{emotion}"
        emotion_impacts[f"impact_{emotion}"] = [float(emotion_score) * ((int(likes) // 10) + int(comments)) for emotion_score, likes, comments in zip(batch[emotion_column], batch['likes'], batch['comments'])]
    return emotion_impacts

hf_dataset = hf_dataset.map(calculate_impact_emotions, batched=True)

# Check for existence of 'texts.csv' and 'words.csv'
def check_and_create_files():
    texts_exists = os.path.exists('texts.csv')
    words_exists = os.path.exists('words.csv')

    if not texts_exists:
        print("'texts.csv' is missing. Generating...")
        create_texts_csv()

    if not words_exists:
        print("'words.csv' is missing. Generating...")
        create_words_csv()

    if texts_exists or words_exists:
        regenerate = input("One or both files already exist. Do you want to regenerate them? (yes/no): ").strip().lower()
        if regenerate == 'yes':
            if texts_exists:
                create_texts_csv()
            if words_exists:
                create_words_csv()

# Create 'texts.csv'
def create_texts_csv():
    print("Creating 'texts.csv'...")
    hf_dataset.to_csv('texts.csv', index=False)

# Create 'words.csv'
def create_words_csv():
    print("Creating 'words.csv'...")
    word_data = defaultdict(lambda: {
        'frequency': 0,
        'likes': 0,
        'comments': 0,
        'tone': 0,
        'impact': 0,
        **{f"emotion_{emotion}": 0 for emotion in emotions},
        **{f"impact_{emotion}": 0 for emotion in emotions}
    })

    for row in hf_dataset:
        words = re.findall(r'\b\w+\b', row['text'].lower())
        for word in words:
            if word not in stopwords:
                word_data[word]['frequency'] += 1
                word_data[word]['likes'] += int(row['likes'])
                word_data[word]['comments'] += int(row['comments'])
                word_data[word]['tone'] += float(sia.polarity_scores(word)['compound'])
                word_data[word]['impact'] += word_data[word]['tone'] * ((int(row['likes']) // 10) + int(row['comments']))
                for emotion in emotions:
                    word_data[word][f"emotion_{emotion}"] += float(row[f"emotion_{emotion}"])
                    word_data[word][f"impact_{emotion}"] += float(row[f"emotion_{emotion}"]) * ((int(row['likes']) // 10) + int(row['comments']))

    word_df = pd.DataFrame.from_dict(word_data, orient='index').reset_index()
    word_df.rename(columns={"index": "word"}, inplace=True)
    word_df.to_csv('words.csv', index=False)

# Execute file check
check_and_create_files()

# Load the main dataset and additional words dataset
texts_df = hf_dataset.to_pandas()
words_df = pd.read_csv('words.csv')

# Initialize global variables for user input
selection = None
sort_by = None
threshold = None
count_val = None
graph_type = None

# Function to group and summarize data based on user selection
def group_and_summarize_data(selection, sort_by):
    if selection == 'words':
        grouped = words_df.groupby('word', as_index=False).sum()
        grouped = grouped[['word', sort_by]].sort_values(by=sort_by, ascending=False)
    elif selection == 'texts':
        grouped = texts_df.groupby('text', as_index=False).sum()
        grouped = grouped[['text', sort_by]].sort_values(by=sort_by, ascending=False)
    else:
        grouped = texts_df.groupby(selection, as_index=False).sum()
        grouped = grouped[[selection, sort_by]].sort_values(by=sort_by, ascending=False)
    return grouped

# Function to slice data based on threshold option
def slice_data(df, threshold, count_val):
    if threshold == 'Highest':
        return df.head(count_val)
    elif threshold == 'Lowest':
        return df.tail(count_val)
    elif threshold == 'Extremes':
        half_n = count_val // 2
        top_n = half_n + (count_val % 2)
        df_top = df.head(top_n)
        df_bottom = df.tail(half_n)
        return pd.concat([df_top, df_bottom])
    else:
        return df.head(count_val)

# Function to plot the data based on user inputs
def plot_data(df, column, actual_count):
    plt.figure(figsize=(10, 6))
    if graph_type == 'Bar':
        plt.bar(df.iloc[:, 0].astype(str), df[column])
    elif graph_type == 'Line':
        plt.plot(df.iloc[:, 0].astype(str), df[column], marker='o')
    elif graph_type == 'Pie':
        plt.pie(df[column], labels=df.iloc[:, 0].astype(str), autopct='%1.1f%%')
    plt.xticks(rotation=45, ha='right')
    plot_title = f"Top {actual_count} {selection} by {sort_by} ({threshold})"
    plt.title(plot_title)
    if graph_type != 'Pie':
        plt.ylabel(column.capitalize())
        plt.xlabel(selection.capitalize() if selection not in ['words', 'texts'] else selection.capitalize())
    plt.tight_layout()
    plt.show()

# Function to handle form submission and process data
def on_submit():
    global selection, sort_by, threshold, count_val, graph_type
    selection = selection_var.get()
    sort_by = sort_by_var.get()
    threshold = threshold_var.get()
    graph_type = graph_type_var.get()

    try:
        count_val = int(count_var.get())
    except ValueError:
        print("Invalid count value. Please enter a valid number.")
        return

    df = group_and_summarize_data(selection, sort_by)
    if df is None or df.empty:
        print("No data available for the given selection and sort_by.")
        return

    df_sliced = slice_data(df, threshold, count_val)
    actual_count = len(df_sliced)

    if actual_count > 0:
        plot_data(df_sliced, sort_by, actual_count)

# Create the GUI using Tkinter
root = tk.Tk()
root.title("Data Selection")
root.resizable(False, False)

# User input fields for data type
selection_var = tk.StringVar(value='texts')
sort_by_var = tk.StringVar(value='impact')
threshold_var = tk.StringVar(value='Highest')
count_var = tk.StringVar(value='10')
graph_type_var = tk.StringVar(value='Bar')

tk.Label(root, text="Select Data Type:").pack()
data_type_menu = ttk.Combobox(root, textvariable=selection_var, values=['agegroup', 'country', 'texts', 'time', 'userid', 'words'], state="readonly")
data_type_menu.pack()

# User input fields for sorting criteria
tk.Label(root, text="Sort By:").pack()
sort_by_menu = ttk.Combobox(root, textvariable=sort_by_var, values=['impact', 'tone', 'likes', 'comments', 'frequency'] + [f'emotion_{e}' for e in emotions] + [f'impact_{e}' for e in emotions], state="readonly")
sort_by_menu.pack()

# User input fields for threshold options
tk.Label(root, text="Threshold:").pack()
threshold_menu = ttk.Combobox(root, textvariable=threshold_var, values=['Highest', 'Lowest', 'Extremes'], state="readonly")
threshold_menu.pack()

# User input fields for count
tk.Label(root, text="Number of Items to Display:").pack()
count_entry = ttk.Entry(root, textvariable=count_var)
count_entry.pack()

# User input fields for graph type
tk.Label(root, text="Select Graph Type:").pack()
graph_type_menu = ttk.Combobox(root, textvariable=graph_type_var, values=['Bar', 'Line', 'Pie'], state="readonly")
graph_type_menu.pack()

# Submit button
tk.Button(root, text="Submit", command=on_submit).pack()

# Run the GUI
root.mainloop()
