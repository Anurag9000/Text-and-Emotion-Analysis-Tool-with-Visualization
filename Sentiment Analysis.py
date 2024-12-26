import pandas as pd
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



# Load the dataset
df = pd.read_csv('sentiment_dataset.csv')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Initialize Hugging Face emotion classification model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-English-distilroberta-base")

# Add "tone" and "impact" columns
df['tone'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['impact'] = df['tone'] * ((df['likes'] // 10) + df['comments'])

# Calculate frequency column
word_frequency = df['text'].str.split().explode().value_counts()
frequency_map = word_frequency.to_dict()
df['frequency'] = df['text'].apply(lambda x: sum(frequency_map.get(word, 0) for word in x.split()))

# Analyze emotions for each text and add as columns
emotions = ["joy", "sadness", "anger", "surprise", "fear"]

def extract_emotions(text):
    results = emotion_classifier(text, top_k=None)
    emotion_scores = {emotion['label']: emotion['score'] for emotion in results}
    return [emotion_scores.get(emotion, 0) for emotion in emotions]

emotion_columns = [f"emotion_{emotion}" for emotion in emotions]
emotion_data = df['text'].apply(extract_emotions)
df[emotion_columns] = pd.DataFrame(emotion_data.tolist(), index=df.index)

# Calculate impact for each emotion
def calc_impact_emotion(row, emotion):
    emotion_column = f"emotion_{emotion}"
    return row[emotion_column] * ((row['likes'] // 10) + row['comments'])

for emotion in emotions:
    df[f"impact_{emotion}"] = df.apply(lambda row: calc_impact_emotion(row, emotion), axis=1)

# Save the "texts" file
df.to_csv('texts.csv', index=False)

# Generate words.csv with 17 columns
word_data = defaultdict(lambda: {
    'frequency': 0,
    'likes': 0,
    'comments': 0,
    'tone': 0,
    'impact': 0,
    **{f"emotion_{emotion}": 0 for emotion in emotions}, **{f"impact_{emotion}": 0 for emotion in emotions}
})

for _, row in df.iterrows():
    words = re.findall(r'\b\w+\b', row['text'].lower())
    for word in words:
        if word not in stopwords:
            word_data[word]['frequency'] += 1
            word_data[word]['likes'] += row['likes']
            word_data[word]['comments'] += row['comments']
            word_data[word]['tone'] += sia.polarity_scores(word)['compound']
            word_data[word]['impact'] += word_data[word]['tone'] * ((row['likes'] // 10) + row['comments'])
            word_emotions = extract_emotions(word)
            for i, emotion in enumerate(emotions):
                word_data[word][f"emotion_{emotion}"] += word_emotions[i]
                word_data[word][f"impact_{emotion}"] += word_emotions[i] * ((row['likes'] // 10) + row['comments'])

word_df = pd.DataFrame.from_dict(word_data, orient='index').reset_index()
word_df.rename(columns={"index": "word"}, inplace=True)

# Save the words.csv file
word_df.to_csv('words.csv', index=False)

# Load the main dataset and additional words dataset
texts_df = pd.read_csv('texts.csv')
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
