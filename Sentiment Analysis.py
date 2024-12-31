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
import torch
from transformers import AutoTokenizer, XLMRobertaTokenizer
import psutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FileHandler:
    def __init__(self, dataset, stopwords, emotions):
        self.dataset = dataset
        self.stopwords = stopwords
        self.emotions = emotions

    def createTextsCsv(self, calculateToneImpact, calculateFrequency, extractEmotions):
        def addToneAndImpact(dataset):
            try:
                return dataset.map(calculateToneImpact, batched=True)
            except Exception as e:
                print(f"Error in 'addToneAndImpact': {e}")
                return dataset

        def addFrequencies(dataset):
            try:
                return dataset.map(calculateFrequency, batched=True)
            except Exception as e:
                print(f"Error in 'addFrequencies': {e}")
                return dataset

        def addEmotions(dataset):
            try:
                return dataset.map(extractEmotions, batched=True)
            except Exception as e:
                print(f"Error in 'addEmotions': {e}")
                return dataset

        print("Creating 'texts.csv'...")
        dataset = self.dataset
        dataset = addToneAndImpact(dataset)
        dataset = addFrequencies(dataset)
        dataset = addEmotions(dataset)
        dataset.to_csv('texts.csv', index=False)

    def createWordsCsv(self):
        sia = SentimentIntensityAnalyzer()

        def processWords(wordData, row):
            words = re.findall(r'\b\w+\b', row['text'].lower())
            for word in words:
                if word not in self.stopwords:
                    wordData[word]['frequency'] += 1
                    wordData[word]['likes'] += int(row['likes'])
                    wordData[word]['comments'] += int(row['comments'])
                    wordData[word]['tone'] += float(sia.polarity_scores(word)['compound'])
                    wordData[word]['impact'] += wordData[word]['tone'] * (
                            (int(row['likes']) // 10) + int(row['comments']))
                    for emotion in self.emotions:
                        wordData[word][f"emotion_{emotion}"] += float(row.get(f"emotion_{emotion}", 0))
                        wordData[word][f"impact_{emotion}"] += float(row.get(f"emotion_{emotion}", 0)) * (
                                (int(row['likes']) // 10) + int(row['comments']))

        print("Creating 'words.csv'...")
        wordData = defaultdict(lambda: {
            'frequency': 0,
            'likes': 0,
            'comments': 0,
            'tone': 0,
            'impact': 0,
            **{f"emotion_{emotion}": 0 for emotion in self.emotions},
            **{f"impact_{emotion}": 0 for emotion in self.emotions}
        })
        for row in self.dataset:
            try:
                processWords(wordData, row)
            except Exception as e:
                print(f"Error processing row: {e}")

        wordDf = pd.DataFrame.from_dict(wordData, orient='index').reset_index()
        wordDf.rename(columns={"index": "word"}, inplace=True)
        wordDf.to_csv('words.csv', index=False)

class DataProcessor:
    def __init__(self, dataset, device, tokenizers, emotionClassifiers, emotions):
        self.dataset = dataset
        self.device = device
        self.tokenizers = tokenizers
        self.emotionClassifiers = emotionClassifiers
        self.emotions = emotions

    @staticmethod
    def getAllStopwords():
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
                print(f"Skipping stopwords for language '{lang}' due to error: {e}")

        # Combine all stopwords into a single set
        return nltkStopwords.union(spacyStopwords)

    @staticmethod
    def getBestGpu():
        bestGpu = -1
        maxFreeMemory = 0
        for i in range(torch.cuda.device_count()):
            freeMemory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
            if freeMemory > maxFreeMemory:
                maxFreeMemory = freeMemory
                bestGpu = i
        return bestGpu

    def calculateToneImpact(self, batch):
        tones = [SentimentIntensityAnalyzer().polarity_scores(text)['compound'] for text in batch['text']]
        impacts = [tone * ((int(likes) // 10) + int(comments)) for tone, likes, comments in zip(tones, batch['likes'], batch['comments'])]
        return {"tone": tones, "impact": impacts}

    def calculateFrequency(self, batch):
        wordFrequency = defaultdict(int)
        for text in batch['text']:
            words = text.split()
            for word in words:
                wordFrequency[word] += 1
        frequencies = [sum(wordFrequency.get(word, 0) for word in text.split()) for text in batch['text']]
        return {"frequency": frequencies}

    def extractEmotions(self, batch):
        batchEmotionScores = defaultdict(list)
        for modelName in self.tokenizers.keys():
            try:
                tokenizer = self.tokenizers[modelName]
                classifier = self.emotionClassifiers[modelName]
                inputs = tokenizer(batch['text'], truncation=True, padding=True, return_tensors="pt").to(self.device)
                outputs = classifier(**inputs)
                scores = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                labels = classifier.config.id2label
                for score in scores:
                    for idx, emotion in labels.items():
                        self.emotions.append(emotion)
                        batchEmotionScores[f"{modelName}_{emotion}"].append(score[idx])
            except Exception as e:
                print(f"Error processing model: {modelName}. Error: {e}")
        return batchEmotionScores

    def calculateImpactEmotions(self, batch):
        emotionImpacts = {}
        for emotion in self.emotions:
            emotionColumn = f"emotion_{emotion}"
            if emotionColumn not in batch:
                print(f"Missing column: {emotionColumn}. Assigning default values.")
                emotionImpacts[f"impact_{emotion}"] = [0.0] * len(batch['text'])
                continue
            emotionImpacts[f"impact_{emotion}"] = [
                float(emotionScore) * ((int(likes) // 10) + int(comments))
                for emotionScore, likes, comments in zip(batch[emotionColumn], batch['likes'], batch['comments'])
            ]
        return emotionImpacts

class Visualizer:
    def __init__(self, textsDf, wordsDf):
        self.textsDf = textsDf
        self.wordsDf = wordsDf

    def groupAndSummarizeData(self, selection, sortBy):
        if selection == 'words':
            grouped = self.wordsDf.groupby('word', as_index=False).sum()
            grouped = grouped[['word', sortBy]].sort_values(by=sortBy, ascending=False)
        elif selection == 'texts':
            grouped = self.textsDf.groupby('text', as_index=False).sum()
            grouped = grouped[['text', sortBy]].sort_values(by=sortBy, ascending=False)
        else:
            grouped = self.textsDf.groupby(selection, as_index=False).sum()
            grouped = grouped[[selection, sortBy]].sort_values(by=sortBy, ascending=False)
        return grouped

    def sliceData(self, df, threshold, countVal):
        if threshold == 'Highest':
            return df.head(countVal)
        elif threshold == 'Lowest':
            return df.tail(countVal)
        elif threshold == 'Extremes':
            halfN = countVal // 2
            topN = halfN + (countVal % 2)
            dfTop = df.head(topN)
            dfBottom = df.tail(halfN)
            return pd.concat([dfTop, dfBottom])
        else:
            return df.head(countVal)

    def plotData(self, df, column, graphType, selection, sortBy, actualCount):
        plt.figure(figsize=(10, 6))
        if graphType == 'Bar':
            plt.bar(df.iloc[:, 0].astype(str), df[column])
        elif graphType == 'Line':
            plt.plot(df.iloc[:, 0].astype(str), df[column], marker='o')
        elif graphType == 'Pie':
            plt.pie(df[column], labels=df.iloc[:, 0].astype(str), autopct='%1.1f%%')
        plt.xticks(rotation=45, ha='right')
        plotTitle = f"Top {actualCount} {selection} by {sortBy} ({column})"
        plt.title(plotTitle)
        if graphType != 'Pie':
            plt.ylabel(column.capitalize())
            plt.xlabel(selection.capitalize() if selection not in ['words', 'texts'] else selection.capitalize())
        plt.tight_layout()
        plt.show()

class GUIHandler:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.selection = None
        self.sortBy = None
        self.threshold = None
        self.countVal = None
        self.graphType = None

    def onSubmit(self, selectionVar, sortByVar, thresholdVar, countVar, graphTypeVar):
        self.selection = selectionVar.get()
        self.sortBy = sortByVar.get()
        self.threshold = thresholdVar.get()
        self.graphType = graphTypeVar.get()

        try:
            self.countVal = int(countVar.get())
        except ValueError:
            print("Invalid count value. Please enter a valid number.")
            return

        df = self.visualizer.groupAndSummarizeData(self.selection, self.sortBy)
        if df is None or df.empty:
            print("No data available for the given selection and sortBy.")
            return

        dfSliced = self.visualizer.sliceData(df, self.threshold, self.countVal)
        actualCount = len(dfSliced)

        if actualCount > 0:
            self.visualizer.plotData(dfSliced, self.sortBy, self.graphType, self.selection, self.sortBy, actualCount)

    
def checkAndCreateFiles(fileHandler):
    textsExists = os.path.exists('texts.csv')
    wordsExists = os.path.exists('words.csv')

    try:
        if not textsExists:
            print("'texts.csv' is missing. Generating...")
            fileHandler.createTextsCsv(fileHandler.dataProcessor.calculateToneImpact,
                                       fileHandler.dataProcessor.calculateFrequency,
                                       fileHandler.dataProcessor.extractEmotions)
        if not wordsExists:
            print("'words.csv' is missing. Generating...")
            fileHandler.createWordsCsv()
    except Exception as e:
        print(f"Error while creating files: {e}")

    if textsExists or wordsExists:
        regenerate = input("One or both files already exist. Do you want to regenerate them? (yes/no): ").strip().lower()
        if regenerate[0].lower() == 'y':
            fileHandler.createTextsCsv(fileHandler.dataProcessor.calculateToneImpact,
                                       fileHandler.dataProcessor.calculateFrequency,
                                       fileHandler.dataProcessor.extractEmotions)
            fileHandler.createWordsCsv()

def main():
    # Load all stopwords using DataProcessor
    allStopwords = DataProcessor.getAllStopwords()

    # Initialize GPU selection
    bestGpu = DataProcessor.getBestGpu()
    if bestGpu != -1:
        device = torch.device(f"cuda:{bestGpu}")
        print(f"Using GPU: {bestGpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Load datasets
    hf_dataset = Dataset.from_csv('sentiment_dataset.csv')

# Load tokenizers and models
    models = [
        "j-hartmann/emotion-English-distilroberta-base",
        "bhadresh-savani/bert-base-go-emotion",
        "monologg/bert-base-cased-goemotions-original",
        "finiteautomata/bertweet-base-emotion-analysis",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    ]

    tokenizers = {}
    emotionClassifiers = {}
    for modelName in models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(modelName)
            if modelName == "cardiffnlp/twitter-xlm-roberta-base-sentiment":
                tokenizers[modelName] = XLMRobertaTokenizer.from_pretrained(modelName)
            else:
            tokenizers[modelName] = tokenizer
            classifier = AutoModelForSequenceClassification.from_pretrained(modelName).to(device)
            emotionClassifiers[modelName] = classifier
        except Exception as e:
            print(f"Error loading model: {modelName}. Error: {e}")

    # Initialize DataProcessor and FileHandler
    emotions = []
    dataProcessor = DataProcessor(hf_dataset, device, tokenizers, emotionClassifiers, emotions)
    fileHandler = FileHandler(hf_dataset, allStopwords, emotions)
    fileHandler.dataProcessor = dataProcessor

    # Check and create files
    checkAndCreateFiles(fileHandler)

    # Load dataframes for visualization
    textsDf = hf_dataset.to_pandas()
    wordsDf = pd.read_csv('words.csv')

    # Initialize Visualizer and GUIHandler
    visualizer = Visualizer(textsDf, wordsDf)
    guiHandler = GUIHandler(visualizer)

    # Launch GUI
    root = tk.Tk()
    root.title("Data Selection")
    root.resizable(False, False)

    # User input fields for data type
    selectionVar = tk.StringVar(value='texts')
    sortByVar = tk.StringVar(value='impact')
    thresholdVar = tk.StringVar(value='Highest')
    countVar = tk.StringVar(value='10')
    graphTypeVar = tk.StringVar(value='Bar')

    tk.Label(root, text="Select Data Type:").pack()
    dataTypeMenu = ttk.Combobox(root, textvariable=selectionVar, values=['agegroup', 'country', 'texts', 'time', 'userid', 'words'], state="readonly")
    dataTypeMenu.pack()

    tk.Label(root, text="Sort By:").pack()
    sortByMenu = ttk.Combobox(root, textvariable=sortByVar, values=['impact', 'tone', 'likes', 'comments', 'frequency'], state="readonly")
    sortByMenu.pack()

    tk.Label(root, text="Threshold:").pack()
    thresholdMenu = ttk.Combobox(root, textvariable=thresholdVar, values=['Highest', 'Lowest', 'Extremes'], state="readonly")
    thresholdMenu.pack()

    tk.Label(root, text="Number of Items to Display:").pack()
    countEntry = ttk.Entry(root, textvariable=countVar)
    countEntry.pack()

    tk.Label(root, text="Select Graph Type:").pack()
    graphTypeMenu = ttk.Combobox(root, textvariable=graphTypeVar, values=['Bar', 'Line', 'Pie'], state="readonly")
    graphTypeMenu.pack()

    tk.Button(root, text="Submit", command=lambda: guiHandler.onSubmit(selectionVar, sortByVar, thresholdVar, countVar, graphTypeVar)).pack()

    root.mainloop()

# Entry point
if __name__ == "__main__":
    main()
