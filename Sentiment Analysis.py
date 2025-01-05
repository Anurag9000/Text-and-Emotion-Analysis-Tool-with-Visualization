import ollama
from ollama import chat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
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
import torch
import psutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FileHandler:
    def __init__(self, dataset, stopwords, emotions, dataProcessor):
        self.dataset = dataset
        self.stopwords = stopwords
        self.emotions = emotions
        self.dataProcessor = dataProcessor  # Store the dataProcessor instance

    def createTextsCsv(self, calculateToneImpact, calculateFrequency, dataProcessor):
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
            def processBatch(batch):
                smallBatchSize = 16  # Reduce if API limitations exist
                emotionScores = defaultdict(list)

                for i in range(0, len(batch['text']), smallBatchSize):
                    smallBatch = {"text": batch['text'][i:i + smallBatchSize]}
                    try:
                        smallBatchScores = dataProcessor.extractEmotions(smallBatch)
                        for emotion, scores in smallBatchScores.items():
                            emotionScores[emotion].extend(scores)
                    except Exception as e:
                        print(f"Error processing sub-batch: {e}")

                for emotion, scores in emotionScores.items():
                    if len(scores) < len(batch['text']):
                        scores.extend([0] * (len(batch['text']) - len(scores)))
                    batch[emotion] = scores

                for emotion in emotionScores.keys():
                    impactKey = f"impact_{emotion}"
                    batch[impactKey] = [
                        score * ((int(likes) // 10) + int(comments))
                        for score, likes, comments in zip(batch[emotion], batch['likes'], batch['comments'])
                    ]

                return batch

            try:
                return dataset.map(processBatch, batched=True)
            except Exception as e:
                print(f"Error in 'addEmotions': {e}")
                return dataset

        print("Creating 'temptexts.csv'...")
        dataset = self.dataset
        dataset = addToneAndImpact(dataset)
        dataset = addFrequencies(dataset)
        dataset = addEmotions(dataset)
        dataset.to_csv('temptexts.csv', index=False)

    def createWordsCsv(self):
        def processWords(wordData, row):
            words = re.findall(r'\b\w+\b', row['text'].lower())
            for word in words:
                try:
                    wordEmotionScores = self.dataProcessor.extractEmotions({"text": [word]})

                    if word not in wordData:
                        wordData[word] = {
                            'frequency': 0,
                            'likes': 0,
                            'comments': 0,
                            'tone': 0,
                            'impact': 0,
                            **{f"emotion_{emotion}": 0 for emotion in wordEmotionScores.keys()},
                            **{f"impact_{emotion}": 0 for emotion in wordEmotionScores.keys()},
                        }

                    wordData[word]['frequency'] += 1
                    wordData[word]['likes'] += int(row['likes'])
                    wordData[word]['comments'] += int(row['comments'])
                    wordData[word]['tone'] += float(SentimentIntensityAnalyzer().polarity_scores(word)['compound'])
                    wordData[word]['impact'] += wordData[word]['tone'] * (
                        (int(row['likes']) // 10) + int(row['comments'])
                    )

                    for emotion, score in wordEmotionScores.items():
                        # Aggregate scores (e.g., take the first value if it's a list)
                        score = score[0] if isinstance(score, list) else score
                        wordData[word][f"emotion_{emotion}"] += score
                        wordData[word][f"impact_{emotion}"] += score * (
                            (int(row['likes']) // 10) + int(row['comments'])
                        )

                except Exception as e:
                    print(f"Error processing word '{word}': {e}")


        print("Creating 'tempwords.csv'...")
        wordData = {}

        for row in self.dataset:
            try:
                processWords(wordData, row)
            except Exception as e:
                print(f"Error processing row: {e}")

        wordDf = pd.DataFrame.from_dict(wordData, orient='index').reset_index()
        wordDf.rename(columns={"index": "word"}, inplace=True)
        wordDf.to_csv('tempwords.csv', index=False)

class DataProcessor:
    def __init__(self, dataset, device, apiModels, emotions):
        self.dataset = dataset
        self.device = device
        self.apiPipelines = {
            model: pipeline("text-classification", model=model, top_k = None, device = DataProcessor.getBestGpu())
            for model in apiModels
        }
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

        return nltkStopwords.union(spacyStopwords)

    @staticmethod
    def getBestGpu():
        bestGpu = -1
        maxFreeMemory = 0
        for i in range(torch.cuda.device_count()):
            freeMemory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            if freeMemory > maxFreeMemory:
                maxFreeMemory = freeMemory
                bestGpu = i
        return bestGpu

    def calculateToneImpact(self, batch):
        tones = [SentimentIntensityAnalyzer().polarity_scores(text)['compound'] for text in batch['text']]
        impacts = [
            tone * ((int(likes) // 10) + int(comments))
            for tone, likes, comments in zip(tones, batch['likes'], batch['comments'])
        ]
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
        emotionScores = defaultdict(list)

        for model, apiPipeline in self.apiPipelines.items():
            try:
                outputs = apiPipeline(batch['text'])
                for textIndex, output in enumerate(outputs):
                    for emotionData in output:
                        label = emotionData['label']
                        score = emotionData['score']

                        if label not in emotionScores:
                            emotionScores[label] = [0] * len(batch['text'])

                        emotionScores[label][textIndex] += score
            except Exception as e:
                print(f"Error processing with model {model}: {e}")

        averagedEmotionScores = {
            emotion: [score / len(self.apiPipelines) for score in scores]
            for emotion, scores in emotionScores.items()
        }

        return averagedEmotionScores

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

class ComplexEmotionProcessor:
    def __init__(self, tempWordsPath, tempTextsPath, outputWordsPath, outputTextsPath, filteredEmotions):
        self.tempWordsPath = tempWordsPath
        self.tempTextsPath = tempTextsPath
        self.outputWordsPath = outputWordsPath
        self.outputTextsPath = outputTextsPath
        self.filteredEmotions = filteredEmotions

    def calculateComplexEmotions(self, row, filteredEmotions):
        complexEmotionScores = {}
        for complexEmotion, baseEmotions in filteredEmotions.items():
            baseScores = []
            for base in baseEmotions:
                # Handle non-prefixed columns (temptexts) and prefixed columns (tempwords)
                colName = f"emotion_{base}" if f"emotion_{base}" in row.index else base
                if colName in row:
                    baseScores.append(row[colName])
                else:
                    print(f"Missing column: {colName}")
            # Calculate average if baseScores are available
            complexEmotionScores[f"emotion_{complexEmotion}"] = (
                sum(baseScores) / len(baseScores) if baseScores else 0
            )
        return complexEmotionScores

    def addImpactColumns(self, df, complexEmotions):
        for emotion in complexEmotions.keys():
            colName = f"emotion_{emotion}" if f"emotion_{emotion}" in df.columns else emotion
            impactColName = f"impact_{emotion}"

            if colName in df.columns:
                if impactColName in df.columns:
                    print(f"Impact column {impactColName} already exists. Skipping calculation.")
                    continue

                df[impactColName] = df[colName] * (
                    (df["likes"] // 10) + df["comments"]
                )
            else:
                print(f"Missing complex emotion column for impact: {colName}")
        return df

    def processWords(self):
        print("Processing complex emotions for words...")
        wordsDf = pd.read_csv(self.tempWordsPath)

        # Calculate complex emotions
        complexEmotionScores = wordsDf.apply(
            lambda row: self.calculateComplexEmotions(row, self.filteredEmotions), axis=1
        )

        # Convert complex emotions into a DataFrame
        complexEmotionDf = pd.DataFrame(complexEmotionScores.tolist())

        # Drop columns from wordsDf if they already exist to prevent duplicates
        overlappingCols = set(wordsDf.columns).intersection(set(complexEmotionDf.columns))
        if overlappingCols:
            print(f"Removing overlapping columns: {overlappingCols}")
            wordsDf = wordsDf.drop(columns=overlappingCols)

        # Concatenate with original DataFrame
        updatedWordsDf = pd.concat([wordsDf, complexEmotionDf], axis=1)

        # Add impact columns for complex emotions
        updatedWordsDf = self.addImpactColumns(updatedWordsDf, self.filteredEmotions)

        # Save the updated DataFrame
        updatedWordsDf.to_csv(self.outputWordsPath, index=False)
        print(f"Updated words saved to {self.outputWordsPath}")


    def processTexts(self):
        print("Processing complex emotions for texts...")
        textsDf = pd.read_csv(self.tempTextsPath)
        print(f"Columns in texts file: {list(textsDf.columns)}")

        # Calculate complex emotions
        complexEmotionScores = textsDf.apply(
            lambda row: self.calculateComplexEmotions(row, self.filteredEmotions), axis=1
        )

        # Convert complex emotions into a DataFrame
        complexEmotionDf = pd.DataFrame(complexEmotionScores.tolist())

        # Concatenate with original DataFrame
        updatedTextsDf = pd.concat([textsDf, complexEmotionDf], axis=1)

        # Add impact columns for complex emotions
        updatedTextsDf = self.addImpactColumns(updatedTextsDf, self.filteredEmotions)

        # Save the updated DataFrame
        updatedTextsDf.to_csv(self.outputTextsPath, index=False)
        print(f"Updated texts saved to {self.outputTextsPath}")

    def process(self):
        self.processWords()
        self.processTexts()

class PoliticalScoreProcessor:
    def __init__(self, sentimentDataset, outputTextsPath):
        self.sentimentDataset = sentimentDataset  # Use sentiment_dataset directly
        self.outputTextsPath = outputTextsPath

    def processTexts(self):
        print(f"Dataset loaded for processing: {self.sentimentDataset.shape}")
        instruction_text = ("You will analyze a series of statements and assign two scores for each based on the specified axes. The X-axis represents economic ideology and ranges from -1 to 1. Negative scores on the X-axis indicate left-wing economic perspectives, which emphasize collective welfare, government regulation, wealth redistribution, and public ownership, while positive scores on the X-axis reflect right-wing economic perspectives, prioritizing free markets, minimal government intervention, privatization, and individual entrepreneurship. The Y-axis represents social ideology, also ranging from -1 to 1. Negative scores on the Y-axis indicate liberal perspectives, characterized by personal freedoms, openness to social change, reduced state control, and the protection of individual rights, while positive scores on the Y-axis represent authoritarian perspectives, emphasizing state control, law and order, adherence to traditional values, and limited individual freedoms for societal goals. For example, a statement advocating for universal healthcare funded through progressive taxation would score approximately -0.8 on the X-axis and -0.6 on the Y-axis, reflecting a left-wing economic perspective with moderately liberal social implications. A statement supporting the deregulation of financial markets and reduced corporate tax would score around 0.9 on the X-axis and 0 on the Y-axis, indicating a strongly right-wing economic position with neutral social implications. A statement calling for strict government surveillance to combat crime would score around 0 on the X-axis and 0.8 on the Y-axis, reflecting a neutral economic stance with a strongly authoritarian social perspective. A statement promoting gender equality and the legalization of same-sex marriage would score approximately 0 on the X-axis and -0.9 on the Y-axis, demonstrating a neutral economic stance and strongly liberal social perspective. Additional examples include a statement advocating for wealth redistribution through high taxes on the rich (-0.9, -0.2), the privatization of public schools (0.8, 0.2), government mandates for wearing uniforms in public schools (0.3, 0.6), support for environmental regulation of businesses (-0.7, -0.3), opposition to immigration (0, 0.7), promoting free college education funded by the state (-1, -0.4), and support for maintaining traditional family structures through policy incentives (0.2, 0.8). A statement proposing a ban on public protests for national security would score around 0.1 on the X-axis and 0.9 on the Y-axis, while one advocating for reducing military budgets to fund social welfare programs would score -0.8 on the X-axis and -0.6 on the Y-axis. For every statement, evaluate its economic and social implications independently, and provide scores within the range of -1 to 1. Your reply must strictly adhere to the format: score on x axis, score on y axis. Do not provide any additional explanation, context, or formatting—only the two scores in the specified format for each statement. You will analyze a series of statements and assign two scores for each based on the specified axes. The X-axis represents economic ideology and ranges from -1 to 1. Negative scores on the X-axis indicate left-wing economic perspectives, which emphasize collective welfare, government regulation, wealth redistribution, and public ownership, while positive scores on the X-axis reflect right-wing economic perspectives, prioritizing free markets, minimal government intervention, privatization, and individual entrepreneurship. The Y-axis represents social ideology, also ranging from -1 to 1. Negative scores on the Y-axis indicate liberal perspectives, characterized by personal freedoms, openness to social change, reduced state control, and the protection of individual rights, while positive scores on the Y-axis represent authoritarian perspectives, emphasizing state control, law and order, adherence to traditional values, and limited individual freedoms for societal goals. If a statement is neutral, normal, a question, or has no clear political or ideological connotation, you must assign the score 0, 0. Examples include The sky is blue, What is your favorite color? I enjoy painting, or The weather is pleasant today. These types of statements, which are descriptive, interrogative, or unrelated to political or ideological frameworks, must always receive the score 0, 0. Under no circumstances should you return anything other than the score in the exact format: score on x axis, score on y axis. For every statement, evaluate its economic and social implications independently, and if it does not align with the ideological framework, always return 0, 0.")
        
        # Initialize new columns for scores and impacts
        economicScores, socialScores, economicImpacts, socialImpacts = [], [], [], []

        for index, row in self.sentimentDataset.iterrows():
            try:
                print(f"Processing row {index}: {row['text']}")
                text = row['text']
                likes = int(row.get('likes', 0))
                comments = int(row.get('comments', 0))

                # Prepend instruction text to the input
                formatted_input = f"{instruction_text} Text: {text}"

                # Pass the text to the scoring model
                response = chat(
                    model="llama3.2",
                    messages=[{"role": "user", "content": formatted_input}]
                )
                print(f"API response: {response}")

                scores = response['message']['content'].split(", ")
                economicScore = float(scores[0])
                socialScore = float(scores[1])

                # Calculate impacts
                economicImpact = economicScore * ((likes // 10) + comments)
                socialImpact = socialScore * ((likes // 10) + comments)

                # Append calculated values
                economicScores.append(economicScore)
                socialScores.append(socialScore)
                economicImpacts.append(economicImpact)
                socialImpacts.append(socialImpact)

            except Exception as e:
                print(f"Error processing row {index}: {e}")
                economicScores.append(0)
                socialScores.append(0)
                economicImpacts.append(0)
                socialImpacts.append(0)

        # Add new columns to the DataFrame
        self.sentimentDataset['economic_score'] = economicScores
        self.sentimentDataset['social_score'] = socialScores
        self.sentimentDataset['economic_impact'] = economicImpacts
        self.sentimentDataset['social_impact'] = socialImpacts

        print("Columns added to DataFrame:")
        print(self.sentimentDataset.head())

        # Save the updated DataFrame
        self.sentimentDataset.to_csv(self.outputTextsPath, index=False)
        print(f"Processed file saved to {self.outputTextsPath}. Columns: {self.sentimentDataset.columns}")

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

    def launchGUI(self):
        root = tk.Tk()
        root.title("Data Selection")
        root.resizable(False, False)

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

        tk.Button(root, text="Submit", command=lambda: self.onSubmit(selectionVar, sortByVar, thresholdVar, countVar, graphTypeVar)).pack()

        root.mainloop()

def checkAndCreateFiles(fileHandler, dataProcessor):
    # Check for the existence of temp files
    textsExists = os.path.exists('temptexts.csv')
    wordsExists = os.path.exists('tempwords.csv')

    try:
        if not textsExists:
            print("'temptexts.csv' is missing. Generating...")
            fileHandler.createTextsCsv(
                dataProcessor.calculateToneImpact,
                dataProcessor.calculateFrequency,
                dataProcessor
            )
        if not wordsExists:
            print("'tempwords.csv' is missing. Generating...")
            fileHandler.createWordsCsv()

        # Process political scores ONLY for texts
        print("Processing political scores...")
        sentimentDataset = pd.read_csv("temptexts.csv")  # Load the dataset for text processing
        politicalScoreProcessor = PoliticalScoreProcessor(
            sentimentDataset=sentimentDataset,
            outputTextsPath="texts.csv"
        )
        # Only process texts for political and economic scores
        print("Processing political scores for texts...")
        politicalScoreProcessor.processTexts()

    except Exception as e:
        print(f"Error while creating files or processing political scores: {e}")

    # Ensure files include necessary economic and social scores
    try:
        print("Verifying final files include economic and social scores...")
        if not os.path.exists("texts.csv"):
            print("Error: 'texts.csv' is missing. Reprocessing 'temptexts.csv'...")
            sentimentDataset = pd.read_csv("temptexts.csv")
            politicalScoreProcessor = PoliticalScoreProcessor(
                sentimentDataset=sentimentDataset,
                outputTextsPath="texts.csv"
            )
            politicalScoreProcessor.processTexts()
    except Exception as e:
        print(f"Error during final verification of files: {e}")

def main():
    # Load all stopwords using DataProcessor
    try:
        allStopwords = DataProcessor.getAllStopwords()
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return

    # Initialize GPU selection
    try:
        bestGpu = DataProcessor.getBestGpu()
        if bestGpu != -1:
            device = torch.device(f"cuda:{bestGpu}")
            print(f"Using GPU: {bestGpu}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
    except Exception as e:
        print(f"Error initializing device: {e}")
        return

    # Load datasets
    try:
        hf_dataset = Dataset.from_csv('sentiment_dataset.csv')
    except FileNotFoundError:
        print("Error: 'sentiment_dataset.csv' not found. Ensure the file is in the correct directory.")
        return
    except pd.errors.EmptyDataError:
        print("Error: 'sentiment_dataset.csv' is empty. Provide a valid dataset.")
        return
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        return

    # List of models for API pipelines
    apiModels = [
        "j-hartmann/emotion-English-distilroberta-base",
        "bhadresh-savani/bert-base-go-emotion",
        "monologg/bert-base-cased-goemotions-original",
    ]

    # Initialize DataProcessor and FileHandler
    try:
        emotions = []
        dataProcessor = DataProcessor(hf_dataset, device, apiModels, emotions)
        fileHandler = FileHandler(hf_dataset, allStopwords, emotions, dataProcessor)
    except Exception as e:
        print(f"Error initializing DataProcessor or FileHandler: {e}")
        return

    # Check and create files with political scores
    try:
        checkAndCreateFiles(fileHandler, dataProcessor)
    except Exception as e:
        print(f"Error in checkAndCreateFiles: {e}")
        return

    # Process complex emotions BEFORE processing political scores
    try:
        complexEmotionProcessor = ComplexEmotionProcessor(
            tempWordsPath="tempwords.csv",
            tempTextsPath="temptexts.csv",
            outputWordsPath="words.csv",
            outputTextsPath="texts_temp.csv",  # Prevent overwriting texts.csv
            filteredEmotions = {
                "compassion": ["caring", "sadness"],
                "elation": ["joy", "excitement"],
                "affection": ["love", "approval"],
                "contentment": ["relief", "joy"],
                "playfulness": ["amusement", "joy"],
                "empathy": ["caring", "sadness"],
                "warmth": ["love", "caring"],
                "frustration": ["annoyance", "anger"],
                "shame": ["embarrassment", "disapproval"],
                "regret": ["remorse", "sadness"],
                "guilt": ["remorse", "grief"],
                "loneliness": ["sadness", "neutral"],
                "disdain": ["disapproval", "disgust"],
                "curiosity": ["confusion", "optimism"],
                "skepticism": ["confusion", "realization"],
                "uncertainty": ["confusion", "neutral"],
                "triumph": ["pride", "joy"],
                "reluctance": ["disapproval", "desire"],
                "apathy": ["neutral", "sadness"],
                "nostalgia": ["joy", "sadness"],
                "intrigue": ["curiosity", "desire"],
                "hopefulness": ["optimism", "joy"],
                "bliss": ["joy", "relief"],
                "fascination": ["curiosity", "admiration"],
                "passion": ["love", "desire"],
                "hopelessness": ["sadness", "disappointment"],
                "bitterness": ["sadness", "anger"],
                "gratefulness": ["gratitude", "relief"],
                "agitation": ["annoyance", "nervousness"],
                "yearning": ["desire", "sadness"],
                "sorrow": ["grief", "sadness"],
                "delight": ["joy", "amusement"],
                "trepidation": ["fear", "nervousness"],
                "amazement": ["surprise", "joy", "admiration"],
                "complacency": ["neutral", "relief", "approval"],
                "disillusionment": ["sadness", "disappointment", "realization"],
                "zeal": ["excitement", "pride", "desire"],
                "reverence": ["admiration", "gratitude"],
                "infatuation": ["love", "desire", "admiration"],
                "composure": ["relief", "neutral", "caring"],
                "ecstasy": ["joy", "excitement", "love"],
                "anticipation": ["excitement", "optimism", "curiosity"],
                "resignation": ["sadness", "relief"],
                "hostility": ["anger", "disgust", "annoyance"],
                "disorientation": ["confusion", "fear", "surprise"],
                "compunction": ["remorse", "sadness", "grief"],
                "humility": ["gratitude", "relief", "approval"],
                "serenity": ["joy", "relief", "caring"],
                "reconciliation": ["relief", "love", "gratitude"],
                "alienation": ["sadness", "disgust", "disapproval"],
                "exultation": ["pride", "joy", "excitement"],
                "affirmation": ["approval", "optimism", "pride"],
                "serendipity": ["joy", "surprise", "relief"],
                "acceptance": ["relief", "approval", "caring"],
                "resentment": ["sadness", "anger", "disapproval"],
                "cheerfulness": ["joy", "amusement", "optimism"],
                "apprehension": ["fear", "nervousness", "curiosity"],
                "eagerness": ["excitement", "curiosity"],
                "clarity": ["relief", "realization", "caring"],
                "hesitation": ["fear", "nervousness", "confusion"],
                "grievance": ["anger", "sadness", "disappointment"],
                "outrage": ["anger", "disapproval", "disgust"],
                "pity": ["sadness", "caring"],
                "shock": ["surprise", "fear", "disgust"],
                "satisfaction": ["relief", "joy", "approval"]
                }

        )
        complexEmotionProcessor.process()
    except Exception as e:
        print(f"Error processing complex emotions: {e}")
        return

    # Re-run PoliticalScoreProcessor to ensure it's the final step
    try:
        sentimentDataset = pd.read_csv("texts_temp.csv")  # Use the output from complex emotions
        politicalScoreProcessor = PoliticalScoreProcessor(
            sentimentDataset=sentimentDataset,
            outputTextsPath="texts.csv"  # Save final output here
        )
        politicalScoreProcessor.processTexts()
    except Exception as e:
        print(f"Error in PoliticalScoreProcessor: {e}")
        return

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()