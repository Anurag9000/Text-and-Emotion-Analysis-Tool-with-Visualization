from ollama import chat
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FileHandler:
    def __init__(self, dataset, stopwords, emotions, dataProcessor):
        self.dataset = dataset
        self.stopwords = stopwords
        self.emotions = emotions
        self.dataProcessor = dataProcessor  # Store the dataProcessor instance

    def createTextsCsv(self, calculateToneImpact, dataProcessor):
        def addToneAndImpact(dataset):
            try:
                return dataset.map(calculateToneImpact, batched=True)
            except Exception as e:
                print(f"Error in 'addToneAndImpact': {e}")
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

        print("Creating 'temp1texts.csv'...")
        dataset = self.dataset
        dataset = addToneAndImpact(dataset)
        dataset = addEmotions(dataset)
        dataset.to_csv('temp1texts.csv', index=False)

    def createWordsCsv(self):
        def processWords(wordData, row):
            temp2words = re.findall(r'\b\w+\b', row['text'].lower())  # Tokenize and normalize case
            for word in temp2words:
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

                    wordData[word]['frequency'] += 1  # Increment global frequency
                    wordData[word]['likes'] += int(row['likes'])
                    wordData[word]['comments'] += int(row['comments'])
                    wordData[word]['tone'] += float(SentimentIntensityAnalyzer().polarity_scores(word)['compound'])
                    wordData[word]['impact'] += wordData[word]['tone'] * (
                        (int(row['likes']) // 10) + int(row['comments'])
                    )

                    for emotion, score in wordEmotionScores.items():
                        score = score[0] if isinstance(score, list) else score
                        wordData[word][f"emotion_{emotion}"] += score
                        wordData[word][f"impact_{emotion}"] += score * (
                            (int(row['likes']) // 10) + int(row['comments'])
                        )

                except Exception as e:
                    print(f"Error processing word '{word}': {e}")

        print("Creating 'temp1words.csv'...")
        wordData = {}

        for row in self.dataset:
            try:
                processWords(wordData, row)
            except Exception as e:
                print(f"Error processing row: {e}")

        wordDf = pd.DataFrame.from_dict(wordData, orient='index').reset_index()
        wordDf.rename(columns={"index": "word"}, inplace=True)
        wordDf.to_csv('temp1words.csv', index=False)

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
    def __init__(self, temp1WordsPath, temp1TextsPath, outputWordsPath, outputTextsPath, filteredEmotions):
        self.temp1WordsPath = temp1WordsPath
        self.temp1TextsPath = temp1TextsPath
        self.outputWordsPath = outputWordsPath
        self.outputTextsPath = outputTextsPath
        self.filteredEmotions = filteredEmotions

    def calculateComplexEmotions(self, row, filteredEmotions):
        complexEmotionScores = {}
        for complexEmotion, baseEmotions in filteredEmotions.items():
            baseScores = []
            for base in baseEmotions:
                # Handle non-prefixed columns (temp1texts) and prefixed columns (temp1words)
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
        print("Processing complex emotions for temp2words...")
        temp2wordsDf = pd.read_csv(self.temp1WordsPath)

        # Calculate complex emotions
        complexEmotionScores = temp2wordsDf.apply(
            lambda row: self.calculateComplexEmotions(row, self.filteredEmotions), axis=1
        )

        # Convert complex emotions into a DataFrame
        complexEmotionDf = pd.DataFrame(complexEmotionScores.tolist())

        # Drop columns from temp2wordsDf if they already exist to prevent duplicates
        overlappingCols = set(temp2wordsDf.columns).intersection(set(complexEmotionDf.columns))
        if overlappingCols:
            print(f"Removing overlapping columns: {overlappingCols}")
            temp2wordsDf = temp2wordsDf.drop(columns=overlappingCols)

        # Concatenate with original DataFrame
        updatedWordsDf = pd.concat([temp2wordsDf, complexEmotionDf], axis=1)

        # Add impact columns for complex emotions
        updatedWordsDf = self.addImpactColumns(updatedWordsDf, self.filteredEmotions)

        # Save the updated DataFrame
        updatedWordsDf.to_csv(self.outputWordsPath, index=False)
        print(f"Updated temp2words saved to {self.outputWordsPath}")


    def processTexts(self):
        print("Processing complex emotions for temp3texts...")
        temp3textsDf = pd.read_csv(self.temp1TextsPath)

        # Calculate complex emotions
        complexEmotionScores = temp3textsDf.apply(
            lambda row: self.calculateComplexEmotions(row, self.filteredEmotions), axis=1
        )

        # Convert complex emotions into a DataFrame
        complexEmotionDf = pd.DataFrame(complexEmotionScores.tolist())

        # Concatenate with original DataFrame
        updatedtemp3textsDf = pd.concat([temp3textsDf, complexEmotionDf], axis=1)

        # Add impact columns for complex emotions
        updatedtemp3textsDf = self.addImpactColumns(updatedtemp3textsDf, self.filteredEmotions)

        # Save the updated DataFrame
        updatedtemp3textsDf.to_csv(self.outputTextsPath, index=False)
        print(f"Updated temp3texts saved to {self.outputTextsPath}")

    def process(self):
        self.processWords()
        self.processTexts()

class ToneAdjuster:
    def __init__(self, positiveEmotions, negativeEmotions):
        self.positiveEmotions = positiveEmotions
        self.negativeEmotions = negativeEmotions

    def adjustToneAndImpact(self, df):
        if "tone" not in df.columns:
            print("Tone column is missing from DataFrame.")
            return df

        positiveSum = df[[col for col in self.positiveEmotions if col in df.columns]].sum(axis=1, skipna=True)
        negativeSum = df[[col for col in self.negativeEmotions if col in df.columns]].sum(axis=1, skipna=True)

        # Adjust tone
        df["adjusted_tone"] = df["tone"] + positiveSum - negativeSum

        # Adjust impact based on new tone
        if "likes" in df.columns and "comments" in df.columns:
            df["adjusted_impact"] = df["adjusted_tone"] * ((df["likes"] // 10) + df["comments"])
        else:
            print("Likes or comments column missing. Impact cannot be recalculated.")

        return df

    def adjustWordsToneAndImpact(self, df):
        if "tone" not in df.columns:
            print("Tone column is missing from temp2words DataFrame.")
            return df

        positiveSum = df[[col for col in self.positiveEmotions if col in df.columns]].sum(axis=1, skipna=True)
        negativeSum = df[[col for col in self.negativeEmotions if col in df.columns]].sum(axis=1, skipna=True)

        # Adjust tone for temp2words
        df["adjusted_tone"] = df["tone"] + positiveSum - negativeSum

        # Adjust impact based on likes and comments instead of frequency
        if "likes" in df.columns and "comments" in df.columns:
            df["adjusted_impact"] = df["adjusted_tone"] * ((df["likes"] // 10) + df["comments"])
        else:
            print("Likes or comments column missing. Adjusted impact cannot be calculated.")

        return df

class PoliticalScoreProcessor:
    def __init__(self, sentimentDataset, outputTextsPath):
        self.sentimentDataset = sentimentDataset  # Use sentiment_dataset directly
        self.outputTextsPath = outputTextsPath

    def processTexts(self):
        instruction_text = ("You will analyze a series of statements and assign two scores for each based on the specified axes. The X-axis represents economic ideology and ranges from -1 to 1. Negative scores on the X-axis indicate left-wing economic perspectives, which emphasize collective welfare, government regulation, wealth redistribution, and public ownership, while positive scores on the X-axis reflect right-wing economic perspectives, prioritizing free markets, minimal government intervention, privatization, and individual entrepreneurship. The Y-axis represents social ideology, also ranging from -1 to 1. Negative scores on the Y-axis indicate liberal perspectives, characterized by personal freedoms, openness to social change, reduced state control, and the protection of individual rights, while positive scores on the Y-axis represent authoritarian perspectives, emphasizing state control, law and order, adherence to traditional values, and limited individual freedoms for societal goals. For example, a statement advocating for universal healthcare funded through progressive taxation would score approximately -0.8 on the X-axis and -0.6 on the Y-axis, reflecting a left-wing economic perspective with moderately liberal social implications. A statement supporting the deregulation of financial markets and reduced corporate tax would score around 0.9 on the X-axis and 0 on the Y-axis, indicating a strongly right-wing economic position with neutral social implications. A statement calling for strict government surveillance to combat crime would score around 0 on the X-axis and 0.8 on the Y-axis, reflecting a neutral economic stance with a strongly authoritarian social perspective. A statement promoting gender equality and the legalization of same-sex marriage would score approximately 0 on the X-axis and -0.9 on the Y-axis, demonstrating a neutral economic stance and strongly liberal social perspective. Additional examples include a statement advocating for wealth redistribution through high taxes on the rich (-0.9, -0.2), the privatization of public schools (0.8, 0.2), government mandates for wearing uniforms in public schools (0.3, 0.6), support for environmental regulation of businesses (-0.7, -0.3), opposition to immigration (0, 0.7), promoting free college education funded by the state (-1, -0.4), and support for maintaining traditional family structures through policy incentives (0.2, 0.8). A statement proposing a ban on public protests for national security would score around 0.1 on the X-axis and 0.9 on the Y-axis, while one advocating for reducing military budgets to fund social welfare programs would score -0.8 on the X-axis and -0.6 on the Y-axis. For every statement, evaluate its economic and social implications independently, and provide scores within the range of -1 to 1. Your reply must strictly adhere to the format: score on x axis, score on y axis. Do not provide any additional explanation, context, or formatting—only the two scores in the specified format for each statement. You will analyze a series of statements and assign two scores for each based on the specified axes. The X-axis represents economic ideology and ranges from -1 to 1. Negative scores on the X-axis indicate left-wing economic perspectives, which emphasize collective welfare, government regulation, wealth redistribution, and public ownership, while positive scores on the X-axis reflect right-wing economic perspectives, prioritizing free markets, minimal government intervention, privatization, and individual entrepreneurship. The Y-axis represents social ideology, also ranging from -1 to 1. Negative scores on the Y-axis indicate liberal perspectives, characterized by personal freedoms, openness to social change, reduced state control, and the protection of individual rights, while positive scores on the Y-axis represent authoritarian perspectives, emphasizing state control, law and order, adherence to traditional values, and limited individual freedoms for societal goals. If a statement is neutral, normal, a question, or has no clear political or ideological connotation, you must assign the score 0, 0. Examples include The sky is blue, What is your favorite color? I enjoy painting, or The weather is pleasant today. These types of statements, which are descriptive, interrogative, or unrelated to political or ideological frameworks, must always receive the score 0, 0. Under no circumstances should you return anything other than the score in the exact format: score on x axis, score on y axis. For every statement, evaluate its economic and social implications independently, and if it does not align with the ideological framework, always return 0, 0.")
        
        # Initialize new columns for scores and impacts
        economicScores, socialScores, economicImpacts, socialImpacts = [], [], [], []

        for index, row in self.sentimentDataset.iterrows():
            try:
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

        # Save the updated DataFrame
        self.sentimentDataset.to_csv(self.outputTextsPath, index=False)

class Visualizer:
    def __init__(self, temp3textsDf, temp2wordsDf):
        self.temp3textsDf = temp3textsDf
        self.temp2wordsDf = temp2wordsDf

    def groupAndSummarizeData(self, selection, sortBy):
        """
        Group and summarize the data based on the user's selection and sort criteria.
        """
        if selection in ['temp2words', 'temp3texts']:
            grouped = self.temp2wordsDf.groupby('word', as_index=False).sum() if selection == 'temp2words' else self.temp3textsDf.groupby('text', as_index=False).sum()
            grouped = grouped[[selection[:-1], sortBy]].sort_values(by=sortBy, ascending=False)
        elif selection in ['agegroup', 'country', 'time', 'userid']:
            if selection in self.temp3textsDf.columns:
                grouped = self.temp3textsDf.groupby(selection, as_index=False).sum()
                grouped = grouped[[selection, sortBy]].sort_values(by=sortBy, ascending=False)
            else:
                print(f"Column '{selection}' not found in the dataset.")
                return pd.DataFrame()
        else:
            print(f"Selection '{selection}' not found in data columns.")
            return pd.DataFrame()

        return grouped


    def sliceData(self, df, threshold, countVal):
        """
        Slice the data based on the specified threshold and count.
        """
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
            print(f"Unknown threshold: {threshold}")
            return pd.DataFrame()

    def plotData(self, df, column, graphType, selection, sortBy, actualCount):
        """
        Plot the data based on user specifications.
        """
        plt.figure(figsize=(10, 6))

        try:
            if graphType == 'Bar':
                plt.bar(df.iloc[:, 0].astype(str), df[column])
            elif graphType == 'Line':
                plt.plot(df.iloc[:, 0].astype(str), df[column], marker='o')
            elif graphType == 'Pie':
                plt.pie(df[column], labels=df.iloc[:, 0].astype(str), autopct='%1.1f%%')
            else:
                print(f"Graph type '{graphType}' not recognized. Defaulting to Bar chart.")
                plt.bar(df.iloc[:, 0].astype(str), df[column])

            plt.xticks(rotation=45, ha='right')
            plotTitle = f"Top {actualCount} {selection} by {sortBy} ({column})"
            plt.title(plotTitle)

            if graphType != 'Pie':
                plt.ylabel(column.capitalize())
                plt.xlabel(selection.capitalize() if selection not in ['temp2words', 'temp3texts'] else selection.capitalize())

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error while plotting data: {e}")

class GUIHandler:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.selection = None
        self.sortBy = None
        self.threshold = None
        self.countVal = None
        self.graphType = None

    def loadDynamicColumns(self, selection):
        """
        Dynamically load column names based on the user's selection.
        """
        if selection in ['temp3texts', 'temp2words']:
            filePath = 'adjusted_texts.csv' if selection == 'temp3texts' else 'adjusted_words.csv'
        elif selection in ['agegroup', 'country', 'time', 'userid']:
            filePath = 'adjusted_texts.csv'  # Assuming these columns are in the same file as `temp3texts`.
        else:
            print(f"Dynamic columns not applicable for selection: {selection}")
            return []

        try:
            data = pd.read_csv(filePath)
            return list(data.columns)
        except Exception as e:
            print(f"Error loading columns for {selection}: {e}")
            return []

    def updateSortByOptions(self, event, sortByMenu, selectionVar):
        """
        Update the Sort By dropdown options based on the selected data type.
        """
        selection = selectionVar.get()
        columns = self.loadDynamicColumns(selection)
        if columns:
            sortByMenu["values"] = columns
        else:
            sortByMenu["values"] = []

    def onSubmit(self, selectionVar, sortByVar, thresholdVar, countVar, graphTypeVar):
        """
        Handle the logic when the Submit button is clicked.
        """
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
        """
        Launch the GUI for user interaction.
        """
        root = tk.Tk()
        root.title("Data Selection")
        root.resizable(False, False)

        selectionVar = tk.StringVar(value='temp3texts')
        sortByVar = tk.StringVar(value='impact')
        thresholdVar = tk.StringVar(value='Highest')
        countVar = tk.StringVar(value='10')
        graphTypeVar = tk.StringVar(value='Bar')

        tk.Label(root, text="Select Data Type:").pack()
        dataTypeMenu = ttk.Combobox(root, textvariable=selectionVar, values=['agegroup', 'country', 'temp3texts', 'time', 'userid', 'temp2words'], state="readonly")
        dataTypeMenu.pack()

        tk.Label(root, text="Sort By:").pack()
        sortByMenu = ttk.Combobox(root, textvariable=sortByVar, values=['impact', 'tone', 'likes', 'comments', 'frequency'], state="readonly")
        sortByMenu.pack()

        # Bind event to update Sort By options dynamically
        dataTypeMenu.bind("<<ComboboxSelected>>", lambda event: self.updateSortByOptions(event, sortByMenu, selectionVar))

        tk.Label(root, text="Threshold:").pack()
        thresholdMenu = ttk.Combobox(root, textvariable=thresholdVar, values=['Highest', 'Lowest', 'Extremes'], state="readonly")
        thresholdMenu.pack()

        tk.Label(root, text="Number of Items to Display:").pack()
        countEntry = ttk.Entry(root, textvariable=countVar)
        countEntry.pack()

        tk.Label(root, text="Select Graph Type:").pack()
        graphTypeMenu = ttk.Combobox(
            root,
            textvariable=graphTypeVar,
            values=['Bar', 'Line', 'Pie'],  # Most common graph types
            state="readonly"
        )
        graphTypeMenu.pack()


        tk.Button(root, text="Submit", command=lambda: self.onSubmit(selectionVar, sortByVar, thresholdVar, countVar, graphTypeVar)).pack()

        root.mainloop()

def checkAndCreateFiles(fileHandler, dataProcessor):
    # Check for the existence of temp files
    textsExists = os.path.exists('temp1texts.csv')
    wordsExists = os.path.exists('temp1words.csv')

    try:
        if not textsExists:
            print("'temp1texts.csv' is missing. Generating...")
            fileHandler.createTextsCsv(
                dataProcessor.calculateToneImpact,
                dataProcessor
            )
        if not wordsExists:
            print("'temp1words.csv' is missing. Generating...")
            fileHandler.createWordsCsv()

        # Process political scores ONLY for temp3texts
        print("Processing political scores...")
        sentimentDataset = pd.read_csv("temp1texts.csv")  # Load the dataset for text processing
        politicalScoreProcessor = PoliticalScoreProcessor(
            sentimentDataset=sentimentDataset,
            outputTextsPath="temp3texts.csv"
        )
        # Only process temp3texts for political and economic scores
        print("Processing political scores for temp3texts...")
        politicalScoreProcessor.processTexts()

    except Exception as e:
        print(f"Error while creating files or processing political scores: {e}")

    # Ensure files include necessary economic and social scores
    try:
        print("Verifying final files include economic and social scores...")
        if not os.path.exists("temp3texts.csv"):
            print("Error: 'temp3texts.csv' is missing. Reprocessing 'temp1texts.csv'...")
            sentimentDataset = pd.read_csv("temp1texts.csv")
            politicalScoreProcessor = PoliticalScoreProcessor(
                sentimentDataset=sentimentDataset,
                outputTextsPath="temp3texts.csv"
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
            temp1WordsPath="temp1words.csv",
            temp1TextsPath="temp1texts.csv",
            outputWordsPath="temp2words.csv",
            outputTextsPath="temp2texts.csv",  # Prevent overwriting temp3texts.csv
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
        sentimentDataset = pd.read_csv("temp2texts.csv")  # Use the output from complex emotions
        politicalScoreProcessor = PoliticalScoreProcessor(
            sentimentDataset=sentimentDataset,
            outputTextsPath="temp3texts.csv"  # Save final output here
        )
        politicalScoreProcessor.processTexts()
    except Exception as e:
        print(f"Error in PoliticalScoreProcessor: {e}")
        return

    positiveEmotions = [
        "joy", "approval", "admiration", "optimism", "caring", "relief", "gratitude", "amusement", "pride",
        "excitement", "desire", "curiosity", "emotion_compassion", "emotion_elation", "emotion_affection",
        "emotion_contentment", "emotion_playfulness", "emotion_empathy", "emotion_warmth", "emotion_triumph",
        "emotion_nostalgia", "emotion_hopefulness", "emotion_bliss", "emotion_fascination", "emotion_passion",
        "emotion_delight", "emotion_amazement", "emotion_zeal", "emotion_reverence", "emotion_infatuation",
        "emotion_composure", "emotion_ecstasy", "emotion_anticipation", "emotion_serendipity", "emotion_acceptance",
        "emotion_cheerfulness", "emotion_eagerness", "emotion_clarity", "emotion_gratefulness",
        "emotion_joy", "emotion_approval", "emotion_excitement", "emotion_admiration", "emotion_caring",
        "emotion_amusement", "emotion_gratitude", "emotion_optimism", "emotion_pride", "emotion_relief"
    ]

    negativeEmotions = [
        "surprise", "sadness", "neutral", "fear", "anger", "disgust", "realization", "disapproval", "annoyance",
        "disappointment", "confusion", "nervousness", "embarrassment", "remorse", "love", "grief",
        "emotion_frustration", "emotion_shame", "emotion_regret", "emotion_guilt", "emotion_loneliness",
        "emotion_disdain", "emotion_skepticism", "emotion_uncertainty", "emotion_reluctance", "emotion_apathy",
        "emotion_bitterness", "emotion_agitation", "emotion_yearning", "emotion_sorrow", "emotion_trepidation",
        "emotion_complacency", "emotion_disillusionment", "emotion_resignation", "emotion_hostility",
        "emotion_disorientation", "emotion_compunction", "emotion_humility", "emotion_serenity", "emotion_reconciliation",
        "emotion_alienation", "emotion_exultation", "emotion_affirmation", "emotion_resentment", "emotion_hesitation",
        "emotion_grievance", "emotion_outrage", "emotion_pity", "emotion_shock", "emotion_satisfaction",
        "emotion_surprise", "emotion_sadness", "emotion_fear", "emotion_anger", "emotion_disgust",
        "emotion_disapproval", "emotion_disappointment", "emotion_confusion", "emotion_nervousness",
        "emotion_embarrassment", "emotion_grief", "emotion_remorse"
    ]

    # Tone Adjustment Step for Sentences

    # Tone Adjustment Step for Texts
    try:
        print("Applying tone adjustments for temp3texts...")
        temp3textsDf = pd.read_csv("temp3texts.csv")

        toneAdjuster = ToneAdjuster(positiveEmotions, negativeEmotions)
        adjustedtemp3textsDf = toneAdjuster.adjustToneAndImpact(temp3textsDf)

        adjustedtemp3textsDf.to_csv("adjusted_texts.csv", index=False)
        print("Adjusted temp3texts saved to 'adjusted_texts.csv'.")
    except Exception as e:
        print(f"Error applying tone adjustments for temp3texts: {e}")

    # Tone Adjustment Step for Words
    try:
        print("Applying tone adjustments for temp2words...")
        temp2wordsDf = pd.read_csv("temp2words.csv")
        toneAdjuster = ToneAdjuster(positiveEmotions, negativeEmotions)
        adjustedWordsDf = toneAdjuster.adjustWordsToneAndImpact(temp2wordsDf)

        adjustedWordsDf.to_csv("adjusted_words.csv", index=False)
        print("Adjusted temp2words saved to 'adjusted_words.csv'.")
    except Exception as e:
        print(f"Error applying tone adjustments for temp2words: {e}")

    # Initialize Visualizer with adjusted data
    try:
        temp3textsDf = pd.read_csv("adjusted_texts.csv")
        temp2wordsDf = pd.read_csv("adjusted_words.csv")
        visualizer = Visualizer(temp3textsDf, temp2wordsDf)
    except Exception as e:
        print(f"Error loading data for Visualizer: {e}")
        return

    # Launch GUI with Visualizer
    try:
        guiHandler = GUIHandler(visualizer)
        guiHandler.launchGUI()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
