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
import pymysql
import csv
nlp = spacy.load("en_core_web_sm")

## FUTURE SCOPE: DETECT NON ENGLISH NATIVE LANGUAGES WRITTEN IN LATIN SCRIPT OR OTHERWISE
## MINE EMOTIONS FOR FINETUNINING AND TRAINING MODELS OF LLMS
# Analyzing
# Monitoring
# Moderating content on social media
# Monitoring mental health of users to send aid
# FUTURE SCOPE: DETECT NON ENGLISH NATIVE LANGUAGES WRITTEN IN LATIN SCRIPT OR OTHERWISE
# NOTE TO SELF: WORDS.CSV AND WORDS TABLE IN DATABASE ARE CREATED FOR THE FOLLOWING REASON:
# 1: PROCESSING A LARGE ENOUGH TEXT SAMPLES WOULD GENERATE ALMOST ALL THE MOST COMMONLY USED WORDS IN CONTENT
# 2: DEPENDING ON THE SCORES AND RATINGS OF THE LEMMATIZED WORDS; OTHER CONTENT CAN BE SCANNED AGAINST THIS AND RATED BY THE WORDS USED IN THE CONTENT;
# 3: CONTENT WITH MORE POSITIVE WORDS CAN BE RATED CHILD FRIENDLY AND NEGATIVE CONNOTATION WORDS CAN BE CENSORED BY THE PLATFORM OR RESTRICTED
# 4: GENERATING A LARGE ENOUGH DATASET FOR INVIDUAL ANNOTATED WORDS WOULD SIMPLIFY FURTHER TEXT PROCESSING AS INSTEAD OF PASSING THE TEXT THROUGH LLMS THE SUM OF SCORES OF WORDS USED IN IT CAN BE USED INSTEAD
# 5: THIS WOULD BE MUCH FASTER AND CHEAPER REQUIRING MUCH LESS RESOURCES AND SIMPLE PROCESS AND A GOOD ESTIMATE OF THE TONE OF THE TEXT
# 6: BUT MAINLY IT CAN BE USED TO MONITOR AND RESTRICT/CENSOR CONTENT BASED ON THE SEVERE NEGATIVE WORDS USED


class FileHandler:
    def __init__(self, dataset, stopwords, emotions, dataProcessor):
        self.dataset = dataset
        self.stopwords = stopwords
        self.emotions = emotions
        self.dataProcessor = dataProcessor  # Store the dataProcessor instance

    @staticmethod
    def cleanData(filePath, outputFilePath):
        try:
            # Load the dataset
            data = pd.read_csv(filePath, quotechar='"')
            print(f"Dataset loaded successfully. Shape: {data.shape}")
        except FileNotFoundError:
            print(f"Error: File not found at {filePath}.")
            return
        except pd.errors.EmptyDataError:
            print(f"Error: Dataset at {filePath} is empty.")
            return
        except Exception as e:
            print(f"Unexpected error loading dataset: {e}")
            return

        # Detect and fill missing values
        print("Checking for missing values...")
        missingCount = data.isnull().sum()
        print("Missing values before cleaning:")
        print(missingCount)

        for column in data.columns:
            if data[column].dtype == "object":  # String or object columns
                data[column] = data[column].fillna("None")
            else:  # Numeric columns
                data[column] = data[column].fillna(0)

        print("Missing values after cleaning:")
        print(data.isnull().sum())

        # Save the cleaned dataset
        try:
            data.to_csv(outputFilePath, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"Cleaned dataset saved to {outputFilePath}.")
        except Exception as e:
            print(f"Error saving cleaned dataset: {e}")


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
        dataset.to_csv('temp1texts.csv', index=False, quoting=csv.QUOTE_MINIMAL)

    def createWordsCsv(self, inputFilePath, outputFilePath):
        print("Processing texts to create words.csv...")

        try:
            textsDf = pd.read_csv(inputFilePath, quotechar='"')

            # Ensure all columns except 'text' are numeric
            for col in textsDf.columns:
                if col != "text":
                    textsDf[col] = pd.to_numeric(textsDf[col], errors="coerce")
        except Exception as e:
            print(f"Error loading or processing input file: {e}")
            return

        if "text" not in textsDf.columns:
            print("Input file must contain a 'text' column.")
            return

        wordData = {}
        nlp = spacy.load("en_core_web_sm")

        for _, row in textsDf.iterrows():
            text = row['text']
            doc = nlp(text.lower())
            lemmatizedWords = set(token.lemma_ for token in doc if token.is_alpha)

            for word in lemmatizedWords:
                if word not in self.stopwords:
                    if word not in wordData:
                        wordData[word] = {col: 0 for col in textsDf.columns if col != "text"}

                    for col in textsDf.columns:
                        if col != "text":
                            wordData[word][col] += row[col]  # Numeric addition

        wordDf = pd.DataFrame.from_dict(wordData, orient="index").reset_index()
        wordDf.rename(columns={"index": "word"}, inplace=True)

        try:
            wordDf.to_csv(outputFilePath, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"Words data saved to {outputFilePath}.")
        except Exception as e:
            print(f"Error saving words.csv: {e}")


    @staticmethod
    def cleanTempFiles():
        tempFiles = ['temp1texts.csv', 'temp2texts.csv', 'temp3texts.csv', 'temp4texts.csv', 'temp5texts.csv']
        for file in tempFiles:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Removed temporary file: {file}")
            except Exception as e:
                print(f"Error removing file {file}: {e}")

class DataProcessor:
    def __init__(self, dataset, device, apiModels, emotions):
        self.dataset = dataset
        self.device = device
        self.apiPipelines = {
            model: pipeline("text-classification", model=model, top_k=None, device=DataProcessor.getBestGpu())
            for model in apiModels
        }
        self.emotions = emotions

    @staticmethod
    def getAllStopwords():
        nltkLanguages = stopwords.fileids()
        nltkStopwords = set()
        for lang in nltkLanguages:
            nltkStopwords.update(stopwords.words(lang))

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
    def __init__(self, tempTextsPath, outputTextsPath, filteredEmotions):
        self.tempTextsPath = tempTextsPath
        self.outputTextsPath = outputTextsPath
        self.filteredEmotions = filteredEmotions

    def calculateComplexEmotions(self, row, filteredEmotions):
        complexEmotionScores = {}
        for complexEmotion, baseEmotions in filteredEmotions.items():
            baseScores = []
            for base in baseEmotions:
                colName = f"emotion_{base}" if f"emotion_{base}" in row.index else base
                if colName in row:
                    baseScores.append(row[colName])
                else:
                    print(f"Missing column: {colName}")
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

    def processTexts(self):
        print("Processing complex emotions for texts...")
        textsDf = pd.read_csv(self.tempTextsPath, quotechar='"')

        complexEmotionScores = textsDf.apply(
            lambda row: self.calculateComplexEmotions(row, self.filteredEmotions), axis=1
        )

        complexEmotionDf = pd.DataFrame(complexEmotionScores.tolist())
        updatedTextsDf = pd.concat([textsDf, complexEmotionDf], axis=1)
        updatedTextsDf = self.addImpactColumns(updatedTextsDf, self.filteredEmotions)

        updatedTextsDf.to_csv(self.outputTextsPath, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Updated texts saved to {self.outputTextsPath}")

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

        df["adjusted_tone"] = df["tone"] + positiveSum - negativeSum

        if "likes" in df.columns and "comments" in df.columns:
            df["adjusted_impact"] = df["adjusted_tone"] * ((df["likes"] // 10) + df["comments"])
        else:
            print("Likes or comments column missing. Impact cannot be recalculated.")

        return df

class PoliticalScoreProcessor:
    def __init__(self, sentimentDataset, outputTextsPath):
        self.sentimentDataset = sentimentDataset
        self.outputTextsPath = outputTextsPath

    def processTexts(self):
        try:
            with open("socio-economic-instructions.txt", 'r') as file:
                instruction_text = file.read().replace('\n', ' ')
        except Exception as e:
            print(f"Error reading file: {e}")

        economicScores, socialScores, economicImpacts, socialImpacts = [], [], [], []

        for index, row in self.sentimentDataset.iterrows():
            try:
                text = row['text']
                likes = int(row.get('likes', 0))
                comments = int(row.get('comments', 0))

                formatted_input = f"{instruction_text} On the basis of ths evaluate the statement TEXT: {text} Just return the scores"

                response = chat(
                    model="llama3",
                    messages=[{"role": "user", "content": formatted_input}]
                )

                scores = response['message']['content'].split(", ")
                economicScore = float(scores[0])
                socialScore = float(scores[1])

                economicImpact = economicScore * ((likes // 10) + comments)
                socialImpact = socialScore * ((likes // 10) + comments)

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

        self.sentimentDataset['economic_score'] = economicScores
        self.sentimentDataset['social_score'] = socialScores
        self.sentimentDataset['economic_impact'] = economicImpacts
        self.sentimentDataset['social_impact'] = socialImpacts

        self.sentimentDataset.to_csv(self.outputTextsPath, index=False, quoting=csv.QUOTE_MINIMAL)
        print("Political scores processing completed.")

class ImpactProcessor:
    def __init__(self, inputFilePath, outputFilePath, modelContextFile, parameters, metricName):
        self.inputFilePath = inputFilePath
        self.outputFilePath = outputFilePath
        self.modelContextFile = modelContextFile
        self.parameters = parameters
        self.metricName = metricName

    def processParameters(self):
        with open(self.modelContextFile, "r") as file:
            context = file.read().replace('\n', ' ')

        try:
            existingData = pd.read_csv(self.inputFilePath, quotechar='"')
        except Exception as e:
            print(f"Error loading input file: {e}")
            return

        processedData = []

        for index, row in existingData.iterrows():
            userInput = row['text']
            likes = int(row.get("likes", 0))
            comments = int(row.get("comments", 0))

            if not userInput:
                print(f"Skipping empty row at index {index}.")
                continue

            try:
                prompt = f"{context} Text: {userInput}"
                response = chat(
                    model="llama3",
                    messages=[{"role": "user", "content": prompt}]
                )

                response_lines = response["message"]["content"].split("\n")
                paramScores = {param: 0 for param in self.parameters}
                for line in response_lines:
                    if ":" in line:
                        try:
                            param, value = line.split(":", 1)
                            param = param.strip()
                            value = float(value.strip())
                            param = re.sub(r'[^a-zA-Z\s]', '', param)
                            if param in paramScores or param.lower() in paramScores or param.upper() in paramScores or param.title() in paramScores:
                                paramScores[param] = value
                        except ValueError:
                            print(f"Skipping malformed line: {line}")

                for param in self.parameters:
                    impactCol = f"impact_{param}"
                    paramScores[impactCol] = paramScores[param] * ((likes // 10) + comments)

                paramScores.update({"text": userInput, "likes": likes, "comments": comments})
                processedData.append(paramScores)

            except Exception as e:
                print(f"Error processing row {index}: {e}")
                processedData.append({**{param: 0 for param in self.parameters},
                                      **{f"impact_{param}": 0 for param in self.parameters},
                                      "text": userInput, "likes": likes, "comments": comments})

        try:
            newData = pd.DataFrame(processedData)

            metricColumns = self.parameters
            impactColumns = [f"impact_{param}" for param in self.parameters]

            # Dynamically calculate metric indices
            newData[f"{self.metricName}_index"] = newData[metricColumns].sum(axis=1)
            newData[f"{self.metricName}_impact"] = newData[impactColumns].sum(axis=1)

            newData = newData.drop(columns=["likes", "comments"], errors="ignore")
            mergedData = pd.merge(existingData, newData, on="text", how="left")
            mergedData.to_csv(self.outputFilePath, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"Processed data saved to {self.outputFilePath}.")
        except Exception as e:
            print(f"Error saving output file: {e}")

class DatabaseHandler:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.conn = None

    def connect(self):
        try:
            self.conn = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
            )
            cursor = self.conn.cursor()
            cursor.execute(f"DROP DATABASE IF EXISTS {self.database};")
            cursor.execute(f"CREATE DATABASE {self.database};")
            self.conn.select_db(f"{self.database}")
            print(f"Connected to MySQL database: {self.database}")
        except pymysql.MySQLError as e:
            print(f"Error connecting to MySQL database: {e}")
            self.conn = None

    def create_tables(self, texts_columns, words_columns):
        if not self.conn:
            print("No connection. Call connect() first.")
            return

        cursor = self.conn.cursor()

        # Create texts table
        create_texts_table = texts_columns
        cursor.execute(create_texts_table)

        # Create words table
        create_words_table = words_columns

        cursor.execute(create_words_table)

        self.conn.commit()
        print("Tables created successfully!")

    def insert_data(self, csv_file, table_name):
        if not self.conn:
            print("No connection. Call connect() first.")
            return

        try:
            df = pd.read_csv(csv_file, quotechar='"')
            df.fillna(0, inplace=True)
            cursor = self.conn.cursor()

            for _, row in df.iterrows():
                placeholders = ', '.join(['%s'] * len(row))
                columns = ', '.join(row.index)
                query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                cursor.execute(query, tuple(row))

            self.conn.commit()
            print(f"Data from '{csv_file}' inserted into '{table_name}'.")
        except Exception as e:
            print(f"Error inserting data into {table_name}: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
            print("MySQL connection closed.")

class Visualizer:
    def __init__(self, textsDf, wordsDf):
        self.textsDf = textsDf
        self.wordsDf = wordsDf

    def groupAndSummarizeData(self, selection, sortBy):
        if selection in ['words', 'texts']:
            grouped = self.wordsDf.groupby('word', as_index=False).sum() if selection == 'words' else self.textsDf.groupby('text', as_index=False).sum()
            grouped = grouped[[selection[:-1], sortBy]].sort_values(by=sortBy, ascending=False)
        elif selection in ['agegroup', 'country', 'time', 'userid']:
            if selection in self.textsDf.columns:
                grouped = self.textsDf.groupby(selection, as_index=False).sum()
                grouped = grouped[[selection, sortBy]].sort_values(by=sortBy, ascending=False)
            else:
                print(f"Column '{selection}' not found in the dataset.")
                return pd.DataFrame()
        else:
            print(f"Selection '{selection}' not found in data columns.")
            return pd.DataFrame()

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
            print(f"Unknown threshold: {threshold}")
            return pd.DataFrame()

    def plotData(self, df, column, graphType, selection, sortBy, actualCount):
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
                plt.xlabel(selection.capitalize() if selection not in ['words', 'texts'] else selection.capitalize())

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
        if selection in ['texts', 'words']:
            filePath = 'texts.csv' if selection == 'texts' else 'words.csv'
        elif selection in ['agegroup', 'country', 'time', 'userid']:
            filePath = 'texts.csv'
        else:
            print(f"Dynamic columns not applicable for selection: {selection}")
            return []

        try:
            data = pd.read_csv(filePath, quotechar='"')
            return list(data.columns)
        except Exception as e:
            print(f"Error loading columns for {selection}: {e}")
            return []

    def updateSortByOptions(self, event, sortByMenu, selectionVar):
        selection = selectionVar.get()
        columns = self.loadDynamicColumns(selection)
        if columns:
            sortByMenu["values"] = columns
        else:
            sortByMenu["values"] = []

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
            values=['Bar', 'Line', 'Pie'],
            state="readonly"
        )
        graphTypeMenu.pack()

        tk.Button(root, text="Submit", command=lambda: self.onSubmit(selectionVar, sortByVar, thresholdVar, countVar, graphTypeVar)).pack()

        FileHandler.cleanTempFiles()

        root.mainloop()

def main():
    try:
        FileHandler.cleanData("sentiment_dataset.csv", "cleaned_dataset.csv")  # Clean data
    except FileNotFoundError:
        print("Error: 'sentiment_dataset.csv' not found. Ensure the file is in the correct directory.")
        return
    except pd.errors.EmptyDataError:
        print("Error: 'cleaned_dataset.csv' is empty. Provide a valid dataset.")
        return
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        return
    
    try:
        allStopwords = DataProcessor.getAllStopwords()
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return

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

    try:
        hf_dataset = Dataset.from_csv('cleaned_dataset.csv')
    except FileNotFoundError:
        print("Error: 'cleaned_dataset.csv' not found. Ensure the file is in the correct directory.")
        return
    except pd.errors.EmptyDataError:
        print("Error: 'cleaned_dataset.csv' is empty. Provide a valid dataset.")
        return
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        return

    apiModels = [
        "j-hartmann/emotion-English-distilroberta-base",
        "bhadresh-savani/bert-base-go-emotion",
        "monologg/bert-base-cased-goemotions-original",
    ]

    try:
        emotions = []
        dataProcessor = DataProcessor(hf_dataset, device, apiModels, emotions)
        fileHandler = FileHandler(hf_dataset, allStopwords, emotions, dataProcessor)
    except Exception as e:
        print(f"Error initializing DataProcessor or FileHandler: {e}")
        return

    # Check and create necessary files
    if not os.path.exists("texts.csv") or not os.path.exists("words.csv"):
        try:
            print("Required files missing. Generating all necessary files...")
            fileHandler.createTextsCsv(dataProcessor.calculateToneImpact, dataProcessor)

            complexEmotionProcessor = ComplexEmotionProcessor(
                tempTextsPath="temp1texts.csv",
                outputTextsPath="temp2texts.csv",
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
            complexEmotionProcessor.processTexts()

            sentimentDataset = pd.read_csv("temp2texts.csv", quotechar='"')
            politicalScoreProcessor = PoliticalScoreProcessor(
                sentimentDataset=sentimentDataset,
                outputTextsPath="temp3texts.csv"
            )
            politicalScoreProcessor.processTexts()

        except Exception as e:
            print(f"Error during file creation: {e}")
            return
    else:
        print("Required files found. Skipping file creation.")

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

    try:
        print("Applying tone adjustments for texts...")
        textsDf = pd.read_csv("temp3texts.csv", quotechar='"')

        toneAdjuster = ToneAdjuster(positiveEmotions, negativeEmotions)
        adjustedTextsDf = toneAdjuster.adjustToneAndImpact(textsDf)

        adjustedTextsDf.to_csv("temp3texts.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        print("Adjusted texts saved to 'temp3texts.csv'.")
    except Exception as e:
        print(f"Error applying tone adjustments for texts: {e}")
    
    try:
        print("Processing flagging parameters and impacts...")
        parameterProcessor = ImpactProcessor(
            inputFilePath="temp3texts.csv",
            outputFilePath="temp4texts.csv",
            modelContextFile="Flagging Prompts.txt",
            parameters=[
                'Ableist', 'Abusive', 'Ageist', 'Aggressive', 'Alienating', 'Antisemitic', 'Belittling', 
                'Belligerent', 'Bullying', 'Caustic', 'Classist', 'Condescending', 'Containing_slurs', 
                'Contemptful', 'Defamatory', 'Degrading', 'Demeaning', 'Demoralizing', 'Derisive', 
                'Derogatory', 'Despising', 'Destructive', 'Discriminatory', 'Disparaging', 'Disturbing', 
                'Enraging', 'Ethnocentric', 'Exclusionary', 'Harassing', 'Harmful', 'Hatespeech', 
                'Homophobic', 'Hostile', 'Hurtful', 'Incendiary', 'Inflammatory', 'Insulting', 
                'Intimidating', 'Intolerable', 'Intolerant', 'Islamophobic', 'Malicious', 'Marginalizing', 
                'Misogynistic', 'Mocking', 'Dehumanizing', 'Objectifying', 'Segregating', 'Nasty', 'Obscene', 
                'Offensive', 'Oppressive', 'Overbearing', 'Pejorative', 'Prejudiced', 'Profane', 'Racist',
                'Sarcastic', 'Scornful', 'Sexist', 'Slanderous', 'Spiteful', 'Threatening', 'Toxic',
                'Transphobic', 'Traumatizing', 'Vindictive', 'Vulgar', 'Xenophobic', 'Manipulative',
                'Exploitative', 'Gaslighting', 'Patronizing', 'Overcritical', 'Fearmongering', 'Shaming',
                'Pathologizing', "Trolling", "Cyberbullying", "Dogpiling", "Sealioning", "Doxxing",
                "Brigading", "Spamming", "Clickbaiting", "Misinformation", "Disinformation",
                "Profanity", "Alarmist", "Hysterical", "Vindictive", "Shocking",
                "Overgeneralizing", "Narcissistic"
            ],
            metricName="toxicity"
        )
        parameterProcessor.processParameters()
        print("Flagging parameters processed and saved to 'temp4texts.csv'.")
    except Exception as e:
        print(f"Error processing flagging parameters: {e}")

    try:
        print("Processing psychological parameters and impacts...")
        mentalHealthProcessor = ImpactProcessor(
            inputFilePath="temp4texts.csv",
            outputFilePath="temp5texts.csv",
            modelContextFile="mental health prompts.txt",
            parameters=[
                'Abandoned', 'Afraid', 'Alienated', 'Alone', 'Anguished', 'Annoyed', 'Anxious', 
                'Apathetic', 'Apologetic', 'Apprehensive', 'Ashamed', 'Awkward', 'Bitter', 'Blameworthy', 
                'Burned_Out', 'Concerned', 'Dejected', 'Demoralized', 'Despondent', 'Detached', 
                'Disconnected', 'Disheartened', 'Dissociative', 'Distraught', 'Doubtful', 'Drained', 
                'Dread', 'Edgy', 'Embarrassed', 'Emptiness', 'Enraged', 'Excluded', 'Exposed', 'Fatigued', 
                'Fearful', 'Forsaken', 'Frustrated', 'Furious', 'Gloomy', 'Heartbroken', 'Helpless', 
                'Hesitant', 'Hopeless', 'Hypervigilant', 'Indifferent', 'Insecure', 'Irritable', 
                'Isolated', 'Judged', 'Lethargic', 'Longing', 'Lost', 'Melancholy', 'Miserable', 'Misunderstood', 
                'Mourning', 'Nervous', 'Numb', 'Overwhelmed', 'Panicked', 'Paranoid', 'Pressured', 
                'Regretful', 'Remorseful', 'Resentful', 'Restless', 'Sad', 'Sarcasm', 'Scared', 'Secluded', 
                'Self_Critical', 'Shaky', 'Shy', 'Sorrowful', 'Startled', 'Stressed', 'Tense', 'Terrified', 
                'Tired', 'Triggered', 'Troubled', 'Uncertain', 'Uneasy', 'Unloved', 'Unmotivated', 
                'Unworthy', 'Vulnerable', 'Withdrawn', 'Worried', 'Worthless', 'Suicidal', 'Self_harm'
            ],
            metricName="distress"
        )
        mentalHealthProcessor.processParameters()
        print("Psychological parameters processed and saved to 'temp5texts.csv'.")
    except Exception as e:
        print(f"Error processing psychological parameters: {e}")

    try:
        print("Processing healing emotions...")
        healingEmotionProcessor = ImpactProcessor(
            inputFilePath="temp5texts.csv",
            outputFilePath="texts.csv",
            modelContextFile="healing prompts.txt",
            parameters=[
                "Calming", "Relaxed", "Safe", "Motivated", 
                "Empowered", "Peaceful", "Confident", "Trusting", 
                "Comforted", "Reassured", "Inspired", "Nurtured",
                "Understanding", "Serene", "Fulfilled", "Energized",
                "Harmonious", "Appreciative", "Openness", "Sociable",
                "Gracious", "Altruistic", "Reflective","Enthusiastic","Adventurous"
                ],
            metricName="healing"
        )
        healingEmotionProcessor.processParameters()
        print("Healing emotions processed and saved to 'texts.csv'.")
    except Exception as e:
        print(f"Error processing healing emotions: {e}")


    fileHandler.createWordsCsv("texts.csv", "words.csv")

    try:
        # Initialize MySQL DatabaseHandler
        db_handler = DatabaseHandler(
            host='localhost',  # Replace with your MySQL host, e.g., 'localhost'
            user='root',  # Replace with your MySQL username
            password='1Anurag2Basistha',  # Replace with your MySQL password
            database='SentimentAnalysis'  # Replace with your MySQL database name
        )

        # Connect to the MySQL database
        db_handler.connect()
        try:
            with open("create_texts_database.txt", 'r') as file:
                texts_columns = file.read().replace('\n', ' ')
        except Exception as e:
            print(f"Error reading file: {e}")

        try:
            with open("create_words_database.txt", 'r') as file:
                words_columns = file.read().replace('\n', ' ')
        except Exception as e:
            print(f"Error reading file: {e}")
 
         # Create tables
        db_handler.create_tables(texts_columns, words_columns)

        # Insert data into tables from CSV files
        db_handler.insert_data('texts.csv', 'texts')
        db_handler.insert_data('words.csv', 'words')

        db_handler.close()
    except Exception as e:
        print(f"Error in database handling: {e}")

    try:
        textsDf = pd.read_csv("texts.csv", quotechar='"')
        wordsDf = pd.read_csv("words.csv", quotechar='"')
        visualizer = Visualizer(textsDf, wordsDf)
    except Exception as e:
        print(f"Error loading data for Visualizer: {e}")
        return

    try:
        guiHandler = GUIHandler(visualizer)
        guiHandler.launchGUI()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()