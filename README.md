# Text-and-Emotion-Analysis-Tool-with-Visualization
This project is an advanced text analysis tool that combines sentiment evaluation, emotion classification, and interactive data visualization to provide comprehensive insights into textual datasets. Built with Python, it leverages state-of-the-art natural language processing techniques to analyze text at both document and word levels.

# Text and Emotion Analysis Tool with Interactive Visualization

## Overview
This project is a powerful Python-based application designed for sentiment and emotion analysis of textual data. It integrates advanced natural language processing (NLP) techniques to generate insights from text at both document and word levels. Additionally, it provides an interactive visualization interface, making it easy to interpret and analyze results.

---

## Features

### 1. **Sentiment Analysis**
- Leverages the `nltk` SentimentIntensityAnalyzer to calculate the sentiment polarity of texts.
- Computes a compound score ranging from -1 (negative sentiment) to +1 (positive sentiment).
- Combines sentiment scores with engagement metrics (likes and comments) to generate an overall impact score.

### 2. **Emotion Classification**
- Utilizes the Hugging Face transformer model `j-hartmann/emotion-English-distilroberta-base` for emotion detection.
- Identifies and quantifies five key emotions: joy, sadness, anger, surprise, and fear.

### 3. **Word-Level Analysis**
- Tracks word frequency, tone, and emotion impact.
- Provides insights into how individual words contribute to the sentiment and emotion of the text.
- Outputs structured data for deeper analysis.

### 4. **Interactive Visualization**
- Built with `Tkinter` for a user-friendly graphical interface.
- Allows users to visualize data using bar, line, and pie charts.
- Supports filtering and grouping options for personalized analysis.

### 5. **Customizable Filters**
- Group data by categories such as words, texts, or user demographics.
- Sort results by metrics like impact, tone, or emotions.
- Filter results based on thresholds (highest, lowest, or extremes).

### 6. **Data Export**
- Saves processed insights into structured CSV files for both text and word levels.
- Outputs include detailed metrics like sentiment scores, emotion scores, and impact values.

---

## Installation

### Prerequisites
- Python 3.7+
- Required Python libraries:
  ```bash
  pip install pandas nltk transformers matplotlib
  ```
- Download the sentiment dataset as `sentiment_dataset.csv` and place it in the project directory.

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### Run the Application
```bash
python app.py
```

---

## Usage

### 1. **Dataset Requirements**
The input dataset (`sentiment_dataset.csv`) should contain the following columns:
- `text`: Text data to be analyzed.
- `likes`: Number of likes associated with the text.
- `comments`: Number of comments associated with the text.

### 2. **Analyzing Text**
- Run the application and let the program process the dataset.
- The processed data will generate two output files:
  - `texts.csv`: Analysis at the text level.
  - `words.csv`: Analysis at the word level.

### 3. **Visualizing Data**
- Use the Tkinter-based GUI to:
  - Select data type (e.g., texts, words).
  - Choose sorting criteria (e.g., impact, tone, emotion scores).
  - Apply thresholds to focus on specific data subsets.
  - Visualize insights using bar, line, or pie charts.

---

## File Descriptions

### **Core Scripts**
- **`app.py`**: Main script containing the entire workflow, including sentiment analysis, emotion classification, data export, and GUI implementation.

### **Generated Files**
- **`texts.csv`**: CSV file with metrics for each text, including sentiment scores, emotion scores, and impact values.
- **`words.csv`**: CSV file with metrics for each word, including frequency, tone, emotion scores, and impact values.

---

## Visualization Examples
### Bar Chart Example
Shows the top 10 words with the highest impact:
![Bar Chart](images/bar_chart_example.png)

### Pie Chart Example
Displays the proportion of emotions across the dataset:
![Pie Chart](images/pie_chart_example.png)

---

## Dependencies
- `pandas`: For data manipulation and processing.
- `nltk`: For sentiment analysis.
- `transformers`: For Hugging Face emotion classification.
- `matplotlib`: For data visualization.
- `Tkinter`: For creating the interactive GUI.

---

## Contributing
Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Create a pull request.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Contact
For questions or feedback, contact:
- **Author:** Anurag Basistha
- **Email:** [your_email@example.com]
- **GitHub:** [GitHub Profile](https://github.com/your_username)

---

Happy Text Analysis! 🎉

