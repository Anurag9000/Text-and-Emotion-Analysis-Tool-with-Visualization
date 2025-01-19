import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

def lemmatize_word(word):
    doc = nlp(word)  # Process the word using spaCy
    return doc[0].lemma_  # Return the lemma of the first (and only) token

while True:
    word = input("Enter a word: ")
    print("Lemmatized Word:", lemmatize_word(word))
