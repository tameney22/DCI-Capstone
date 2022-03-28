# Text Classification of 19th Century Newspapers by Publication Press

**Project Site:** [https://iwrite.wludci.info/capstone/](https://iwrite.wludci.info/capstone/)

This project uses supervised classification methods in the field of digital humanities. I am using a corpus of 19th Century US Newspapers provided by the Gale Database to train a model that takes in text and predicts what publication it belongs to. The research question I am attempting to answer is whether or not it is possible to do such classification and create a model that performs at a high accuracy.

## Libraries Used

This project uses the following python libraries:

- pandas
- numpy
- nltk
- scipy

## Getting Started

- Assuming the `19thCenturyUSNewspapers.xlsx` file is in this directory, the first step is to run the `prepArticles.py` script to generate `articles.csv`.
- The next step is to run the `preprocess.py` script to extract and preprocess the text from the xml files. This will output `preprocessed.csv`, which is a large csv with the preprocessed text and labels.
- Lastly, `model.py` is ran to split the preprocessed text into a 70-30 train and test split then train the model. Using the trained model, it predicts the test data in order to determine the model's accuracy.

## License

MIT
