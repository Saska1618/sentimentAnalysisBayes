# sentimentAnalysisBayes

In this repo I created a simple sentiment analysis tool based on a bag of words and Naive Bayes approach.

Basically, i assigned a positive and a negative score to each word (or token if you wish) and I will calculated conditional probabilities.

The main idea is that i want to calculate P(positive|"Specific text with positive or negative sentiment") values

I am using the imdb dataset from Huggingface: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)

Note1:
    this approach on its own was not too successfull, i achiaved 64% accuracy on 2 classes (positive or negative)
    To further enhance this model i am planning on introducing connection between words and impact, not just polarity