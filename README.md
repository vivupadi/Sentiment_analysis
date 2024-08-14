1. The project uses a GUI for better interactive and ease purpose.
2. The Project delves into NLP concepts and the pre-processing involved before training a Logistic Regression model.
3. Dataset used- Kaggle Twitter Sentiment Analysis which consists of 4 columns -  Tweet ID, Entity, Text and Sentiment
4. Visualization includes using the Word cloud mapping for the user-selected sentiment.
5. Preprocessing includes- Retaining only alphanumeric letters, Switching everything to lower-case, Removing stopwords(e.g 'The', 'is'), Lemmatizing (changing words to root form)
6. The Text to Vector conversion is done using Bag of Words Or TF_IDF(User Selected)
7. Main.py - Includes GUI parts
8. Train.py - Includes loading, preprocessing, vectorization, model training
9. Predict.py - Prediction on the user input
10. vectorizer.py - Includes the frequency based vector embeddings(BOW , TF_IDF)
11. model.py - Different models initialized
Next steps:
1. To see the difference in metrics if Word2Vec is used. Read that Word2Vec is better in highlighting the sentiments involved in the text.
2. Hyperparameter tuning
3. Use Prediction based Vector embeddings (Word2Vec, Glove,etc). Might need to change few concepts


--Screenshot of the GUI
![image](https://github.com/user-attachments/assets/cabdc3a0-1abc-4750-be06-49a7e56c56fa)

