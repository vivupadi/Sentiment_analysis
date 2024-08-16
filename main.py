from train import *
from predict import *


import sys
import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout,QHBoxLayout, QWidget, QLineEdit,QPushButton, QListWidget, QFileDialog, QComboBox, QLabel
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from PyQt5.QtCore import Qt

from io import BytesIO

from wordcloud import WordCloud

import matplotlib.pyplot as plt

class Sentiment_Analysis(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentiment Analysis")
        self.setGeometry(200, 200, 800, 800)
        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout1 = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout3 = QHBoxLayout()
        layout4 = QHBoxLayout()

        self.load_button = QPushButton('Select File',self)
        self.load_button.clicked.connect(self.load_file)
        self.load_button.setFixedSize(100, 30)
        layout1.addWidget(self.load_button, alignment=Qt.AlignCenter)

        self.Load_status = QLabel()
        self.Load_status.setText('Select File')
        self.Load_status.setAlignment(Qt.AlignCenter)
        self.Load_status.setFixedSize(100,20)
        layout1.addWidget(self.Load_status, alignment=Qt.AlignCenter)

        #display data

        self.preprocess_button = QPushButton('Preprocess data',self)
        self.preprocess_button.clicked.connect(self.prepare_data)
        self.preprocess_button.setFixedSize(200,30)
        layout1.addWidget(self.preprocess_button, alignment=Qt.AlignCenter)

        self.preprocess_status = QLabel()
        self.preprocess_status.setAlignment(Qt.AlignCenter)
        self.preprocess_status.setFixedSize(200,20)
        layout1.addWidget(self.preprocess_status, alignment=Qt.AlignCenter)

        #Display Sentiment selection option
        self.sentiment_text = QLabel()
        #self.sentiment_text.setAlignment(Qt.AlignCenter)
        self.sentiment_text.setText('Select the Sentiment to show its Word Cloud:')
        layout2.addWidget(self.sentiment_text)

        #select sentiment for the wordmap
        self.Senti_wordmap = QComboBox(self)
        self.Senti_wordmap.addItems(["Positive", "Negative", "Neutral", "Irrelevant"])
        #self.Senti_wordmap.setFixedSize(200,30)
        layout2.addWidget(self.Senti_wordmap)
        
        layout1.addLayout(layout2)
        
        #display Wordmap
        self.wordmap_button = QPushButton('Show Word Map', self)
        self.wordmap_button.clicked.connect(self.show_wordmap)
        self.wordmap_button.setFixedSize(150,50)
        layout1.addWidget(self.wordmap_button, alignment=Qt.AlignCenter)

        #Display WOrdCloud
        self.wordmap_label = QLabel(self)
        self.wordmap_label.setAlignment(Qt.AlignCenter)
        layout1.addWidget(self.wordmap_label)

        #Display vector selection option
        self.vect_text = QLabel()
        self.vect_text.setText('Select the type of vectorizer:')
        layout3.addWidget(self.vect_text)

        #Select the vectorizer
        self.select_vectorizer = QComboBox(self)
        self.select_vectorizer.addItems(["Bow", "TF_IDF"])
        self.select_vectorizer.setFixedSize(200,30)
        layout3.addWidget(self.select_vectorizer)

        layout1.addLayout(layout3)

        #Button to vectorize Text data
        self.vector_button = QPushButton('Vectorize Data')
        self.vector_button.clicked.connect(self.vectorize)
        self.vector_button.setFixedSize(200,30)
        layout1.addWidget(self.vector_button, alignment=Qt.AlignCenter)

        #Display vector selection option
        self.model_text = QLabel()
        self.model_text.setText('Select model to be trained:')
        layout4.addWidget(self.model_text)

        #Select model
        self.select_model = QComboBox(self)
        self.select_model.addItems(["LogisticRegression", "XGBoost", "Random Forest"])
        self.select_model.setFixedSize(200,30)
        layout4.addWidget(self.select_model)

        layout1.addLayout(layout4)

        #Button to Train model
        self.train_button = QPushButton('Train model')
        self.train_button.clicked.connect(self.training)
        self.train_button.setFixedSize(200, 30)
        layout1.addWidget(self.train_button, alignment= Qt.AlignCenter)

        #Dipslay accuracy
        self.accuracy = QLabel(self)
        self.accuracy.setAlignment(Qt.AlignCenter)
        layout1.addWidget(self.accuracy)

        #Field to enter a User defined New tweet
        self.prompt = QLineEdit(self)
        self.prompt.setPlaceholderText("Enter a Tweet")
        layout1.addWidget(self.prompt)

        #predict button
        self.predict_button = QPushButton('Predict the text Sentiment')
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setFixedSize(200, 30)
        layout1.addWidget(self.predict_button, alignment=Qt.AlignCenter)


        #Display the sentiment of the entered Text 
        self.display_sentiment = QLabel(self)
        self.display_sentiment.setAlignment(Qt.AlignCenter)
        layout1.addWidget(self.display_sentiment)

        central_widget.setLayout(layout1)
        

    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;Text Files (*.txt)", options=options)
        if file_path:
            self.data = load_data(file_path)
            self.Load_status.setText('File Selected')
        
    def prepare_data(self):
        self.preprocess_status.setText('Preprocessing in Progress....')
        self.df = preprocess(self.data)
        self.preprocess_status.setText('Preprocessing Done')
        self.df['Text_copy'] = self.df['Text']
        self.df['Sentiment_copy'] =self.df['Sentiment']
        print(self.df)

    #Display wordcloud
    def show_wordmap(self):
        senti = self.Senti_wordmap.currentText()
        df_sentiment = self.get_text_data(senti)

        text_data = ' '.join(df_sentiment['Text_copy'])
        font_path = r'C:/Windows/Fonts/arial.ttf'
        
        word_cloud = WordCloud(width = 600, height = 400,background_color = 'black', min_font_size=6, font_path= font_path).generate(text_data)
        plt.figure(figsize = (8,5))
        plt.imshow(word_cloud)
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        self.wordmap_label.setPixmap(pixmap)
    
    #Either one hot encoding or below function to encode labels
    def get_text_data(self, selected_senti):
        df = self.df
        if selected_senti == 'Positive':
            return df[df['Sentiment_copy'] == 'Positive']
        elif selected_senti == 'Negative':
            return df[df['Sentiment_copy'] == 'Negative']
        elif selected_senti == 'Neutral':
            return df[df['Sentiment_copy'] == 'Neutral']
        elif selected_senti == 'Irrelevant':
            return df[df['Sentiment_copy'] == 'Irrelevant']
        else:
            print('Select the type of sentiment to plot Wordcloud')
    

    def vectorize(self):
        self.selected_vect =self.select_vectorizer.currentText()
        self.X, self.y, self.vectorizer = vectorize_data(self.df, self.selected_vect)
        print('Done Vectorization')

    def training(self):
        self.selected_model = self.select_model.currentText()
        self.model, self.acc = train_model(self.selected_model, self.X, self.y)
        self.accuracy.clear()
        self.accuracy.setText(f'Accuracy_Score: {self.acc}')  

    def predict(self):
        text = self.prompt.text()
        self.display_sentiment.clear()
        y_sentiment = predict_sentiment(self.model, self.vectorizer, text) 
        self.display_sentiment.setText(f'Predicted Sentiment is: {y_sentiment}')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Sentiment_Analysis()
    window.show()
    sys.exit(app.exec_())
