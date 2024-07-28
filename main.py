from train import *
from predict import *


import sys
import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit,QPushButton, QListWidget, QFileDialog, QComboBox, QLabel
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from PyQt5.QtCore import Qt

from io import BytesIO

from wordcloud import WordCloud

from sklearn.linear_model import LogisticRegression


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

        layout = QVBoxLayout()

        self.load_button = QPushButton('Select File',self)
        self.load_button.clicked.connect(self.load_file)
        #self.load_button.setAlignment(Qt.AlignCenter)
        self.load_button.setFixedSize(300, 30)
        layout.addWidget(self.load_button, alignment=Qt.AlignCenter)

        self.Load_status = QLabel()
        self.Load_status.setText('Select File')
        self.Load_status.setAlignment(Qt.AlignCenter)
        self.Load_status.setFixedSize(300,30)
        layout.addWidget(self.Load_status, alignment=Qt.AlignCenter)

        #display data

        self.preprocess_button = QPushButton('Preprocess data',self)
        self.preprocess_button.clicked.connect(self.prepare_data)
        #self.preprocess_button.setAlignment(Qt.AlignCenter)
        self.preprocess_button.setFixedSize(300,30)
        layout.addWidget(self.preprocess_button, alignment=Qt.AlignCenter)

        self.preprocess_status = QLabel()
        #self.preprocess_status.setText('Preprocessing in Progress...')
        self.preprocess_status.setAlignment(Qt.AlignCenter)
        self.preprocess_status.setFixedSize(300,30)
        layout.addWidget(self.preprocess_status, alignment=Qt.AlignCenter)
        
        #display Wordmap
        self.wordmap_button = QPushButton('Show Word Map', self)
        self.wordmap_button.clicked.connect(self.show_wordmap)
        self.wordmap_button.setFixedSize(300,30)
        layout.addWidget(self.wordmap_button, alignment=Qt.AlignCenter)

        #select sentiment for the wordmap
        self.Senti_wordmap = QComboBox(self)
        self.Senti_wordmap.addItems(["Positive", "Negative", "Neutral", "Irrelevant"])
        self.Senti_wordmap.setFixedSize(200,30)
        layout.addWidget(self.Senti_wordmap)

        #Display WOrdCloud
        self.wordmap_label = QLabel(self)
        self.wordmap_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.wordmap_label)


        #Button to vectorize Text data
        self.vector_button = QPushButton('Vectorize Data')
        self.vector_button.clicked.connect(self.vectorize)
        self.vector_button.setFixedSize(200,30)
        layout.addWidget(self.vector_button)

        #Button to Train model
        self.train_button = QPushButton('Train model')
        self.train_button.clicked.connect(self.training)
        self.train_button.setFixedSize(200, 30)
        layout.addWidget(self.train_button)

        #Dipslay accuracy
        self.accuracy = QLabel(self)
        self.accuracy.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.accuracy)

        #FIeld to enter a User defined New tweet
        self.prompt = QLineEdit(self)
        self.prompt.setPlaceholderText("Enter a Tweet")
        layout.addWidget(self.prompt)

        #predict button
        self.predict_button = QPushButton('Predict the text Sentiment')
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setFixedSize(200, 30)
        layout.addWidget(self.predict_button)


        #Display the sentiment of the entered Text 
        self.display_sentiment = QLabel(self)
        self.display_sentiment.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.display_sentiment)

        central_widget.setLayout(layout)
        

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
        self.X, self.y, self.vectorizer = vectorize_data(self.df)
        print('Done Vectorization')

    def training(self):
        self.model = LogisticRegression(C=1, solver = 'liblinear',max_iter=150)
        self.model, self.acc = train_model(self.model, self.X, self.y)
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
