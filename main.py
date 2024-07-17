

from train import *

import sys
import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit,QPushButton, QListWidget, QFileDialog
from PyQt5.QtGui import QPalette, QBrush, QPixmap

class Sentiment_Analysis(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentiment Analysis")
        self.setGeometry(200, 200,500, 500)
        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.load_button = QPushButton('Select File',self)
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)

        #display data

        self.preprocess_button = QPushButton('Preprocess data',self)
        self.preprocess_button.clicked.connect(self.prepare_data)
        layout.addWidget(self.preprocess_button)

        central_widget.setLayout(layout)
        

    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;Text Files (*.txt)", options=options)
        if file_path:
            self.data = load_data(file_path)
            #self.df = preprocess(self.data)
            #print(self.df)
            #return df
        
    def prepare_data(self):
        self.df = preprocess(self.data)
        print(self.df)

    def show_wordmap(self):
        #show wordmap


    #def load_model():

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Sentiment_Analysis()
    window.show()
    sys.exit(app.exec_())
