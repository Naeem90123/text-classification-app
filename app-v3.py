import customtkinter as ctk
import tkinter as tk
import nltk
import joblib

# for text preprocessing
from nltk.stem import SnowballStemmer  # for stemming
from nltk import word_tokenize  # for tokenization
from nltk.corpus import stopwords  # for getting stop words
import re  # for regular expression

# download the stopwords
nltk.download('stopwords')

# import the models
tfidf_vectorizer = joblib.load("models/vectorizer.pkl")
classifer_model = joblib.load("models/classifier_model.pkl")

# set the target mapping:
targets = ['alt.atheism',
           'comp.graphics',
           'comp.os.ms-windows.misc',
           'comp.sys.ibm.pc.hardware',
           'comp.sys.mac.hardware',
           'comp.windows.x',
           'misc.forsale',
           'rec.autos',
           'rec.motorcycles',
           'rec.sport.baseball',
           'rec.sport.hockey',
           'sci.crypt',
           'sci.electronics',
           'sci.med',
           'sci.space',
           'soc.religion.christian',
           'talk.politics.guns',
           'talk.politics.mideast',
           'talk.politics.misc',
           'talk.religion.misc']


def preprocess(text):
    # case folding (converting the string to lower case)
    text = text.lower()

    # removing html tags
    obj = re.compile(r"<.*?>")
    text = obj.sub(r" ", text)

    # removing url
    obj = re.compile(r"https://\S+|http://\S+")
    text = obj.sub(r" ", text)

    # removing punctuations
    obj = re.compile(r"[^\w\s]")
    text = obj.sub(r" ", text)

    # removing multiple spaces
    obj = re.compile(r"\s{2,}")
    text = obj.sub(r" ", text)

    # loading english stop words
    en_stopwords = stopwords.words('english')

    # removing stop words and stemming
    stemmer = SnowballStemmer("english")
    words = []

    text = [stemmer.stem(word) for word in text.split() if word not in en_stopwords]
    return " ".join(text)


def vectorize(preprocessed_text):
    x = tfidf_vectorizer.transform([preprocessed_text])
    return x


def predict_class(text):
    preprocessed_text = preprocess(text)
    x = vectorize(preprocessed_text)
    class_int = classifer_model.predict(x)[0]
    return targets[class_int]


class TextClassifierApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Text Classifier")
        self.root.geometry("800x550")

        ctk.set_appearance_mode("dark")  # Default to dark theme
        ctk.set_default_color_theme("blue")

        self.create_widgets()

    def create_widgets(self):
        # Title frame
        title_frame = ctk.CTkFrame(self.root, corner_radius=0)
        title_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        title_label = ctk.CTkLabel(title_frame, text="Text Classification App", font=("Helvetica", 24))
        title_label.pack(pady=20)

        # Theme switch
        self.theme_switch = ctk.CTkSwitch(title_frame, text="Dark Mode", command=self.toggle_theme)
        self.theme_switch.pack(pady=10)
        self.theme_switch.select()  # Start with dark mode on

        # Input frame (left side)
        input_frame = ctk.CTkFrame(self.root)
        input_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        input_label = ctk.CTkLabel(input_frame, text="Enter text to classify:", font=("Helvetica", 16))
        input_label.pack(pady=10)

        self.text_input = ctk.CTkTextbox(input_frame, height=200, width=300, font=("Helvetica", 12))
        self.text_input.pack(pady=10, padx=10, fill="both", expand=True)

        classify_button = ctk.CTkButton(input_frame, text="Classify", command=self.classify)
        classify_button.pack(pady=10)

        # Result frame (right side)
        result_frame = ctk.CTkFrame(self.root)
        result_frame.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

        result_label = ctk.CTkLabel(result_frame, text="Classification Result:", font=("Helvetica", 16))
        result_label.pack(pady=10)

        self.result_box = ctk.CTkTextbox(result_frame, height=200, width=300, font=("Helvetica", 12))
        self.result_box.pack(pady=10, padx=10, fill="both", expand=True)
        self.result_box.configure(state="disabled")

        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

    def classify(self):
        text = self.text_input.get("0.0", "end").strip()
        if text:
            category = predict_class(text)
            result = f"Classification: {category}\n"
            self.result_box.configure(state="normal")
            self.result_box.delete("0.0", "end")
            self.result_box.insert("0.0", result)
            self.result_box.configure(state="disabled")
        else:
            self.result_box.configure(state="normal")
            self.result_box.delete("0.0", "end")
            self.result_box.insert("0.0", "Please enter some text to classify.")
            self.result_box.configure(state="disabled")

    def toggle_theme(self):
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TextClassifierApp()
    app.run()
