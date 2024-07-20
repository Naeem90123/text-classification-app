# Text Classification App

This is a simple text classification application built using Python and Tkinter. The application uses a pre-trained machine learning model to classify input text into different categories. It provides a user interface where users can enter text and get the classification result.

## Installation

To run the application, you need to have the following dependencies installed:

- Python 3.9.x
- customtkinter
- nltk
- joblib

You can install the dependencies by running the following command:
```
pip install customtkinter nltk joblib
```


## Usage

1. Run the application:
```
python text_classifier.py
```



2. The application window will open. Enter the text you want to classify in the input area and click the "Classify" button.

3. The classification result will be displayed in the result area.

## Customization

You can customize the application by modifying the following parts of the code:

- Preprocessing: The `preprocess()` function in the code performs text preprocessing tasks such as converting the text to lowercase, removing HTML tags, URLs, punctuations, and stop words. You can modify this function to add or remove preprocessing steps according to your needs.

- Model and Data: The code uses a pre-trained machine learning model for classification. You can replace the `vectorizer.pkl` and `classifier_model.pkl` files with your own trained models. Make sure the models are compatible with the code.

- Target Mapping: The `targets` list in the code defines the mapping between class indices and class labels. Modify this list to match your classification problem.

- User Interface: The code uses customtkinter library to create the user interface. You can modify the appearance, layout, and functionality of the UI by modifying the `TextClassifierApp` class.

## License

This project is licensed under the [MIT License](LICENSE).
Make sure to replace <repository-url> and <project-directory> with the actual URL of the repository and the directory where the code is located, respectively. Also, make sure to include the LICENSE file in the project directory and specify the appropriate license in the License section of the readme.md file.