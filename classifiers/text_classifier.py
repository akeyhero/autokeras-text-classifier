"""
autokeras/text/text_supervised.py の TextClassifier を、 preprocess を何も行わずに実行するよう改変
"""

from abc import ABC
import pandas as pd
from autokeras.nn.loss_function import classification_loss, regression_loss
from autokeras.nn.metric import Accuracy, MSE
from autokeras.preprocessor import OneHotEncoder, TextDataTransformer
from autokeras.supervised import DeepSupervised
from autokeras.text.text_preprocessor import tokenlize_text, processing
from autokeras.utils import temp_path_generator, ensure_dir
from autokeras.constant import Constant

class TextPreprocessor:
    def __init__(self,
                 store_path=Constant.STORE_PATH,
                 max_seq_length=Constant.MAX_SEQUENCE_LENGTH,
                 max_num_words=Constant.MAX_NB_WORDS):
        self.store_path     = store_path
        self.max_seq_length = max_seq_length
        self.max_num_words  = max_num_words

    def clean_str(self, string):
        return string.strip().lower()

    def preprocess(self, x_train):
        """
        It takes an raw string, clean it and processing it into tokenlized numpy array.
        """
        if self.store_path == '':
            temp_path = temp_path_generator()
            path = temp_path + '_store'
        else:
            path = self.store_path

        ensure_dir(path)

        x_train = [self.clean_str(x) for x in x_train]
        x_train, word_index = tokenlize_text(max_seq_length=self.max_seq_length,
                                             max_num_words=self.max_num_words,
                                             x_train=x_train)

        print("generating preprocessing model...")
        x_train = processing(path=path, word_index=word_index, input_length=self.max_seq_length, x_train=x_train)
        return x_train

class TextSupervised(DeepSupervised, ABC):
    """TextClassifier class.

    Attributes:
        cnn: CNN module from net_module.py.
        path: A path to the directory to save the classifier as well as intermediate results.
        y_encoder: Label encoder, used in transform_y or inverse_transform_y for encode the label. For example,
                    if one hot encoder needed, y_encoder can be OneHotEncoder.
        data_transformer: A transformer class to process the data. See example as ImageDataTransformer.
        verbose: A boolean value indicating the verbosity mode which determines whether the search process
                will be printed to stdout.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        The classifier will be loaded from the files in 'path' if parameter 'resume' is True.
        Otherwise it would create a new one.
        Args:
            verbose: A boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
            resume: A boolean. If True, the classifier will continue to previous work saved in path.
                Otherwise, the classifier will start a new search.
            searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
        """
        super().__init__(**kwargs)
        self.preprocessor = TextPreprocessor()

    def fit(self, x, y, x_test=None, y_test=None, time_limit=None):
        """Find the best neural architecture and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset is in numpy.ndarray format.
        So they training data should be passed through `x_train`, `y_train`.

        Args:
            x: A numpy.ndarray instance containing the training data.
            y: A numpy.ndarray instance containing the label of the training data.
            y_test: A numpy.ndarray instance containing the testing data.
            x_test: A numpy.ndarray instance containing the label of the testing data.
            time_limit: The time limit for the search in seconds.
        """
        x = self.preprocessor.preprocess(x)
        super().fit(x, y, x_test, y_test, time_limit)

    def init_transformer(self, x):
        # Wrap the data into DataLoaders
        if self.data_transformer is None:
            self.data_transformer = TextDataTransformer()

    def preprocess(self, x):
        return self.preprocessor.preprocess(x)

class TextClassifier(TextSupervised):
    @property
    def metric(self):
        return Accuracy

    @property
    def loss(self):
        return classification_loss

    def transform_y(self, y_train):
        # Transform y_train.
        if self.y_encoder is None:
            self.y_encoder = OneHotEncoder()
            self.y_encoder.fit(y_train)
        y_train = self.y_encoder.transform(y_train)
        return y_train

    def inverse_transform_y(self, output):
        return self.y_encoder.inverse_transform(output)

    def get_n_output_node(self):
        return self.y_encoder.n_classes
