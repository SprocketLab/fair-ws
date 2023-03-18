from abc import ABC, abstractmethod
import numpy as np
import tqdm
 


class AbstractLF(ABC):
    """
    LF abstraction
    """
    @abstractmethod
    def apply(self, feature_array):
        pass

class FeatureMeanLF(AbstractLF):
    """
    Binary labeling function that takes feature array as the input
    and gives weak label matrix L, by comparing each instance feature
    with the mean of each feature.
    """
    def __init__(self):
        pass

    def apply(self, feature_array, feature_indices=None):
        if feature_indices is None:
            feature_indices = list(range(feature_array.shape[1]))
        
        L = np.zeros((feature_array.shape[0], len(feature_indices)))
        
        for i, feature_idx in enumerate(feature_indices):
            L[:, i] = (feature_array[:, feature_idx] >= feature_array[:, feature_idx].mean()).astype(int)
        
        return L
    
class WordInclusionLF(AbstractLF):
    """
    This class is a sub-class of AbstractLF. It takes a list of words as an input
    and checks if any of these words appear in a given text.
    """

    def __init__(self, word_list):
        """
        Initializes the class with a list of words to be searched for in texts.

        Parameters:
            word_list (list): A list of words to be searched for in texts.
        """
        self.word_list = word_list

    def apply(self, text_list):
        """
        Takes a list of texts as input and returns a list of Boolean values,
        indicating whether any of the words in word_list appear in the corresponding text.

        Parameters:
            text_list (list): A list of texts to be checked for the presence of words.

        Returns:
            l_values (list): A list of Boolean values indicating the presence of words in the text.
        """
        word_list = self.word_list

        l_values = []

        for text in tqdm.tqdm(text_list):
            search_result = False

            for word in word_list:
                if word in text.lower():
                    search_result = True

            l_values.append(search_result)

        return l_values