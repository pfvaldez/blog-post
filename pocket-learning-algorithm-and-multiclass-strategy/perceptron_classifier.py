import numpy as np

class PerceptronClassifier:
    '''Preceptron Binary Classifier uses Perceptron Learning Algorithm
        to  classify two classes data.

    Parameters
    ----------
    number_of_attributes : int
        The number of attributes of the data set.

    class_labels : tuple of the class labels
        The class labels can be anything as long as it has only two types of labels.

    Attributes
    ----------
    weights : list of float
        The list of weights corresponding input attributes.

    misclassify_record : list of int
        The number of misclassification for each training sample.
    '''
    def __init__(self, number_of_attributes: int, class_labels: ()):
        # Initialize the weights to zero
        # The size is the number of attributes plus the bias, i.e. x_0 * w_0
        self.weights = np.zeros(number_of_attributes + 1)

        # Record of the number of misclassify for each train sample
        self.misclassify_record = []

        # Build the label map to map the original labels to numerical labels
        # For example, ['a', 'b'] -> {0: 'a', 1: 'b'}
        self._label_map = {1: class_labels[0], -1: class_labels[1]}
        self._reversed_label_map = {class_labels[0]: 1, class_labels[1]: -1}

    def _linear_combination(self, sample):
        '''linear combination of sample and weights'''
        return np.inner(sample, self.weights[1:])

    def train(self, samples, labels, max_iterator=10):
        '''Train the model

        Parameters
        ----------
        samples : two dimensions list
            Training data set
        labels : list of labels
            The class labels of the training data
        max_iterator : int
            The max iterator to stop the training process
            in case the training data is not converaged.
        '''
        # Transfer the labels to numerical labels
        transferred_labels = [self._reversed_label_map[index] for index in labels]

        for _ in range(max_iterator):
            misclassifies = 0
            for sample, target in zip(samples, transferred_labels):
                linear_combination = self._linear_combination(sample)
                update = target - np.where(linear_combination >= 0.0, 1, -1)

                # use numpy.multiply to multiply element-wise
                self.weights[1:] += np.multiply(update, sample)
                self.weights[0] += update

                # record the number of misclassification
                misclassifies += int(update != 0.0)

            if misclassifies == 0:
                break
            self.misclassify_record.append(misclassifies)

    def classify(self, new_data):
        '''Classify the sample based on the trained weights

        Parameters
        ----------
        new_data : two dimensions list
            New data to be classified

        Return
        ------
        List of int
            The list of predicted class labels.
        '''
        predicted_result = np.where((self._linear_combination(new_data) + self.weights[0]) >= 0.0, 1, -1)
        return [self._label_map[item] for item in predicted_result]
