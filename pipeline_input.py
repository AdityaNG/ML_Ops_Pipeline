import numpy as np
import tensorflow as tf


class pipeline_dataset_interpreter:
    # TODO: Impractical to have entire dataset to be loaded into memory
    # look into alterative architectures which load things into memory chunk by chunk
    __dataset = {
        'test': {
            'x':   [],
            'y': []
        },
        'train': {
            'x':   [],
            'y': []
        }
    }

    def __init__(self, input_dir, load=True) -> None:
        self.input_dir = input_dir
        if load:
            self.load()

    def get_dataset(self) -> dict:
        return self.__dataset

    def load(self) -> None:
        # Load the dataset into self.__dataset
        pass


class pipeline_model:
    __model = None

    def __init__(self, weights_path, load=True) -> None:
        self.weights_path = weights_path
        if load:
            self.load()

    def get_model(self) -> tf.keras.Model:
        return self.__model

    def load(self) -> None:
        # Load the model into self.__model
        pass

    def predict(self, x: np.array) -> np.array:
        # Runs prediction on list of values x of length n
        # Returns a list of values of length n
        pass

class pipeline_ensembler:

    def merge(self, x: np.array) -> np.array:
        # Given a list of lists of predictions from multiple learners
        # Say m learners and n predictions
        # x[m][n] -> the nth prediction for the mth learner
        # Returns a single result
        # y[n] -> merged nth prediction
        pass


class pipeline_input:
    __pipeline_name = None
    __pipeline_dataset_interpreter = None
    __pipeline_model = None
    __pipeline_ensembler = None

    def __init__(self, p_name: str, p_dataset_interpreter: type, p_model: type, p_ensembler: type) -> None:
        assert isinstance(p_name, str)
        assert issubclass(p_dataset_interpreter,pipeline_dataset_interpreter)
        assert issubclass(p_model,pipeline_model)
        assert issubclass(p_ensembler,pipeline_ensembler)
        self.__pipeline_name = p_name
        self.__pipeline_dataset_interpreter = p_dataset_interpreter
        self.__pipeline_model = p_model
        self.__pipeline_ensembler = p_ensembler
    
    def get_pipeline_name(self) -> str:
        return self.__pipeline_name

    def get_pipeline_dataset_interpreter(self) -> type:
        return self.__pipeline_dataset_interpreter
    
    def get_pipeline_model(self) -> type:
        return self.__pipeline_model

    def get_pipeline_ensembler(self) -> type:
        return self.__pipeline_ensembler
