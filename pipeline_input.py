import numpy as np
import tensorflow as tf


class pipeline_dataset_interpreter:
    # TODO: Impractical to have entire dataset to be loaded into memory
    # look into alterative architectures which load things into memory chunk by chunk
    dataset = {
        'test': {
            'x':   None,
            'y': None
        },
        'train': {
            'x':   None,
            'y': None
        }
    }

    def __init__(self, input_dir, load=True) -> None:
        self.input_dir = input_dir
        if load:
            self.load()

    def get_dataset(self) -> dict:
        return self.dataset

    def load(self) -> None:
        # Load the dataset into self.dataset
        pass


class pipeline_model:
    model = None

    def __init__(self, load=True) -> None:
        if load:
            self.load()

    def get_model(self) -> tf.keras.Model:
        return self.model

    def load(self) -> None:
        # Load the model into self.__model
        pass

    def predict(self, x: np.array) -> np.array:
        # Runs prediction on list of values x of length n
        # Returns a list of values of length n
        pass
        
    def evaluate(self, x, y) -> np.array:
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
    __pipeline_name = {}
    __pipeline_dataset_interpreter = {}
    __pipeline_model = {}
    __pipeline_ensembler = {}

    def __init__(self, p_name: str, p_dataset_interpreter: dict, p_model: dict, p_ensembler: dict) -> None:
        assert isinstance(p_name, str)
        assert isinstance(p_dataset_interpreter,dict)
        for p in p_dataset_interpreter:
	        assert issubclass(p_dataset_interpreter[p],pipeline_dataset_interpreter)
        assert isinstance(p_model,dict)
        for p in p_model:
	        assert issubclass(p_model[p],pipeline_model)
        assert isinstance(p_ensembler,dict)
        for p in p_ensembler:
	        assert issubclass(p_ensembler[p],pipeline_ensembler)
        self.__pipeline_name = p_name
        self.__pipeline_dataset_interpreter = p_dataset_interpreter
        self.__pipeline_model = p_model
        self.__pipeline_ensembler = p_ensembler
    
    def get_pipeline_name(self) -> str:
        return self.__pipeline_name

    def get_pipeline_dataset_interpreter_by_name(self, name: str) -> type:
        return self.__pipeline_dataset_interpreter[name]
    
    def get_pipeline_model_by_name(self, name: str) -> type:
        return self.__pipeline_model[name]

    def get_pipeline_ensembler_by_name(self, name: str) -> type:
        return self.__pipeline_ensembler[name]

    def get_pipeline_dataset_interpreter(self) -> dict:
        return self.__pipeline_dataset_interpreter
    
    def get_pipeline_model(self) -> dict:
        return self.__pipeline_model

    def get_pipeline_ensembler(self) -> dict:
        return self.__pipeline_ensembler
