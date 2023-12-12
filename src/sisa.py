from classifier.neural_network import NeuralNetwork
from techniques.retraining import circuit_checkpoint_retraining, circuit_train_retraining
from checkpoints import CheckpointManager
from pathlib import Path
from dataset import Dataset
import uuid
import json
from loguru import logger as log
from utils import setup_working_dir
from pathlib import Path

NUM_SHARDS = 6

class SISA_Model:
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
        self.model = NeuralNetwork()
        self.weights = None
        self.id = str(uuid.uuid4())

    def train(self, config, X, y):
        # initialize weights
        log.info('EPOCH 0')
        dataset = Dataset(train=list(zip(X[0:1], y[0:1]))).shift(1e5)
        self.weights = self.model.init_model(config, dataset.no_features)
        self.model.train(config, dataset, self.weights)
        self.checkpoint_manager.add_checkpoint(self.model, dataset, 1)

        # train model
        if len(X) > 2:
            for i in range(2, len(X)):
                log.info(f'EPOCH {i-1}')
                dataset = Dataset(train=list(zip(X[0:i], y[0:i]))).shift(1e5)
                self.weights = self.model.weights
                self.model.train(config, dataset, self.weights)
                self.checkpoint_manager.add_checkpoint(self.model, dataset, i)

    def retrain(self, config, min_idx, D_prev, U_prev, D_plus):
        self.model = self.checkpoint_manager.getCheckpoint(min_idx).model
        proof_src, params = circuit_train_retraining(config, self.model, D_prev, U_prev, D_plus)

        return {'proof': proof_src, 'params': params}

    def unlearn(self, config, datapoints):
        datapoint_indices = datapoints
        min_idx = min(datapoint_indices)
        datasets = []
        dataset = None
        deleted_prev = []
        for idx in datapoint_indices:
            deleted_point, dataset = self.checkpoint_manager.delete_points_from_dataset(idxs=[idx], dataset=dataset)
            deleted_prev.append(deleted_point)
        for i in range(min_idx, dataset.size):
            train_to_index = list(zip([dataset.X[i]], [dataset.Y[i]]))
            datasets.append(Dataset(train=train_to_index))
        self.checkpoint_manager.delete_checkpoints_after(min_idx)

        foundation_dataset = datasets[0]
        proof_params = []
        while len(datasets) > 1:
            X, y = zip(*deleted_prev)
            U_prev = Dataset(train=list(zip(X, y)))
            D_plus = datasets.pop()
            proof_param = self.retrain(config, min_idx, foundation_dataset, U_prev, D_plus)
            proof_params.append(proof_param)
            foundation_dataset = D_plus
        return proof_params

    def predict(self, datapoint, config):
        return self.model.predict(datapoint, config)

class SISA:
    def __init__(self, config):
        self.models = [SISA_Model() for _ in range(NUM_SHARDS)]
        self.shards = []
        self.config = config

    def train(self, X, y):
        shard_size = len(X) // NUM_SHARDS
        sharded_X = list(shard_dataset(X, shard_size))
        sharded_y = list(shard_dataset(y, shard_size))

        for i in range(NUM_SHARDS):
            self.shards.append({"X": sharded_X[i], "y": sharded_y[i]})

        for i, model in enumerate(self.models):
            model.train(self.config, self.shards[i]['X'], self.shards[i]['y'])

    def unlearn(self, datapoint_ids):

        model_datapoints = {}

        for model in self.models:
            model_datapoints[model.id] = []
        
        for datapoint_id in datapoint_ids:
            model_id = datapoint_id['model_id']
            datapoint_idx = int(datapoint_id['datapoint_idx'])
            model_datapoints[model_id].append(datapoint_idx)

        proof_params = []

        for model_id, datapoints in model_datapoints.items():
            if len(datapoints) == 0:
                continue
            model = next((model for model in self.models if model.id == model_id), None)
            proof_params += model.unlearn(self.config, datapoints)

        return proof_params

    def predict(self, datapoint, scaler):
        tmp = Dataset(train=[])
        dpt = scaler.transform([datapoint])[0]

        X_sh = [ tmp.add_shift(x_i, self.config['precision']) for x_i in dpt]
        predictions = {'0': 0, '1': 0}
        list_pred = []
        
        for sisa_model in self.models:
            
            prediction = sisa_model.predict(X_sh, self.config)
            list_pred.append(prediction)

            thresh = sisa_model.model.add_shift(0.5, self.config['precision'])
            
            if prediction < thresh:
                predictions['0'] += 1
            else:
                predictions['1'] += 1

        return max(predictions, key=predictions.get), list_pred

    def examples(self):
        ex = []
        for sisa_model in self.models:
            datapoint = {
                'id': sisa_model.id,
                'datapoints': [0, 1, 2, 3],
                'data': [sisa_model.checkpoint_manager.getCheckpoint(5).dataset[i] for i in [0, 1, 2, 3]],
            }
            ex.append(datapoint)
        return ex


def shard_dataset(dataset, shard_size):
    for i in range(0, len(dataset), shard_size):
        yield dataset[i:i + shard_size]