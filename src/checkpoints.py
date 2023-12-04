import copy

class Checkpoint:
    def __init__(self, model, dataset, idx):
        self.model = copy.deepcopy(model)
        self.dataset = dataset
        self.idx = idx

    def model(self):
        return self.model
    
    def dataset(self):
        return self.dataset
    
    def idx(self):
        return self.idx

    def __eq__(self, other):
        return self.idx == other.idx
    
    def __str__(self):
        return f'Checkpoint(id={self.idx}, model={self.model}, dataset={self.dataset})'


class CheckpointManager:

    def __init__(self):
        self.checkpoints = []
    
    def add_checkpoint(self, model, dataset, idx):
        checkpoint = Checkpoint(model, dataset, idx)
        self.checkpoints.append(checkpoint)
    
    def checkpoints(self):
        return self.checkpoints
    
    def getCheckpoint(self, idx):
        for checkpoint in self.checkpoints:
            if checkpoint.idx == idx:
                return checkpoint
        return None
    
    def delete_checkpoints_after(self, idx):
        deleted = []
        for checkpoint in self.checkpoints:
            if checkpoint.idx > idx:
                deleted.append(checkpoint)
                self.checkpoints.remove(checkpoint)
        return deleted
    
    def delete_points_from_dataset(self, idxs, dataset=None):
        if dataset is not None:
            tmp = copy.deepcopy(dataset)
            return tmp[idxs[0]], tmp.remove(idxs)
        else:
            tmp = copy.deepcopy(self.checkpoints[-1])
            return tmp.dataset[idxs[0]], tmp.dataset.remove(idxs)

