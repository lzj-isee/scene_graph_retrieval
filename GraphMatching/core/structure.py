import numpy as np
import torch

class graph(object):
    def __init__(self, w: torch.Tensor, label: torch.Tensor) -> None:
        super().__init__()
        self.w = w
        self.node_label = label

class scene_graph(object):
    def __init__(self, pred, device = 'cpu', temperature = 0.1) -> None:
        super().__init__()
        self.device = device
        self.boxes = torch.as_tensor(pred['box'], device = device)
        __temp = torch.as_tensor(pred['area'], device = device)
        self.pair_weights = torch.exp(__temp / temperature) / torch.sum(torch.exp(__temp / temperature))
        self.box_labels = torch.as_tensor(pred['box_labels'], device = device)
        # self.box_scores = torch.as_tensor(pred['box_scores'], device = device)
        self.rel_pairs = torch.as_tensor(pred['rel_pairs'], device = device)
        # self.rel_scores = torch.as_tensor(pred['rel_scores'], device = device)
        self.rel_labels = torch.as_tensor(pred['rel_labels'], device = device)
        self.node_num = len(self.box_labels)
        self.intra_simi = None
        self.rel_score_matrix = None
        self.rel_label_matrix = None

    def get_node_wise_simi(self):
        # calculate the similarity between two nodes via the average of rel_score
        if self.intra_simi is None:
            self.intra_simi = torch.zeros(self.node_num, self.node_num, device = self.device)
            self.intra_simi[self.rel_pairs[:, 0], self.rel_pairs[:, 1]] += self.rel_scores
            self.intra_simi = (self.intra_simi + self.intra_simi.t()) / 2
            self.intra_simi = self.intra_simi / self.intra_simi.max()
        return self.intra_simi

    def get_rel_score_matrix(self):
        if self.rel_score_matrix is None:
            self.rel_score_matrix = torch.zeros(self.node_num, self.node_num, device = self.device)
            self.rel_score_matrix[self.rel_pairs[:, 0], self.rel_pairs[:, 1]] = self.rel_scores
        return self.rel_score_matrix

    def get_rel_label_matrix(self):
        if self.rel_label_matrix is None:
            self.rel_label_matrix = torch.zeros(self.node_num, self.node_num, device = self.device)
            self.rel_label_matrix[self.rel_pairs[:, 0], self.rel_pairs[:, 1]] = self.rel_labels.float()
        return self.rel_label_matrix