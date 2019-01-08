class Vocabulary:
    def __init__(self):
        self.labels_to_idx = {chr(char): idx for idx, char in enumerate(range(ord("a"), ord("z") + 1))}
        self.idx_to_labels = {self.labels_to_idx[char]: char for char in self.labels_to_idx}

    def convert_indices_to_sequences(self, indices):
        return "".join([self.idx_to_labels[idx.data.item()] for idx in indices])
