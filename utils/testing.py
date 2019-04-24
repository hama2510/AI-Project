import numpy as np

def create_array_score(ids, target, label, label_number, score):
    ids = np.repeat(ids, len(score))
    target = np.repeat(target, len(score))
    label = np.repeat(label, len(score))
    label_number = np.repeat(label_number, len(score))
    return np.transpose(np.vstack([ids, target, score, label, label_number]))