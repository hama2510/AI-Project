import numpy as np

def create_array_score(ids, target, label, label_number, score):
    """Create array score for visualizing testing

    Parameters:
    ----------
    ids: string
        ID of item
    target: string
        Measurement Target
    label: string
        Label of item
    label_number: int
        Numeric label of item
    score: array
        Scores between ground truth and prediction
    Returns:
    scores: array
        array score

    """
    ids = np.repeat(ids, len(score))
    target = np.repeat(target, len(score))
    label = np.repeat(label, len(score))
    label_number = np.repeat(label_number, len(score))
    return np.transpose(np.vstack([ids, target, score, label, label_number]))