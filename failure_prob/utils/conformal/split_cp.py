import numpy as np
import torch

def quantile_threshold(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes the threshold as the ceil((N+1)*(1 - alpha))-th smallest value of the provided scores.
    
    Args:
        scores: 1D tensor of nonconformity scores (for a given class).
        alpha: significance level (e.g., 0.1 means we want at least 90% coverage).
        
    Returns:
        A scalar tensor representing the threshold.
    """
    N = scores.numel()
    # Calculate rank: note that we need to use 1-indexing for the quantile
    k = int(torch.ceil(torch.tensor((N + 1) * (1 - alpha), dtype=torch.float)))
    k = np.clip(k, 1, N)  # Ensure k is within bounds
    sorted_scores, _ = torch.sort(scores)
    threshold = sorted_scores[k - 1]  # k-1 because of 0-indexing
    return threshold

def split_conformal_binary(cal_scores: torch.Tensor | list,
                           cal_labels: torch.Tensor | list,
                           test_scores: torch.Tensor | list,
                           alpha: float):
    """
    Performs split conformal prediction for binary classification.
    
    For each calibration example, a nonconformity score is computed according to:
      - If the true label is 1 (positive):  alpha = 1 - s(x)
      - If the true label is 0 (negative):  alpha = s(x)
    
    Then, for each candidate label, a threshold is computed from the calibration set.
    For a test example with score s, the nonconformity scores are:
      - For candidate label 1: 1 - s
      - For candidate label 0: s
    
    The prediction set for the test example includes a label if its test nonconformity score is below the corresponding threshold.
    
    Args:
        cal_scores: 1D tensor of shape (N_cal,) containing scores (from e.g. a sigmoid) for calibration examples.
        cal_labels: 1D tensor of shape (N_cal,) containing true binary labels (0 or 1) for calibration examples.
        test_scores: 1D tensor of shape (N_test,) containing scores for test examples.
        alpha: Significance level (e.g., 0.1 for 90% coverage).
        
    Returns:
        A list of length N_test, where each element is a set containing one or both of the candidate labels (0 and/or 1).
    """
    if isinstance(cal_scores, list):
        cal_scores = torch.tensor(cal_scores)
    if isinstance(cal_labels, list):
        cal_labels = torch.tensor(cal_labels)
    if isinstance(test_scores, list):
        test_scores = torch.tensor(test_scores)
    
    # Compute thresholds for each candidate label
    thresholds = {}
    
    # For positive class (label 1): use nonconformity score = 1 - score.
    pos_mask = (cal_labels == 1)
    if pos_mask.sum() > 0:
        cal_pos_scores = cal_scores[pos_mask]
        # Compute nonconformity values for positive examples.
        cal_pos_nconf = 1 - cal_pos_scores
        threshold_pos = quantile_threshold(cal_pos_nconf, alpha)
        thresholds[1] = threshold_pos.item()
    else:
        thresholds[1] = float('inf')
    
    # For negative class (label 0): use nonconformity score = score.
    neg_mask = (cal_labels == 0)
    if neg_mask.sum() > 0:
        cal_neg_scores = cal_scores[neg_mask]
        # Nonconformity values for negative examples.
        cal_neg_nconf = cal_neg_scores
        threshold_neg = quantile_threshold(cal_neg_nconf, alpha)
        thresholds[0] = threshold_neg.item()
    else:
        thresholds[0] = float('inf')
    
    prediction_sets = []
    
    # For each test example, compute nonconformity scores for each candidate label.
    for s in test_scores:
        pred_set = set()
        # For candidate label 1, nonconformity is 1 - s
        if (1 - s) <= thresholds[1]:
            pred_set.add(1)
        # For candidate label 0, nonconformity is s
        if s <= thresholds[0]:
            pred_set.add(0)
        prediction_sets.append(pred_set)
    
    return prediction_sets, thresholds

# --- Example usage ---
if __name__ == "__main__":
    torch.manual_seed(42)
    N_cal = 100  # number of calibration examples
    N_test = 5   # number of test examples
    
    # Simulate calibration scores (for instance, outputs from a sigmoid)
    cal_scores = torch.rand(N_cal)
    # Simulate binary labels (0 or 1) for calibration data
    cal_labels = torch.randint(0, 2, (N_cal,))
    
    # Simulate test scores
    test_scores = torch.rand(N_test)
    
    # Set significance level, e.g., 0.1 for 90% coverage
    alpha = 0.1
    
    pred_sets = split_conformal_binary(cal_scores, cal_labels, test_scores, alpha)
    
    for i, pred in enumerate(pred_sets):
        print(f"Test example {i}: score = {test_scores[i]:.3f}, prediction set = {pred}")
