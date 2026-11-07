import re


def edit_distance(a,b):
    previous = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row = [i]
        for j,y in enumerate(b,1):
            row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x != y)))
        previous = row
    return previous[-1]


def compute_reading_order_ned(predicted_order,ground_truth_order):
    """Normalized edit similarity, 1 for perfect order (the plan's convention)."""
    return 1-edit_distance(predicted_order,ground_truth_order)/max(1,len(predicted_order),len(ground_truth_order))


def compute_parsing_metrics(predicted_text,ground_truth_text):
    chars = edit_distance(predicted_text,ground_truth_text)/max(1,len(ground_truth_text))
    pred_words,gt_words = predicted_text.split(),ground_truth_text.split()
    words = edit_distance(pred_words,gt_words)/max(1,len(gt_words))
    return dict(cer=chars,wer=words)
