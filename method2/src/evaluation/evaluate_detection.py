from ..spatial.coordinates import intersection_area,area


def bbox_iou(a,b):
    intersection = intersection_area(a,b)
    return intersection/max(1,area(a)+area(b)-intersection)


def compute_detection_metrics(predictions,ground_truth,iou_threshold=.5):
    def get(item,key):
        if isinstance(item,dict):
            return item[key]
        return getattr(item,key)
    unmatched = set(range(len(ground_truth)))
    true_positive = 0
    for pred in predictions:
        candidates = [(bbox_iou(get(pred,'bbox'),get(ground_truth[i],'bbox')),i) for i in unmatched
                      if get(pred,'label') == get(ground_truth[i],'label')]
        if candidates:
            overlap,index = max(candidates)
            if overlap >= iou_threshold:
                true_positive += 1
                unmatched.remove(index)
    precision = true_positive/max(1,len(predictions))
    recall = true_positive/max(1,len(ground_truth))
    return dict(precision=precision,recall=recall,f1=2*precision*recall/max(1e-12,precision+recall))
