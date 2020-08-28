import numpy as np
import json
import re
from .actions import get_action_classes, get_comp_action_classes


def assess_by_class(video_infos, results, classes):
    true_positives = np.zeros(len(classes), dtype=np.float)
    false_positives = np.zeros(len(classes), dtype=np.float)
    true_negatives = np.zeros(len(classes), dtype=np.float)
    false_negatives = np.zeros(len(classes), dtype=np.float)

    for i in range(len(video_infos)):
        labels = video_infos[i]['label']
        for j in range(len(classes)):
            if labels[j] == 1:
                if results[i][j] > 0:
                    true_positives[j] += 1
                else:
                    false_negatives[j] += 1
            else:
                if results[i][j] > 0:
                    false_positives[j] += 1
                else:
                    true_negatives[j] += 1

    positives = true_positives + false_positives
    negatives = true_negatives + false_negatives

    true_positives /= positives
    false_positives /= positives
    true_negatives /= negatives
    false_negatives /= negatives

    greater_key = lambda x: -x[1]
    lower_key = lambda x: x[1]

    tp_list = sorted([(classes[i], true_positives[i]) for i in range(len(classes))], key=greater_key)
    tn_list = sorted([(classes[i], true_negatives[i]) for i in range(len(classes))], key=greater_key)
    fp_list = sorted([(classes[i], false_positives[i]) for i in range(len(classes))], key=lower_key)
    fn_list = sorted([(classes[i], false_negatives[i]) for i in range(len(classes))], key=lower_key)

    return {'true_positives': tp_list,
            'false_positives': fp_list,
            'true_negatives': tn_list,
            'false_negatives': fn_list}


def assess_by_video(video_infos, results, classes):
    true_positives = np.zeros(len(video_infos), dtype=np.float)
    false_positives = np.zeros(len(video_infos), dtype=np.float)
    true_negatives = np.zeros(len(video_infos), dtype=np.float)
    false_negatives = np.zeros(len(video_infos), dtype=np.float)
    classifications = [[] for _ in range(len(video_infos))]
    predictions = [[] for _ in range(len(video_infos))]

    for i in range(len(video_infos)):
        labels = video_infos[i]['label']
        for j in range(len(classes)):
            if labels[j] == 1:
                classifications[i].append(classes[j])
                if results[i][j] > 0:
                    predictions[i].append(classes[j])
                    true_positives[i] += 1
                else:
                    false_negatives[i] += 1
            else:
                if results[i][j] > 0:
                    predictions[i].append(classes[j])
                    false_positives[i] += 1
                else:
                    true_negatives[i] += 1

    names = [re.search(r'CATER_new_(\d+)', info['filename']).group(0) for info in video_infos]

    positives = true_positives + false_positives
    negatives = true_negatives + false_negatives

    true_positives /= positives
    false_positives /= positives
    true_negatives /= negatives
    false_negatives /= negatives

    greater_key = lambda x: -x[1]
    lower_key = lambda x: x[1]

    tp_list = sorted([(names[i], true_positives[i]) for i in range(len(names))], key=greater_key)
    tn_list = sorted([(names[i], true_negatives[i]) for i in range(len(names))], key=greater_key)
    fp_list = sorted([(names[i], false_positives[i]) for i in range(len(names))], key=lower_key)
    fn_list = sorted([(names[i], false_negatives[i]) for i in range(len(names))], key=lower_key)

    return {'true_positives': tp_list,
            'false_positives': fp_list,
            'true_negatives': tn_list,
            'false_negatives': fn_list,
            'classifications': classifications,
            'predictions': predictions}


def assess_results(video_infos, results, filename, is_comp=False):
    classes = get_comp_action_classes() if is_comp else get_action_classes()

    class_stats = assess_by_class(video_infos, results, classes)
    video_stats = assess_by_video(video_infos, results, classes)

    with open(filename, 'w') as fp:
        json.dump({'class_analysis': class_stats, 'video_analysis': video_stats}, fp)
