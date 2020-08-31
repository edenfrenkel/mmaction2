import numpy as np
import json
import re
from .actions import get_action_classes, get_comp_action_classes


def assess_by_class(video_infos, results, classes):
    true_positives = np.zeros(len(classes), dtype=np.float)
    false_positives = np.zeros(len(classes), dtype=np.float)
    true_negatives = np.zeros(len(classes), dtype=np.float)
    false_negatives = np.zeros(len(classes), dtype=np.float)
    confusions = np.zeros((len(classes), len(classes)), dtype=np.float)

    for i in range(len(video_infos)):
        labels = video_infos[i]['label'].cpu().numpy()
        result = np.array([1 if results[i][j] > 0 else 0
                           for j in range(len(classes))], dtype=np.float)
        both = labels * result
        just_labels = labels - both
        just_result = result - both

        true_positives += both
        false_negatives += just_labels
        false_positives += just_result
        true_negatives += (1 - labels) * (1 - result)
        confusions[just_labels == 1] += just_result

    conf_prop = confusions.max(axis=1) / confusions.sum(axis=1)
    conf_ind = confusions.argmax(axis=1)

    greater_key = lambda x: -x[1]

    tp_list = sorted([(classes[i], true_positives[i]) for i in range(len(classes))], key=greater_key)
    tn_list = sorted([(classes[i], true_negatives[i]) for i in range(len(classes))], key=greater_key)
    fp_list = sorted([(classes[i], false_positives[i]) for i in range(len(classes))], key=greater_key)
    fn_list = sorted([(classes[i], false_negatives[i]) for i in range(len(classes))], key=greater_key)
    conf_list = sorted([(classes[i], conf_prop[i], classes[conf_ind[i]]) for i in range(len(classes))], key=greater_key)

    return {'true_positives': tp_list,
            'false_positives': fp_list,
            'true_negatives': tn_list,
            'false_negatives': fn_list,
            'confusion': conf_list}


def assess_by_video(video_infos, results, classes):
    true_positives = np.zeros(len(video_infos), dtype=np.float)
    false_positives = np.zeros(len(video_infos), dtype=np.float)
    true_negatives = np.zeros(len(video_infos), dtype=np.float)
    false_negatives = np.zeros(len(video_infos), dtype=np.float)
    classifications = {}
    predictions = {}
    names = [re.search(r'CATER_new_(\d+)', info['filename']).group(0) for info in video_infos]

    for i in range(len(video_infos)):
        labels = video_infos[i]['label']
        cls = []
        preds = []
        for j in range(len(classes)):
            if labels[j] == 1:
                cls.append(classes[j])
                if results[i][j] > 0:
                    preds.append(classes[j])
                    true_positives[i] += 1
                else:
                    false_negatives[i] += 1
            else:
                if results[i][j] > 0:
                    preds.append(classes[j])
                    false_positives[i] += 1
                else:
                    true_negatives[i] += 1
        classifications[names[i]] = cls
        predictions[names[i]] = preds

    greater_key = lambda x: -x[1]

    tp_list = sorted([(names[i], true_positives[i]) for i in range(len(names))], key=greater_key)
    tn_list = sorted([(names[i], true_negatives[i]) for i in range(len(names))], key=greater_key)
    fp_list = sorted([(names[i], false_positives[i]) for i in range(len(names))], key=greater_key)
    fn_list = sorted([(names[i], false_negatives[i]) for i in range(len(names))], key=greater_key)

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
