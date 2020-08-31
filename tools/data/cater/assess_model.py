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

    conf_max = confusions.max(axis=1)
    conf_sum = confusions.sum(axis=1)
    conf_prop = np.divide(conf_max, conf_sum, out=np.zeros_like(conf_max), where=conf_max!=0)
    conf_ind = confusions.argmax(axis=1)
    conf_sum = conf_sum[:, np.newaxis]
    conf_dense = np.divide(confusions, conf_sum, out=np.zeros_like(confusions), where=conf_sum!=0)

    greater_key = lambda x: -x[1]

    tp_list = sorted([(classes[i], true_positives[i]) for i in range(len(classes))], key=greater_key)
    tn_list = sorted([(classes[i], true_negatives[i]) for i in range(len(classes))], key=greater_key)
    fp_list = sorted([(classes[i], false_positives[i]) for i in range(len(classes))], key=greater_key)
    fn_list = sorted([(classes[i], false_negatives[i]) for i in range(len(classes))], key=greater_key)
    conf_list = sorted([(classes[i], conf_prop[i], classes[conf_ind[i]]) for i in range(len(classes))], key=greater_key)
    full_conf_list = sorted([(classes[i], conf_prop[i],
                              sorted([(classes[j], conf_dense[i, j]) for j in range(len(classes))],
                                     key=greater_key))
                             for i in range(len(classes))], key=greater_key)

    return {'true_positives': tp_list,
            'false_positives': fp_list,
            'true_negatives': tn_list,
            'false_negatives': fn_list,
            'confusion': conf_list,
            'full_confusion': full_conf_list}


def assess_by_video(video_infos, results, classes):
    just_labels = np.zeros((len(video_infos), len(classes)), dtype=np.float)
    just_results = np.zeros((len(video_infos), len(classes)), dtype=np.float)
    true_pos = np.zeros((len(video_infos), len(classes)), dtype=np.float)
    names = [re.search(r'CATER_new_(\d+)', info['filename']).group(0) for info in video_infos]

    for i in range(len(video_infos)):
        label = video_infos[i]['label'].cpu().numpy()
        result = np.array([1 if results[i][j] > 0 else 0
                           for j in range(len(classes))], dtype=np.float)
        both = label * result
        just_label = label - both
        just_result = result - both

        just_labels[i] += just_label
        just_results[i] += just_result
        true_pos[i] += both

    len_key = lambda x: -len(x[1])

    jl_list = sorted([[names[i], [classes[j] for j in range(len(classes)) if just_labels[i, j] == 1]]
                      for i in range(len(names))], key=len_key)
    jr_list = sorted([[names[i], [classes[j] for j in range(len(classes)) if just_results[i, j] == 1]]
                      for i in range(len(names))], key=len_key)
    both_list = sorted([[names[i], [classes[j] for j in range(len(classes)) if true_pos[i, j] == 1]]
                        for i in range(len(names))], key=len_key)

    return {'just_label': jl_list,
            'just_result': jr_list,
            'both': both_list}


def assess_results(video_infos, results, filename, is_comp=False):
    classes = get_comp_action_classes() if is_comp else get_action_classes()

    class_stats = assess_by_class(video_infos, results, classes)
    video_stats = assess_by_video(video_infos, results, classes)

    with open(filename, 'w') as fp:
        json.dump({'class_analysis': class_stats, 'video_analysis': video_stats}, fp)
