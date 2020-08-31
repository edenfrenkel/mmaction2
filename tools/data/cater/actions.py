from itertools import product

#######################################################
# Some code taken from the original CATER repository: #
#       https://github.com/rohitgirdhar/CATER         #
#######################################################

ACTION_CLASSES = [
    # object, movement
    'sphere slide',
    'sphere pick_place',
    'spl slide',
    'spl pick_place',
    'spl rotate',
    'cylinder pick_place',
    'cylinder slide',
    'cylinder rotate',
    'cube slide',
    'cube pick_place',
    'cube rotate',
    'cone contain',
    'cone pick_place',
    'cone slide',
]

_BEFORE = 'BEFORE'
_AFTER = 'AFTER'
_DURING = 'DURING'
ORDERING = [
    _BEFORE,
    _DURING,
    _AFTER,
]


def get_action_classes():
    return ACTION_CLASSES


def get_comp_action_classes():
    def reverse(order):
        if order == _DURING:
            return _DURING
        elif order == _BEFORE:
            return _AFTER
        elif order == _AFTER:
            return _BEFORE

    action_sets = list(product(ACTION_CLASSES, repeat=2))
    classes = list(product(action_sets, ORDERING))

    classes_uniq = []
    for el in classes:
        cls = el[0][0] + ' ' + el[1] + ' ' + el[0][1]
        reverse_cls = el[0][1] + ' ' + reverse(el[1]) + ' ' + el[0][0]
        if cls not in classes_uniq and reverse_cls not in classes_uniq:
            classes_uniq.append(cls)

    return classes_uniq
