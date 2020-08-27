from itertools import product

#######################################################
# Some code taken from the original CATER repository: #
#       https://github.com/rohitgirdhar/CATER         #
#######################################################

ACTION_CLASSES = [
    # object, movement
    ('sphere', '_slide'),
    ('sphere', '_pick_place'),
    ('spl', '_slide'),
    ('spl', '_pick_place'),
    ('spl', '_rotate'),
    ('cylinder', '_pick_place'),
    ('cylinder', '_slide'),
    ('cylinder', '_rotate'),
    ('cube', '_slide'),
    ('cube', '_pick_place'),
    ('cube', '_rotate'),
    ('cone', '_contain'),
    ('cone', '_pick_place'),
    ('cone', '_slide'),
]

_BEFORE = 'before'
_AFTER = 'after'
_DURING = 'during'
ORDERING = [
    _BEFORE,
    _DURING,
    _AFTER,
]


def get_action_classes():
    return ACTION_CLASSES


def get_comp_action_classes():
    def reverse(el):
        if el == _DURING:
            return el
        elif el == _BEFORE:
            return _AFTER
        elif el == _AFTER:
            return _BEFORE
        else:
            raise ValueError('This should not happen')

    action_sets = list(product(ACTION_CLASSES, repeat=2))
    classes = list(product(action_sets, ORDERING))

    classes_uniq = []
    for el in classes:
        if el not in classes_uniq and ((el[0][1], el[0][0]), reverse(el[1])) not in classes_uniq:
            classes_uniq.append(el)

    return classes_uniq
