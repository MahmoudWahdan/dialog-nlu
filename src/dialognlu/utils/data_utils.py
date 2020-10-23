# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from itertools import chain


def flatten(y):
    """
    Flatten a list of lists.

    >>> flatten([[1,2], [3,4]])
    [1, 2, 3, 4]
    """
    return list(chain.from_iterable(y))


def convert_to_slots(slots_arr, no_class_tag='O', begin_prefix='B-', in_prefix='I-'):
    previous = None
    slots = []
    start = -1
    end = -1

    def add(name, s, e):
        if e < s:
            e = s
        slots.append((name, s, e))

    for i, slot in enumerate(slots_arr):
        if slot == no_class_tag:
            current = None
            if previous is not None:
                add(previous, start, end)
        if slot.startswith(begin_prefix):
            current = slot[len(begin_prefix):]
            if previous is not None:
                add(previous, start, end)
            start = i
        elif slot.startswith(in_prefix):
            current = slot[len(in_prefix):]
            if current != previous:
                if previous is not None:
                    add(previous, start, end)
                for j, sl in enumerate(slots):
                    if current == sl[0]:
                        sl = list(sl)
                        sl[2] = i
                        slots[j] = tuple(sl)
                current = None
            else:
                end = i
            
        previous = current
        
    if previous is not None:
        add(previous, start, end)
        
    return slots


if __name__ == '__main__':
    result = convert_to_slots(['B-action', 'B-object', 'B-state'])
    assert result == [('action', 0, 0), ('object', 1, 1), ('state', 2, 2)]
    result = convert_to_slots(['O', 'B-action', 'O', 'B-object', 'B-state'])
    assert result == [('action', 1, 1), ('object', 3, 3), ('state', 4, 4)]
    result = convert_to_slots(['O', 'B-action', 'O', 'B-object', 'I-action'])
    assert result == [('action', 1, 4), ('object', 3, 3)]
    result = convert_to_slots(['O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O'])
    assert result == [('artist', 1, 2), ('playlist', 5, 6)]
    result = convert_to_slots(['O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'O'])
    assert result == [('artist', 1, 2), ('playlist', 5, 5)]
    result = convert_to_slots(['O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist'])
    assert result == [('artist', 1, 2), ('playlist', 5, 5)]
    result = convert_to_slots(['O', 'B-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O'])
    assert result == [('artist', 1, 1), ('playlist', 4, 5)]
    result = convert_to_slots(['O', 'I-artist', 'I-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O'])
    assert result == [('playlist', 5, 6)]
