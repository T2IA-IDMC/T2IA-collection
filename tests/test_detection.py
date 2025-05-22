import pytest
from t2ia_collection import BoundingBox
import math


def compare_float_sequences(sequence_1, sequence_2, rel_tol=1e-06, abs_tol=0.0):
    """compare des séquences de float élément par élément avec une certaine tolérance"""
    res = True
    for el_1, el_2 in zip(sequence_1, sequence_2):
        res &= math.isclose(el_1, el_2, rel_tol=rel_tol, abs_tol=abs_tol)
    return res


def test_bounding_box():
    test_img_size = (4646, 3028)

    test_bbox = {
        'xywhn': (0.7443957924842834,
                  0.1604120135307312,
                  0.15120507776737213,
                  0.2762804627418518),
        'xywh': (3458, 486, 702, 837),
        'xyxyn': (0.668793261051178,
                  0.022271793335676193,
                  0.8199983835220337,
                  0.2985522449016571),
        'xyxy': (3107, 67, 3810, 904),
    }

    bbox = BoundingBox(*test_bbox['xywhn'])
    invalid_bbox = BoundingBox(0.056, 0.005, 0.452, 0.004)

    assert bbox.is_valid()
    assert not invalid_bbox.is_valid()
    assert bbox == bbox
    assert bbox != invalid_bbox
    assert BoundingBox(0.5, 0.4, 0.2, 0.2) > BoundingBox(0.2, 0.3, 0.1, 0.1)
    assert compare_float_sequences(bbox.xywhn(), test_bbox['xywhn'])
    assert compare_float_sequences(bbox.xyxyn(), test_bbox['xyxyn'])
    assert bbox.xywh(test_img_size) == test_bbox['xywh']
    assert bbox.xyxy(test_img_size) == test_bbox['xyxy']

