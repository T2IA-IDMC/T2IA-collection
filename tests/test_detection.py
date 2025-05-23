import pytest
from t2ia_collection import BoundingBox, bbox_from_coord
import math


# ======================================================================================================================
# FUNCTIONS
# ======================================================================================================================

def isclose_float_sequences(sequence_1, sequence_2, rel_tol=1e-06, abs_tol=0.0):
    """compare des séquences de float élément par élément avec une certaine tolérance"""
    res = True
    for el_1, el_2 in zip(sequence_1, sequence_2):
        res &= math.isclose(el_1, el_2, rel_tol=rel_tol, abs_tol=abs_tol)
    return res


# ======================================================================================================================
# FIXTURES
# ======================================================================================================================

@pytest.fixture #test fixture decorator
def test_img_size():
  return 4646, 3028

@pytest.fixture
def test_bboxes():
    return {'xywhn': [0.7443957924842834, 0.1604120135307312, 0.15120507776737213, 0.2762804627418518],
            'xywh': [3458.462890625, 485.7275695800781, 702.498779296875, 836.5772094726562],
            'xyxy': [3107.213623046875, 67.4389877319336, 3809.71240234375, 904.0161743164062],
            'xyxyn': [0.668793261051178, 0.022271793335676193, 0.8199983835220337, 0.2985522449016571]}

@pytest.fixture
def bbox(test_bboxes):
  return BoundingBox(*test_bboxes['xywhn'])  #create an instance

@pytest.fixture
def invalid_bbox(test_bboxes):
  return BoundingBox(0.056, 0.005, 0.452, 0.004)


# ======================================================================================================================
# TESTS
# ======================================================================================================================

class TestClassBoundingBox:
    """test of Class BoundingBox"""

    def test_validity(self, bbox, invalid_bbox):
        """test validity of BoundingBox"""
        assert bbox.is_valid()
        assert not invalid_bbox.is_valid()

    def test_equality(self, bbox, invalid_bbox):
        """test equality of BoundingBox"""
        assert bbox == bbox
        assert bbox != invalid_bbox

    def test_comparison(self):
        """test comparison of BoundingBox"""
        assert BoundingBox(0.5, 0.4, 0.2, 0.2) > BoundingBox(0.2, 0.3, 0.1, 0.1)

    def test_format_conversion(self, bbox, test_bboxes, test_img_size):
        """test conversion of BoundingBox in different formats"""
        # normalized
        assert isclose_float_sequences(bbox.xywhn(), test_bboxes['xywhn'])
        assert isclose_float_sequences(bbox.xyxyn(), test_bboxes['xyxyn'])
        # absolute
        assert bbox.xywh(test_img_size) == tuple(round(coord) for coord in test_bboxes['xywh'])
        assert bbox.xyxy(test_img_size) == tuple(round(coord) for coord in test_bboxes['xyxy'])


class TestFuncBboxFromCoord:
    """test of function bbox_from_coord(coords, format, img_size) -> BoundingBox"""

    def test_instantiation(self, bbox, test_bboxes, test_img_size):
        """test instantiation of BoundingBox using function bbox_from_coord()"""
        for bbox_format in test_bboxes.keys():
            assert bbox == bbox_from_coord(test_bboxes[bbox_format], bbox_format, test_img_size)


    @pytest.mark.parametrize("coords, format, img_size", [([56], 'xywh', (1000, 1000)),
                                                          ([56, 5], 'xywh', (1000, 1000)),
                                                          ([56, 5, 452], 'xywh', (1000, 1000)),
                                                          ([56, 5, 452, 4], 'invalid_format', (1000, 1000)),
                                                          ([56, 5, 452, 4], 'xyxy', None),
                                                          ([56, 5, 452, 4], 'xywh', (1000, 1000))])
    def test_invalid(self, coords, format, img_size):
        """test if bbox_from_coord() call raises a ValueError with invalid entries"""
        print(coords, format, img_size)
        with pytest.raises(ValueError):
            bbox_from_coord(coords, format, img_size)