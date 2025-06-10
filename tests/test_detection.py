import pytest
from t2ia_collection.detection import *
import math

# ======================================================================================================================
# FUNCTIONS
# ======================================================================================================================

def isclose_float_sequences(sequence_1, sequence_2, rtol=1e-05, atol=1e-08):
    """compare des séquences de float élément par élément avec une certaine tolérance"""
    res = True
    for el_1, el_2 in zip(sequence_1, sequence_2):
        res &= math.isclose(el_1, el_2, rel_tol=rtol, abs_tol=atol)
    return res


# ======================================================================================================================
# FIXTURES
# ======================================================================================================================

# BoundingBox
# -----------
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


# Detection
# ---------
@pytest.fixture
def dict_empty_det(test_bboxes):
  return {'bbox': {coord: value for coord, value in zip('xywh', test_bboxes['xywhn'])},
          'is_manual': False,
          'confidence': 0.75,
          'content': None}

@pytest.fixture
def dict_text_det():
  return {'bbox': {'x': 0.239518, 'y': 0.038474, 'w': 0.23065, 'h': 0.033355},
          'is_manual': True,
          'confidence': None,
          'content': {'PrintedText': {'is_manual': True,
                      'confidence': None,
                      'ocr_result': "ALLAND'HUY. - L’Eglise.",
                      'keywords': ['église'],
                      'orientation': 0,
                      'is_editor': False}}}

@pytest.fixture
def dict_datestamp_det():
  return {'bbox': {'x': 0.694804, 'y': 0.164404, 'w': 0.183935, 'h': 0.275776},
          'is_manual': True,
          'confidence': None,
          'content': {'DateStamp': {'is_manual': True,
                                    'postal_agency': 'ATTIGNY',
                                    'date': 'XXXX-07-30TXX:XX',
                                    'department': 'ARDENNES',
                                    'starred_hour': False,
                                    'collection': '3E',
                                    'mark_type': 'post office',
                                    'quality': 'mediocre'}}}

@pytest.fixture
def empty_det(bbox):
    return Detection(bbox, is_manual=False, confidence=0.75)

@pytest.fixture
def text_det(dict_text_det):
  return Detection(BoundingBox.from_dict(dict_text_det['bbox']),
                   is_manual=dict_text_det['is_manual'],
                   confidence=dict_text_det['confidence'],
                   content=Content.from_json_object(dict_text_det['content']))

@pytest.fixture
def datestamp_det(dict_datestamp_det):
  return Detection(BoundingBox.from_dict(dict_datestamp_det['bbox']),
                   is_manual=dict_datestamp_det['is_manual'],
                   confidence=dict_datestamp_det['confidence'],
                   content=Content.from_json_object(dict_datestamp_det['content']))

# ======================================================================================================================
# TESTS BoundingBox
# ======================================================================================================================

class TestClassBoundingBox:
    """tests for BoundingBox Class"""

    def test_validity(self, bbox, invalid_bbox):
        """test validity of BoundingBox"""
        assert bbox.isvalid()
        assert not invalid_bbox.isvalid()

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

    def test_rotation(self, bbox):
        """test rotation of BoundingBox"""
        assert bbox == bbox.rotate(360)
        assert bbox == bbox.rotate(90).rotate(180).rotate(90)
        assert bbox.rotate(-90) == bbox.rotate(270)
        assert bbox.rotate(180) == bbox.rotate(-180)
        assert bbox.rotate(90) == bbox.rotate(450)
        assert bbox.rotate(90) == BoundingBox(x=0.1604120135307312, y=0.25560420751571655,
                                              w=0.2762804627418518, h=0.15120507776737213)
        assert bbox.rotate("90") == bbox.rotate(90)
        assert bbox == bbox.rotate(None)
        # test inplace
        bbox_test = bbox.copy()
        bbox_test.rotate(90, inplace=True)
        assert bbox_test != bbox
        assert bbox_test == bbox.rotate(90)

    def test_invalid_rotation(self, bbox):
        """test rotation of BoundingBox avec des angles invalides"""
        with pytest.raises(ValueError):
            bbox.rotate(68)
        with pytest.raises(ValueError):
            bbox.rotate("lolilol")
        with pytest.raises(TypeError):
            bbox.rotate([0, 90, 0])


    def test_to_dict(self, bbox, test_bboxes):
        """test BoundingBox transformation to dict"""
        assert bbox.to_dict() == {coord: value for coord, value in zip('xywh', test_bboxes['xywhn'])}

    def test_from_dict(self, bbox, test_bboxes):
        """test BoundingBox instantiation from dict"""
        assert bbox == BoundingBox.from_dict({coord: value for coord, value in zip('xywh', test_bboxes['xywhn'])})
        assert bbox == BoundingBox.from_dict(bbox.to_dict())


class TestBboxFromCoord:
    """test for method from_coords(coords, coord_format, img_size) -> BoundingBox"""

    def test_instantiation(self, bbox, test_bboxes, test_img_size):
        """test instantiation of BoundingBox using method bbox_from_coord()"""
        for bbox_format in test_bboxes.keys():
            assert bbox == BoundingBox.from_coords(test_bboxes[bbox_format], bbox_format, test_img_size)


    @pytest.mark.parametrize("coords, coord_format, img_size", [([56], 'xywh', (1000, 1000)),
                                                                ([56, 5], 'xywh', (1000, 1000)),
                                                                ([56, 5, 452], 'xywh', (1000, 1000)),
                                                                ([56, 5, 452, 4], 'invalid_format', (1000, 1000)),
                                                                ([56, 5, 452, 4], 'xyxy', None),
                                                                ([56, 5, 452, 4], 'xywh', (1000, 1000))])
    def test_invalid(self, coords, coord_format, img_size):
        """test if bbox_from_coord() call raises a ValueError with invalid entries"""
        with pytest.raises(ValueError):
            BoundingBox.from_coords(coords, coord_format, img_size)


class TestCoordFromBbox:
    """test for method to_coords(self, coord_format, img_size) -> Tuple[float, float, float, float]"""

    def test_retrival(self, bbox, test_bboxes, test_img_size):
        """test retrival of coordinates from BoundingBox using method to_coords()"""
        for bbox_format in test_bboxes.keys():
            if 'n' in bbox_format:
                assert isclose_float_sequences(test_bboxes[bbox_format], bbox.to_coords(bbox_format, test_img_size))
            else:  # sinon problème avec les parties entières des coordonnées non normalisées
                assert tuple(round(el) for el in test_bboxes[bbox_format]) == bbox.to_coords(bbox_format, test_img_size)


    @pytest.mark.parametrize("coord_format, img_size", [('invalid_format', (1000, 1000)),
                                                        ('xyxy', None)])
    def test_invalid(self, bbox, coord_format, img_size):
        """test if bbox_from_coord() call raises a ValueError with invalid entries"""
        with pytest.raises(ValueError):
            bbox.to_coords(coord_format, img_size)

# ======================================================================================================================
# TESTS Detection
# ======================================================================================================================

class TestClassDetection:
    """tests for Detection Class"""

    def test_instantiation(self, bbox):
        """test instantiation of Detection Class"""
        assert Detection(bbox)
        assert Detection(bbox, is_manual=False, confidence=0.76)
        # TODO : add with a content

    def test_invalid(self, bbox):
        """test invalid instantiation of a non-manual Detection without confidence"""
        with pytest.raises(ValueError):
            Detection(bbox, is_manual=False)

    def test_isempty(self, bbox, empty_det, text_det, datestamp_det):
        """test if Detection has content"""
        assert Detection(bbox).isempty()
        assert empty_det.isempty()
        assert not Detection(bbox, content=DateStamp()).isempty()  # pas vide, mais pas process
        assert not text_det.isempty()
        assert not datestamp_det.isempty()

    def test_isprocessed(self, bbox, empty_det, text_det, datestamp_det):
        """test if Detection has content"""
        assert not Detection(bbox).isprocessed()
        assert not empty_det.isprocessed()
        assert not Detection(bbox, content=DateStamp()).isprocessed()  # pas vide, mais pas process
        assert text_det.isprocessed()
        assert datestamp_det.isprocessed()

    def test_get_content_cls(self, bbox, datestamp_det):
        """test Content class retrieval"""
        assert Detection(bbox).get_content_cls() == "Content"
        assert Detection(bbox).get_content_cls() != "PrintedText"
        assert Detection(bbox, content=DateStamp()).get_content_cls() != "Content"
        assert Detection(bbox, content=DateStamp()).get_content_cls() == "DateStamp"
        assert Detection(bbox, content=PrintedText()).get_content_cls() == "PrintedText"

    def test_to_dict(self, empty_det, text_det, datestamp_det, dict_empty_det, dict_text_det, dict_datestamp_det):
        """test Detection transformation to dict"""
        assert empty_det.to_dict() == dict_empty_det
        assert text_det.to_dict() == dict_text_det
        assert datestamp_det.to_dict() == dict_datestamp_det

    def test_from_dict(self, empty_det, text_det, datestamp_det, dict_empty_det, dict_text_det, dict_datestamp_det):
        """test Detection instantiation from dict"""
        assert empty_det == Detection.from_dict(dict_empty_det)
        assert empty_det == Detection.from_dict(empty_det.to_dict())
        assert text_det == Detection.from_dict(dict_text_det)
        assert text_det == Detection.from_dict(text_det.to_dict())
        assert datestamp_det == Detection.from_dict(dict_datestamp_det)
        assert datestamp_det == Detection.from_dict(datestamp_det.to_dict())


