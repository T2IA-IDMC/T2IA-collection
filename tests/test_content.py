import pytest
from t2ia_collection.content import Text


# ======================================================================================================================
# FIXTURES
# ======================================================================================================================

# Text
# ----
@pytest.fixture
def dict_manual_text():
    return {'is_manual': True,
            'confidence': None,
            'ocr_result': 'test du contenu manuel',
            'keywords': ['test', 'manuel'],
            'orientation': 90}

@pytest.fixture
def dict_pred_text():
    return {'is_manual': False,
            'confidence': 0.8,
            'ocr_result': 'test du contenu prédit',
            'keywords': [],
            'orientation': 90}

@pytest.fixture
def manual_text():
    return Text(ocr_result='test du contenu manuel', orientation=90, keywords=['test', 'manuel'])

@pytest.fixture
def pred_text():
    return Text(False, 0.8, ocr_result='test du contenu prédit', orientation=90)


# ======================================================================================================================
# TESTS Text and subclasses
# ======================================================================================================================
class TestClassText:
    """test for class content of BoundingBox"""

    def test_instantiation(self):
        """test instantiation of Detection Class"""
        assert Text(ocr_result='test du contenu manuel', orientation=90, keywords=['test', 'manuel'])
        assert Text(False, 0.8, ocr_result='test du contenu prédit', orientation=90)

    def test_invalid(self):
        """test invalid instantiation of a non-manual Text without confidence"""
        with pytest.raises(ValueError):
            Text(False, ocr_result='test du contenu prédit', orientation=90)

    def test_contains(self, pred_text, manual_text):
        """test of __contains__ method"""
        assert 'prédit' in pred_text
        assert not 'manuel' in pred_text
        assert 'contenu' in manual_text
        assert not 'prédit' in manual_text

    def test_iter(self, pred_text, manual_text):
        """test of __iter__ method"""
        res_manual = []
        for kw in manual_text:
            res_manual.append(kw)
        assert res_manual == ['test', 'manuel']

        res_pred = []
        for kw in pred_text:
            res_pred.append(kw)
        assert len(res_pred) == 0

    def test_set_keywords(self):
        # TODO : implémenter la recherche de mots-clés dans les résultats d'OCR
        pass

    def test_get_keywords(self, pred_text, manual_text):
        assert manual_text.get_keywords() == ['test', 'manuel']
        assert pred_text.get_keywords() == []

    def test_word_list(self, pred_text, manual_text):
        # TODO : voir comment se débarrasser des caractères spéciaux
        assert pred_text.word_list() == ['test', 'du', 'contenu', 'prédit']
        assert manual_text.word_list() == ['test', 'du', 'contenu', 'manuel']

    def test_to_dict(self, pred_text, manual_text, dict_pred_text, dict_manual_text):
        assert manual_text.to_dict() == dict_manual_text # test manuel
        assert pred_text.to_dict() == dict_pred_text # test prédit


    def test_from_dict(self, pred_text, manual_text, dict_pred_text, dict_manual_text):
        assert Text.from_dict({'is_manual': False, 'confidence': 0.75}) == Text(is_manual=False, confidence=0.75, ocr_result='', keywords=[], orientation=0)  # défaut depuis Content
        assert Text.from_dict(dict_manual_text) == manual_text
        assert Text.from_dict(dict_pred_text) == pred_text