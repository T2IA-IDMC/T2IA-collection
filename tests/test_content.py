import pytest
from t2ia_collection.content import *
import importlib.util  # pour détecter si d'autres librairies sont installées


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
# CONTENT Abstract Class
# ======================================================================================================================
class TestClassContent:
    """test for class Content"""

    def test_instantiation(self):
        """test instantiation of Content Class"""
        assert Content()
        assert Content(False, 0.8)

    def test_invalid(self):
        """test invalid instantiation of a non-manual Content without confidence"""
        with pytest.raises(ValueError):
            Content(False)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert Content().get_cls_name() == 'Content'

    def test_to_dict(self):
        """test to_dict() method"""
        assert Content().to_dict() == {'is_manual': True, 'confidence': None}
        assert Content(False, 0.8).to_dict() == {'is_manual': False, 'confidence': 0.8}

    def test_from_dict(self):
        """test from_dict() method"""
        assert Content.from_dict({'is_manual': False, 'confidence': 0.75}) == Content(False, 0.75)
        assert Content.from_dict({}) == Content()
        assert Content.from_dict(Content(False, 0.8).to_dict()) == Content(False, 0.8)

    def test_to_series(self):
        """test to_series() method"""
        if importlib.util.find_spec("pandas") is not None:
            import pandas as pd
            assert Content().to_series().name == 'Content'
            assert Content(False, 0.8).to_series().equals(pd.Series({'is_manual': False, 'confidence': 0.8}, name='Content'))
        else:
            with pytest.raises(NotImplementedError):
                Content().to_series()

    def test_from_series(self):
        """test from_series() method"""
        if importlib.util.find_spec("pandas") is not None:
            import pandas as pd
            assert Content.from_series(Content().to_series()) == Content()
            assert Content.from_series(Content(False, 0.8).to_series()) == Content(False, 0.8)
        else:
            with pytest.raises(NotImplementedError):
                Content().to_series()


# ======================================================================================================================
# TEXT Abstract Class & subclasses
# ======================================================================================================================
class TestClassText:
    """test for class Text"""

    def test_instantiation(self):
        """test instantiation of Text Class"""
        assert Text(ocr_result='test du contenu manuel', orientation=90, keywords=['test', 'manuel'])
        assert Text(False, 0.8, ocr_result='test du contenu prédit', orientation=90)

    def test_invalid(self):
        """test invalid instantiation of a non-manual Text without confidence"""
        with pytest.raises(ValueError):
            Text(False, ocr_result='test du contenu prédit', orientation=90)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert Text().get_cls_name() == 'Text'

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


# Text Subclasses
# ---------------
class TestClassPrinted:
    """test for class Printed"""

    def test_instantiation(self):
        """test instantiation of Printed Class"""
        assert Printed()
        assert Printed(False, 0.8, ocr_result='test du contenu prédit', orientation=90, is_editor=True)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert Printed().get_cls_name() == 'Printed'


class TestClassHandwritten:
    """test for class Handwritten"""

    def test_instantiation(self):
        """test instantiation of Handwritten Class"""
        assert Handwritten()
        assert Handwritten(False, 0.8, ocr_result='test du contenu prédit', orientation=90)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert Handwritten().get_cls_name() == 'Handwritten'


class TestClassSceneText:
    """test for class SceneText"""

    def test_instantiation(self):
        """test instantiation of SceneText Class"""
        assert SceneText()
        assert SceneText(False, 0.8, ocr_result='test du contenu prédit', orientation=90)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert SceneText().get_cls_name() == 'SceneText'


# ======================================================================================================================
# POSTMARKS Abstract Class & subclasses
# ======================================================================================================================
class TestClassPostmark:
    """test for class Postmark"""

    def test_instantiation(self):
        """test instantiation of Postmark Class"""
        assert Postmark()
        assert Postmark(False, 0.78)

    def test_invalid(self):
        """test invalid instantiation of a non-manually annotated Postmark without confidence"""
        with pytest.raises(ValueError):
            Postmark(False)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert Postmark().get_cls_name() == 'Postmark'

    def test_to_dict(self, pred_text, manual_text, dict_pred_text, dict_manual_text):
        assert Postmark().to_dict() == {'is_manual': True, 'confidence': None} # test manuel
        assert Postmark(False, 0.78).to_dict() == {'is_manual': False, 'confidence': 0.78} # test prédit

    def test_from_dict(self):
        """test from_dict() method"""
        assert Postmark.from_dict({'is_manual': False, 'confidence': 0.75}) == Postmark(False, 0.75)
        assert Postmark.from_dict({}) == Postmark()
        assert Postmark.from_dict(Postmark(False, 0.8).to_dict()) == Postmark(False, 0.8)


# Text Subclasses
# ---------------
class DateStamp:
    """test for class DateStamp"""
    # TODO
    pass


class Stamp:
    """test for class Stamp"""

    def test_instantiation(self):
        """test instantiation of Stamp Class"""
        assert Stamp()
        assert Stamp(False, 0.65, country= "France", color='red', price=0.5)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert Stamp().get_cls_name() == 'Stamp'


class TestClassOtherMark:
    """test for class OtherMark"""

    def test_instantiation(self):
        """test instantiation of OtherMark Class"""
        assert OtherMark()
        assert OtherMark(False, 0.82, is_editor=True)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert OtherMark().get_cls_name() == 'OtherMark'