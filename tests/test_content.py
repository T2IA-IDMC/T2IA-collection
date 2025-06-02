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
            'keywords': ['manuel', 'test'],
            'orientation': 0}

@pytest.fixture
def dict_pred_text():
    return {'is_manual': False,
            'confidence': 0.8,
            'ocr_result': 'test du contenu prédit',
            'keywords': [],
            'orientation': 90}

@pytest.fixture
def manual_text():
    return Text(is_manual=True, ocr_result='test du contenu manuel', orientation=0, keywords=['manuel', 'test'])

@pytest.fixture
def pred_text():
    return Text(False, 0.8, ocr_result='test du contenu prédit', orientation=90)

@pytest.fixture
def datestamp_json():
    return {'postal_agency': 'EPERNAY',
            'date': '1907-08-05T22:30',
            'department': 'MARNE',
            'starred_hour': False,
            'collection': None,
            'mark_type': 'post office',
            'quality': 'good'}

@pytest.fixture
def datestamp_bad_json():
    return {'postal_agency': 'EPERNAY',
            'date': '1907-08-05T22:30',
            'department': 'MARNE',
            'starred_hour': False,
            'collection': None,
            'mark_type': 'datestamp',
            'quality': 'good'}

# ======================================================================================================================
# CONTENT Abstract Class
# ======================================================================================================================
class TestClassContent:
    """test for class Content"""

    def test_instantiation(self):
        """test instantiation of Content Class"""
        assert Content()
        assert Content(None, 0.1) == Content()
        assert Content(True)
        assert Content(True, 0.2) == Content(True)
        assert Content(False, 0.8)

    def test_invalid(self):
        """test invalid instantiation of a non-manual Content without confidence"""
        with pytest.raises(ValueError):
            Content(False)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert Content().get_cls_name() == 'Content'

    def test_isprocessed(self):
        """test isprocessed() method"""
        assert not Content().isprocessed()
        assert Content(True).isprocessed()
        assert Content(False, 0.8).isprocessed()

    def test_to_dict(self):
        """test to_dict() method"""
        assert Content().to_dict() == {'is_manual': None, 'confidence': None}
        assert Content(True).to_dict() == {'is_manual': True, 'confidence': None}
        assert Content(False, 0.8).to_dict() == {'is_manual': False, 'confidence': 0.8}

    def test_from_dict(self):
        """test from_dict() method"""
        assert Content.from_dict({'is_manual': False, 'confidence': 0.75}) == Content(False, 0.75)
        assert Content.from_dict({}) == Content()
        assert Content.from_dict({'is_manual': True}) == Content(True)
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
            assert Content.from_series(Content(True).to_series()) == Content(True)
            assert Content.from_series(Content(False, 0.8).to_series()) == Content(False, 0.8)
        else:
            with pytest.raises(NotImplementedError):
                Content().to_series()

    def test_process_content(self, pred_text):
        test_p = Content()
        assert not test_p.isprocessed()
        assert test_p.process_content(confidence=0.64) == Content(False, 0.64)
        assert not test_p.isprocessed()
        # inplace
        test_p2 = test_p.copy()
        assert not test_p2.isprocessed()
        test_p2.process_content(confidence=0.64, inplace=True)
        assert not test_p.isprocessed()
        assert test_p2.isprocessed()
        assert test_p2 == Content(False, 0.64)


# ======================================================================================================================
# TEXT Abstract Class & subclasses
# ======================================================================================================================
class TestClassText:
    """test for class Text"""

    def test_instantiation(self, manual_text):
        """test instantiation of Text Class"""
        assert Text()
        assert Text(confidence=0.8) == Text()
        assert Text(is_manual=True, ocr_result='test du contenu manuel', orientation=90, keywords=['manuel', 'test'])
        assert Text(is_manual=True,
                    confidence=0.54,
                    ocr_result='test du contenu manuel',
                    orientation=0,
                    keywords=['manuel', 'test']) == manual_text
        assert Text(is_manual=True,
                    confidence=0.54,
                    ocr_result='test du contenu manuel',
                    orientation=Orientation.ZERO,
                    keywords=['manuel', 'test']) == manual_text
        assert Text(False, 0.8, ocr_result='test du contenu prédit', orientation=90)
        # test normalisation de l'angle
        assert Text() == Text(orientation=None)
        assert Text(ocr_result='test', orientation=90) == Text(ocr_result='test', orientation='90')
        assert Text(ocr_result='test', orientation=90) == Text(ocr_result='test', orientation=-270)
        assert Text(ocr_result='test', orientation=90) == Text(ocr_result='test', orientation=90.0)


    def test_invalid(self):
        """test invalid instantiation of a non-manual Text without confidence"""
        with pytest.raises(ValueError):
            Text(False, ocr_result='test du contenu prédit', orientation=90)
        with pytest.raises(ValueError):
            Text(orientation=45)
        with pytest.raises(ValueError):
            Text(orientation=143)
        with pytest.raises(ValueError):
            Text(orientation='lol')
        with pytest.raises(TypeError):
            Text(orientation=[0, 90, 0])


    def test_get_cls_name(self, manual_text, pred_text):
        """test get_cls_name() method"""
        assert Text().get_cls_name() == 'Text'
        assert manual_text.get_cls_name() == 'Text'
        assert pred_text.get_cls_name() == 'Text'

    def test_isprocessed(self, manual_text, pred_text):
        """test isprocessed() method"""
        assert not Text().isprocessed()
        assert manual_text.isprocessed()
        assert pred_text.isprocessed()

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
        assert res_manual == ['manuel', 'test']

        res_pred = []
        for kw in pred_text:
            res_pred.append(kw)
        assert len(res_pred) == 0

    def test_get_keywords(self, pred_text, manual_text):
        assert manual_text.get_keywords() == {'test', 'manuel'}
        assert pred_text.get_keywords() == set()

    def test_to_dict(self, pred_text, manual_text, dict_pred_text, dict_manual_text):
        assert manual_text.to_dict() == dict_manual_text # test manuel
        assert pred_text.to_dict() == dict_pred_text # test prédit

    def test_from_dict(self, pred_text, manual_text, dict_pred_text, dict_manual_text):
        assert Text.from_dict({'is_manual': False, 'confidence': 0.75}) == Text(is_manual=False, confidence=0.75, ocr_result='', keywords=[], orientation=0)  # défaut depuis Content
        assert Text.from_dict(dict_manual_text) == manual_text
        assert Text.from_dict({'is_manual': None, 'confidence': 0.75}) == Text()
        assert Text.from_dict({'is_manual': True, 'confidence': 0.75}) == Text(True)
        assert Text.from_dict(dict_pred_text) == pred_text

    @pytest.mark.parametrize("init_orient, rotation, final_orient", [(90, 0, 90),
                                                                     (0, None, 0),
                                                                     (0, 270, "90"),
                                                                     (0, 270, -270),
                                                                     (90, 90, 0),
                                                                     (90, -90, 180),
                                                                     (90, "-180", 270)])
    def test_rotate(self, init_orient, rotation, final_orient):
        assert Text(orientation=init_orient).rotate(rotation) == Text(orientation=final_orient)

    def test_process_content(self, pred_text):
        test_p = Text()
        assert not test_p.isprocessed()
        assert test_p.process_content(ocr_result='test du contenu prédit', confidence=0.8, orientation=90) == pred_text
        assert not test_p.isprocessed()
        # test inplace :
        assert test_p != pred_text
        test_p.process_content(ocr_result='test du contenu prédit', confidence=0.8, orientation=90, inplace=True)
        assert test_p.isprocessed()
        assert test_p == pred_text

    def test_word_list(self, pred_text, manual_text):
        # TODO : voir comment se débarrasser des caractères spéciaux
        assert pred_text.word_list()._word_list == ['test', 'du', 'contenu', 'prédit']
        assert pred_text._word_list is None  # pas de modification en place ici
        assert manual_text.word_list()._word_list == ['test', 'du', 'contenu', 'manuel']
        assert manual_text._word_list is None  # pas de modification en place ici
        # pour tests :
        test_wl = Text(
            is_manual=True,
            ocr_result="ALLAND'HUY. - L’Eglise.",
            keywords=["église"],
            orientation=0,
        )
        assert test_wl.word_list()._word_list == ['alland', 'huy', 'l', 'eglise']
        assert test_wl.word_list(preprocessing=lambda x: x.lower())._word_list == ['alland\'huy.', '-', 'l’eglise.']
        # test inplace
        test_wl2 = test_wl.copy()
        assert test_wl == test_wl2
        assert test_wl is not test_wl2  # car on a une copie
        test_wl2.word_list(inplace=True)
        assert test_wl2._word_list == ['alland', 'huy', 'l', 'eglise']
        assert test_wl != test_wl2
        assert test_wl._word_list is None


    def test_lemmatize(self):
        test_lem = Text(
            is_manual=True,
            ocr_result="Wilmet, phot., Rethel. - Livoir, édit., Vouziers.",
            keywords=None,
            orientation=90,
        )
        assert test_lem.word_list().lemmatize()._lemmas == ['wilmet', 'phot', 'rethel', 'livoir', 'édit', 'vouziers']
        assert test_lem.lemmatize(preprocessing=lambda x: x)._lemmas == ['wilmet,', 'phot.,', 'rethel.', '-', 'livoir,', 'édit.,', 'vouziers.']
        assert test_lem.lemmatize(warn=False)._lemmas == ['wilmet', 'phot', 'rethel', 'livoir', 'édit', 'vouziers']
        # test inplace
        test_lem2 = test_lem.copy()
        assert test_lem == test_lem2
        assert test_lem is not test_lem2  # car on a une copie
        test_lem2.lemmatize(inplace=True, warn=False)
        assert test_lem2._lemmas == ['wilmet', 'phot', 'rethel', 'livoir', 'édit', 'vouziers']
        assert test_lem != test_lem2
        assert test_lem._word_list is None
        assert test_lem._lemmas is None

    def test_set_keywords(self):
        test_kw = Text(
            is_manual=True,
            ocr_result="ALLAND'HUY. - L’Eglise.",
            orientation=0,
        )
        assert test_kw.word_list().lemmatize().set_keywords({'eglise', 'chateau'}).keywords == {'eglise'}
        assert test_kw.set_keywords({'eglise', 'chateau', 'l’eglise.'}, lemmatizer=lambda x: x, preprocessing=lambda x: x.lower()).keywords == {'l’eglise.'}
        assert test_kw.set_keywords({'eglise', 'chateau'}, warn=False).keywords == {'eglise'}
        # test inplace
        test_kw2 = test_kw.copy()
        assert test_kw == test_kw2
        assert test_kw is not test_kw2  # car on a une copie
        test_kw2.set_keywords({'eglise', 'chateau'}, inplace=True, warn=False)
        assert test_kw2.keywords == {'eglise'}
        assert test_kw != test_kw2
        assert test_kw._word_list is None
        assert test_kw._lemmas is None
        assert test_kw.keywords == set()



# Text Subclasses
# ---------------
class TestClassPrintedText:
    """test for class PrintedText"""

    def test_instantiation(self):
        """test instantiation of PrintedText Class"""
        assert PrintedText()
        assert PrintedText(is_manual=True, ocr_result='test du contenu manuel', orientation=180)
        assert PrintedText(False, 0.8, ocr_result='test du contenu prédit', orientation=90, is_editor=True)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert PrintedText().get_cls_name() == 'PrintedText'


class TestClassHandwrittenText:
    """test for class HandwrittenText"""

    def test_instantiation(self):
        """test instantiation of HandwrittenText Class"""
        assert HandwrittenText()
        assert HandwrittenText(False, 0.8, ocr_result='test du contenu prédit', orientation=90)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert HandwrittenText().get_cls_name() == 'HandwrittenText'


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
        assert Postmark(confidence=0.3) == Postmark()
        assert Postmark(True)
        assert Postmark(True, confidence=0.5) == Postmark(True)
        assert Postmark(False, 0.78)

    def test_invalid(self):
        """test invalid instantiation of a non-manually annotated Postmark without confidence"""
        with pytest.raises(ValueError):
            Postmark(False)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert Postmark().get_cls_name() == 'Postmark'

    def test_isprocessed(self):
        """test isprocessed() method"""
        assert not Postmark().isprocessed()
        assert Postmark(True).isprocessed()
        assert Postmark(False, 0.8).isprocessed()

    def test_to_dict(self, pred_text, manual_text, dict_pred_text, dict_manual_text):
        assert Postmark().to_dict() == {'is_manual': None, 'confidence': None} # test manuel
        assert Postmark(confidence=0.69).to_dict() == {'is_manual': None, 'confidence': None} # test manuel
        assert Postmark(True).to_dict() == {'is_manual': True, 'confidence': None} # test manuel
        assert Postmark(True, confidence=0.71).to_dict() == {'is_manual': True, 'confidence': None} # test manuel
        assert Postmark(False, 0.78).to_dict() == {'is_manual': False, 'confidence': 0.78} # test prédit

    def test_from_dict(self):
        """test from_dict() method"""
        assert Postmark.from_dict({'is_manual': False, 'confidence': 0.75}) == Postmark(False, 0.75)
        assert Postmark.from_dict({}) == Postmark()
        assert Postmark.from_dict({'confidence': 0.87}) == Postmark()
        assert Postmark.from_dict({'is_manual': True}) == Postmark(True)
        assert Postmark.from_dict({'is_manual': True, 'confidence': 0.87}) == Postmark(True)
        assert Postmark.from_dict(Postmark(False, 0.8).to_dict()) == Postmark(False, 0.8)


# Text Subclasses
# ---------------
class TestClassDateISO8601:
    """test for class DateISO8601"""

    def test_instantiation(self):
        """test instantiation of DateISO8601 Class"""
        assert DateISO8601()
        assert DateISO8601("1912-08-30T22:10")
        assert DateISO8601("1908-12-28T12:XX")
        assert str(DateISO8601())
        assert str(DateISO8601("1912-08-30T22:10"))

    def test_eq(self):
        """test __eq__ method"""
        assert DateISO8601() == DateISO8601()
        assert DateISO8601("1912-08-30T22:10") == DateISO8601("1912-08-30T22:10")
        assert DateISO8601("1908-12-28T12:XX") == DateISO8601("1908-12-28T12:XX")
        assert DateISO8601("1912-08-30T22:10") != DateISO8601()
        assert DateISO8601("1912-08-30T22:10") != DateISO8601("1908-12-28T12:XX")
        assert DateISO8601() != DateISO8601("1908-12-28T12:XX")

    def test_compare(self):
        """test __lt__ et __le__ methods"""
        assert not DateISO8601() < DateISO8601()
        assert DateISO8601() <= DateISO8601()
        assert DateISO8601() >= DateISO8601()
        assert DateISO8601("1912-08-30T22:10") < DateISO8601()  # effet de bord
        assert DateISO8601("1912-08-30T22:10") > DateISO8601("1908-12-28T12:XX")


class TestClassDateStamp:
    """test for class DateStamp"""

    def test_instantiation(self):
        """test instantiation of DateStamp Class"""
        assert DateStamp()
        assert DateStamp(True, None,"EPERNAY","1907-08-05T22:30","MARNE",False,None,"post office","good")
        assert DateStamp(False, None,"GIVENY","XXXX-10-17T15:XX","MARNE",False,None,"post office","mediocre")

    def test_invalid(self):
        """test invalid instantiation of a non-manually annotated Postmark without confidence"""
        with pytest.raises(ValueError):
            # anciens type de DateStamp
            DateStamp(True, None, "EPERNAY", "1907-08-05T22:30", "MARNE", False, None, "date stamp", "good")
        with pytest.raises(ValueError):
            # qualité inconnue
            DateStamp(True, None, "EPERNAY", "1907-08-05T22:30", "MARNE", False, None, "post office", "ok")

        # wrong types
        with pytest.raises(TypeError):
            DateStamp(True, None, "EPERNAY", "1907-08-05T22:30", "MARNE", False, None, 1, "good")
        with pytest.raises(TypeError):
            DateStamp(True, None, "EPERNAY", "1907-08-05T22:30", "MARNE", False, None, "post office", 1)
        with pytest.raises(TypeError):
            DateStamp(True, None, "EPERNAY", 1907, "MARNE", False, None, "post office", "good")

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert DateStamp().get_cls_name() == 'DateStamp'

    def test_to_dict(self):
        """test to_dict() method"""
        assert DateStamp().to_dict() == {'is_manual': None, 'postal_agency': None, 'date': 'XXXX-XX-XXTXX:XX',
                                         'department': None, 'starred_hour': False, 'collection': None,
                                         'mark_type': 'post office', 'quality': 'poor'}
        assert DateStamp(True,None,"EPERNAY","1907-08-05T22:30","MARNE",
                         False,None,"post office","good").to_dict() == {'is_manual': True,
                                                                        'postal_agency': 'EPERNAY',
                                                                        'date': '1907-08-05T22:30',
                                                                        'department': 'MARNE',
                                                                        'starred_hour': False,
                                                                        'collection': None,
                                                                        'mark_type': 'post office',
                                                                        'quality': 'good'}

    def test_from_dict(self):
        """test from_dict() method"""
        assert DateStamp() == DateStamp.from_dict({'is_manual': None})
        assert DateStamp() == DateStamp.from_dict(DateStamp().to_dict())
        assert DateStamp(True, None, "EPERNAY", "1907-08-05T22:30",
                         "MARNE", False, None, "post office", "good") == DateStamp.from_dict({'is_manual': True,
                                                                                              'postal_agency': 'EPERNAY',
                                                                                              'date': '1907-08-05T22:30',
                                                                                              'department': 'MARNE',
                                                                                              'starred_hour': False,
                                                                                              'collection': None,
                                                                                              'mark_type': 'post office',
                                                                                              'quality': 'good'})

    def test_to_series(self):
        """test to_series() method"""
        assert DateStamp().to_series().any()
        assert DateStamp(True, None, "EPERNAY", "1907-08-05T22:30", "MARNE",
                         False, None, "post office", "good").to_series().any()

    def test_from_series(self):
        """test from_series() method"""
        assert DateStamp() == DateStamp.from_series(DateStamp().to_series())

    def test_process_content(self, datestamp_json, datestamp_bad_json):
        test_p = DateStamp()
        assert not test_p.isprocessed()
        assert DateStamp().process_content() == DateStamp(is_manual=False)
        assert not test_p.isprocessed()
        # inplace:
        test_p2 = test_p.copy()
        assert not test_p2.isprocessed()
        assert test_p == test_p2
        assert test_p is not test_p2
        test_p2.process_content(datestamp_json, inplace=True)
        assert test_p2.isprocessed()
        assert not test_p.isprocessed()
        assert test_p != test_p2
        assert test_p2 == test_p.process_content(datestamp_json)
        assert not test_p.isprocessed()
        # test des vérifications
        with pytest.raises(ValueError):
            test_p.process_content(datestamp_bad_json)




class TestClassPostageStamp:
    """test for class PostageStamp"""

    def test_instantiation(self):
        """test instantiation of PostageStamp Class"""
        assert PostageStamp()
        assert PostageStamp(False, 0.65, country="France", color='red', price=0.5)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert PostageStamp().get_cls_name() == 'PostageStamp'


class TestClassOtherMark:
    """test for class OtherMark"""

    def test_instantiation(self):
        """test instantiation of OtherMark Class"""
        assert OtherMark()
        assert OtherMark(False, 0.82, is_editor=True)

    def test_get_cls_name(self):
        """test get_cls_name() method"""
        assert OtherMark().get_cls_name() == 'OtherMark'