from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict
from enum import StrEnum, IntEnum
import importlib.util  # pour détecter si d'autres librairies sont installées

# ======================================================================================================================
# CONTENT Abstract Class
# ======================================================================================================================

@dataclass
class Content(ABC):
    """Classe abstraite pour toutes les autres formes de contenu text et marque postales"""
    is_manual: bool | None = None
    confidence: float | None = None

    def __post_init__(self):
        """Vérification de la présence d'un score de confiance si contenu non annoté manuellement"""
        if (self.is_manual is False) and (self.confidence is None):
            raise ValueError("Confidence must be set if the content is not manually set.")
        elif self.is_manual or self.is_manual is None:
            self.confidence = None  # si manuel ou non processé pas de confiance à associer

    @classmethod
    def get_cls_name(cls):
        """Retourne le nom de la classe"""
        return cls.__name__

    def is_processed(self):
        """Vérifie si le contenu a été extrait"""
        return self.is_manual is not None

    def to_dict(self) -> Dict:
        """Renvoie un dictionnaire avec le contenu de la classe"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """Permet d'instancier la classe à partir d'un dictionnaire"""
        return cls(**data)

    def to_series(self):
        """Renvoie un dataframe pandas avec le contenu de la classe"""
        if importlib.util.find_spec("pandas") is not None:
            import pandas as pd
            return pd.Series(self.to_dict(), name=self.get_cls_name())
        raise NotImplementedError("pandas library is not installed, use 'pip install pandas'")

    @classmethod
    def from_series(cls, data):
        """Permet d'instancier la classe à partir d'une Series pandas"""
        if importlib.util.find_spec("pandas") is not None:
            return cls(**data.to_dict())
        raise NotImplementedError("pandas library is not installed, use 'pip install pandas'")


# ======================================================================================================================
# TEXT Abstract Class & subclasses
# ======================================================================================================================
# Définir l'énumération pour les orientations
class Orientation(IntEnum):
    """Enumération des orientations pour les Text, les BoundingBox et les Postcard"""
    ZERO = 0
    NINETY = 90
    ONE_EIGHTY = 180
    TWO_SEVENTY = 270

    def __repr__(self):
        return self.value


@dataclass
class Text(Content):
    """Sous-classe de contenu pour les textes"""
    ocr_result: str = ""
    keywords: List[str] | None = None
    orientation: Orientation | int = 0

    def __post_init__(self):
        super().__post_init__()
        self.keywords = self.keywords or []
        # check orientation
        if isinstance(self.orientation, int):
            try:
                self.orientation = Orientation(self.orientation)
            except ValueError:
                raise ValueError(f"orientation must be one of : {[e.value for e in Orientation]}")
        elif not isinstance(self.orientation, Orientation):
            raise TypeError(f"orientation must be an int or {Orientation.__module__}.{Orientation.__name__}")

    def __contains__(self, word):
        """tests de contenance sur ocr"""
        return word in self.ocr_result  # garder aini ou faire sur les mots-clés ?

    def __iter__(self):
        """iteration sur les mots clés"""
        return iter(self.keywords)  # garder ça ou faire key: value (pour dict() par exemple)

    def set_keywords(self):
        # TODO : implémenter la recherche de mots-clés dans les résultats d'OCR
        pass

    def get_keywords(self):
        return self.keywords

    def word_list(self):
        # TODO : voir comment se débarrasser des caractères spéciaux
        return self.ocr_result.split()


# Text Subclasses
# ---------------

@dataclass
class PrintedText(Text):
    """Subclass of Text for printed text"""
    is_editor: bool = False
    # TODO : autres attributs et méthodes ?


@dataclass
class HandwrittenText(Text):
    """Subclass of Text for handwritten text"""
    pass
    # TODO : autres attributs et méthodes ?

@dataclass
class SceneText(Text):
    """Subclass of Text for scene text"""
    pass
    # TODO : autres attributs et méthodes ?


# ======================================================================================================================
# POSTMARK Abstract Class & subclasses
# ======================================================================================================================

@dataclass
class Postmark(Content):
    """Sous-classe de contenu pour les marqueurs postaux"""
    pass
    # TODO : autres attributs et méthodes ?


# DateStamp Subclasse
# -------------------

# Définir l'énumération pour les types de DateStamp
class DateStampType(StrEnum):
    """Enumération des types de tampons"""
    POST_OFFICE = "post office"
    AUXILIARY_OFFICE = "auxiliary post office"
    LINE_CONVEYOR = "line conveyor"
    DISTRIBUTION_OFFICE = "distribution office"  # TODO : à ajouter dans le dataset

    def __repr__(self):
        return self.value

# Définir l'énumération pour les qualités de DateStamp
class DateStampQuality(StrEnum):
    """Enumération des qualités de tampons"""
    POOR = "poor"
    MEDIOCRE = "mediocre"
    GOOD = "good"

    def __repr__(self):
        return self.value

# classe pour la date
@dataclass
class DateISO8601:
    date_str: str | None = None
    # TODO : ajouter conversion pour objets datetime ?
    # TODO : ajouter vérification du format ISO 8601

    def __post_init__(self):
        if self.date_str is None:
            self.date_str = "XXXX-XX-XXTXX:XX"

    def __repr__(self):
        return self.date_str

    def __eq__(self, other):
        return self.date_str == other.date_str

    def __le__(self, other):
        return self.date_str <= other.date_str

    def __lt__(self, other):
        return self.date_str < other.date_str

    def is_valid(self):
        pass
        # TODO


@dataclass
class DateStamp(Postmark):
    """Subclass of Postmark for obliteration stamps (date stamps)"""
    postal_agency: str | None = None
    date: DateISO8601 | str | None = None
    department:  str | None = None
    starred_hour: bool = False
    collection: str | None = None
    mark_type: DateStampType | str = DateStampType.POST_OFFICE
    quality: DateStampQuality | str = DateStampQuality.POOR

    def __post_init__(self):
        # pas de super().__post_init__() car on n'obtient pas de confiance en sortie de GPT4o
        # check mark_type
        if isinstance(self.mark_type, str):
            try:
                self.mark_type = DateStampType(self.mark_type)
            except ValueError:
                raise ValueError(f"mark_type must be one of : {[e.value for e in DateStampType]}")
        elif not isinstance(self.mark_type, DateStampType):
            raise TypeError(f"mark_type must be a str or {DateStampType.__module__}.{DateStampType.__name__}")

        # check quality
        if isinstance(self.quality, str):
            try:
                self.quality = DateStampQuality(self.quality)
            except ValueError:
                raise ValueError(f"mark_type must be one of : {[e.value for e in DateStampQuality]}")
        elif not isinstance(self.quality, DateStampQuality):
            raise TypeError(f"quality must be a str or {DateStampQuality.__module__}.{DateStampQuality.__name__}")

        # str date to DateISO8601 object
        if isinstance(self.date, str):
            self.date = DateISO8601(self.date)
        elif self.date is None:
            self.date = DateISO8601(self.date)
        elif not isinstance(self.date, DateISO8601):
            # TODO : ajouter conversion pour objets datetime ?
            raise TypeError(f"date must be an str or {DateISO8601.__module__}.{DateISO8601.__name__}")

    def to_dict(self):
        """Renvoie un dictionnaire avec le contenu de la classe"""
        res = self.__dict__
        del res['confidence']  # pas utile car on n'obtient pas de confiance en sortie de GPT4o
        res['mark_type'] = self.mark_type.value  # pour accéder à la valeur
        res['quality'] = self.quality.value  # pour accéder à la valeur
        res['date'] = str(self.date)
        return res


# Other Postmark Subclasses
# -------------------------

@dataclass
class PostageStamp(Postmark):
    """Subclass of Postmark for stamps"""
    country: str = ""
    color: str | None = None
    price: float | None = None
    # TODO : autres attributs et méthodes ?

@dataclass
class OtherMark(Postmark):
    """Subclass of Postmark for other marks"""
    is_editor: bool = False
    # TODO : autres attributs et méthodes ?