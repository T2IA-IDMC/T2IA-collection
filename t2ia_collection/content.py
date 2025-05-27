from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
import importlib.util  # pour détecter si d'autres librairies sont installées

# ======================================================================================================================
# CONTENT Abstract Class
# ======================================================================================================================

@dataclass
class Content(ABC):
    """Classe abstraite pour toutes les autres formes de contenu text et marque postales"""
    is_manual: bool = None
    confidence: float = None

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

@dataclass
class Text(Content):
    """Sous-classe de contenu pour les textes"""
    ocr_result: str = ""
    keywords: List[str] = None
    orientation: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.keywords = self.keywords or []

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
class Printed(Text):
    """Subclass of Text for printed text"""
    is_editor: bool = False
    # TODO : autres attributs et méthodes ?


@dataclass
class Handwritten(Text):
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


# Postmark Subclasses
# -------------------



@dataclass
class Stamp(Postmark):
    """Subclass of Postmark for stamps"""
    country: str = ""
    color: str = None
    price: float = None
    # TODO : autres attributs et méthodes ?

@dataclass
class OtherMark(Postmark):
    """Subclass of Postmark for other marks"""
    is_editor: bool = False
    # TODO : autres attributs et méthodes ?