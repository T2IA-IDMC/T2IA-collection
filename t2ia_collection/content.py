from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

# ======================================================================================================================
# CONTENT & abstract subclasses
# ======================================================================================================================

@dataclass
class Content(ABC):
    """Classe abstraite pour toutes les autres formes de contenu text et marque postales"""
    is_manual: bool = True
    confidence: float = None

    def __post_init__(self):
        """Vérification de la présence d'un score de confiance si contenu non annoté manuellement"""
        if not self.is_manual and self.confidence is None:
            raise ValueError("Confidence must be set if the content is not manually set.")

    def to_dict(self) -> Dict:
        """Renvoie un dictionnaire avec le contenu de la classe"""
        return {'is_manual': self.is_manual, 'confidence': self.confidence}

    @classmethod
    def from_dict(cls, data):
        """Permet d'instancier la classe à partir d'un dictionnaire"""
        return cls(is_manual=data["is_manual"], confidence=data["confidence"])


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
        return iter(self.keywords)

    def set_keywords(self):
        # TODO : implémenter la recherche de mots-clés dans les résultats d'OCR
        pass

    def get_keywords(self):
        return self.keywords

    def word_list(self):
        # TODO : voir comment se débarrasser des caractères spéciaux
        return self.ocr_result.split()

    def to_dict(self) -> Dict:
        res =  {**super().to_dict(),
                'ocr_result': self.ocr_result,
                'keywords': self.keywords,
                'orientation': self.orientation}
        return res

    @classmethod
    def from_dict(cls, data):
        res = cls(is_manual=data['is_manual'], # depuis la classe Content
                  confidence= data['confidence'], # depuis la classe Content
                  ocr_result=data.get('ocr_result', ""),
                  keywords=data.get('keywords', None),
                  orientation=data.get('orientation', 0))
        return res