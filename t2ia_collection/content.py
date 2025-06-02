from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import List, Iterator, Set, Callable, LiteralString, SupportsIndex
from enum import StrEnum, IntEnum
from copy import deepcopy
import re
import math
import warnings
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
    def get_cls_name(cls) -> str:
        """Retourne le nom de la classe"""
        return cls.__name__

    def copy(self):
        """retourne une copie de l'instance"""
        return deepcopy(self)

    # Les tests :
    # -----------
    def isprocessed(self) -> bool:
        """Vérifie si le contenu a été extrait"""
        return self.is_manual is not None

    # pour exporter/importer :
    # ------------------------
    def to_dict(self) -> dict:
        """Renvoie un dictionnaire avec le contenu de la classe"""
        res = {}
        for key, value in self.__dict__.items():
            if key[0] != "_":  # pour ne pas ajouter les attributs privés et protégés
                res[key] = value
        return res

    def _to_full_dict(self) -> dict:
        """Renvoie un dictionnaire avec la totalité du contenu de la classe, sauf les attributs privés ou protégés"""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict) -> "Content":
        """Permet d'instancier la classe à partir d'un dictionnaire"""
        return cls(**data)

    def to_series(self):
        """Renvoie un dataframe pandas avec le contenu de la classe"""
        if importlib.util.find_spec("pandas") is not None:
            import pandas as pd
            return pd.Series(self.to_dict(), name=self.get_cls_name())
        raise NotImplementedError("pandas library is not installed, use 'pip install pandas'")

    @classmethod
    def from_series(cls, data) -> "Content":
        """Permet d'instancier la classe à partir d'une Series pandas"""
        if importlib.util.find_spec("pandas") is not None:
            return cls(**data.to_dict())
        raise NotImplementedError("pandas library is not installed, use 'pip install pandas'")

    # créer n'importe quelle sous-classe
    # ----------------------------------
    @classmethod
    def create_instance(cls, class_name: str | None = None, class_dict: dict | None = None):
        """Permet de créer n'importe quelle sous-classe à partir de son nom (et d'un dict optionnel)"""
        # Récupérer toutes les sous-classes et la classe courante
        def get_all_subclasses(cls):
            """permet de récupérer toutes les sous-classes de la classe courante"""
            subclasses = {cls}  # Inclure la classe courante
            for subclass in cls.__subclasses__():
                subclasses.update(get_all_subclasses(subclass))
            return subclasses

        dict_subcls = {subcls.__name__: subcls for subcls in get_all_subclasses(cls)}

        # Vérifier si le nom de classe existe
        if class_name is None:
            class_name = cls.__name__  # si pas spécifié, on prend le nom de la classe courante
        elif class_name not in dict_subcls.keys():
            raise ValueError(f"Classe {class_name} non trouvée")

        # Instancier la classe avec les attributs du dictionnaire
        res = dict_subcls[class_name]() if class_dict is None else dict_subcls[class_name].from_dict(class_dict)
        return res

    # Les traitements :
    # -----------------
    def process_content(self, confidence: float = 0, inplace: bool = False, *args, **kwargs):
        """Méthode pour les différents traitements des contenus renvoie soit une copie, soit modifie en place"""
        res = self if inplace else self.copy()
        # modifications
        res.is_manual = False
        res.confidence = confidence

        res.__post_init__()  # pour les vérifs
        return None if inplace else res



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

    def __repr__(self) -> str:
        return str(self.value)

    @staticmethod
    def from_input(theta: int | float | str | None) -> "Orientation":
        """Orientation à partir d'un input correspondant à un angle multiple de 90°"""
        # si theta est None == 0
        if theta is None:
            theta = Orientation.ZERO
        # Test de la validité de l'angle
        try:
            # normalisation de l'angle
            theta = float(theta) % 360
            round(theta / 90) * 90 % 360
            theta = Orientation(theta)
        except Exception as e:
            if type(e) is ValueError:
                raise ValueError(f"theta must be an angle in degrees, multiple of 90°")
            elif type(e) is TypeError:
                raise TypeError(f"theta must be a int, str, float, None or {Orientation.__module__}.{Orientation.__name__}")

        return Orientation(theta)


@dataclass
class Text(Content):
    """Sous-classe de contenu pour les textes"""
    ocr_result: str = ""
    _word_list: List[str] | None = field(default=None, repr=False)  # les mots ne seront pas affichés lors de la repr
    _lemmas: List[str] | None = field(default=None, repr=False)  # les lemmes ne seront pas affichés lors de la repr
    keywords: Set[str] | List[str] | None = None
    # angle par lequel l'image doit être rotatée pour que le texte soit dans le bon sens :
    orientation: Orientation | int | float | str | None = 0

    def __post_init__(self):
        super().__post_init__()  # vérifications de la classe Content
        self.keywords = self.keywords or set()
        if isinstance(self.keywords, list):
            self.keywords = set(self.keywords)
        if not isinstance(self.keywords, set):
            raise TypeError("keywords must be a set or None")
        # Test de la validité de l'angle
        if not isinstance(self.orientation, Orientation):
            self.orientation = Orientation.from_input(self.orientation)

    def __contains__(self, word) -> bool:
        """tests de contenance sur ocr"""
        return word in self.ocr_result  # garder aini ou faire sur les mots-clés ?

    def __iter__(self) -> Iterator[str]:
        """iteration sur les mots-clés"""
        return iter(sorted(self.keywords))  # garder ça ou faire key: value (pour dict() par exemple)

    def get_keywords(self) -> Set[str]:
        return self.keywords

    # pour exporter/importer :
    # ------------------------
    def to_dict(self) -> dict:
        res = super().to_dict()
        res['keywords'] = sorted(self.keywords)  # les sets doivent être sous forme de listes pour les json
        return res

    def _to_full_dict(self) -> dict:
        res = super()._to_full_dict()
        res['keywords'] = sorted(self.keywords)  # les sets doivent être sous forme de listes pour les json
        return res

    # Les rotations :
    # ---------------
    def rotate(self, theta: Orientation | int | float | str | None = 90) -> "Text":
        """
        Rotation de l'orientation du texte par rapport à une rotation de l'image
        """
        dict_text = self._to_full_dict()
        # Test de la validité de l'angle
        if not isinstance(theta, Orientation):
            theta = Orientation.from_input(theta)

        # nouvelle orientation
        dict_text["orientation"] = self.orientation.value - theta.value

        return self.__class__.from_dict(dict_text)

    # Les traitements :
    # -----------------
    def process_content(self, ocr_result: str = "", orientation: Orientation | int | float | str | None =  None,
                        confidence: float = 0, inplace: bool = False):
        """Méthode pour les différents traitements des textes renvoie soit une copie, soit modifie en place"""
        res = super().process_content(confidence, inplace=inplace)
        if res is None:  # si la modification se fait en place
            res = self

        # modifications
        res.ocr_result = ocr_result
        res.orientation = orientation

        res.__post_init__()  # pour les vérifs
        return None if inplace else res

    def word_list(self, preprocessing: Callable | None = None, inplace: bool = False,
                  sep: LiteralString | None = None, maxsplit: SupportsIndex =-1):
        """Découpe les résultats d'ocr en liste de mots selon un séparateur, après préprocessing si une fonction est
        spécifiée. Renvoie soit une copie, soit modifie en place. Par défaut, supprime la ponctuation et transforme en
        minuscules."""
        res = self if inplace else self.copy()
        # préprocessing des résultats d'OCR
        if preprocessing is None:
            ocr_result = re.sub(r'[^\w\s]|_', ' ', res.ocr_result).lower()  # supprime la ponctuation
        else:
            ocr_result = preprocessing(res.ocr_result)  # TODO : voir comment se débarrasser des caractères spéciaux
        # découpage
        res._word_list = ocr_result.split(sep=sep, maxsplit=maxsplit)

        res.__post_init__()  # pour les vérifs
        return  None if inplace else res

    def lemmatize(self, lemmatizer: Callable | None = None, preprocessing: Callable | None = None,
                  inplace: bool = False, warn: bool=True, **kwargs):
        """Méthode pour lemmatizer une liste de mots, utilise la liste de mots dans _word_list si déjà générée, sinon
        appelle la méthode word_list()"""
        res = self if inplace else self.copy()

        if self._word_list is None:  # si pas de liste de mots
            if preprocessing is None and warn:
                warnings.warn("the 'word_list' method was not called on current instance before lemmatizing and no "
                              "preprocessing was set. If it's intentional please ignore this warning.")
            res.word_list(preprocessing, inplace=True, **kwargs)  # on modifie l'objet en place
        # TODO : scabreux, demander avant au user de générer la liste de mots ?
        # TODO : et retourner les listes de mots / lemmes si en place ?

        # lemmatisation
        if lemmatizer is None:
            res._lemmas = [word.lower() for word in res._word_list]
        else:
            res._lemmas = lemmatizer(res._word_list)  # TODO : implémenter la lemmatisation

        res.__post_init__()  # pour les vérifs
        return  None if inplace else res

    def set_keywords(self, ref_keywords: Set[str] | None = None, lemmatizer: Callable | None = None,
                     preprocessing: Callable | None = None, inplace: bool = False, warn: bool=True, **kwargs):
        """Méthode pour obtenir l'ensemble des mots clés à partir mots, utilise la liste de mots dans _word_list si déjà générée, sinon
        appelle la méthode word_list()"""
        res = self if inplace else self.copy()

        if self._lemmas is None:  # si pas de liste lemmes
            if lemmatizer is None and warn:
                warnings.warn("the 'lemmatize' method was not called on current instance before keyword recuperation "
                              "and no lemmatizer was set. If it's intentional please ignore this warning.")
            res.lemmatize(lemmatizer, preprocessing, inplace=True, warn=warn, **kwargs)  # on modifie l'objet en place

        # lemmatisation
        if ref_keywords is None:
            ref_keywords = set()
        res.keywords = ref_keywords & set(res._lemmas)

        res.__post_init__()  # pour les vérifs
        return  None if inplace else res




# Text Subclasses
# ---------------

@dataclass
class PrintedText(Text):
    """Subclass of Text for printed text"""
    is_editor: bool = False
    # TODO : autres attributs et méthodes ?

    def set_editor(self, is_editor: bool = False, inplace: bool = False):
        """Permet de spécifier si un text correspond à l'éditeur ou non"""
        res = self if inplace else self.copy()
        res.is_editor = is_editor
        return None if inplace else res


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

    def __repr__(self) -> str:
        return str(self.value)

# Définir l'énumération pour les qualités de DateStamp
class DateStampQuality(StrEnum):
    """Enumération des qualités de tampons"""
    POOR = "poor"
    MEDIOCRE = "mediocre"
    GOOD = "good"

    def __repr__(self) -> str:
        return str(self.value)

# classe pour la date
@dataclass
class DateISO8601:
    date_str: str | None = None
    # TODO : ajouter conversion pour objets datetime ?
    # TODO : ajouter vérification du format ISO 8601

    def __post_init__(self):
        if self.date_str is None:
            self.date_str = "XXXX-XX-XXTXX:XX"

    def __repr__(self) -> str:
        return self.date_str

    def __eq__(self, other: "DateISO8601") -> bool:
        return self.date_str == other.date_str

    def __le__(self, other: "DateISO8601") -> bool:
        return self.date_str <= other.date_str

    def __lt__(self, other: "DateISO8601") -> bool:
        return self.date_str < other.date_str

    # Les tests :
    # -----------
    def isvalid(self):
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

    def to_dict(self) -> dict:
        """Renvoie un dictionnaire avec le contenu de la classe"""
        res = super().to_dict()
        del res['confidence']  # pas utile car on n'obtient pas de confiance en sortie de GPT4o
        res['mark_type'] = self.mark_type.value  # pour accéder à la valeur
        res['quality'] = self.quality.value  # pour accéder à la valeur
        res['date'] = str(self.date)
        return res

    def _to_full_dict(self) -> dict:
        """Renvoie un dictionnaire avec la totalité du contenu de la classe"""
        res = super()._to_full_dict()
        del res['confidence']  # pas utile car on n'obtient pas de confiance en sortie de GPT4o
        res['mark_type'] = self.mark_type.value  # pour accéder à la valeur
        res['quality'] = self.quality.value  # pour accéder à la valeur
        res['date'] = str(self.date)
        return res

    # Les traitements :
    # -----------------
    def process_content(self, datestamp_dict: dict | None = None, inplace: bool = False, *args, **kwargs):
        """Méthode pour le traitement du contenu des tampons d'oblitération"""
        res = self if inplace else self.copy()
        # modifications
        res.is_manual = False
        if datestamp_dict is not None:  # si un dictionnaire est passé
            for key in datestamp_dict.keys():
                res.__dict__[key] = datestamp_dict[key]

        res.__post_init__()  # pour les vérifs
        return None if inplace else res



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

    def set_editor(self, is_editor: bool = False, inplace: bool = False):
        """Permet de spécifier si un text correspond à l'éditeur ou non"""
        res = self if inplace else self.copy()
        res.is_editor = is_editor
        return None if inplace else res
