from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Iterator, Tuple, Optional, Dict
from enum import StrEnum
from copy import deepcopy
import math
from t2ia_collection.content import *
import importlib.util  # pour détecter si d'autres librairies sont installées

# ======================================================================================================================
# BOUNDING BOXES
# ======================================================================================================================

# Définir les différents formats de coordonnées
class CoordFormat(StrEnum):
    """Enumération des formats de coordonnées"""
    XYWH = 'xywh'
    XYWHN = 'xywhn'
    XYXY = 'xyxy'
    XYXYN = 'xyxyn'
    XXYY = 'xxyy'
    XXYYN = 'xxyyn'

    def __repr__(self) -> str:
        return str(self.value)

    def __contains__(self, item) -> bool:
        return item in self.value

    def is_normalized(self) -> bool:
        """Retourne s'il s'agit de coordonnées normalisées"""
        return self.value[-1] == 'n'


@dataclass
class BoundingBox:
    """
    Class for bounding boxes, with [x, y, w, h] normalized coordinates with (x, y) the central point of the bbox and (w, h) its width and height."""
    x: float
    y: float
    w: float
    h: float

    def __eq__(self, other: "BoundingBox") -> bool:
        return self.isclose(other)

    def __lt__(self, other: "BoundingBox") -> bool:
        # Norme euclidienne au point central de la bbox : sqrt(x^2 + y^2)
        norm_self = math.sqrt(self.x**2 + self.y**2)
        norm_other = math.sqrt(other.x**2 + other.y**2)
        return norm_self < norm_other

    def __le__(self, other: "BoundingBox") -> bool:
        # Norme euclidienne au point central de la bbox : sqrt(x^2 + y^2)
        norm_self = math.sqrt(self.x**2 + self.y**2)
        norm_other = math.sqrt(other.x**2 + other.y**2)
        return norm_self <= norm_other

    def __iter__(self: "BoundingBox") -> Iterator[float]:
        return iter(self.xywhn())

    def copy(self):
        """retourne une copie de l'instance"""
        return deepcopy(self)

    # Les tests :
    # -----------
    def isclose(self, other: "BoundingBox", rtol=1e-05, atol=1e-08) -> bool:
        """Comparaison de BoundingBox avec une certaine tolérance (similaire à celle de numpy)"""
        res = True
        for a, b in zip(self, other):
            res &= math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
        return res

    def isvalid(self, atol=1e-9) -> bool:
        """Vérifie si la bounding box possède des coordonnées valides (bbox inclue dans l'image) avec une certaine tolérance"""
        x, y, w, h = self.xywhn()
        return (w / 2 - atol <= x <= 1 - w / 2 + atol) and (h / 2 - atol <= y <= 1 - h / 2 + atol)

    # Les rotations :
    # ---------------
    def rotate(self, theta: Orientation | int | float | str | None = Orientation.NINETY) -> "BoundingBox":
        """
        Rotation d'un point de coordonnées xy d'un angle theta en degrés, multiple de 90°, par rapport au centre de
        rotation xy_center.
        """
        # Centre de rotation
        cx, cy = 0.5, 0.5
        # Test de la validité de l'angle
        if not isinstance(theta, Orientation):
            theta = Orientation.from_input(theta)
        # Angle négatif en radians pour matcher PIL.Image.rotate()
        theta_radians = - theta.value * math.pi / 180
        # Angle négatif pour matcher PIL.Image.rotate() :
        # Translation vers l'origine
        x_trans = self.x - cx
        y_trans = self.y - cy
        # Rotation
        cos_theta = math.cos(theta_radians)
        sin_theta = math.sin(theta_radians) # Angle négatif pour matcher PIL.Image.rotate()
        x_rot = x_trans * cos_theta - y_trans * sin_theta
        y_rot = x_trans * sin_theta + y_trans * cos_theta
        w_rot = self.w * cos_theta ** 2 + self.h * sin_theta ** 2
        h_rot = self.w * sin_theta ** 2 + self.h * cos_theta ** 2
        return BoundingBox(x_rot + cx, y_rot + cy, w_rot, h_rot)

    # Les différentes coordonnées :
    # -----------------------------
    def xywhn(self) -> Tuple[float, float, float, float]:
        """Coordonnées normalisées avec point central, largeur et hauteur de la bbox"""
        return self.x, self.y, self.w, self.h

    def xywh(self, img_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Coordonnées avec point central, largeur et hauteur de la bbox en fonction de la taille de l'image"""
        x, y, w, h = self.xywhn()
        img_w, img_h = img_size
        return round(x * img_w), round(y * img_h), round(w * img_w), round(h * img_h)

    def xyxyn(self) -> Tuple[float, float, float, float]:
        """Coordonnées [x_min, y_min, x_max, y_max] normalisées avec points haut-gauche (x_min, y_min) et bas-droit (x_max, y_max)"""
        x, y, w, h = self.xywhn()
        x_min = max(x - w/2, 0)
        y_min = max(y - h/2, 0)
        x_max = min(x + w/2, 1)
        y_max = min(y + h/2, 1)
        return x_min, y_min, x_max, y_max

    def xyxy(self, img_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Coordonnées [x_min, y_min, x_max, y_max] avec points haut-gauche (x_min, y_min) et bas-droit (x_max, y_max) en fonction de la taille de l'image"""
        x, y, w, h = self.xywhn()
        img_w, img_h = img_size
        x_min = max(round((x - w/2) * img_w), 0)
        y_min = max(round((y - h/2) * img_h), 0)
        x_max = min(round((x + w/2) * img_w), img_w)
        y_max = min(round((y + h/2) * img_h), img_h)
        return x_min, y_min, x_max, y_max

    def xxyyn(self) -> Tuple[float, float, float, float]:
        """Coordonnées [x_min, x_max, y_min, y_max] normalisées avec points haut-gauche (x_min, y_min) et bas-droit (x_max, y_max)"""
        x_min, y_min, x_max, y_max = self.xyxyn()
        return x_min, x_max, y_min, y_max

    def xxyy(self, img_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Coordonnées [x_min, x_max, y_min, y_max] avec points haut-gauche (x_min, y_min) et bas-droit (x_max, y_max) en fonction de la taille de l'image"""
        x_min, y_min, x_max, y_max = self.xyxy(img_size)
        return x_min, x_max, y_min, y_max

    # pour exporter/importer :
    # ------------------------
    def to_dict(self) -> Dict[str, float]:
        """Renvoie un dictionnaire avec les coordonnées xywhn"""
        return {coord: value for coord, value in zip('xywhn', self.xywhn())}

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "BoundingBox":
        """Permet d'instancier une BoundingBox à partir d'un dictionnaire"""
        return BoundingBox(**data)

    @staticmethod
    def from_coords(
            coords: Sequence[float],
            coord_format: CoordFormat | str = CoordFormat.XYWHN,
            img_size: Optional[Tuple[float, float]] = None) -> "BoundingBox":
        """Renvoie un objet BoundingBox en fonction des coordonnées données en entrée, de leur format, ainsi que des
        dimensions de l'image (largeur, hauteur) si les coordonnées ne sont pas normalisées."""
        # test de la longueur des coordonnées (forcément égal à 4)
        if len(coords) != 4:
            raise ValueError(f"coords must have 4 coordinates, got {len(coords)}")
        # test format de coordonnées
        if isinstance(coord_format, str):
            try:
                coord_format = CoordFormat(coord_format)
            except ValueError:
                raise ValueError(f"coord_format must be one of : {[e.value for e in CoordFormat]}")

        # coordonnées normalisées ou non
        if coord_format.is_normalized():
            img_w, img_h = (1, 1)  # pas de modification
        else:
            if img_size is not None:
                img_w, img_h = img_size
            else:
                raise ValueError(f"If the coordinates are not normalized (one of : {[
                    e.value for e in CoordFormat if 'n' in e
                ]}), you must specify an img_size.")

        # format de coordonnées
        if 'xywh' in coord_format:
            x, y, w, h = coords
        else:
            if 'xyxy' in coord_format:
                x_min, y_min, x_max, y_max = coords
            else:
                x_min, x_max, y_min, y_max = coords
            # calcul des coordonnées
            w = x_max - x_min
            h = y_max - y_min
            x = x_min + w/2
            y = y_min + h/2

        # normalisation
        x, w = [el/img_w for el in [x, w]]
        y, h = [el/img_h for el in [y, h]]

        bbox = BoundingBox(x, y, w, h)
        if not bbox.isvalid():
            raise ValueError(f"The coordinates '{coords}' are invalid, the bbox is outside the image.")

        return bbox

    def to_coords(
            self,
            coord_format: CoordFormat | str = CoordFormat.XYWHN,
            img_size: Optional[Tuple[int, int]] = None) -> Tuple[float, float, float, float]:
        """Renvoie un tuple de coordonnées en fonction du format de coordonnées donné en entrée, ainsi que des
        dimensions de l'image (largeur, hauteur) si les coordonnées ne sont pas normalisées."""
        # test format de coordonnées
        if isinstance(coord_format, str):
            try:
                coord_format = CoordFormat(coord_format)
            except ValueError:
                raise ValueError(f"coord_format must be one of : {[e.value for e in CoordFormat]}")

        # coordonnées normalisées ou non
        if coord_format.is_normalized():
            if 'xywh' in coord_format:
                res = self.xywhn()
            else:
                if 'xyxy' in coord_format:
                    res = self.xyxyn()
                else:
                    res = self.xxyyn()
        else:
            if img_size is not None:
                # format de coordonnées
                if 'xywh' in coord_format:
                    res = self.xywh(img_size)
                else:
                    if 'xyxy' in coord_format:
                        res = self.xyxy(img_size)
                    else:
                        res = self.xxyy(img_size)
            else:
                raise ValueError(f"If the coordinates are not normalized (one of : {[
                    e.value for e in CoordFormat if 'n' in e
                ]}), you must specify an img_size.")

        return res


# ======================================================================================================================
# DETECTIONS
# ======================================================================================================================

@dataclass
class Detection:
    """Classe pour les différentes détections"""
    bbox: BoundingBox
    is_manual: bool = True
    confidence: float | None = None
    content: Content | None = None  # TODO : faire le lien avec les classes Content

    def __post_init__(self):
        """Vérification de la présence d'un score de confiance si contenu non annoté manuellement"""
        if (self.is_manual is False) and self.confidence is None:
            raise ValueError("Confidence must be set if the detection is not manually set.")

    def copy(self):
        """retourne une copie de l'instance"""
        return deepcopy(self)

    # Les tests :
    # -----------
    def isprocessed(self) -> bool:
        """Vérifie si un contenu a été extrait à partir de la détection"""
        return self.content is not None

    # pour exporter/importer :
    # ------------------------
    def to_dict(self) -> dict:
        """Renvoie un dictionnaire avec le contenu de la classe"""
        res = {
            'bbox': self.bbox.to_dict(),
            'is_manual': self.is_manual,
            'confidence': self.confidence,
            'content': self.content.to_dict() if self.isprocessed() else None
        }
        return res

    @staticmethod
    def from_dict(data: dict) -> "Detection":
        """Permet d'instancier la classe à partir d'un dictionnaire"""
        return Detection(bbox=BoundingBox.from_dict(data['bbox']),
                         is_manual=data['is_manual'],
                         confidence=data['confidence'],
                         content=data['content'])


