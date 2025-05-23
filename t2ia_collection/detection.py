from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence
from typing import List, Dict, Tuple, Optional, Literal
from enum import Enum
import math

# ======================================================================================================================
# BOUNDING BOXES
# ======================================================================================================================

@dataclass
class BoundingBox:
    """
    Class for bounding boxes, with [x, y, w, h] normalized coordinates with (x, y) the central point of the bbox and (w, h) its width and height."""
    x: float
    y: float
    w: float
    h: float

    def __eq__(self, other: 'BoundingBox'):
        res = True
        for a, b in zip(self, other):
            res &= math.isclose(a, b, rel_tol=1e-06)
        return res

    def __lt__(self, other):
        # Norme euclidienne au point central de la bbox : sqrt(x^2 + y^2)
        norm_self = math.sqrt(self.x**2 + self.y**2)
        norm_other = math.sqrt(other.x**2 + other.y**2)
        return norm_self < norm_other

    def __iter__(self):
        return iter(self.xywhn())

    def rotate(self, angle: float) -> 'BoundingBox':
        #TODO : rotation of bboxes and test
        pass

    def xywhn(self) -> Tuple[float, float, float, float]:
        """Coordonnées normalisées avec point central, largeur et hauteur de la bbox"""
        return self.x, self.y, self.w, self.h

    def xywh(self, img_size: Tuple[float, float]) -> Tuple[float, float, float, float]:
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

    def xyxy(self, img_size: Tuple[float, float]) -> Tuple[float, float, float, float]:
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

    def xxyy(self, img_size: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """Coordonnées [x_min, x_max, y_min, y_max] avec points haut-gauche (x_min, y_min) et bas-droit (x_max, y_max) en fonction de la taille de l'image"""
        x_min, y_min, x_max, y_max = self.xyxy(img_size)
        return x_min, x_max, y_min, y_max

    def is_valid(self, tol=1e-9):
        """Vérifie si la bounding box possède des coordonnées valides (bbox inclue dans l'image) avec une certaine tolérance"""
        x, y, w, h = self.xywhn()
        return (w/2 - tol <= x <= 1 - w/2 + tol) and (h/2 - tol <= y <= 1 - h/2 + tol)


def bbox_from_coord(
        coords: Sequence[float],
        format: Literal['xywh', 'xywhn', 'xyxy', 'xyxyn', 'xxyy', 'xxyyn'] = 'xywhn',
        img_size: Optional[Tuple[float, float]] = None
) -> BoundingBox:
    """Renvoie un objet BoundingBox en fonction des coordonnées données en entrée, de leur format, ainsi que de les dimensions de l'image (largeur, hauteur) si les coordonnées ne sont pas normalisées."""
    # test de la longueur des coordonnées (forcément égal à 4)
    if len(coords) != 4:
        raise ValueError(f"'coords' argument must have 4 coordinates, got {len(coords)}")
    # test format de coordonnées
    list_formats = ['xywh', 'xywhn', 'xyxy', 'xyxyn', 'xxyy', 'xxyyn']
    if format not in list_formats:
        raise ValueError(f"'format' argument must be one of {list_formats}, got '{format}'")

    # coordonnées normalisées ou non
    if format[-1] == 'n':
        img_w, img_h = (1, 1)  # pas de modification
        format = format[:-1]
    else:
        if img_size is not None:
            img_w, img_h = img_size
        else:
            raise ValueError(f"If the coordinates are not normalized ('{format}'), you must specify an 'img_size'.")

    # format de coordonnées
    if format == 'xywh':
        x, y, w, h = coords
    else:
        if format == 'xyxy':
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
    if not bbox.is_valid():
        raise ValueError(f"The coordinates '{coords}' are invalid, the bbox is outside the image.")

    return bbox