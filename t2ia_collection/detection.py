from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, List
from enum import Enum
import math

@dataclass
class BoundingBox:
    x: float
    y: float
    w: float
    h: float

    def __eq__(self, other: 'BoundingBox'):
        res = True
        for a, b in zip(self, other):
            res &= math.isclose(a, b, rel_tol=1e-06)
        return res

    def __iter__(self):
        return iter(self.xywhn())

    def rotate(self, angle: float) -> 'BoundingBox':
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