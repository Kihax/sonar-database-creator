import numpy as np


class FuzzyDistanceClassifier:
    """Génère N fonctions d'appartenance floues bornées strictement sur [0.0, 1.0].

    Toute distance d >= 1.0 bascule à 100% dans la dernière catégorie.
    """

    def __init__(self, categories: list[str], min_dist: float = 0.0, max_dist: float = 1.0):
        self.categories = categories
        self.num_classes = len(categories)
        self.min_dist = min_dist
        self.max_dist = max_dist  # Fixé strictement à 1.0

        # Positionnement des 5 centres : [0.0, 0.25, 0.50, 0.75, 1.0]
        self.centers = np.linspace(min_dist, max_dist, self.num_classes)
        self.step = (max_dist - min_dist) / max(1, (self.num_classes - 1))

    @staticmethod
    def _trimbf(x: float, a: float, b: float, c: float) -> float:
        """Fonction d'appartenance triangulaire."""
        return max(min((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)), 0.0)

    @staticmethod
    def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
        """Fonction d'appartenance trapézoïdale."""
        return max(min((x - a) / (b - a + 1e-6), 1.0, (d - x) / (d - c + 1e-6)), 0.0)

    def evaluate(self, d: float) -> dict[str, float]:
        """Calcule le degré d'appartenance [0.0, 1.0] pour chaque catégorie floue."""
        degrees = {}
        for idx, cat in enumerate(self.categories):
            center = self.centers[idx]

            if idx == 0:
                # 1. Premier ensemble : Trapèze plat à gauche (d <= 0)
                right = center + self.step  # 0.25
                deg = self._trapmf(d, -1.0, -0.5, center, right)

            elif idx == self.num_classes - 1:
                # 2. Dernier ensemble : Trapèze plat à droite (d >= 1.0)
                # Monte entre 0.75 et 1.0, puis reste à 1.0 jusqu'à l'infini
                left = center - self.step  # 0.75
                deg = self._trapmf(d, left, center, 100.0, 100.0)

            else:
                # 3. Ensembles intermédiaires : Triangles centrés
                left = center - self.step
                right = center + self.step
                deg = self._trimbf(d, left, center, right)

            degrees[cat] = deg
        return degrees

    def predict(self, d: float) -> dict:
        """Prédit la classe dominante et renvoie le degré de certitude en %."""
        degrees = self.evaluate(d)
        dominant_category = max(degrees, key=degrees.get)
        confidence_degree = degrees[dominant_category]

        return {
            "distance": float(d),
            "dominant_category": dominant_category,
            "confidence": round(confidence_degree * 100, 1),
            "degrees": degrees,
        }