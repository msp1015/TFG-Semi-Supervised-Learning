"""
Algoritmos semi-supervisados.
"""

from .selftraining import SelfTraining
from .cotraining import CoTraining
from .democraticcolearning import DemocraticCoLearning
from .tritraining import TriTraining
from .coforest import CoForest

__all__ = ['SelfTraining', 'CoTraining',
           'DemocraticCoLearning', 'TriTraining', 'CoForest']
