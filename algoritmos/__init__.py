"""
Algoritmos semi-supervisados.
"""

from .selftraining import SelfTraining
from .cotraining import CoTraining
from .democraticcolearning import DemocraticCoLearning
from .tritraining import TriTraining
from .coforest import CoForest
from .gbili import Gbili
from .localglobalconsistency import LGC

__all__ = ['SelfTraining', 'CoTraining',
           'DemocraticCoLearning', 'TriTraining', 
           'CoForest', 'Gbili', 'LGC']
