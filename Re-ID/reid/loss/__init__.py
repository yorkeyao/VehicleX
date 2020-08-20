from __future__ import absolute_import

from .label_smooth import LSR_loss
from .triplet import TripletLoss

__all__ = [
    'TripletLoss',
    'LSR_loss'
]
