from numpy import ndarray
from enum import Enum
import numpy

class Direction(Enum):
    RIGHT = "right"
    LEFT = "left"
    UP = "up"
    DOWN = "down"

class RawImage(ndarray):

    def __int__(self):
        super()

    def addWhite(self, direction: Direction, howMany: int):
        if (direction in [Direction.LEFT, Direction.RIGHT]):
            white = numpy.full((self.shape[0], howMany, 3), 255)
            if (direction == direction.LEFT):
                self = numpy.concatenate([white, self], 0)
            else:
                self = numpy.concatenate([self, white], 0)
            return

        if (direction in [Direction.UP, Direction.DOWN]):
            white = numpy.full((howMany, self.shape[1], 3), 255)
            if (direction == direction.UP):
                self = numpy.concatenate([white, self], 1)
            else:
                self = numpy.concatenate([self, white], 1)
        return

class LabeldImage:

    def __int__(self, rawImage: RawImage, label ):
        rawImage = rawImage
        label = label;