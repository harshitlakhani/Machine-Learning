import turtle

import os
from generator.shapes import *


class GeometricShapes:

    __GENERATORS__ = [
        Triangle, Circle, Heptagon, Octagon, Hexagon, Square, Star,
        Nonagon, Pentagon
    ]

    def __init__(self, num_imgs, destination, animation=False):
        """
        num_imgs - 'dict' with keys 'train'
                                value - number of examples per class in the train set
                                keys 'valid'
                                value - number of examples per class in the valid set
                                keys 'test'
                                value - number of examples per class in the test set

        """

        self.__num_imgs__ = num_imgs
        self.destination = destination
        self.animation = animation
    
    def _generate(self, mode):
        destination = os.path.join(self.destination, mode)
        if not os.path.exists(destination):
            os.makedirs(destination)

        turtle.colormode(255)

        # the canvas substract a pixel from the height
        turtle.setup(width=200, height=200)
        turtle.hideturtle()
        turtle.tracer(self.animation)

        container = turtle.Turtle()

        shapes = [
            generator(
                destination, container, mode
            ) for generator in self.__GENERATORS__
        ]

        for _ in range(self.__num_imgs__[mode]):
            for shape in shapes:
                shape.generate()


    def generate_all_splits(self):
        
        self._generate('train')
        self._generate('valid')
        self._generate('test')       