import pygame as pg
import numpy as np


class Vehicle(pg.sprite.Sprite):

    image = pg.Surface((15, 15), pg.SRCALPHA)

    def __init__(self, position, speed, phi, is_predator, is_food):

        super().__init__()

        self.is_predator = is_predator
        self.is_food = is_food
        # food source is not moving
        if self.is_food:
            speed = 0.0
            phi = 0.0
        # set speed
        self.speed = speed
        self.phi = phi
        self.position = pg.Vector2(position)
        self.rect = self.image.get_rect(center=self.position)

    def update(self):
        # draws the birds/predators/food sources with steering angles
        pg.draw.polygon(Vehicle.image, pg.Color(self.color), [(15, 5), (0, 2), (0, 8)])
        self.image = pg.transform.rotate(Vehicle.image, -np.degrees(self.phi))
        self.rect = self.image.get_rect(center=self.position)

    @staticmethod
    def set_boundary():
        info = pg.display.Info()
        Vehicle.max_x = info.current_w
        Vehicle.max_y = info.current_h
