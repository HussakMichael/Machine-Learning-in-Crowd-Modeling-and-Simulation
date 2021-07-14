import pygame as pg
import random
from math import pi, cos, sin, atan2
from vehicle_vicsek import Vehicle


class Bird(Vehicle):

    # CONFIG
    speed = 2.
    noise = 0.5
    radius = 20.
    # set random start orientation
    phi = 2 * pi * random.random()

    def __init__(self, is_predator, is_food):
        Bird.set_boundary()

        # Randomize starting position and velocity
        start_position = pg.math.Vector2(
            random.random() * Bird.max_x,
            random.random() * Bird.max_y)

        self.color = 'white'
        if is_predator:
            self.color = 'red'
        if is_food:
            self.color = 'green'

        super().__init__(start_position,
                         Bird.speed, Bird.phi, is_predator, is_food)

        self.rect = self.image.get_rect(center=self.position)

    def calc_new_phi(self, birds):
        sin_tot = 0.
        cos_tot = 0.

        # loop over neighbours
        neighbours = self.get_neighbors(birds)
        found_predator = False
        found_food = False
        for n in neighbours:
            if n.is_predator:
                found_predator = True
                # escape from predator in opposite direction and leaving iteration
                self.phi = -atan2(n.position[1] - self.position[1], n.position[0] - self.position[0])
                break
            elif n.is_food:
                found_food = True
                # heading towards the food source and leaving iteration
                self.phi = atan2(n.position[1] - self.position[1], n.position[0] - self.position[0])
                break
            else:
                # summing up the neighbours phi
                sin_tot += sin(n.phi)
                cos_tot += cos(n.phi)

        # everything except of food source is updating its Phi
        if not self.is_food:
            # Phi is already calculated while looping over neighbours
            if found_predator or found_food:
                pass
            # in case neighbours found but no predator or food
            elif len(neighbours) > 0:
                self.phi = atan2(sin_tot, cos_tot) + (self.noise / 2.) * (1 - 2. * random.random())
            # in case no neighbours found (not even food or predator)
            else:
                self.phi += (self.noise / 2.) * (1 - 2. * random.random())

    def calc_new_pos(self):
        # calculating delta(position)
        steering = pg.Vector2(self.speed * cos(self.phi), self.speed * sin(self.phi))
        # move
        self.position += steering
        # if bird exits on one side, appearing on the opposite side of the pygame window (instead of edge avoidance)
        self.position = pg.Vector2(self.position[0] % self.max_x,
                                   self.position[1] % self.max_y)

    def get_neighbors(self, birds):
        neighbors = []
        for bird in birds:
            if bird != self:
                dist = self.position.distance_to(bird.position)
                if dist < bird.radius:
                    neighbors.append(bird)
        return neighbors
