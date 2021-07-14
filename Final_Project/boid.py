import pygame as pg
from random import uniform
from vehicle_boid import Vehicle


class Boid(Vehicle):

    # CONFIG
    min_speed = .01
    max_speed = .2
    # defines repulsive force: higher values cause less strong flocking
    max_force = 1
    # angle in degree of maximum adjustment
    max_turn = 5
    # Maximum distance to check neighbors or radius
    perception = 40
    # Defines how close the boids are to each other
    crowding = 15
    ###############

    def __init__(self, is_predator, is_food):
        Boid.set_boundary()

        # Randomize starting position and velocity
        start_position = pg.math.Vector2(
            uniform(0, Boid.max_x),
            uniform(0, Boid.max_y))
        start_velocity = pg.math.Vector2(
            uniform(-1, 1) * Boid.max_speed,
            uniform(-1, 1) * Boid.max_speed)
        # Default color of boids is white
        self.color = 'white'
        # Predators are red
        if is_predator:
            self.color = 'red'
        # Food is green
        if is_food:
            self.color = 'green'

        super().__init__(start_position, start_velocity,
                         Boid.min_speed, Boid.max_speed,
                         Boid.max_force, is_predator, is_food)

    def separation(self, boids):
        """
        Dependent only on position: minimum distance to neighbors comes into play
        Separation vector depends on the negative total sum of the boids within a certain distance (crowding constant)
        if a boid is too close or less distant than crowding value, separation becomes higher.
        :param boids:
        :return: pg.Vector
        """
        # initialize an empty vector [0, 0]
        steering = pg.Vector2()
        for boid in boids:
            if not boid.is_predator:
                dist = self.position.distance_to(boid.position)
                if dist < self.crowding:
                    steering -= boid.position - self.position
        steering = self.clamp_force(steering)
        # steering is of type pg.Vector
        return steering

    def alignment(self, boids):
        """
        Here only the speed is considered, you adapt to the speed of the neighbors
        Alignment vector is basically the total sum of both speed and angles of the boids
        :param boids:
        :return:
        """
        steering = pg.Vector2()
        for boid in boids:
            if not boid.is_predator:
                steering += boid.velocity
        steering /= len(boids)
        steering -= self.velocity
        steering = self.clamp_force(steering)
        # seems like /8 is a prioritizing factor
        return steering/8

    def cohesion(self, boids):
        """
        Calculates the mean of the neighbors within the radius and moves in its direction
        Cohesion vector is basically the mean of the boids
        :param boids:
        :return:
        """
        steering = pg.Vector2()
        for boid in boids:
            if not boid.is_predator:
                steering += boid.position
        steering /= len(boids)
        steering -= self.position
        steering = self.clamp_force(steering)
        # seems like /100 is a prioritizing factor
        return steering/100

    def avoid_pred(self, boids):
        """
        Function to avoid predator. If found any predator, adjusts the steering to the direct opposite of it and returns the vector.
        :param boids:
        :return:steering, found_pred
        """
        found_pred = False
        steering = pg.Vector2()
        for boid in boids:
            if boid.is_predator:
                steering -= pg.Vector2(boid.position[0] - self.position[0], boid.position[1] - self.position[1])
                steering = self.clamp_force(steering)
                found_pred = True
                break
        return steering, found_pred

    def food_detected(self, boids):
        """
        Function to move toward food source. If found any food source, adjusts the steering to toward it and returns the vector.
        :param boids:
        :return:steering, found_pred
        """
        found_food = False
        steering = pg.Vector2()
        for boid in boids:
            if boid.is_food and not self.is_predator:
                steering += pg.Vector2(boid.position[0] - self.position[0], boid.position[1] - self.position[1])
                steering = self.clamp_force(steering)
                found_food = True
                break
        return steering, found_food

    def update(self, dt, boids):
        # Calculates the steering vector based on separation, alignment and cohesion value within the radius
        # If there is a predator near
        steering = pg.Vector2()
        # steering += self.avoid_edge()

        neighbors = self.get_neighbors(boids)
        found_pred = False
        found_food = False
        if neighbors:
            separation = self.separation(neighbors)
            alignment = self.alignment(neighbors)
            cohesion = self.cohesion(neighbors)
            pred_avoidance, found_pred = self.avoid_pred(neighbors)
            food_source, found_food = self.food_detected(neighbors)

            if found_pred:
                steering = pred_avoidance
            elif found_food:
                steering = food_source
            else:
                steering += separation + alignment + cohesion

        super().update(dt, steering, found_pred, found_food)

    def get_neighbors(self, boids):
        """
        Function to calculate neighbor boids within given radius
        :param boids:
        :return:neighbors
        """
        neighbors = []
        for boid in boids:
            if boid != self:
                dist = self.position.distance_to(boid.position)
                if dist < self.perception:
                    neighbors.append(boid)
        return neighbors