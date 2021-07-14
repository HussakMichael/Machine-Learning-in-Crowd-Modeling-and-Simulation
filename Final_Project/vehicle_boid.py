import pygame as pg

class Vehicle(pg.sprite.Sprite):
    image = pg.Surface((15, 15), pg.SRCALPHA)

    def __init__(self, position, velocity, min_speed, max_speed,
                 max_force, is_predator, is_food):

        super().__init__()
        self.is_food = is_food
        self.is_predator = is_predator
        if self.is_food:
            velocity = 0.
            min_speed = 0.
            max_speed = 0.
            # position = pg.math.Vector2(Vehicle.max_x/2, Vehicle.max_y/2)
        # set limits
        self.min_speed = min_speed
        self.max_speed = max_speed
        # Defines the repulsive force: the higher, the less swarming behavior
        self.max_force = max_force

        # set position
        self.position = pg.Vector2(position)
        self.acceleration = pg.Vector2(0, 0)
        self.velocity = pg.Vector2(velocity)

        self.heading = 0.0

        self.rect = self.image.get_rect(center=self.position)

    def update(self, dt, steering, found_pred, found_food):
        self.acceleration = steering * dt

        # enforce turn limit
        old_speed, old_heading = self.velocity.as_polar()
        if found_pred:
            new_velocity = self.acceleration * dt
            _, new_heading = new_velocity.as_polar()
            # increasing the speed to max if predator is close
            speed = self.max_speed
        elif found_food and not self.is_predator:  # predators are only hunting moving boids
            _, new_heading = steering.as_polar()
            # lowering the speed if close to food source
            speed = old_speed * 0.95
        else:
            new_velocity = self.velocity + self.acceleration * dt
            speed, new_heading = new_velocity.as_polar()

        heading_diff = 180 - (180 - new_heading + old_heading) % 360
        # defines angle of maximum adjustment, in case food source or predator is close, higher angles are allowed
        if abs(heading_diff) > self.max_turn and not found_pred and not found_food:
            if heading_diff > self.max_turn:
                new_heading = old_heading + self.max_turn
            else:
                new_heading = old_heading - self.max_turn

        self.velocity.from_polar((speed, new_heading))

        # enforce speed limit
        speed, self.heading = self.velocity.as_polar()
        if speed < self.min_speed:
            self.velocity.scale_to_length(self.min_speed)
        if speed > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)

        self.position += self.velocity * dt
        self.position = pg.Vector2(self.position[0] % self.max_x,
                                   self.position[1] % self.max_y)

        pg.draw.polygon(Vehicle.image, pg.Color(self.color), [(15, 5), (0, 2), (0, 8)])
        self.image = pg.transform.rotate(Vehicle.image, -self.heading)
        self.rect = self.image.get_rect(center=self.position)

    @staticmethod
    def set_boundary():
        info = pg.display.Info()
        Vehicle.max_x = info.current_w
        Vehicle.max_y = info.current_h
        Vehicle.edges = [0, 0, Vehicle.max_x,
                         Vehicle.max_y]

    def clamp_force(self, force):
        # normalizes force to self.max_force
        if 0 < force.magnitude() > self.max_force and force.length_squared() > 0:
            force.scale_to_length(self.max_force)
        return force

