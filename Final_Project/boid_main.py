# Import standard modules.
import argparse
import sys

# Import non-standard modules.
import pygame as pg
from pygame.locals import *

# Import local modules
from boid import Boid

default_boids = 100
window_width, window_height = 512, 512
simulation_ongoing = True


def listener(boids, screen):
    """
    Listens to the mouse during simulation, checks if 'start' or 'pause' button is clicked.
    Pause : Stops the simulation
    Start : Continues the simulation

    Listens to the keyboard buttons that change the simulation:
    up : Adds birds to model
    down : Remove birds from the model
    q : Quits
    """

    # stores the width of the
    # screen into a variable
    width = screen.get_width()

    # stores the height of the
    # screen into a variable
    height = screen.get_height()
    # defining a font
    smallfont = pg.font.SysFont('Corbel', 35)

    # rendering a text written in
    # this font
    text1 = smallfont.render('Pause', True, 'Red')
    text2 = smallfont.render('Start', True, 'Green')
    screen.blit(text1, (width - 100, 20))
    screen.blit(text2, (width - 100, 55))
    info = "Crowding Factor : " + str(Boid.crowding) + " Radius :" + str(Boid.perception)
    text3 = pg.font.SysFont('Sans', 20).render(info, True, 'orange')
    screen.blit(text3, (0, height - 40))

    for event in pg.event.get():
        # closing window
        mouse = pg.mouse.get_pos()
        if event.type == QUIT:
            pg.quit()
            sys.exit(0)
        # if the mouse is clicked
        if event.type == pg.MOUSEBUTTONDOWN:

            # status of simulation
            global simulation_ongoing
            # Pause button location
            if width - 100 <= mouse[0] <= width and 20 <= mouse[1] <= 55:
                simulation_ongoing = False
            # Start button location
            elif width - 100 <= mouse[0] <= width and 55 <= mouse[1] <= 90:
                simulation_ongoing = True
        # in case any key is pressed
        elif event.type == KEYDOWN:
            mods = pg.key.get_mods()
            # pressing 'q' to quit game
            if event.key == pg.K_q:
                # quit
                pg.quit()
                sys.exit(0)
            elif event.key == pg.K_UP:
                # add boids
                add_boids(boids, 10)
            elif event.key == pg.K_DOWN:
                # remove boids
                boids.remove(boids.sprites()[:10])


def draw(screen, background, boids):
    """
    Draws things to the window. Called once per frame.
    """

    # Redraw screen here

    boids.clear(screen, background)
    dirty = boids.draw(screen)

    # Flip the display so that the things we drew actually show up.
    pg.display.update(dirty)


def main():
    # Initialise pygame.
    pg.init()

    pg.event.set_allowed([pg.QUIT, pg.KEYDOWN, pg.KEYUP])

    # Set up the clock to maintain a relatively constant framerate.
    # frames per second
    fps = 60.0
    fpsClock = pg.time.Clock()

    # Set up the window.
    pg.display.set_caption("Boids Model Simulation")
    flags = DOUBLEBUF

    screen = pg.display.set_mode((window_width, window_height), flags)
    screen.set_alpha(None)
    background = pg.Surface(screen.get_size()).convert()
    background.fill(pg.Color('black'))

    boids = pg.sprite.RenderUpdates()

    add_boids(boids, default_boids)
    # add_predator(boids)
    # add_food(boids)

    # Main game loop.
    dt = 1 / fps  # dt is the time since last frame.
    while True:
        listener(boids, screen)
        if simulation_ongoing:
            for b in boids:
                b.update(dt, boids)
            draw(screen, background, boids)
            dt = fpsClock.tick(fps)


# adds boids to the model
def add_boids(boids, num_boids):
    for _ in range(num_boids):
        boids.add(Boid(is_predator=False, is_food=False))


# adds food source to the model
def add_food(boids):
    boids.add(Boid(is_predator=False, is_food=True))


# adds predator to the model
def add_predator(boids):
    boids.add(Boid(is_predator=True, is_food=False))


if __name__ == "__main__":
    main()