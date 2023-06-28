import pygame, random, math, sys, matplotlib.pyplot
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from numpy import savetxt
from numpy import loadtxt


#np.random.seed(random.randint(100, 1000))


pygame.init()

set_seed = False
display = True
graph = False
non_adjust_speed = 50
fps = 60
simulation_speed = 1
draw_circles = True
draw_lines = True

stop = 5000/simulation_speed
modified = False

WIN_W = 960
WIN_H = 540
WHITE = (255, 255, 255)
PINK = (255, 150, 150)
RED = (255, 0, 0)
BLUE = (125, 125, 255)
BLACK = (0, 0, 0)

character_size = 3
character_speed = 300*simulation_speed/fps
character_avoidance = 100*simulation_speed/fps
num_characters = 1000
node_reassign_chance = 0.005*simulation_speed/fps
node_assign_chance = 0.7

node_size = 8
node_radius = 150
node_enabled = 5*fps/simulation_speed
node_disabled = 0*fps/simulation_speed
node_attraction = 40*simulation_speed/fps
num_nodes = 1
start_in_node = True

incubation_time = 2*fps/simulation_speed
recovery_time = 10*fps/simulation_speed
infection_radius = 10
infection_rate = 0.078 #0.07
infection_cooldown = 0.1*fps/simulation_speed
immunity_multiplier = 0.1
death_chance = 0.003
infected_slowdown = 0.5

quarantine_chance = 0
mask_chance = 0
mask_effectiveness = 1

mutation_variance = 0.2
mutation = False
v_pressed = False

popt = 0

def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))


class Character(pygame.sprite.Sprite):
    def __init__(self, pos, infected_chance=0, node=None, variant=None, mask=1):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((character_size, character_size))
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.x = self.rect.x
        self.y = self.rect.y
        self.image.fill(BLUE)
        self.immunity_dict = {}
        if np.random.rand() < infected_chance:
            self.status = 1
            self.timer = 0
        else:
            self.status = 0
        #0 = susceptible
        #1 = incubation
        #2 = infectious
        self.cooldown = 0
        self.timer = 0
        self.node = node
        self.moved = False
        self.variant = variant
        self.quarantined = False
        self.mask = mask
        self.r = 0
        self.prev_x = self.x
        self.prev_y = self.y

    def update(self, character_group, mutation, screen, variant_group, node_group):
        if draw_lines:
            self.prev_x = self.x
            self.prev_y = self.y
        if (self.status == 2 or self.status == 1) and self.cooldown <= 0:
            if draw_circles and display and not self.quarantined:
                pygame.draw.circle(screen, RED, self.rect.center, infection_radius, width=1)
            for c in character_group:
                if c.status == 0 and math.dist(self.rect.center, c.rect.center) <= self.variant.infection_radius and not c.moved and not self.quarantined and self.status == 2:
                    vector = pygame.math.Vector2(x=c.rect.x - self.rect.x, y=c.rect.y - self.rect.y)
                    if vector.x or vector.y:
                        vector = pygame.math.Vector2.normalize(vector) * character_avoidance
                    else:
                        vector = pygame.math.Vector2.normalize(pygame.math.Vector2(x=(np.random.rand() * 2 - 1), y=(np.random.rand() * 2 - 1))) * character_avoidance
                    c.x += vector.x
                    c.y += vector.y
                    c.moved = True
                    if self.variant in c.immunity_dict:
                        immunity = c.immunity_dict[self.variant]
                    else:
                        immunity = 0

                    if np.random.rand() < self.variant.infection_rate*immunity_multiplier**immunity * c.mask * self.mask:
                        c.status = 1
                        c.timer = incubation_time
                        c.image.fill(PINK)
                        self.r += 1
                        c.r = 0
                        if mutation:
                            v = Variant(parent=self.variant)
                            variant_group.add(v)
                            c.variant = v
                            mutation = False
                        else:
                            c.variant = self.variant
                        if c.variant in c.immunity_dict:
                            c.immunity_dict[self.variant] += 1
            if np.random.rand() < death_chance * simulation_speed:
                self.kill()
            self.cooldown = self.variant.infection_cooldown
            if self.variant not in self.immunity_dict:
                self.immunity_dict[self.variant] = 1
        elif self.status == 1 and self.timer <= 0:
            r = np.random.rand()
            #print(r)
            if r < quarantine_chance:
                #print("quarantine")
                self.quarantined = True
            self.status = 2
            self.cooldown = self.variant.infection_cooldown
            self.timer = self.variant.recovery_time
            self.image.fill(RED)
        elif self.status == 2 and self.timer <= 0:
            self.status = 0
            self.cooldown = 0
            self.r = 0
            self.quarantined = False
            self.immunity_dict[self.variant] += 1
            self.image.fill((125 * 0.8**self.immunity_dict[self.variant], 125 * 0.8**self.immunity_dict[self.variant], 255  * 0.8**self.immunity_dict[self.variant]))
        if self.timer > 0:
            self.timer -= 1
        if self.cooldown > 0:
            self.cooldown -= 1
        if not self.quarantined:
            if self.status == 2:
                self.x += (np.random.rand()*2-1) * character_speed * self.variant.infected_slowdown
                self.y += (np.random.rand()*2-1) * character_speed * self.variant.infected_slowdown
            else:
                self.x += (np.random.rand()*2-1) * character_speed
                self.y += (np.random.rand()*2-1) * character_speed
            if self.node and self.node.enabled:
                vector = pygame.math.Vector2(x=self.node.rect.centerx - self.rect.centerx, y=self.node.rect.centery - self.rect.centery)
                if vector.magnitude() > node_radius:
                    vector = pygame.math.Vector2.normalize(vector) * node_attraction #/ (math.dist(self.node.rect.center, self.rect.center))
                    if self.status == 2:
                        self.x += vector.x * self.variant.infected_slowdown
                        self.y += vector.y * self.variant.infected_slowdown
                    else:
                        self.x += vector.x
                        self.y += vector.y
        if np.random.rand() < node_reassign_chance:
            if np.random.rand() < node_assign_chance:
                self.node = random.choice(node_group.sprites())
            else:
                self.node = None
        if self.x < 0:
            self.x = 0
        if self.x > WIN_W - self.rect.width:
            self.x = WIN_W - self.rect.width
        if self.y < 0:
            self.y = 0
        if self.y > WIN_H - self.rect.height:
            self.y = WIN_H - self.rect.height
            #pygame.draw.line(screen, RED, (self.prev_x, self.prev_y),
             #                (self.x, self.y))

        self.rect.x = self.x
        self.rect.y = self.y
        self.moved = False
        if draw_lines:
            length = 1
            pygame.draw.line(screen, BLACK, (self.x, self.y),
                             ((length + 1) * self.x - length * self.prev_x, (length + 1) * self.y - length * self.prev_y))


class Node(pygame.sprite.Sprite):
    def __init__(self, pos, enabled=False):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((node_size, node_size))
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.x = self.rect.x
        self.y = self.rect.y
        self.image.fill(BLACK)
        self.enabled = enabled
        if enabled:
            self.timer = node_enabled
        else:
            self.timer = node_disabled

    def update(self):
        self.timer -= 1
        if self.timer <= 0:
            if self.enabled:
                self.timer = node_disabled
            else:
                self.timer = node_enabled
            self.enabled = not self.enabled


class Variant(pygame.sprite.Sprite):
    def __init__(self, parent=None):
        pygame.sprite.Sprite.__init__(self)
        if parent:
            self.incubation_time = parent.incubation_time * (np.random.rand()*mutation_variance*2 + 1 - mutation_variance)
            self.recovery_time = parent.recovery_time * (np.random.rand()*mutation_variance*2 + 1 - mutation_variance)
            self.infection_radius = parent.infection_radius * (np.random.rand()*mutation_variance*2 + 1 - mutation_variance)
            self.infection_rate = parent.infection_rate * (np.random.rand()*mutation_variance*2 + 1 - mutation_variance)
            self.infection_cooldown = parent.infection_cooldown * (np.random.rand()*mutation_variance*2 + 1 - mutation_variance)
            self.death_chance = parent.death_chance * (np.random.rand()*mutation_variance*2 + 1 - mutation_variance)
            self.infected_slowdown = parent.infected_slowdown * (np.random.rand()*mutation_variance*2 + 1 - mutation_variance)

        else:
            self.incubation_time = incubation_time
            self.recovery_time = recovery_time
            self.infection_radius = infection_radius
            self.infection_rate = infection_rate
            self.infection_cooldown = infection_cooldown
            self.death_chance = death_chance
            self.infected_slowdown = infected_slowdown


clock = pygame.time.Clock()

def simulation():
    if set_seed:
        np.random.seed(999)
    else:
        np.random.seed(random.randint(100,1000))
    run = True
    screen = pygame.display.set_mode(size=(WIN_W, WIN_H))
    character_group = pygame.sprite.Group()
    node_group = pygame.sprite.Group()
    variant_group = pygame.sprite.Group()
    global popt
    global infection_rate
    global v_pressed
    global mutation
    immune = 0
    infected = 0
    incubation = 0
    alive = 0
    dead = 0
    data = np.zeros((1, 7))
    print(num_nodes)
    if num_nodes == 1:
        n = Node((WIN_W*0.5, 0.5 * WIN_H))
        node_group.add(n)
    elif num_nodes == 2:
        n = Node((WIN_W*0.2, 0.5 * WIN_H))
        node_group.add(n)
        n = Node((WIN_W*0.8, 0.5 * WIN_H))
        node_group.add(n)
    else:
        for i in range(num_nodes):
            n = Node(((i + 1)/(num_nodes + 1) * WIN_W, 0.5 * WIN_H))
            node_group.add(n)

    v = Variant()
    variant_group.add(v)
    if np.random.rand() < node_assign_chance or start_in_node:
        c = Character((np.random.rand() * WIN_W, np.random.rand() * WIN_H), infected_chance=1,
                      node=random.choice(node_group.sprites()), variant=v)

        t = 2 * math.pi * np.random.rand()
        u = np.random.rand() + np.random.rand()
        if u > 1:
            r = 2 - u
        else:
            r = u
        c.x = c.node.x + r * math.cos(t) * node_radius
        c.y = c.node.y + r * math.sin(t) * node_radius
    else:
        c = Character((np.random.rand() * WIN_W, np.random.rand() * WIN_H), infected_chance=1, node=None, variant=v)
    character_group.add(c)
    for i in range(num_characters - 1):
        if np.random.rand() < node_assign_chance:
            c = Character((np.random.rand() * WIN_W, np.random.rand() * WIN_H), infected_chance=0,
                          node=random.choice(node_group.sprites()))
            t = 2 * math.pi * np.random.rand()
            u = np.random.rand() + np.random.rand()
            if u > 1:
                r = 2 - u
            else:
                r = u
            c.x = c.node.x + r * math.cos(t) * node_radius
            c.y = c.node.y + r * math.sin(t) * node_radius
        else:
            c = Character((np.random.rand() * WIN_W, np.random.rand() * WIN_H), infected_chance=0, node=None)
        character_group.add(c)
        if np.random.rand() < mask_chance:
            c.mask = mask_effectiveness

    while run:
        pygame.display.set_caption(str(np.shape(data)[0]))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if pygame.key.get_pressed()[pygame.K_SPACE]:
                infected_arr, incubation_arr, immune_arr, alive_arr, dead_arr, r_arr, immunity_arr = data.T
                x = np.arange(len(stop))
                #x_axis = np.zeros(infected_arr.size)
                # axs[0].plot(x, infected_arr, 'r')
                # axs[1].plot(x, np.sum(infected_arr, incubation_arr, immune_arr, dead_arr), 'b')
                # axs[2].plot(x, dead_arr, 'k')
                # axs[3].plot(x, immunity_arr, 'b')
                # axs[4].plot(x, r_arr, 'r')
                # plot
                # fig, axs = matplotlib.pyplot.subplots(3, 3)
                # for ax in axs.flat:
                #     ax.set(xlim=(0, infected_arr.size), ylim=(0, num_characters))
                #
                # axs[0, 0].plot(x, np.subtract(alive_arr, sum((infected_arr, incubation_arr, immune_arr))))
                # axs[0, 1].plot(x, incubation_arr, color=(0.9, 0.4, 0.4))
                # axs[0, 2].plot(x, infected_arr, 'r')
                # axs[1, 0].plot(x, immune_arr, color=(0.5, 0.5, 0.5))
                # axs[1, 1].plot(x, dead_arr, color='k')
                #
                # axs[1, 2].plot(x, infected_arr, 'r', linewidth=1)
                # axs[1, 2].fill_between(x, infected_arr, x_axis, color=(0.9, 0.1, 0.1))
                # incubation_line = np.add(incubation_arr, infected_arr)
                # axs[1, 2].plot(x, incubation_line, color=(0.9, 0.4, 0.4), linewidth=1)
                # axs[1, 2].fill_between(x, incubation_line, infected_arr, color=(0.9, 0.6, 0.6))
                # dead_line = np.add(dead_arr, incubation_line)
                # axs[1, 2].plot(x, dead_line, 'k', linewidth=1)
                # axs[1, 2].fill_between(x, dead_line, incubation_line, color=(0, 0, 0))
                # immune_line = np.add(dead_line, immune_arr)
                # axs[1, 2].plot(x, immune_line, color=(0.5, 0.5, 0.5), linewidth=1)
                # axs[1, 2].fill_between(x, immune_line, dead_line, color=(0.6, 0.6, 0.6))
                # axs[2, 0].plot(x, immune_line, color=(0.5, 0.5, 0.5), linewidth=1)
                # axs[2, 1].set(ylim=(0, 20))
                # axs[2, 1].plot(x, r_arr, color='r')
                # matplotlib.pyplot.show()
                # cum_cases = immune_line
                # cum_cases = (cum_cases - cum_cases.min()) / (cum_cases.max() - cum_cases.min())
                # cum_cases = cum_cases[:np.argmax(cum_cases > 0.5) * 2]
                # popt, pcov = curve_fit(fsigmoid, np.linspace(-3, 3, len(cum_cases)), cum_cases)
                # sample_line = fsigmoid(np.linspace(-3, 3, len(cum_cases)), popt[0], popt[1])
                # plt.scatter(np.linspace(-3, 3, len(cum_cases)), cum_cases, s=.1)
                # plt.plot(np.linspace(-3, 3, len(cum_cases)), sample_line)
                # plt.show()

                run = False
                #return popt[0]
                break

            if pygame.key.get_pressed()[pygame.K_v] and not v_pressed:
                mutation = True
                v_pressed = True
            if not pygame.key.get_pressed()[pygame.K_v] and v_pressed:
                v_pressed = False

        if display:
            screen.fill(WHITE)
        for n in node_group:
            n.update()
        if 0 < stop <= np.shape(data)[0]:
            #infected_arr, incubation_arr, immune_arr, alive_arr, dead_arr, r_arr, immunity_arr = data.T

            run = False
            #print("Max infected:")
            #print(immune_line.max())
            #print("progression time:")
            #print(cum_cases.shape[0])
            return data
            break

        if np.shape(data)[0] == 1000 and np.max(data.T[0]) < 20:
            filler_arr = np.zeros((5000, 7))
            return filler_arr
            break

        infected = 0
        incubation = 0
        immune = 0
        alive = 0
        mask = 0

        immunity = 0
        r_num = 0
        r_sum_temp = 0
        for c in character_group:
            c.update(character_group, mutation, screen, variant_group, node_group)
            immunity += sum(c.immunity_dict.values())/num_characters
            if c.status == 2:
                r_num += 1
                r_sum_temp += c.r
                infected += 1
                immunity += 1/num_characters
            elif c.status == 1:
                incubation += 1
                r_num += 1
                r_sum_temp += c.r
            else:
                recovered = False
                for i in c.immunity_dict:
                    if not recovered and c.immunity_dict[i]:
                        recovered = True
                if recovered:
                    immune += 1
            alive += 1
        if r_num == 0:
            r_num = 1
        r_sum = r_sum_temp / r_num

        dead = num_characters - alive
        data = np.append(data, [[infected, incubation, immune, alive, dead, r_sum, immunity]], axis=0)

        if display:
            for n in node_group:
                if n.enabled:
                    screen.blit(n.image, n.rect)
            for c in character_group:
                screen.blit(c.image, c.rect)
            pygame.display.flip()

        clock.tick(fps*non_adjust_speed)
    pygame.display.quit()

mask_chance = 0
quarantine_chance = 0
tests = 5 #5
trials = 5
num_nodes = 2
node_radius = 120
test_values = [0.001, 0.0005, 0.0003, 0.0002, 0.0001]
data = np.zeros((1, 5000, 7, trials))
test_arrays = []
for a in range(tests):
    node_reassign_chance = test_values[a]
    for i in range(trials):
        sim = simulation()

        print(type(sim))
        print(sim.shape)
        print(sim[sim.shape[0] - 1][2])
        while sim[sim.shape[0] - 1][2] < 20:
            print("dud, redo")
            sim = simulation()
        test_arrays.append(sim)
        print("Test " + str(a + 1) + ", trial " + str(i + 1) + " complete")
    test_data = np.stack(test_arrays, axis=-1)
    test_arrays = []
    print(test_data.shape)
    print("Testing condition " + str(a + 1) + " complete")
    data = np.vstack((data, test_data[np.newaxis, :, :, :]))
    #data = np.append(data, sum_data, axis=1)
    print(data.shape)

np.save('MigrationChance.npy', data, allow_pickle=False)
print("data saved")
    #print(sum_data.T.shape)
    #infected_arr, incubation_arr, immune_arr, alive_arr, dead_arr, r_arr, immunity_arr = sum_data.T
    #print(infected_arr)
    #axs[0].plot(x, infected_arr, 'r')
    #print(type(infected_arr))
    #print(np.sum((infected_arr, incubation_arr, immune_arr, dead_arr), axis=0).shape)
    #print(type(np.sum((infected_arr, incubation_arr, immune_arr, dead_arr), axis=0)))

    # axs[2].plot(x, dead_arr, 'k')
    # axs[3].plot(x, immunity_arr, 'b')
    # axs[4].plot(x, r_arr, 'r')


#matplotlib.pyplot.show()