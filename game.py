import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from snake_game_human import Direction, Point, SnakeGame

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 10000

class SnakeGameAI(SnakeGame):

    def __init__(self, w=20, h=20, fps = SPEED, starting_num_apples = 10):
        self.starting_num_apples = starting_num_apples
        super().__init__(w=BLOCK_SIZE*w, h=BLOCK_SIZE*h)
        self.fps = fps

    def reset(self):
        # init game state
        self._reset_snake()
        
        self.score = 0
        self.food = []
        self.frame_iteration = 0
        try:
            self.game_iteration += 1
        except:
            self.game_iteration = 0
        self._place_food()


    def _place_food(self):
        tot_num_food = max(1,self.starting_num_apples-self.game_iteration//100) if self.game_iteration<=1000 else 1
        if self.game_iteration == 0:
            need_to_create_num_food = tot_num_food
        else:
            need_to_create_num_food = max(tot_num_food - len(self.food), 0)

        def _new_food():    
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            new_food = Point(x, y)
            if new_food in self.snake:
                _new_food()
            return new_food
        
        for i in range(0, need_to_create_num_food):
            self.food.append(_new_food())


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        # if self.frame_iteration % 50:
        #     reward = -1
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head in self.food:
            self.score += 1
            reward = 10
            self.food.remove(self.head)
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.fps)
        # 6. return game over and score
        return reward, game_over, self.score


    def _update_ui_food(self):
        for food in self.food:
            pygame.draw.rect(self.display, RED, 
                             pygame.Rect(
                                 food.x, food.y, 
                                 BLOCK_SIZE, BLOCK_SIZE))


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        super()._move(self.direction)