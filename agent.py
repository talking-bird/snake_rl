import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, Conv_QNet, QTrainer
from helper import plot
from icecream import ic
import os
import argparse

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, r=0, is_train=True,savefile=None, model_folder_path = './model'):
        self.n_games = 0
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.r = r
        self.is_train = is_train
        if r==0:
            self.model = Linear_QNet(11, 256, 3)
        else:
            self.model = Conv_QNet(11, 256, 3, self.r)
        if savefile is not None:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(model_folder_path, savefile)
                    )
                )
        if not is_train:
            self.model.eval()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        # tail = game.snake[-1]
        # body = game.snake[len(game.snake)//2]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        # point_2l = Point(head.x - 2*BLOCK_SIZE, head.y)
        # point_2r = Point(head.x + 2*BLOCK_SIZE, head.y)
        # point_2u = Point(head.x, head.y - 2*BLOCK_SIZE)
        # point_2d = Point(head.x, head.y + 2*BLOCK_SIZE)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # overall_state = np.zeros((int(game.w/BLOCK_SIZE+2),int(game.h/BLOCK_SIZE+2)), dtype=int)
        # for i in range(0,overall_state.shape[0]):
        #     for j in range(0, overall_state.shape[1]):
        #         overall_state[i,j] = game.is_collision(Point((i-1)*BLOCK_SIZE, (j-1)*BLOCK_SIZE))
                # print(i, j, i*BLOCK_SIZE, j*BLOCK_SIZE, overall_state[i,j])
        overall_state = np.zeros((2,2*self.r+1,2*self.r+1), dtype=int)
        
        for i in range(-self.r,self.r+1):
            for j in range(-self.r,self.r+1):
                pt = Point(head.x+(i)*BLOCK_SIZE, head.y+(j)*BLOCK_SIZE)
                overall_state[0,i,j] = game.is_collision(pt)
                if overall_state[0,i,j] != 1 and pt in game.snake:
                    overall_state[1,i,j] = 2
                    
                # print(i, j, i*BLOCK_SIZE, j*BLOCK_SIZE, overall_state[i,j])                
        # plt.imshow(overall_state)
        if dir_u:
            pass
        elif dir_r:
            overall_state=np.rot90(overall_state,axes=(1, 2), k=1)
        elif dir_d:
            overall_state=np.rot90(overall_state,axes=(1, 2), k=2)
        elif dir_l:
            overall_state=np.rot90(overall_state,axes=(1, 2), k=3)
        
        proximities = np.array(map(lambda food: ((food.x-head.x)**2+(food.y-head.y)**2)**0.5,
                game.food))
        closest_food_idx = proximities.argmax()
        
        # overall_state = overall_state.flatten()
                
            
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            
            # # Danger straight 2
            # (dir_r and game.is_collision(point_2r)) or 
            # (dir_l and game.is_collision(point_2l)) or 
            # (dir_u and game.is_collision(point_2u)) or 
            # (dir_d and game.is_collision(point_2d)),

            # # Danger right 2
            # (dir_u and game.is_collision(point_2r)) or 
            # (dir_d and game.is_collision(point_2l)) or 
            # (dir_l and game.is_collision(point_2u)) or 
            # (dir_r and game.is_collision(point_2d)),

            # # Danger left 2
            # (dir_d and game.is_collision(point_2r)) or 
            # (dir_u and game.is_collision(point_2l)) or 
            # (dir_r and game.is_collision(point_2u)) or 
            # (dir_l and game.is_collision(point_2d)),        
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # # Tail location
            # tail.x < head.x,  # tail left
            # tail.x > head.x,  # tail right
            # tail.y < head.y,  # tail up
            # tail.y > head.y,   # tail down
            
            # # Body location
            # body.x < head.x,  # body left
            # body.x > head.x,  # body right
            # body.y < head.y,  # body up
            # body.y > head.y,   # body down
                        
            # # Food location 
            game.food[closest_food_idx].x < head.x,  # food left
            game.food[closest_food_idx].x > head.x,  # food right
            game.food[closest_food_idx].y < head.y,  # food up
            game.food[closest_food_idx].y > head.y   # food down
            # head.x - game.food[closest_food_idx].x,
            # head.y - game.food[closest_food_idx].y
            
            ]
        # print(overall_state.shape)
        return np.array(state, dtype=int), overall_state

    def remember(self, state1,state2,  action, reward, next_state1,next_state2,  done):
        self.memory.append((state1,state2, action, reward, next_state1,next_state2, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states1,states2, actions, rewards, next_states1,next_states2, dones = zip(*mini_sample)

        self.trainer.train_step(states1,states2, actions, rewards, next_states1,next_states2, dones)

    def train_short_memory(self, state1,state2, action, reward, next_state1,next_state2, done):
        self.trainer.train_step(state1,state2, action, reward, next_state1,next_state2, done)

    def get_action(self, state1,state2):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon and self.is_train:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # print(state.shape)
            # state, state_overall = state
            if not self.is_train:
                with torch.no_grad():
                    state10 = torch.tensor(state1, dtype=torch.float).unsqueeze(0)
                    state20 = torch.tensor(state2.copy(), dtype=torch.float).unsqueeze(0)
                    # ic(state10.shape, state20.shape)
                    prediction = self.model(state10, state20)
                    move = torch.argmax(prediction).item()
                    final_move[move] = 1
            else:
                state10 = torch.tensor(state1, dtype=torch.float).unsqueeze(0)
                state20 = torch.tensor(state2.copy(), dtype=torch.float).unsqueeze(0)
                # ic(state10.shape, state20.shape)
                prediction = self.model(state10, state20)
                move = torch.argmax(prediction).item()
                final_move[move] = 1

        return final_move


def train(w=20,h=20,fps=10000,starting_num_apples=10,r=0, is_train=True, savefile=None,save_to=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(r, is_train, savefile)
    game = SnakeGameAI(w,h,fps,starting_num_apples)
    while True:
        # get old state
        state_old1,state_old2  = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old1,state_old2)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new1,state_new2 = agent.get_state(game)

        if is_train:
            # train short memory
            agent.train_short_memory(state_old1,state_old2, final_move, reward, state_new1,state_new2, done)
            # remember
            agent.remember(state_old1,state_old2, final_move, reward, state_new1,state_new2, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            if is_train:
                agent.train_long_memory()

            if score > record:
                record = score
                if save_to is not None:
                    agent.model.save(save_to)
                else:
                    agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', dest='w',type=int, default=20,
                    help=" width of game's field in cells")
    parser.add_argument('--height', dest='h',type=int, default=20,
                    help=" height of game's field in cells")
    parser.add_argument('--fps',type=int, default=20,
                    help=" game's fps. high for training, low for testing")
    parser.add_argument('--apples',type=int, default=10,
                    help=" starting number of apples. Set to bigger than 1 for training ")
    parser.add_argument('--train', dest='is_train',action='store_true',
                    help=" train mode ")
    parser.add_argument('--test', dest='is_train',action='store_false',
                    help=" test mode ")
    parser.add_argument('--r', type=int,
                    help=" 'radius' of area aroung snake's which is included in state matrix ")
    parser.add_argument('--savefile',type=str, default=None,
                    help=" load model from save file ")
    parser.add_argument('--saveto',type=str, default=None,
                    help="specify to which file save the model ")
    return parser

if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()
    
    train(args.w,args.h,args.fps,args.apples,
          args.r, args.is_train, args.savefile,
          args.saveto) 