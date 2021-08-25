import torch
import random
import numpy as np
from GameSnake.game import  Direction, Point
from src.model import Linear_QNet
from src.agent import Agent

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class AgentSnake(Agent):

    def __init__(self, model=Linear_QNet(11, 256, 3), alpha=80):
        super().__init__(model)
        self.alpha = alpha


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

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
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.alpha - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            print(state0)
            prediction = self.model(state0)
            print(prediction)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def training(self, game, record):
        # get old state
        state_old = self.get_state(game)

        # get move
        final_move = self.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = self.get_state(game)

        # train short memory
        self.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        self.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            self.n_games += 1
            self.train_long_memory()

            if score > record:
                record = score
                self.model.save('snake.pth')

            print('Game', self.n_games, 'Score', score, 'Record:', record)
        return record, score, done
        
