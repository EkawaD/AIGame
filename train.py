from Game5000.agent import Agent5000
from Game5000.game import AI5000
from Game421.agent import Agent421
from Game421.game import AI421
from GameSnake.agent import AgentSnake
from GameSnake.game import SnakeGameAI
from GameDice.game import AIDice
from GameDice.agent import AgentDice

from src.plot import plot
from src.model import Linear_QNet
import torch

from collections import Counter
import pandas as pd 


class Train():

    def __init__(self, agent, nb_train=500):
        self.agent = agent
        self.nb_train = nb_train
        self.plot_scores = []
        self.plot_mean_scores = []

    def plot_graph(self, score, n_games, total):
        """ Plot """
        total += score
        self.plot_scores.append(score)
        mean_score = total / n_games
        self.plot_mean_scores.append(mean_score)
        plot(self.plot_scores, self.plot_mean_scores)
        return total
    
    def format(self, stats, record):
        keep_total = []
        all_roll = []
        score_roll = []
        for game in stats:
            for keep in game['keep']:
                keep_total += keep
            all_roll += game['roll']
            score_roll += game['roll_score']

        occur_keep = [keep_total.count(1), keep_total.count(2), keep_total.count(3), keep_total.count(4), keep_total.count(5), keep_total.count(6)]
        most_rolled = (Counter(tuple(r) for r in all_roll)).most_common()[0]
        scores = Counter(score_roll)

        text = "Dé gardé le plus souvent : {} \nLancer le plus courant : {} \nScore les plus courants : {}\nRecord pour une partie : {}".format(occur_keep.index(max(occur_keep)) + 1, most_rolled, scores.most_common()[0:5], record)
        return {'text': text, 'keeped': occur_keep, 'most_rolled': most_rolled, 'scores':  scores, 'record': record}

    def trainSnake(self, plot):
        record = 0
        total = 0
        game = SnakeGameAI()
        for i in range(self.nb_train):
            while True:
                record, score, done = self.agent.training(game, record)
                if done and plot:
                    total = self.plot_graph(score, self.agent.n_games, total)
                    break
            
    def train421(self, plot):
        record = 0
        total = 0
        stats = []
        for i in range(self.nb_train):
            ai = AI421()
            record = self.agent.training(ai, record)
            stats.append(ai.stats)
            if plot: 
                print('Game', self.agent.n_games, 'Score', ai.total, 'Record:', record)
                total = self.plot_graph(ai.total, self.agent.n_games, total)
        return stats, record

    def train_421_with_human(self, players, tab_score, sum_score):
        record = 0
        quit = False
        stats = []
        players.append(self.agent)
        ai = AI421()
        while not quit:
            for player in players:
                if not isinstance(player, str):
                    score = player.train_human(ai)
                    print(score)
                    stats.append(ai.stats)
                    player = 'AI'
                    if quit: score = 0
                else:
                    score = input('Score ' + player)
                    if score == 'X':
                        quit = True
                        score = 0
                    else:
                        score = int(score)

                tab_score[player].append(score)
                # print(tab_score)
                sum = sum_score(tab_score)
                print(pd.DataFrame([sum]))

        return sum

    def train_5000(self, plot):
        total = 0
        stats = []
        for i in range(self.nb_train):
            ai = AI5000()
            game = self.agent.training(ai)
            stats.append(game)
            print('Game number ', i)
            if plot: 
                print('Nb de tours: ', game['Nombre tour'])
                total = self.plot_graph(game['Nombre tour'], i+1, total)
        return stats




def train_dice():
    agent = AgentDice(lr=0.1)
    trainer = Train(agent, nb_train=100)
    k = []
    for i in range(trainer.nb_train):
            ai = AIDice()
            reward, keep = trainer.agent.training(ai)
            # print(reward)
            k.append(keep)
    return k

def train_human(players, tab_score, sum_score):
    model_421 = Linear_QNet(3, 256, 43)
    model_421.load_state_dict(torch.load('model/AI421.pth'))
    agent = Agent421(model=model_421, alpha=80)
    trainer = Train(agent)
    return trainer.train_421_with_human(players, tab_score, sum_score)

def snake_training(plot=True):
    model_snake = Linear_QNet(11, 256, 3)
    model_snake.load_state_dict(torch.load('model/snake.pth'))
    agent = AgentSnake(model=model_snake, alpha=10)
    trainer = Train(agent)
    trainer.trainSnake(plot)

def new_snake_training(plot=True):
    agent = AgentSnake()
    trainer = Train(agent)
    trainer.trainSnake(plot)

def ai421_training(plot=True, nb_train=500):
    model_421 = Linear_QNet(3, 256, 43)
    model_421.load_state_dict(torch.load('model/AI421.pth'))
    agent = Agent421(lr=0.1, model=model_421, alpha=40)
    trainer = Train(agent, nb_train=nb_train)
    stats, record = trainer.train421(plot)
    return trainer.format(stats, record)

def new_421_training(plot=True, nb_train=100):
    agent = Agent421(lr=0.1)
    trainer = Train(agent, nb_train=nb_train)
    stats, record = trainer.train421(plot)
    return trainer.format(stats, record)

def new_5000_training(plot=True, nb_train=500):
    agent = Agent5000(lr=0.01)
    trainer = Train(agent, nb_train=nb_train)
    stats = trainer.train_5000(plot)
    return stats

def ai5000_training(plot=True, nb_train=500):
    model_5000 = Linear_QNet(1, 256, 2)
    model_5000.load_state_dict(torch.load('model/AI5000.pth'))
    agent = Agent5000(lr=0.1, model=model_5000, alpha=80)
    trainer = Train(agent, nb_train=nb_train)
    stats = trainer.train_5000(plot)
    return stats

if __name__ == '__main__':
    # snake_training()
    # new_snake_training()
    # ai421_training()
    stats = ai5000_training(nb_train=1000)
    print(stats)
            
    

    