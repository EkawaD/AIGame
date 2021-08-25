import numpy as mp
import pandas as pd
from collections import Counter

class AI5000:

    def __init__(self):
        self.dice_left = 5
        self.keep = []

    def rolls(self):
        des = []
        for i in range(self.dice_left):
            des.append(mp.random.randint(1,6))
        return des

    def play(self, point):
        score = [
        {1: 100, 2: 200, 3: 1000, 4: 2000, 5: 5000},
        {1: 0, 2: 0, 3: 200, 4: 400, 5: 1000},
        {1: 0, 2: 0, 3: 300, 4: 600, 5: 1500},
        {1: 0, 2: 0, 3: 400, 4: 800, 5: 2000},
        {1: 50, 2: 100, 3: 500, 4: 1000, 5: 2500},
        {1: 0, 2: 0, 3: 600, 4: 1200, 5: 3000}
        ]
        success = False
        end = False
        main_pleine = False

        result = self.rolls()
        if Counter(result) == Counter([1,2,3,4,5]) or Counter(result) == Counter([2,3,4,5,6]):
            point += 650
            success = True
            main_pleine = True
        else:
            for item in [1,2,3,4,5,6]:
                occur = Counter(result)[item] # nb occurences de 1, 2, 3...
                
                try:   
                    s = score[item-1][occur] # score associé 
                except KeyError:
                    s = 0
                if s > 0:
                    for i in range(occur):
                        self.keep.append(item)
                    if item == 5 and occur == 2:
                        self.keep.remove(item)
                    point += s
                    success = True

        if set([4,2]) <= set(result) and len(self.keep) == 3:
            self.keep.append(4) 
            self.keep.append(2)
            success = True

        if len(self.keep) >= 5:
            self.dice_left = 5
            self.keep = []
            main_pleine = True
        else:
            self.dice_left = 5 - len(self.keep)

        if not success and self.dice_left == 5:
            end = True
            point = -500

        if not success:
            point = 0
            end = True

        print(result, self.dice_left, point)
        return result, point, main_pleine, end
        

    

if __name__ == '__main__':
    ai = AI5000()
    turn = ai.play()
    print(turn)
    # new_snake_training()
    # ai421_training()
            