import random
from collections import Counter
from src.helper import Helper


class AI421(Helper) : 
    """ Main class of the 421 game: generate the rolls and scores
     For now simulate choices with randomness
     Args: 
        params : Dict of the ai params
    """

    def __init__(self):
        self.NB_ROLLS = 3
        self.df_score, self.list_score = self.csv_to_list('Game421/data/score421.csv')
        self.total = 0
        self.multiply = 1
        self.stats = {'keep' : [], 'roll': [], 'roll_score': []}
        
    def roll(self, keeped):       
        dices = []
        dices = dices + keeped
        while len(dices) <= 2 :
            dices.append(random.randint(1,6))
        return dices

    def save_stats(self,keep, roll):
        self.stats['keep'].append(keep)
        self.stats['roll'].append(roll)
        score = self.get_score(roll, 'T3')
        self.stats['roll_score'].append(score)


    def play(self, keeped, strat, turn):
        has_combi = False
        roll = self.roll(keeped)
        nb_rolls = 'T'+str(turn+1)
        if self.check_keep_score(roll, nb_rolls):
            has_combi = True
            return self.get_results(roll, nb_rolls, keeped, has_combi)


        keeped = []
        for i, item in enumerate(strat):
            if item in roll:
                keeped.append(roll[i])

        return self.get_results(roll, 'T'+str(self.NB_ROLLS), keeped, has_combi)

    def get_results(self, roll, nb_rolls, keeped, has_combi):   
        self.save_stats(keeped, roll) 
        score = self.int_str(self.get_score(roll, nb_rolls))
        if isinstance(score, str):
            score = 0
        elif self.multiply != 1:
            score = score * self.multiply
            self.multiply = 1
        self.total += score
        # str_roll = self.int_list_to_str(roll)
        return roll, score, keeped, has_combi

    def get_score(self, roll, nb_rolls):      
        combinaison = self.int_list_to_str(roll)
        df = self.df_score
        df['result'] = df['combinaison'].apply(lambda c: Counter(c) == Counter(combinaison))
        try:
            return df[df['result'] == True][nb_rolls].values[0]
        except (KeyError, IndexError):
            return 1

    def check_keep_score(self, roll, nb_rolls):   
        score = self.int_str(self.get_score(roll, nb_rolls))
        for combinaison in self.list_score:
            if Counter(roll) == Counter(combinaison):
                if isinstance(score, str):
                    if score == 'X2':
                        self.multiply *= 2
                    else: 
                        self.multiply *= -2
                return True
        return False
        
