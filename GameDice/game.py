import random


class AIDice() : 
    """ Main class of the 421 game: generate the rolls and scores
     For now simulate choices with randomness
     Args: 
        params : Dict of the ai params
    """

    def __init__(self):
        self.obj = random.randint(1,6)

    
    def play(self, action):
        reward = 0
        done = False
        roll = [random.randint(1,6), random.randint(1,6), random.randint(1,6)]
        keep = action
        if keep == 6:
            reward = 1000
            done = True
        else: 
            reward = -1000
            done = False
        
        return roll, reward, keep, done

        
        

  
        
