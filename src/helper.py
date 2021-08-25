import numpy as np 
import pandas as pd
from pandas.core.frame import DataFrame


class Helper():

    def int_list_to_str(self, roll):
        """String representation of a list with integer

        Args:
            roll (list): current list

        Returns:
            string: string representation of the list
        """        
        return ','.join([str(int) for int in roll])

    def csv_to_list(self, path):
        """transform a csv into a pandas DataFrame and into a comprehensive list

        Args:
            path ([path]): relative or absolute path to .csv

        Returns:
            [type]: [description]
        """        
        df = pd.read_csv(path, sep=';')
        new_list = []
        str_list = [i.split(',') for i in df['combinaison'].to_list()]
        for item in str_list:
            new_list.append([int(i) for i in item])
        return df, new_list

    def int_str(self, value):
        try:
            return int(value)
        except ValueError:
            return str(value)
