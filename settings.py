"""
Version: 20/12/2022
"""
from getpass import getuser
from pathlib import Path

# the user_name should be the name of your session
user_name = getuser()
paths = {'mcanela': {'basepath': Path('C:/Users/mcanela/Desktop/Python'), # Folder to your Python folder
                     'dlcpath': Path('C:/Users/mcanela/Desktop/Python/Data/dlc'), # Folder with the CSV files from DLC
                     'table_path': Path('C:/Users/mcanela/Desktop/Python/Mice.txt'), # Mice.txt file
                     'figures': Path('C:/Users/mcanela/Desktop/Python/Figures'), # Folder where to save your images
                     'poly': Path('C:/Users/mcanela/Desktop/Python/Data/dat'), # Folder with the DAT files from the polybox
                     'video_path': Path('C:/Users/mcanela/Desktop/Python/Data/videos'), # Folder with the AVI videos from the polybox
                     'lengths_path': Path('C:/Users/mcanela/Desktop/Python/Freezing periods') # Folder where to store your freezing lengths.
                     }}

upaths = paths[user_name]

sites_names={1: 'dHipp', 2: 'vHipp'}
