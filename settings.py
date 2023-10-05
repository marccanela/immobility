from getpass import getuser
from pathlib import Path

# the user_name should be the name of your session
user_name = getuser()
paths = {'mcanela': {'basepath': Path('C:/Users/mcanela/Desktop/Python'),
                     'dlcpath': Path('C:/Users/mcanela/Desktop/Python/Data/dlc'),   # CSV files from Deeplabcut
                     # 'fppath': Path('C:/Users/mcanela/Desktop/Python/Data/photometry'),
                     'table_path': Path('C:/Users/mcanela/Desktop/Python/Mice.txt'),   # Mice file
                     'figures': Path('C:/Users/mcanela/Desktop/Python/Figures'),
                     # 'succinate': Path('C:/Users/mcanela/Desktop/Python/Data/Serotonin_Succinate'),
                     # 'cfc': Path('C:/Users/mcanela/Desktop/Python/Data/CFC'),
                     'poly': Path('C:/Users/mcanela/Desktop/Python/Data/dat'),  # Data files from the polybox
                     'video_path': Path('C:/Users/mcanela/Desktop/Python/Data/videos'),
                     'lengths_path': Path('C:/Users/mcanela/Desktop/Python/Freezing periods')
                     }}

upaths = paths[user_name]

sites_names={1: 'dHipp', 2: 'vHipp'}
