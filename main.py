# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:18:28 2023

@author: Chunhui TU

---------------------------
FILE DESCRIPTION:
---------------------------
Execution file, including executing various experiments
"""
# make sure that MCTS.py, experiment.py, utils.py and main.py are under the same directory and set a correct work dir
import MCTS
import experiment

##################################
###    initialization phase    ###
##################################

# Initialize bonus/penalty points and birth point
mapSize = [10,10]    # map size
nPoint  = 33         # Number of randomly bonus/penalty points

# generate randomPoints, rbirthPoint, randomMarkList
randomPoints, rbirthPoint = MCTS.generate_Map(mapSize, nPoint)
randomMarkList = MCTS.generate_Random_Marks(nList=10, ratio=0.5, repeat=True)
# Initialize state
state = MCTS.GameState(mapSize, randomPoints, randomMarkList, rbirthPoint)
# visualize the initial map
experiment.plot_Scatter_Initial(mapSize, randomPoints, rbirthPoint, saveName='Initial_Map')


##################################
###     experimental phase     ###
##################################

### 1. analysis the resource usages, number of victory and time consumption
winRateResourceAna = experiment.win_Rate_Resource_Analysis(state, min_sim=100, max_sim=2000, num_exp=5, step=100)

    
### 2. win rate analysis
result_winRate, summary_winRate = experiment.win_Rate_Analysis(state, num_sim=30, num_exp=50) 


### 3. diff algorithm compare
result_UCB, summary_UCB = result_winRate, summary_winRate # sama as above
result_UCT, summary_UCT = experiment.win_Rate_Analysis(state, num_sim=30, num_exp=50, select_method='UCT', C_p=1/2**0.5) 


### 4. ratio analysis   
mapParams = [mapSize, randomPoints, rbirthPoint]   
ratioAna = experiment.ratio_Effect_Analysis(mapParams, step=0.1, num_sim=20, num_exp=20)  


### 5. diff number of randomly bonus/penalty points
rbirthPoint = (0,0) # start at position [0,0]
mapParams = [mapSize, rbirthPoint]
nPointAna = experiment.diff_nPoint_Analysis(mapParams, step=10, ratio=0.5, num_sim=20, num_exp=20)


### 6. diff C_p compare
CpAna = experiment.diff_Cp_Analysis(state, cp_step = 0.1, num_exp=50)

################################### END #######################################
