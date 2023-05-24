# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:18:28 2023

@author: Chunhui TU
"""
# make sure that MCTS.py, experiment.py, utils.py and main.py are under the same directory and set a correct work dir
import MCTS
import experiment

# Initialize bonus/punishment points and birth point
mapSize   = [10,10] # map size
nPoint = 33         # Number of randomly penalty/bonus points
randomPoints, rbirthPoint = MCTS.generate_Map(mapSize, nPoint)
randomMarkList = MCTS.generate_Random_Marks(nList=10, ratio=0.5)

state = MCTS.GameState(mapSize, randomPoints, randomMarkList, rbirthPoint)

# 0. calculate the resource usages
result, resources = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_Iterations = 1000, select_method='UCB1', C_p=0.7)

# 1. win rate
results, summary = experiment.win_Rate_Analysis(state, num_sim=20, num_exp=50) 

# 2. diff algorithm compare
result_UCB, summary_UCB = experiment.win_Rate_Analysis(state, num_sim=20, num_exp=50) 
result_UCT, summary_UCT = experiment.win_Rate_Analysis(state, num_sim=20, num_exp=50, select_method='UCT', C_p=0.9) 

# 3. ratio analysis   
mapParams = [mapSize, randomPoints, rbirthPoint]   
ratioAna = experiment.ratio_Effect_Analysis(mapParams, step=0.1, num_sim=20, num_exp=50)  

# 4. diff C_p compare
CpAna = experiment.diff_Cp_Analysis(state, cp_step = 0.1, num_experiment=10)

# 5.1 score confidence interval
experiment.score_Confidence_Interval_Ana(state, num_sim=20, num_exp=200, num_bootstrap=1000)

"""
Confidence Interval (95.0%): [-0.11651500000000001, 0.026807499999999998]

您獲得的置信區間 [-0.116515, 0.0268075] 表示真實總體參數（在本例中為最好孩子的分數）估計處於 95% 置信水平的值範圍。

以下是置信區間中的值的含義：

區間的下限 -0.116515 表示真實總體參數的估計範圍的下限。它表明，在 95% 的置信度下，最好的孩子的真實分數預計大於或等於 -0.116515。

區間的上限 0.0268075 表示真實總體參數的估計範圍上限。它表明，在 95% 的置信度下，最好的孩子的真實分數預計小於或等於 0.0268075。

請記住，置信區間的解釋假設採樣和估計過程已正確執行，並且滿足基本假設（例如數據的獨立性和隨機性）。此外，置信區間的解釋可能會因您所處理的特定上下文和問題而異。

總之，您可以將獲得的置信區間解釋為您有理由相信（在 95% 的置信水平下）有限空間探索中最好的孩子的真實分數所在的值範圍。
"""
# 5.2 diff num_sim
experiment.score_Confidence_Interval_Ana(state, num_sim=50, num_exp=200, num_bootstrap=1000)   
# Confidence Interval (95.0%): [-0.17421, -0.013580000000000026]
