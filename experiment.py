# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:00:01 2023

@author: Chunhui TU
"""

import MCTS
import pandas as pd
import matplotlib.pyplot as plt



def win_Rate_Analysis(state, num_experiment=10, select_method = 'UCB1', C_p = 0.9):
    # get birth point
    birthPoint = state.currentPoint
    
    # create empty dataframe to store experiment results
    columns_name = ['best child', 'victory', 'failure', 'score']
    results = pd.DataFrame(columns = columns_name)

    # repeat experiment
    for i in range(num_experiment):
        result = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_Iterations = 100, select_method=select_method, C_p = C_p)
        
        # analysis result
        bestChild = result.currentPoint
        score = result.score
        if score >= 1:
            victory = 1
        else:
            victory = 0
        if score <= -0.5:
            failure = 1
        else:
            failure = 0
        result_tmp = [bestChild, victory, failure, score]
        results.loc[len(results)] = result_tmp
    
    # summarize experiment results
    # create another empty dataframe to summarize experiment results
    columns_name = ['birth point', 'best child', 'number of choices', 'Number of victories',
                    'Number of failures', 'max score', 'min score', 'average score']
    winRateAna = pd.DataFrame(columns = columns_name)
    
    bestChilds = set(results['best child'])
    # analyze each best child
    for child in bestChilds:
        subData = results[results['best child'] == child]
        # summary
        numChoice = subData.shape[0]
        numVic    = subData['victory'].sum()
        numFail   = subData['failure'].sum()
        max_score = subData['score'].max()
        min_score = subData['score'].min()
        avg_score = subData['score'].mean()
        
        summary = [birthPoint, child, numChoice, numVic, numFail, max_score, min_score, avg_score]
        
        winRateAna.loc[len(winRateAna)] = summary
        
    # plot
    labels_bn = ['best child', 'number of choices', 'The next choice and the corresponding number of chosen']
    plotBar(df = winRateAna, x='best child', y='number of choices', labels = labels_bn)
    
    labels_ba = ['best child', 'average score', 'The next choice and the corresponding average score']
    plotBar(df = winRateAna, x='best child', y='average score', labels = labels_ba)
 
    return results, winRateAna
    
    
def ratio_Effect_Analysis(mapParams, step=0.1):
    """
    Analyze the effect of the ratio of random reward and punishment points 
    on the results of Monte Carlo search trees
    """
    # initialize
    ratio        = 0
    mapSize      = mapParams[0]
    randomPoints = mapParams[1]
    rbirthPoint  = mapParams[2]
    # create empty dataframe to store experiment results
    columns_name = ['birth point', 'ratio', 'best child', 'victory', 'failure', 'score']
    ratioAna     = pd.DataFrame(columns = columns_name)
    
    # Loop to generate state information under different ratios
    while ratio <= 1:
        randomMarkList = MCTS.generate_Random_Marks(nList=10, ratio=ratio)
        state = MCTS.GameState(mapSize, randomPoints, randomMarkList, rbirthPoint)
        
        result = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_Iterations = 100)
        
        # analysis result
        bestChild = result.currentPoint
        score = result.score
        if score >= 1:
            victory = 1
        else:
            victory = 0
        if score <= -0.5:
            failure = 1
        else:
            failure = 0
        result_tmp = [rbirthPoint, ratio, bestChild, victory, failure, score]
        ratioAna.loc[len(ratioAna)] = result_tmp
        
        # update ratio
        ratio += step
    
    return ratioAna
    

def diff_Cp_Analysis(state, cp_step = 0.1, num_experiment=20):
    
    # initialize C_p
    C_p = 1
    # create empty dataframe to store results
    columns_name = ['birth point', 'best child', 'number of choices', 'Number of victories',
                    'Number of failures', 'max score', 'min score', 'average score', 'C_p']
    CpAna = pd.DataFrame(columns = columns_name)
    
    while C_p >= 0:
        _, summary = win_Rate_Analysis(state, num_experiment=num_experiment, select_method = 'UCT', C_p = C_p)
        summary['C_p'] = C_p
        # save the summary result
        CpAna = pd.concat([CpAna, summary], axis=0)
        # update C_p
        C_p -= cp_step
        
    return CpAna
        

def plotBar(df, x, y, labels):
    
    df.plot(x=x, y=y, kind='bar')
    
    # Adding labels and title
    # labels[0] = xlabel, labels[1] = ylabel, labels[2] = title
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(labels[2])
    
    # Display the plot
    plt.show()
    
    return

###############################################################################

# Initialize bonus/punishment points and birth point
mapSize   = [10,10] # map size
nPoint = 10         # Number of randomly penalty/bonus points
randomPoints, rbirthPoint = MCTS.generate_Map(mapSize, nPoint)
randomMarkList = MCTS.generate_Random_Marks(nList=10, ratio=0.5)

state = MCTS.GameState(mapSize, randomPoints, randomMarkList, rbirthPoint)
result = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_Iterations = 100, select_method='UCT', C_p=0.7)

#result.score
#result.currentPoint

#len(root.children)

#root.state.currentPoint    

# 1. win rate
results, summary = win_Rate_Analysis(state, num_experiment=20) 
# 2. diff algorithm compare
result_UCB, summary_UCB = win_Rate_Analysis(state, num_experiment=20) 
result_UCT, summary_UCT = win_Rate_Analysis(state, num_experiment=20, select_method='UCT', C_p=0.9) 
# 3. ratio analysis   
mapParams = [mapSize, randomPoints, rbirthPoint]   
ratioAna = ratio_Effect_Analysis(mapParams, step=0.05)  
# 4. diff C_p compare
CpAna = diff_Cp_Analysis(state, cp_step = 0.1, num_experiment=10)
    
    
    
    