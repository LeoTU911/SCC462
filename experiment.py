# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:00:01 2023

@author: Chunhui TU
"""

import MCTS # make sure that MCTS.py, experiment.py, utils.py and main.py are under the same directory and set a correct work dir
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt



def win_Rate_Analysis(state, num_sim=20, num_exp=50, select_method = 'UCB1', C_p = 0.9):
    # get birth point
    birthPoint = state.currentPoint
    
    # create empty dataframe to store experiment results
    columns_name = ['best child', 'victory', 'failure', 'score']
    results = pd.DataFrame(columns = columns_name)

    # repeat experiment
    for i in range(num_exp):
        result,_ = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_Iterations = num_sim, select_method=select_method, C_p = C_p)
        
        # analysis result
        bestChild = result.currentPoint
        score = result.score
        if score >= 0.5:
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
    
    
def ratio_Effect_Analysis(mapParams, step=0.1, num_sim=20, num_exp=50):
    """
    Analyze the effect of the ratio of random reward and punishment points 
    on the results of Monte Carlo Tree Search
    
    Parameters
    ----------
    mapParams :
        [mapSize, randomPoints, rbirthPoint] 
    step:
        The stride of ratio change
    num_sim : 
        number of MCTS iterations
    num_exp : 
        number of repeat experiments
        
    """
    # initialize
    ratio        = 0
    mapSize      = mapParams[0]
    randomPoints = mapParams[1]
    rbirthPoint  = mapParams[2]
    # initial game settings
    randomMarkList = MCTS.generate_Random_Marks(nList=10, ratio=ratio)
    state = MCTS.GameState(mapSize, randomPoints, randomMarkList, rbirthPoint)
    # create empty dataframe to summarize experiment results
    columns_name = ['birth point', 'ratio', 'best child', 'number of choices', 
                    'Number of victories', 'Number of failures', 'average score',
                    'lower bound CI', 'upper bound CI']
    ratioAna = pd.DataFrame(columns = columns_name)
    
    # Loop to generate state information under different ratios
    while ratio <= 1:

        # create empty dataframe to store experiment results
        columns_name = ['best child', 'victory', 'failure', 'score']
        results      = pd.DataFrame(columns = columns_name)
        for i in range(num_exp):
            # use default UCB1 algorithm
            result,_ = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_Iterations = num_sim)
        
            # analysis result
            bestChild = result.currentPoint
            score = result.score
            if score >= 0.5:
                victory = 1
            else:
                victory = 0
            if score <= -0.5:
                failure = 1
            else:
                failure = 0
            
            result_tmp = [bestChild, victory, failure, score]
            results.loc[len(results)] = result_tmp
    
        # summarize the result of current ratio     
        bestChilds = set(results['best child'])
        # analyze each best child
        for child in bestChilds:
            subData = results[results['best child'] == child]
            # summary
            numChoice = subData.shape[0]
            numVic    = subData['victory'].sum()
            numFail   = subData['failure'].sum()
            avg_score = subData['score'].mean()
            scores    = list(subData['score']) 
            # calculate confidence interval based on current ratio
            lowerBound_ci, upperBound_ci = score_Confidence_Interval_Ana(scores, num_bootstrap=1000)
            # concat the result
            summary = [rbirthPoint, ratio, child, numChoice, numVic, numFail, avg_score, lowerBound_ci, upperBound_ci]
            # store it
            ratioAna.loc[len(ratioAna)] = summary
        
        # update the ratio
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


def score_Confidence_Interval_Ana(scores, num_bootstrap):
    """
    Parameters
    ----------
    scores :
        the best childs' scores of each MCTS
    num_bootstrap :
        number of bootstrap iterations
    """

    # Set the desired confidence level
    confidenceLevel = 0.95
    # Initialize list to store bootstrap sample scores
    bootstrapSampleScores = []
    
    # Perform bootstrap iterations
    for _ in range(num_bootstrap):
        # Generate a bootstrap sample by sampling with replacement
        bootstrapSample = random.choices(scores, k=len(scores))
        # Calculate the statistic of mean
        statistic = np.mean(bootstrapSample)  
        # Store the statistic in the list of bootstrap sample scores
        bootstrapSampleScores.append(statistic)
    
    # Sort the list of bootstrap sample scores
    bootstrapSampleScores.sort()
    
    # Calculate the lower and upper percentiles based on the confidence level
    lowerPercentile = (1 - confidenceLevel) / 2
    upperPercentile = 1 - lowerPercentile

    # Calculate the lower and upper bounds of the confidence interval
    lowerBound_ci = np.percentile(bootstrapSampleScores, lowerPercentile * 100)
    upperBound_ci = np.percentile(bootstrapSampleScores, upperPercentile * 100)

    # Print the results
    print(f"Confidence Interval ({confidenceLevel * 100}%): [{lowerBound_ci}, {upperBound_ci}]")
    
    return lowerBound_ci, upperBound_ci

################################### END #######################################
    