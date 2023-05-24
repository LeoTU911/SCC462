# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:00:01 2023

@author: Chunhui TU

---------------------------
FILE DESCRIPTION:
---------------------------
The file contains all the experimental methods, 
that is, each experiment is completed by building a Monte Carlo tree search.
Including experiment function, 
experiment data analysis 
and experiment result visualization
"""

import MCTS # make sure that MCTS.py, experiment.py, utils.py and main.py are under the same directory and set a correct work dir
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt


def win_Rate_Resource_Analysis(state, min_sim=100, max_sim=3000, num_exp=10, step=100):
    """
    Analyze the relationship between the number of MCTS iterations, 
    the number of wins, and the consumption of running time
    
    Params
    ----------
    state : 
        initial state
    min_sim : optional
        the initial number of MCTS iterations. The default is 100.
    max_sim : optional
        the final number of MCTS iterations. The default is 3000.
    num_exp : optional
        number of repeat experiments of MCTS. The default is 10
    step: optional
        the step size of MCTS iterations increase. The default is 100
    """
    
    # initial start number of simulation
    simCurrent = min_sim
    
    # create empty dataframe to store experiment results
    columns_name = ['number_of_simulations', 'victory', 'time_consumption']
    winRateResourceAna = pd.DataFrame(columns = columns_name)
    
    # repeat experiment
    while simCurrent <= max_sim:
        
        print('current number of simulations:{}'.format(simCurrent))
        
        # create empty dataframe to store experiment results
        columns_name = ['victory', 'time_consumption']
        results = pd.DataFrame(columns = columns_name)
        
        for i in range(num_exp):
            result, resource = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_sim = simCurrent)
        
            # analysis result
            score = result.score
            if score >= 0.5:
                victory = 1
            else:
                victory = 0
                
            result_tmp = [victory, resource[0]]
            results.loc[len(results)] = result_tmp
        
        # summarize the result
        numWin  = results['victory'].sum()
        avgTime = results['time_consumption'].mean()
        
        summary_tmp = [simCurrent, numWin, avgTime]
        winRateResourceAna.loc[len(winRateResourceAna)] = summary_tmp
        
        # update the number of simulation
        simCurrent += step

    # plot multi-line chart
    # Define colors for each line
    colors = ['red', 'blue']
    
    plt.plot(winRateResourceAna['number_of_simulations'], winRateResourceAna['time_consumption'], color=colors[0], label='time_consumption')
    plt.plot(winRateResourceAna['number_of_simulations'], winRateResourceAna['victory'], color=colors[1], label='victory')
    
    # Adding labels and title
    plt.xlabel('number_of_simulations')
    plt.ylabel('number')
    plt.title('the relationship of number of simulations,\ntime consumption and victory')
    
    # save the image
    savePath = './images'
    saveName = 'resource_win_time.jpg'
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    saveFile = '{}/{}'.format(savePath, saveName)
    plt.savefig(saveFile)
    
    # Displaying the legend
    plt.legend()

    # Displaying the graph
    plt.show()
    
    return winRateResourceAna
    

def win_Rate_Analysis(state, num_sim=20, num_exp=50, select_method = 'UCB1', C_p = 0.9):
    """
    Repeat the experiment with the MCTS algorithm to 
    view the winning analysis under the current game rules
    
    Params
    ----------
    state : 
        initial state
    num_sim : optional
        number of MCTS iterations. The default is 20.
    num_exp : optional
        number of repeat experiments of MCTS. The default is 50
    select_method : optional, ('UCB1', 'UCT')
        the MCTS selection algorithm. The default is 'UCB1'. 
    C_p : TYPE, optional
        Constant for the UCT algorithm. The default is 0.9.
    """
    
    # get birth point
    birthPoint = state.currentPoint
    
    # create empty dataframe to store experiment results
    columns_name = ['best child', 'victory', 'failure', 'score']
    results = pd.DataFrame(columns = columns_name)

    # repeat experiment
    for i in range(num_exp):
        result,_ = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_sim = num_sim, select_method=select_method, C_p = C_p)
        
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
    plot_Bar(df = winRateAna, 
             x='best child', 
             y='number of choices', 
             labels = labels_bn, 
             saveName='{}-choice.jpg'.format(select_method))
    
    labels_ba = ['best child', 'average score', 'The next choice and the corresponding average score']
    plot_Bar(df = winRateAna, 
             x='best child', 
             y='average score', 
             labels = labels_ba, 
             saveName='{}-score.jpg'.format(select_method))
 
    return results, winRateAna
    
    
def ratio_Effect_Analysis(mapParams, step=0.1, num_sim=20, num_exp=50):
    """
    Analyze the effect of the ratio of random reward/penalty points 
    on the results of Monte Carlo Tree Search
    
    Params
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
    
    # create empty dataframe to summarize experiment results
    columns_name = ['birth point', 'ratio', 'best child', 'number of choices', 
                    'Number of victories', 'Number of failures', 'average score',
                    'lower bound CI', 'upper bound CI']
    ratioAna = pd.DataFrame(columns = columns_name)
    
    # Loop to generate state information under different ratios
    while ratio <= 1:
        # generate random score list
        randomMarkList = MCTS.generate_Random_Marks(nList=10, ratio=ratio, repeat=False)
        state = MCTS.GameState(mapSize, randomPoints, randomMarkList, rbirthPoint)
        # create empty dataframe to store experiment results
        columns_name = ['best child', 'victory', 'failure', 'score']
        results      = pd.DataFrame(columns = columns_name)
        # create a dataframe for visualization
        columns_name_visit = ['node', 'visit']
        nodeVisited_df = pd.DataFrame(columns = columns_name_visit)
        
        for i in range(num_exp):
            # use default UCB1 algorithm
            result, resources, nodeVisited = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_sim = num_sim, visualize=True)
        
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
            nodeVisited_df = pd.concat([nodeVisited_df, nodeVisited], ignore_index=True)
    
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
            
        # visualize
        merged_nodeVisited_df = nodeVisited_df.groupby('node').sum().reset_index()
        coordinates = list(merged_nodeVisited_df['node'])
        visits = list(merged_nodeVisited_df['visit'])
        saveName = 'node_visits_with_ratio_{}.jpg'.format(ratio)
        plot_Scatter_Visit(mapSize, coordinates, visits, rbirthPoint, saveName=saveName, subtitle='ratio:{}'.format(ratio))
        
        # update the ratio
        ratio += step
    
    return ratioAna
    

def diff_nPoint_Analysis(mapParams, step=10, ratio=0.5, num_sim=20, num_exp=50):
    """
    Analyze the effect of the number of random reward/penalty points 
    on the results of Monte Carlo Tree Search
    
    Params
    ----------
    mapParams :
        [mapSize, rbirthPoint] 
    step:
        The stride of nPoint change
    num_sim : 
        number of MCTS iterations
    num_exp : 
        number of repeat experiments
        
    """
    # initialize
    nPoint       = 0
    mapSize      = mapParams[0]
    rbirthPoint  = mapParams[1]
    
    # create empty dataframe to summarize experiment results
    columns_name = ['birth point', 'nPoint', 'best child', 'number of choices', 
                    'Number of victories', 'Number of failures', 'average score',
                    'lower bound CI', 'upper bound CI']
    nPointAna = pd.DataFrame(columns = columns_name)
    
    # Loop to generate state information under different ratios
    while nPoint < mapSize[0]*mapSize[1]:
        # generate initial map info
        randomPoints, _ = MCTS.generate_Map(mapSize, nPoint)
        # generate random score list
        randomMarkList = MCTS.generate_Random_Marks(nList=10, ratio=ratio, repeat=True)
        state = MCTS.GameState(mapSize, randomPoints, randomMarkList, rbirthPoint)
        # create empty dataframe to store experiment results
        columns_name = ['best child', 'victory', 'failure', 'score']
        results      = pd.DataFrame(columns = columns_name)
        # create a dataframe for visualization
        columns_name_visit = ['node', 'visit']
        nodeVisited_df = pd.DataFrame(columns = columns_name_visit)
        
        for i in range(num_exp):
            # use default UCB1 algorithm
            result, resources, nodeVisited = MCTS.monte_Carlo_Tree_Search(initial_State = state, num_sim = num_sim, visualize=True)
        
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
            nodeVisited_df = pd.concat([nodeVisited_df, nodeVisited], ignore_index=True)
    
        # summarize the result of current nPoint     
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
            summary = [rbirthPoint, nPoint, child, numChoice, numVic, numFail, avg_score, lowerBound_ci, upperBound_ci]
            # store it
            nPointAna.loc[len(nPointAna)] = summary
            
        # visualize
        merged_nodeVisited_df = nodeVisited_df.groupby('node').sum().reset_index()
        coordinates = list(merged_nodeVisited_df['node'])
        visits = list(merged_nodeVisited_df['visit'])
        saveName = 'node_visits_with_nPoint_{}.jpg'.format(nPoint)
        plot_Scatter_Visit(mapSize, coordinates, visits, rbirthPoint, saveName=saveName, subtitle='nPoint:{}'.format(nPoint))
        
        # update the ratio
        nPoint += step
    
    return nPointAna

def diff_Cp_Analysis(state, cp_step = 0.1, num_exp=50):
    
    """
    Analyze the effect of the constant C_p of UCT algorithm 
    on the results of Monte Carlo Tree Search
    
    Params
    ----------
    state : 
        initial state
    cp_step:
        The stride of C_p change
    num_exp : 
        number of repeat experiments
    """
    
    # initialize C_p
    C_p = 1
    # create empty dataframe to store results
    columns_name = ['birth point', 'best child', 'number of choices', 'Number of victories',
                    'Number of failures', 'max score', 'min score', 'average score',
                    'lower bound CI', 'upper bound CI', 'C_p']
    CpAna = pd.DataFrame(columns = columns_name)
    
    while C_p >= 0:
        results, summary = win_Rate_Analysis(state, num_sim=20, num_exp=num_exp, select_method = 'UCT', C_p = C_p)
        
        # calculate the confidence interval
        # summarize the result of current ratio     
        bestChilds = set(results['best child'])
        # analyze each best child
        for child in bestChilds:
            subData = results[results['best child'] == child]
            scores = list(subData['score']) 
            lowerBound_ci, upperBound_ci = score_Confidence_Interval_Ana(scores, num_bootstrap=1000)
            # Find the corresponding row where 'best child' column is current child
            rowToUpdate = summary['best child'] == child
            # Add new columns of lower and upper confidence interval
            summary.loc[rowToUpdate, 'lower bound CI'] = lowerBound_ci
            summary.loc[rowToUpdate, 'upper bound CI'] = upperBound_ci
            # Add the C_p column
            summary.loc[rowToUpdate, 'C_p'] = C_p
            
        # save the summary result
        CpAna = pd.concat([CpAna, summary], axis=0)
        # update C_p
        C_p -= cp_step
        
    return CpAna


def score_Confidence_Interval_Ana(scores, num_bootstrap):
    """
    Params
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


### PLOT PART ###

def plot_Bar(df, x, y, labels, saveName):
    
    df.plot(x=x, y=y, kind='bar')
    
    # Adding labels and title
    # labels[0] = xlabel, labels[1] = ylabel, labels[2] = title
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(labels[2])
    
    # save the image
    savePath = './images'
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    saveFile = '{}/{}'.format(savePath, saveName)
    plt.savefig(saveFile)
    
    # Display the plot
    plt.show()
    
    return


def plot_Scatter_Visit(mapSize, coordinates, visits, birthPoint, saveName, subtitle=None):
    # Plot each node and its access intensity
    
    # Extract x and y coordinates from the list of coordinates
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]
    
    # Create a scatter plot
    plt.scatter(x, y, c=visits, cmap='viridis')
    
    # Add a colorbar
    plt.colorbar(label='Number of Visits')
    
    # Set x-axis and y-axis grid density to 1
    plt.xticks(np.arange(0, mapSize[0], 1))
    plt.yticks(np.arange(0, mapSize[1], 1))
    
    # Add a grid
    plt.grid(True)
    
    # Add the birth point with red color
    plt.scatter(birthPoint[0], birthPoint[1], c='red', marker='o')
    
    # Set x-axis and y-axis limits
    plt.xlim(0, mapSize[0]-1)
    plt.ylim(0, mapSize[1]-1)
    
    # Set labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot of Visits')
    if subtitle:
        plt.suptitle(subtitle)
    
    # save the image
    savePath = './images'
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    saveFile = '{}/{}'.format(savePath, saveName)
    plt.savefig(saveFile)
    
    # Show the plot
    plt.show()
    
def plot_Scatter_Initial(mapSize, randomPoints, birthPoint, saveName):
    # Plot the birth point and all the random B/P points
    
    # Extract x and y coordinates from the list of coordinates
    x = [coord[0] for coord in randomPoints]
    y = [coord[1] for coord in randomPoints]
    
    # Create a scatter plot
    plt.scatter(x, y, c='green')
    
    # Set x-axis and y-axis grid density to 1
    plt.xticks(np.arange(0, mapSize[0], 1))
    plt.yticks(np.arange(0, mapSize[1], 1))
    
    # Add a grid
    plt.grid(True)
    
    # Add the birth point with red color
    plt.scatter(birthPoint[0], birthPoint[1], c='red', marker='o')
    
    # Set x-axis and y-axis limits
    plt.xlim(0, mapSize[0]-1)
    plt.ylim(0, mapSize[1]-1)
    
    # Set labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot of initial Map')
    
    # save the image
    savePath = './images'
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    saveFile = '{}/{}'.format(savePath, saveName)
    plt.savefig(saveFile)
    
    # Show the plot
    plt.show()


################################### END #######################################
    