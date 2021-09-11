#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from functools import partial
from multiprocessing import Pool, Lock
import pandas as pd

def calc_total_cells(mask):
    x = np.unique(mask, return_counts=True)[-1][-1]
    return x
    
def calc_agreement(mask, map1, map2, Pe1_array, Pe2_array, Po_array, pair):
    i = pair[0]
    j = pair[1]
    if mask[i,j] != 0:
        x = map1[i, j]
        y = map2[i, j]
        # Amend the expected array count
        Pe1_array[x]=Pe1_array[x]+1
        Pe2_array[y]=Pe2_array[y]+1
        # If there is agreement, amend the observed count
        if x == y:
            Po_array[x] = Po_array[x] + 1


def kappa(map1, map2, mask, total_cells=0):
    # Determine the map dimensions and number of land-use classes.
    shape_map1 = np.shape(map1)
    row = shape_map1[0]
    column = shape_map1[1]
    luc = np.amax(map1) + 1
    # Determine the total number of cells to be considered.
    if total_cells==0:
        for i in range(0, row):
            for j in range(0, column):
                x = mask[i, j]
                if x != 0:
                    total_cells = total_cells + 1
    # Initialise an array to store the observed agreement probability.
    Po_array = np.zeros(shape=luc)
    # Initialise a set of arrays to store the expected agreement probability,
    # for both maps, and then combined.
    Pe1_array = np.zeros(shape=luc)
    Pe2_array = np.zeros(shape=luc)
    Pe_array = np.zeros(shape=luc)
    # Initialise an array to store the maximum possible agreement probability.
    Pmax_array=np.zeros(shape=luc)
    # Analyse the agreement between the two maps.
                
    arg_pairs = [(i,j) for i in range(row) for j in range(column)]
    pool = Pool()
    func = partial(calc_agreement, mask, map1, map2, Pe1_array, Pe2_array, Po_array)
    pool.map(func, arg_pairs)
        
    
    for i in range(0, row):
        for j in range(0, column):
            if mask[i,j] != 0:
                x = map1[i, j]
                y = map2[i, j]
                # Amend the expected array count
                Pe1_array[x]=Pe1_array[x]+1
                Pe2_array[y]=Pe2_array[y]+1
                # If there is agreement, amend the observed count
                if x == y:
                        Po_array[x] = Po_array[x] + 1
    # Convert to probabilities.
    Po_array[:] = [x/total_cells for x in Po_array]
    Pe1_array[:] = [x/total_cells for x in Pe1_array]
    Pe2_array[:] = [x/total_cells for x in Pe2_array]
    # Now process the arrays to determine the maximum and expected
    # probabilities.
    for i in range(0, luc):
        Pmax_array[i] = min(Pe1_array[i], Pe2_array[i])
        Pe_array[i] = Pe1_array[i]*Pe2_array[i]
    # Calculate the values of probability observed, expected, and max.
    Po = np.sum(Po_array)
    Pmax = np.sum(Pmax_array)
    Pe = np.sum(Pe_array)
    # Now calculate the Kappa histogram and Kappa location.
    Khist = (Pmax - Pe)/(1 - Pe)
    Kloc = (Po - Pe)/(Pmax - Pe)
    # Finally, calculate Kappa.
    Kappa=Khist*Kloc
    # Return the value of Kappa.
    return Kappa


def ksim(omap, map1, map2, mask):
    # Determine the map dimensions and number of land-use classes.
    shape = np.shape(map1)
    row = shape[0]
    column = shape[1]
    luc = np.amax(map1) + 1
    # Determine the total number of cells to be considered.
    total_cells = 0
    for i in range(0, row):
        for j in range(0, column):
            x = mask[i, j]
            if x != 0:
                total_cells = total_cells + 1
    # Initialise an array to store the observed and expected agreement for the
    # transitions between the two maps.
    Po_o_array = np.zeros(shape=luc)
    Po_a_array = np.zeros(shape=luc)
    Pe1trans_array = np.zeros(shape=luc**2)
    Pe2trans_array = np.zeros(shape=luc**2)
    Petrans_array = np.zeros(shape=luc)
    Pmaxtrans_array = np.zeros(shape=luc)
    # Evaluate the maps via couting.
    for i in range(0, row):
        for j in range(0, column):
            if mask[i, j] != 0:
                x=omap[i, j]
                y=map1[i, j]
                z=map2[i, j]
                
                Po_o_array[x] = Po_o_array[x] + 1
                
                Pe1trans_array[x*luc + y] = Pe1trans_array[x*luc + y] + 1
                Pe2trans_array[x*luc + z] = Pe2trans_array[x*luc + z] + 1
                
                if y==z:
                    Po_a_array[z] = Po_a_array[z] + 1
    # Convert the counts to proportions
    Po_o_array[:] = [x/total_cells for x in Po_o_array]
    Po_a_array[:] = [x/total_cells for x in Po_a_array]
    Pe1trans_array[:] = [x/total_cells for x in Pe1trans_array]
    Pe2trans_array[:] = [x/total_cells for x in Pe2trans_array]
    # Analyse the observed agreements for calculation of the expected
    # agreements.
    for i in range(0, luc):
        for j in range(0, luc):
            if Po_o_array[i] > 0:
                Pe1trans_array[i*luc + j] = (
                    Pe1trans_array[i * luc + j] / Po_o_array[i]
                )
                Pe2trans_array[i*luc + j] = (
                    Pe2trans_array[i * luc + j] / Po_o_array[i]
                )

    for i in range(0, luc):
        for j in range(0, luc):
            Petrans_array[i] = (
                Petrans_array[i] + Po_o_array[i] *
                Pe1trans_array[luc * i + j] * Pe2trans_array[luc * i + j]
            )
            Pmaxtrans_array[i] = (
                Pmaxtrans_array[i] + Po_o_array[i] *
                min(Pe1trans_array[luc * i + j], Pe2trans_array[luc * i + j])
            )
    # Calculate the values of observed expected, and max transition agreement.
    Po_a = np.sum(Po_a_array)
    Petrans = np.sum(Petrans_array)
    Pmaxtrans = np.sum(Pmaxtrans_array)
    # Print error feedback if the solution cannot be evaluated.
    if Petrans == 1:
        return 'Calculation results in undefined solution'
        
    if Petrans == Pmaxtrans:
        return 'Calculation results in undefined solution'
    # Calculate the histogram and location of Kappa Simulation.
    Ktransition = (Pmaxtrans - Petrans)/(1 - Petrans)
    Ktransloc = (Po_a - Petrans)/(Pmaxtrans - Petrans)
    # Finally, calculate Kappa Simulation, and return the value.
    KSIM = Ktransition*Ktransloc
    return KSIM
    
def kappa_ori(mask, total_cells, arg_pairs):
    # Determine the map dimensions and number of land-use classes.
    m = arg_pairs[0]
    n = arg_pairs[1]
    map1 = arg_pairs[2]
    map2 = arg_pairs[3]
    
    if m <= n:
        #print(m,n)
    
        shape_map1 = np.shape(map1)
        row = shape_map1[0]
        column = shape_map1[1]
        luc = np.amax(map1) + 1
        # Determine the total number of cells to be considered.
        if total_cells==0:
            for i in range(0, row):
                for j in range(0, column):
                    x = mask[i, j]
                    if x != 0:
                        total_cells = total_cells + 1
        # Initialise an array to store the observed agreement probability.
        Po_array = np.zeros(shape=luc)
        # Initialise a set of arrays to store the expected agreement probability,
        # for both maps, and then combined.
        Pe1_array = np.zeros(shape=luc)
        Pe2_array = np.zeros(shape=luc)
        Pe_array = np.zeros(shape=luc)
        # Initialise an array to store the maximum possible agreement probability.
        Pmax_array=np.zeros(shape=luc)
        # Analyse the agreement between the two maps.
        for i in range(0, row):
            for j in range(0, column):
                if mask[i,j] != 0:
                    x = map1[i, j]
                    y = map2[i, j]
                    # Amend the expected array count
                    Pe1_array[x]=Pe1_array[x]+1
                    Pe2_array[y]=Pe2_array[y]+1
                    # If there is agreement, amend the observed count
                    if x == y:
                            Po_array[x] = Po_array[x] + 1
        
        #print(Pe1_array)
        #print(Pe2_array)
        #print(Po_array)
        # Convert to probabilities.
        Po_array[:] = [x/total_cells for x in Po_array]
        Pe1_array[:] = [x/total_cells for x in Pe1_array]
        Pe2_array[:] = [x/total_cells for x in Pe2_array]
        # Now process the arrays to determine the maximum and expected
        # probabilities.
        for i in range(0, luc):
            Pmax_array[i] = min(Pe1_array[i], Pe2_array[i])
            Pe_array[i] = Pe1_array[i]*Pe2_array[i]
        # Calculate the values of probability observed, expected, and max.
        Po = np.sum(Po_array)
        Pmax = np.sum(Pmax_array)
        Pe = np.sum(Pe_array)
        # Now calculate the Kappa histogram and Kappa location.
        Khist = (Pmax - Pe)/(1 - Pe)
        Kloc = (Po - Pe)/(Pmax - Pe)
        # Finally, calculate Kappa.
        Kappa=Khist*Kloc
        # Return the value of Kappa.
        
        #print(Kappa)
        
        return (m,n,Kappa)
    
def kappa_ema_multiprocess(outcome, mask):
    
    
    total_cells = calc_total_cells(mask)
    
    df = pd.DataFrame(np.ones((len(outcome), len(outcome))))
    
    shape_map1 = np.shape(outcome[0])
    row = shape_map1[0]
    column = shape_map1[1]
    luc = np.amax(outcome[0]) + 1
    # Initialise an array to store the observed agreement probability.
    Po_array = np.zeros(shape=luc)
    # Initialise a set of arrays to store the expected agreement probability,
    # for both maps, and then combined.
    Pe1_array = np.zeros(shape=luc)
    Pe2_array = np.zeros(shape=luc)
    Pe_array = np.zeros(shape=luc)
    # Initialise an array to store the maximum possible agreement probability.
    Pmax_array=np.zeros(shape=luc)
    
    arg_pairs = []
    for i, m_i in enumerate(outcome):
        for j, m_j in enumerate(outcome):
            if i <= j:
                arg_pairs.append((i, j, m_i, m_j))
    
    #arg_pairs = [(i, j, m_i,m_j) for i, m_i in enumerate(outcome) for j, m_j in enumerate(outcome)]
    
    max_task = 10000
    
    #return arg_pairs
    
    if len(arg_pairs) > max_task:
        iter_num = len(arg_pairs) / max_task
        last_iter = 0
        if np.mod(len(arg_pairs), max_task) > 0:
            iter_num = int(iter_num)+1
        print('number of iteration will be ' + str(iter_num))
        for iter in range(int(iter_num)):
            print('iteration no ' + str(iter+1))
            try:
                arg_pairs_ = arg_pairs[last_iter:last_iter+max_task]
            except:
                arg_pairs_ = arg_pairs[last_iter:]
            pool = Pool(24)
            func = partial(kappa_mod, mask, total_cells, shape_map1, row, column, luc, Po_array, Pe1_array, Pe2_array, Pe_array, Pmax_array)
            all_kappa = pool.map(func, arg_pairs_)
            for item in all_kappa:
                try:
                    i = item[0]
                    j = item[1]
                    k = item[2]
                    df[i][j] = k
                    #df[item[0]][item[1]] = item[2]
                except:
                    pass
            del all_kappa
            try:
                pool.terminate()
            except:
                pass
            try:
                pool.close()
            except:
                pass
            last_iter += max_task
    
    
    else:
        pool = Pool(6)
        func = partial(kappa_mod, mask, total_cells)
        all_kappa = pool.map(func, arg_pairs)
        
        for item in all_kappa:
            try:
                i = item[0]
                j = item[1]
                k = item[2]
                df[i][j] = k
            except:
                pass
            
    print('iteration done, storing Kappa values to a dataframe')
    for i in range(len(outcome)):
        for j in range(len(outcome)):
            df[j][i] = df[i][j]
    
    return df
    
def kappa_mod(mask, total_cells, shape_map1, row, column, luc, Po_array, Pe1_array, Pe2_array, Pe_array, Pmax_array, arg_pairs):
    # Determine the map dimensions and number of land-use classes.
    m = arg_pairs[0]
    n = arg_pairs[1]
    map1 = arg_pairs[2]
    map2 = arg_pairs[3]
    
    #if m <= n:
    print(m,n)
    
    # Determine the total number of cells to be considered.
    if total_cells==0:
        for i in range(0, row):
            for j in range(0, column):
                x = mask[i, j]
                if x != 0:
                    total_cells = total_cells + 1
    
    #Analyse the agreement between the two maps
    map1_unique = np.unique(map1, return_counts=True)
    map1_unique = dict(zip(map1_unique[0], map1_unique[1]))
    map2_unique = np.unique(map2, return_counts=True)
    map2_unique = dict(zip(map2_unique[0], map2_unique[1]))
    #print(map2_unique)
    map_mask = 1-mask
    map1_mask = np.ma.masked_array(map1, mask=mask).compressed()
    map2_mask = np.ma.masked_array(map2, mask=mask).compressed()
    map1_mask_unique = np.unique(map1_mask, return_counts=True)
    map1_mask_unique = dict(zip(map1_mask_unique[0], map1_mask_unique[1])) 
    map2_mask_unique = np.unique(map2_mask, return_counts=True)
    map2_mask_unique = dict(zip(map2_mask_unique[0], map2_mask_unique[1])) 
    for key in map1_mask_unique.keys():
        map1_unique[key] = map1_unique[key] - map1_mask_unique[key]
    for key in map2_mask_unique.keys():
        map2_unique[key] = map2_unique[key] - map2_mask_unique[key]
    agreement = np.where(map1==map2, map1, -1)
    agreement = np.where(mask==1, agreement, -1)
    agreement_unique = np.unique(agreement, return_counts=True)
    agreement_unique = dict(zip(agreement_unique[0], agreement_unique[1]))
    
    
    
    for l in range(0, luc):
        try:
            Pe1_array[l] = map1_unique[l]
        except:
            pass
        try:
            Pe2_array[l] = map2_unique[l]
        except:
            pass
        try:
            Po_array[l] = agreement_unique[l]
        except:
            pass
            
    #print(Pe1_array)
    #print(Pe2_array)
    #print(Po_array)
        
    Po_array[:] = [x/total_cells for x in Po_array]
    Pe1_array[:] = [x/total_cells for x in Pe1_array]
    Pe2_array[:] = [x/total_cells for x in Pe2_array]
    # Now process the arrays to determine the maximum and expected
    # probabilities.
    for i in range(0, luc):
        Pmax_array[i] = min(Pe1_array[i], Pe2_array[i])
        Pe_array[i] = Pe1_array[i]*Pe2_array[i]
    # Calculate the values of probability observed, expected, and max.
    Po = np.sum(Po_array)
    Pmax = np.sum(Pmax_array)
    Pe = np.sum(Pe_array)
    # Now calculate the Kappa histogram and Kappa location.
    Khist = (Pmax - Pe)/(1 - Pe)
    Kloc = (Po - Pe)/(Pmax - Pe)
    # Finally, calculate Kappa.
    Kappa=Khist*Kloc
    # Return the value of Kappa.
    print(Kappa)
    
    return (m,n,Kappa)

