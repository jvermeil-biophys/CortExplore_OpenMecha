# -*- coding: utf-8 -*-
"""
Main_3DTracker.py - Script to call the mainTracker function from BeadTracker.py.
Joseph Vermeil, 2024


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

# %% Imports

import os
import numpy as np
import pandas as pd

from BeadTracker_V4 import smallTracker

# %% Utility functions


def makeMetaData_type1(T_raw, B_set, loopStructure):
    """
    Parameters
    ----------
    T_raw : array-like
        The list of all time points corresponding to the images. len = N_Frames_total.
    B_set : array-like
        The list of all magnetic field values corresponding to the images. len = N_Frames_total.
    loopStructure : list of dicts
        Describe the phases composing each loop: 'Status' is the type of phase, 'NFrames' is the number of images it contains.
        The sum of all 'NFrames' must be equal to N_Frames_perLoop = N_Frames_total//N_Loops.
        Example of a loopStructure: 
        loopStructure = [
                         {'Status':'Passive',     'NFrames':30},  # First constant field phase
                         {'Status':'Action',      'NFrames':40},  # Pre-compression: sigmoïd down + constant field
                         {'Status':'Action_main', 'NFrames':125}, # Compression
                         {'Status':'Action',      'NFrames':15},  # Relaxation
                         {'Status':'Passive',     'NFrames':30},  # Second constant field phase
                        ]

    Returns
    -------
    metaDf: pandas DataFrame
        Table with 4 columns: 'T_raw', 'B_set', 'iL', 'Status'. Each row correspond to one frame. 
        Its length is N_Frames_total = N_Loops * N_Frames_perLoop
        - T_raw is the list of all time points.
        - B_set is the list of all magnetic field values.
        - iL stands for 'index loop'. It is an integer that gets incremented by 1 for each new loop.
        - Status is the list of the status of each image, as given by the loopStructure list.

    """
    
    # Columns from the field file
    metaDf = pd.DataFrame({'T_raw':T_raw, 'B_set':B_set})
    
    # Columns from the loopStructure
    N_Frames_total = metaDf.shape[0]
    N_Frames_perLoop = np.sum([phase['NFrames'] for phase in loopStructure])
    consistency_test = ((N_Frames_total/N_Frames_perLoop) == (N_Frames_total//N_Frames_perLoop))
    
    if not consistency_test:
        print('Error in the loop structure: the length of described loop do not match the Field file data!')
    
    else:
        N_Loops = (N_Frames_total//N_Frames_perLoop)
        StatusCol = []
        for phase in loopStructure:
            Status = phase['Status']
            NFrames = phase['NFrames']
            L = [Status] * NFrames # Repeat
            StatusCol += L # Append
        StatusCol *= N_Loops # Repeat
        StatusCol = np.array(StatusCol)
        
        iLCol = (np.ones((N_Frames_perLoop, N_Loops)) * np.arange(1, N_Loops+1)).flatten(order='F')
        
        metaDf['iL'] = iLCol.astype(int)
        metaDf['Status'] = StatusCol
    
    return(metaDf)


def makeMetaData_type2(fieldPath, loopStructure):
    """
    Parameters
    ----------
    fieldPath : string
        Path toward the '_Field.txt' file generated with the Labview software.
        This file is a table with  4 columns: ['B_meas', 'T_raw', 'B_set', 'Z_piezo']; and one row for each frame.
        From these, 'T_raw' & 'B_set' are read and used as columns for metaDf
        
    loopStructure : list of dicts
        Describe the phases composing each loop: 'Status' is the type of phase, 'NFrames' is the number of images it contains.
        The sum of all 'NFrames' must be equal to N_Frames_perLoop = N_Frames_total//N_Loops.
        Example of a loopStructure: 
        loopStructure = [
                         {'Status':'Passive',     'NFrames':30},  # First constant field phase
                         {'Status':'Action',      'NFrames':40},  # Pre-compression: sigmoïd down + constant field
                         {'Status':'Action_main', 'NFrames':125}, # Compression
                         {'Status':'Action',      'NFrames':15},  # Relaxation
                         {'Status':'Passive',     'NFrames':30},  # Second constant field phase
                        ]

    Returns
    -------
    metaDf: pandas DataFrame
        Table with 4 columns: 'T_raw', 'B_set', 'iL', 'Status'. Each row correspond to one frame. 
        Its length is N_Frames_total = N_Loops * N_Frames_perLoop
        - T_raw is the list of all time points.
        - B_set is the list of all magnetic field values.
        - iL stands for 'index loop'. It is an integer that gets incremented by 1 for each new loop.
        - Status is the list of the status of each image, as given by the loopStructure list.

    """
    
    # Columns from the field file
    fieldDf = pd.read_csv(fieldPath, sep='\t', names=['B_meas', 'T_raw', 'B_set', 'Z_piezo'])
    metaDf = fieldDf[['T_raw', 'B_set']]
    
    # Columns from the loopStructure
    N_Frames_total = metaDf.shape[0]
    N_Frames_perLoop = np.sum([phase['NFrames'] for phase in loopStructure])
    consistency_test = ((N_Frames_total/N_Frames_perLoop) == (N_Frames_total//N_Frames_perLoop))
    
    if not consistency_test:
        print(os.path.split(fieldPath)[-1])
        print('Error in the loop structure: the length of described loop do not match the Field file data!')
    
    else:
        N_Loops = (N_Frames_total//N_Frames_perLoop)
        StatusCol = []
        for phase in loopStructure:
            Status = phase['Status']
            NFrames = phase['NFrames']
            L = [Status] * NFrames # Repeat
            StatusCol += L # Append
        StatusCol *= N_Loops # Repeat
        StatusCol = np.array(StatusCol)
        
        iLCol = (np.ones((N_Frames_perLoop, N_Loops)) * np.arange(1, N_Loops+1)).flatten(order='F')
        
        metaDf['iL'] = iLCol
        metaDf['Status'] = StatusCol
    
    return(metaDf)



def makeMetaData_type3(fieldPath, statusPath):
    """
    Parameters
    ----------
    fieldPath : string
        Path toward the '_Field.txt' file generated with the Labview software.
        This file is a table with  4 columns: ['B_meas', 'T_raw', 'B_set', 'Z_piezo']; and one row for each frame.
        From these, 'T_raw' & 'B_set' are read and used as columns for metaDf
        
    statusPath : string
        Path toward the '_Status.txt' file generated with the Labview software.
        This file has 1 column containing infos about each frame. The infos are separated with a '_'.
        The format is usually the following: 'iL_PhaseType_ExtraInfos'
        From these, iL the loop index and the phase type related infos are read to build the columns 'Status' & 'iL' in metaDf

    Returns
    -------
    metaDf: pandas DataFrame
        Table with 4 columns: 'T_raw', 'B_set', 'iL', 'Status'. Each row correspond to one frame. 
        Its length is N_Frames_total = N_Loops * N_Frames_perLoop
        - T_raw is the list of all time points.
        - B_set is the list of all magnetic field values.
        - iL stands for 'index loop'. It is an integer that gets incremented by 1 for each new loop.
        - Status is the list of the status of each image, as given by the loopStructure dict.

    """
    
    # Columns from the field file
    fieldDf = pd.read_csv(fieldPath, sep='\t', names=['B_meas', 'T_raw', 'B_set', 'Z_piezo'])
    metaDf = fieldDf[['T_raw', 'B_set']]
    
    # Format the status file
    statusDf = pd.read_csv(statusPath, sep='_', names=['iL', 'Status', 'Status details'])
    Ns = len(statusDf)
    
    statusDf['Action type'] = np.array(['' for i in range(Ns)], dtype = '<U16')
    statusDf['deltaB'] = np.zeros(Ns, dtype = float)
    statusDf['B_diff'] = np.array(['' for i in range(Ns)], dtype = '<U4')
    
    indexAction = statusDf[statusDf['Status'] == 'Action'].index
    Bstart = statusDf.loc[indexAction, 'Status details'].apply(lambda x : float(x.split('-')[1]))
    Bstop = statusDf.loc[indexAction, 'Status details'].apply(lambda x : float(x.split('-')[2]))
    
    statusDf.loc[indexAction, 'deltaB'] =  Bstop - Bstart
    statusDf.loc[statusDf['deltaB'] == 0, 'B_diff'] =  'none'
    statusDf.loc[statusDf['deltaB'] > 0, 'B_diff'] =  'up'
    statusDf.loc[statusDf['deltaB'] < 0, 'B_diff'] =  'down'
    
    statusDf.loc[statusDf['Status details'].apply(lambda x : x.startswith('t^')), 'Action type'] = 'power'
    statusDf.loc[statusDf['Status details'].apply(lambda x : x.startswith('sigmoid')), 'Action type'] = 'sigmoid'
    statusDf.loc[statusDf['Status details'].apply(lambda x : x.startswith('constant')), 'Action type'] = 'constant'
    
    statusDf.loc[indexAction, 'Action type'] = statusDf.loc[indexAction, 'Action type'] + '_' + statusDf.loc[indexAction, 'B_diff']
    statusDf = statusDf.drop(columns=['deltaB', 'B_diff'])
    
    # Columns from the status file
    mainActionStep = 'power_up'
    metaDf['iL'] = statusDf['iL']
    metaDf['Status'] = statusDf['Status']
    metaDf.loc[statusDf['Action type'] == mainActionStep, 'Status'] = 'Action_main'
    return(metaDf)




# %% General template

# %%% Define paths

dictPaths = {'sourceDirPath' : '',
             'imageFileName' : '',
             'resultsFileName' : '',
             'depthoDir':'',
             'depthoName':'',
             'resultDirPath' : '',
             }


# %%% Define constants

dictConstants = {'microscope' : 'labview',
                 # If 1 type of beads
                  # 'bead type' : 'M450', # 'M450' or 'M270'
                  # 'bead diameter' : 4493, # nm
                  # 'bead magnetization correction' : 0.969, # nm
                 # If 2 types of beads
                 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 'inside bead diameter' : 4493, # nm
                 'inside bead magnetization correction' : 0.969, # nm
                 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 'outside bead diameter' : 4506, # nm
                 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


# %%% Additionnal options

dictOptions = {'redoAllSteps' : False,
               'saveFluo' : True,
               'plotZ' : False,
               'plotZ_range' : [0, 0],
               }

# %%% Make metaDataFrame

# metaDf = pd.DataFrame(columns=['T_raw', 'B_set', 'iL', 'Status'])

#### Method 1 - User defines everything
T_raw = []
B_set = []
loopStructure = [
                 {'Status':'',     'NFrames':0} # First phase
                ]
metaDf = makeMetaData_type1(T_raw, B_set, loopStructure)
    


#### Method 2 - User defines the loopStructure, 'T_raw' and 'B_set' are read from the field file.   
fieldPath = ''
loopStructure = [
                 {'Status':'',     'NFrames':0} # First phase
                ]
metaDf = makeMetaData_type2(fieldPath, loopStructure)
    


#### Method 3 - User defines nothing, the four columns are read from the field file (for 'T_raw' and 'B_set') 
#               and status file (for 'iL' and 'Status').
fieldPath = ''
statusPath = ''
metaDf = makeMetaData_type3(fieldPath, statusPath)
    
    

# %%% Call mainTracker()

smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)












# %% Example 1

# %%% Define paths

dictPaths = {'sourceDirPath'   : './Example_Data_2024/03_ExampleCell1',
             'imageFileName'   :   '24-04-11_M1_P1_C3_L50_disc20um_12ms.tif',
             'resultsFileName' :   '24-04-11_M1_P1_C3_L50_disc20um_12ms_Results.txt',
             'depthoDir'       : './Example_Data_2024/02_Depthograph/',
             'depthoName'      :   '24-04-11_Deptho_M1.tif',
             'resultDirPath'   : './Example_Data_2024/06_Results',
             }


# %%% Define constants

dictConstants = {'microscope' : 'labview',
                 # If 1 type of beads
                  # 'bead type' : 'M450', # 'M450' or 'M270'
                  # 'bead diameter' : 4493, # nm
                  # 'bead magnetization correction' : 0.969, # nm
                 # If 2 types of beads
                 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 'inside bead diameter' : 4493, # nm
                 'inside bead magnetization correction' : 0.969, # nm
                 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 'outside bead diameter' : 4506, # nm
                 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
                 'beads bright spot delta' : 0, # Rarely useful, do not change
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


# %%% Additionnal options

dictOptions = {'redoAllSteps' : True,
               'saveFluo' : False,
               'plotZ' : False,
               'plotZ_range' : [0, 0],
               }

# %%% Make metaDataFrame
# metaDf # ['T_raw', 'B_set', 'iL', 'Status']

fieldPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Field.txt'

loopStructure = [
                 {'Status':'Passive',     'NFrames':30},  # First constant field phase
                 {'Status':'Action',      'NFrames':40},  # Pre-compression: sigmoïd down + constant field
                 {'Status':'Action_main', 'NFrames':125}, # Compression
                 {'Status':'Action',      'NFrames':15},  # Relaxation
                 {'Status':'Passive',     'NFrames':30},  # Second constant field phase
                ]                                         # The sum of NFrames is 240, which is the loop size.

metaDf = makeMetaData_type2(fieldPath, loopStructure)


# %%% Call mainTracker()

tsDf = smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)
# The result is a timeSeries DataFrame. It gets saved in dictPaths['resultDirPath'] automatically.















# %% Example 2


# %%% Define paths

dictPaths = {'sourceDirPath'   : './Example_Data_2024/04_ExampleCell2',
             'imageFileName'   :   '24-04-11_M3_P1_C6_L50_disc20um_12ms.tif',
             'resultsFileName' :   '24-04-11_M3_P1_C6_L50_disc20um_12ms_Results.txt',
             'depthoDir'       : './Example_Data_2024/02_Depthograph/',
             'depthoName'      :   '24-04-11_Deptho_M1.tif',
             'resultDirPath'   : './Example_Data_2024/06_Results',
             }


# %%% Define constants

dictConstants = {'microscope' : 'labview',
                 # If 1 type of beads
                  # 'bead type' : 'M450', # 'M450' or 'M270'
                  # 'bead diameter' : 4493, # nm
                  # 'bead magnetization correction' : 0.969, # nm
                 # If 2 types of beads
                 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 'inside bead diameter' : 4493, # nm
                 'inside bead magnetization correction' : 0.969, # nm
                 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 'outside bead diameter' : 4506, # nm
                 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
                 'beads bright spot delta' : 0, # Rarely useful, do not change
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


# %%% Additionnal options

dictOptions = {'redoAllSteps' : True,
               'saveFluo' : False,
               'plotZ' : True,
               'plotZ_range' : [10, 30],
               }

# %%% Make metaDataFrame
# metaDf # ['T_raw', 'B_set', 'iL', 'Status']

fieldPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Field.txt'
statusPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Status.txt'

metaDf = makeMetaData_type3(fieldPath, statusPath)



# %%% Call mainTracker()

tsDf = smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)








# %% Example 3


# %%% Define paths

dictPaths = {'sourceDirPath'   : './Example_Data_2024/05_ExampleCell3',
             'imageFileName'   :   '24-04-11_M5_P1_C4_L50_disc15um_12ms.tif',
             'resultsFileName' :   '24-04-11_M5_P1_C4_L50_disc15um_12ms_Results.txt',
             'depthoDir'       : './Example_Data_2024/02_Depthograph/',
             'depthoName'      :   '24-04-11_Deptho_M1.tif',
             'resultDirPath'   : './Example_Data_2024/06_Results',
             }


# %%% Define constants

dictConstants = {'microscope' : 'labview',
                 # If 1 type of beads
                  # 'bead type' : 'M450', # 'M450' or 'M270'
                  # 'bead diameter' : 4493, # nm
                  # 'bead magnetization correction' : 0.969, # nm
                 # If 2 types of beads
                 'inside bead type' : 'M450-2025', # 'M450' or 'M270'
                 'inside bead diameter' : 4493, # nm
                 'inside bead magnetization correction' : 0.969, # nm
                 'outside bead type' : 'M450-Strept', # 'M450' or 'M270'
                 'outside bead diameter' : 4506, # nm
                 'outside bead magnetization correction' : 1.056, # nm
                 #
                 'normal field multi images' : 3, # Number of images
                 'multi image Z step' : 500, # nm
                 'multi image Z direction' : 'upward', # Either 'upward' or 'downward'
                 #
                 'scale pixel per um' : 15.8, # pixel/µm
                 'optical index correction' : 0.85, # ratio, without unit
                 'beads bright spot delta' : 0, # Rarely useful, do not change
                 'magnetic field correction' : 1.0, # ratio, without unit
                 }


# %%% Additionnal options

dictOptions = {'redoAllSteps' : True,
               'saveFluo' : False,
               'plotZ' : True,
               'plotZ_range' : [25, 40],
               }

# %%% Make metaDataFrame
# metaDf # ['T_raw', 'B_set', 'iL', 'Status']

fieldPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Field.txt'
statusPath = dictPaths['sourceDirPath'] + '//' + dictPaths['imageFileName'][:-4] + '_Status.txt'

metaDf = makeMetaData_type3(fieldPath, statusPath)



# %%% Call mainTracker()

tsDf = smallTracker(dictPaths, metaDf, dictConstants, NB = 2, **dictOptions)