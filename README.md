# SimpleOfflineRL

Installation instructions (Run the following commands_.
- Run git clone git@github.com:sanathsk009/SimpleOfflineRL.git
- conda env create -f environment.yml
- source activate offlineSRL
- python setup.py develop
- Now, you should be able to run the notebooks within the offlineSRL environment. All notebooks are in SimpleOfflineRL/Notebooks.

File structure in OfflineSRL:
- Agent:
  - Implements the key algorithms.
  - To understand the classes implemented, both files in the folder have lotsofcommentsdescribingthe classes and their methods.
- BPolicy:
  - Implements behavioral policies. Only rudimentary policies have been implemented so far.
- MDP:
  - Implements environments.
  - old_MDP.py and old_State.py implement the key abstract classes (TakenfromSimpleRL). 
  - ChainBandit is well commented.
- MDPDataset:
  - Implements the MDPDataset class. Taken from an older version of d3rlpy.
- OfflineLearners:
  - Implements all learners that fit to MDPDatasets.
  - Relies algorithms in Agent.
  - All files have lots of comments to make the class and methods understandable.
  
  
Modifications:
- Small modifications to BaseFiniteHorizonFiniteTabularAgent.
- Created EvalAgent within OfflineSRL/Agent
- Create OfflineEvaluators
