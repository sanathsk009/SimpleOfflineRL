# SimpleOfflineRL

Installationinstructions:
■ Navigateto"MidasRLforSearch/src/midas_rl_for_search/sanathkk_files/SimpleOfflineRL". ■ Thenrunthefollowingcommands.
■ condaenvcreate-fenvironment.yml
■ sourceactivateofflineSRL
■ pythonsetup.pydevelop
○ Now,youshouldbeabletorunthenotebookswithintheofflineSRLenvironment. ○ All notebooks are in SimpleOfflineRL/Notebooks
○ UnderstandingmainfoldersinOfflineSRL:
■ Agent:
□ Implementsthekeyalgorithms.
□ Tounderstandtheclassesimplemented,bothfilesinthefolderhavelotsofcommentsdescribingthe classes and their methods.
■ BPolicy:
□ Implementsbehavioralpolicies.Onlyrudimentarypolicieshavebeenimplementedsofar.
■ MDP:
□ Implementsenvironments.
□ old_MDP.pyandold_State.pyimplementthekeyabstractclasses.TakenfromSimpleRL. □ ChainBanditiswellcommented.Willaddcommentstotherest.
■ MDPDataset:
□ ImplementstheMDPDatasetclass.Takenfromanolderversionofd3rlpy.
■ OfflineLearners:
□ ImplementsalllearnersthatfittoMDPDatasets.
□ ReliesalgorithmsinAgent.
□ All files have lots of comments to make the class and methods understandable.
