from OfflineSRL.OfflineEvaluators.offlineBaseEvaluator import offlineTabularBaseEvaluator
from OfflineSRL.Agent.EvalAgent import StandardPesEvalAgent

import numpy as np
import math

class StandardPesEval(offlineTabularBaseEvaluator):

    def set_agent(self):
        self.agent = StandardPesEvalAgent(self.n_states, self.n_actions, self.epLen)