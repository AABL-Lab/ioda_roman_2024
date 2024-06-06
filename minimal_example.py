"""
Minimal example of using the ood detector class and the IODA algorithm 
"""
import numpy as np
from ood_detector import OODDetector
from ioda import IodaStateModifier

#  Here we create a simple example state history
#  for deployment, this would be made up of real states that the user
#  has either seen or is familiar with.


# create a dummy state history
state_history = np.linspace(1, 5, 10).reshape(-1, 2)
print("state history: ", state_history)
# state history:  [[1., 1.44444444]
#  [1.88888889, 2.33333333]
#  [2.77777778, 3.22222222]
#  [3.66666667, 4.11111111]
#  [4.55555556, 5.        ]]

# create ood detector
class SimpleOODDetector(OODDetector):
    def detect(self, state):
        if state[0] > 4:
            return -1, 0.9
        else:
            return 1, 0.1
        
ood_detector = SimpleOODDetector(state_size=2)

# create IODA state modifier
ioda_state_modifier = IodaStateModifier(ood_detector=ood_detector, state_history=state_history)

# test IODA state modifier
for state in [[1, 1], [3, 3], [5, 5]]:
    ioda_state, ood_label = ioda_state_modifier.get_ioda_state(state)
    print("state: ", state, "ioda state: ", ioda_state, "ood label: ", ood_label)