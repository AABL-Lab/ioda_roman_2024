# Imaginary Out-of-Distrbution Actions (IODA)

import numpy as np

class IodaStateModifier():
    def __init__(self, ood_detector=None, state_history_path=None, state_history=None):
        """
            Implements the IODA state modifier algorithm with distance based similarity metrics.
        """ 
        assert (state_history_path is not None) or (state_history is not None), "ERROR: Either state history path or state history should be provided"
        self.ood_detector = ood_detector
        self.state_history = state_history
        self.state_history_path = state_history_path
        if state_history_path is not None:
            # check if state history path is csv or np file
            if self.state_history_path.endswith('.csv'):
                # read csv file
                self.state_history = np.loadtxt(self.state_history_path, delimiter=',')
            elif self.state_history_path.endswith('.npy'):
                # read np file
                self.state_history = np.load(self.state_history_path)
            else:
                print("ERROR: File type not supported")
                exit()

    def get_ioda_state(self, state, distance_function='l1'):
        '''
            Returns the IODA state for the given state and the ood label.
            If the state is in distribution, returns the state itself.
            If the state is out of distribution, returns the most similar state in state history.
        '''
        # get ood label for state
        ood_label, _ = self.ood_detector.detect(state)
        if ood_label == 1:
            # state is in distribution
            return state, ood_label
        else:
            # find most similar state in state history
            if distance_function == 'l1':
                state_distances = np.sum(np.abs(self.state_history - state), axis=1)
            elif distance_function == 'l2':
                state_distances = np.linalg.norm(self.state_history - state, axis=1)
            else:
                print("ERROR: Distance function not supported")
                exit()
            # find index of most similar state
            most_similar_state_index = np.argmin(state_distances)
            # return most similar state
            return self.state_history[most_similar_state_index], ood_label
        