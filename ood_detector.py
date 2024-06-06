# generic out-of-distribution (ood) detector class

class OODDetector():
    def __init__(self, state_size):
        self.state_size = state_size

    def detect(self, state):
        """
            Returns the ood label and score for the state.
            Label is 1 if in distribution, -1 if out of distribution.
        """
        return 0, 0