import numpy as np
class MemoryReplay:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.storage = []

    def additem(self, item):
        self.storage.append(item)
        if len(self.storage) > (self.max_size):
            self.storage = self.storage[1:]

    def sample_random(self):
        idx = np.random.randint(low=0, high=len(self.storage))

        return self.storage[idx]