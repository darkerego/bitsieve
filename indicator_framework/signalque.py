class SignalQue:
    def __init__(self):
        self.signals = []

    def __next__(self):
        try:
            return self.signals.pop()
        except IndexError:
            return None

    def append(self, data):
        self.signals.append(data)