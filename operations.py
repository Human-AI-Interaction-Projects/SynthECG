class Operation:
    def __init__(self, type, value, ext):
        self.typename = type # inc/dec, smooth
        self.val = value # Magnitude of change (like inc/dec, smoothing factor)
        self.extras = ext # Extras in the case of multiple data points being modified for inc/dec