class GlobalCounter:
    _counter = 0

    @classmethod
    def next(cls):
        value = cls._counter
        cls._counter += 1
        return value

    @classmethod
    def next_str(cls):
        value = cls._counter
        cls._counter += 1
        return str(value)
