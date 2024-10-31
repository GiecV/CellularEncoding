class GlobalCounter:
    _counter = 0

    @classmethod
    def next(cls):
        """
        Get the next integer value from the global counter.

        Returns:
            int: The next integer value.
        """
        value = cls._counter
        cls._counter += 1
        return value

    @classmethod
    def next_str(cls):
        """
        Get the next integer value from the global counter as a string.

        Returns:
            str: The next integer value as a string.
        """
        value = cls._counter
        cls._counter += 1
        return str(value)
