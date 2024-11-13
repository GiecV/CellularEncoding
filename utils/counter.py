class GlobalCounter:
    """
    A class that provides a global counter with methods to generate 
    sequential integer values, either as integers or as strings.

    The counter is shared across all instances of the class, and its 
    value is incremented with each call to the `next` or `next_str` methods.
    """
    _counter = 0

    @classmethod
    def next(cls):
        """
        Get the next integer value from the global counter.

        This method increments the counter by 1 and returns the next value as an integer.

        :return: The next integer value in the sequence.
        :rtype: int
        """
        value = cls._counter
        cls._counter += 1
        return value

    @classmethod
    def next_str(cls):
        """
        Get the next integer value from the global counter as a string.

        This method increments the counter by 1 and returns the next value as a string.

        :return: The next integer value in the sequence as a string.
        :rtype: str
        """
        value = cls._counter
        cls._counter += 1
        return str(value)
