class Point:
    """
        Point on the earth

        Args:
            - eastern (float) : eastern reference using UTM system of the point
            - nothern (float) : nothern reference using UTM system of the point
    """

    def __init__(self, eastern : float, nothern : float):
        self.eastern : float = eastern;
        self.nothern : float = nothern;
