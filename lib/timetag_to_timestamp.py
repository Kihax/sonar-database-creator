import numpy as np
def timetag_to_timestamp(timetags):
    """
        Transfrom a list of timetag (given in ns) to a list of timestamp
    """
    return timetags.astype('datetime64[ns]').astype(np.int64)