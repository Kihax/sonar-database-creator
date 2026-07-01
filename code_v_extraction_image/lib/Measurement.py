import numpy as np
from .timetag_to_timestamp import timetag_to_timestamp
from xarray import DataTree
from typing import List

class Measurement:
    """
        Uniform class to extract all measure 

        - Args :
            - path (String) : path to the required ressources
            - dts (List<DataTree>) : A list of DataTree from NetCDF files
            - timestamp_key (String) : the key to the time property from the ressources (eg : sometime it is timetag or time)
            - extra_key (String) : The key to access the value from the captor (eg : heave, depth, ...), it was including because sometime we can't access the ressources directly with dt[path].value and we need to provide an extra key other path throw an error
            - approximate_method (String) : Method that should be utilized to calculate the measure for a given timestamp
            - correct_offset (Boolean) : Automatically catch the offset and remove it.
            - offset_path (String) : Path to catch offset data in the DataTree
            - offset_index (int between 0 and 2) : Index where is stored the offset in DeltaXYZ
    """
    def __init__(self, path : str, dts : List[DataTree], timestamp_key : str, extra_key : str = None, approximate_method : str="linear_interpolation", correct_offset : bool =False, offset_path : str=None, offset_index : int=0):
        self.path : str = path
        self.dts : List[DataTree] = dts
        self.timestamp_key : str = timestamp_key
        self.extra_key : str = extra_key
        self.timestamp : List[int] = np.array([])
        self.value : List[float] = np.array([])
        self.loaded : bool = False
        self.approximate_method : str = approximate_method
        self.correct_offset : bool = correct_offset
        self.offset_path : str = offset_path
        self.offset_index : int = offset_index

    """
        A function to load the data stored in each dts, it open each dt and extract the timestamp and the value
    """
    def load(self):
        # if values has already been loaded we skip this step
        if(self.loaded):
            return

        # browse each DataTree in the list
        for dt in self.dts:
            offset = 0
            # load the offset
            if(self.correct_offset and self.offset_path is not None):
                offset = dt[self.offset_path].deltaXYZ[self.offset_index]
            
            # load the timestamp
            self.timestamp = np.append(self.timestamp, timetag_to_timestamp(dt[self.path][self.timestamp_key].values))

            # load the values
            if(self.extra_key is not None):
                self.value = np.append(self.value, dt[self.path][self.extra_key].values - offset)
            else:
                self.value = np.append(self.value, dt[self.path].values - offset)


        # sort lists
        sorted_indices = np.argsort(self.timestamp)
        
        self.timestamp = self.timestamp[sorted_indices]
        self.value = self.value[sorted_indices]

        self.loaded = True
    
    """
        get values stored in the data with the timestamp provided, if the timestamp is before the first timestamp of our data, it returns the first value otherwise it calculate the value through with a linear interpolation between previous and next value
        Args:
            - timestamp (int) : timestamp in nanosecond to the required value (eg: 1712858300567000000)
    """
    def get_value_from_timestamp(self, timestamp : int):
        self.load()

        # if there is no value we return None
        if(self.value.size == 0):
            return None
        # if there is only one value we return this one
        elif(self.value.size == 1):
            return self.value[0]
        
        idx = np.searchsorted(self.timestamp, timestamp) - 1 # get the index that is just superior to the timestamp provided so we can calculate interpolation without error
        
        if(self.approximate_method == "linear_interpolation"):
            
            # get previous id
            idx = idx-1
            if(idx < 0): # if it was null we returns the first value in the list
                return self.value[0]
            
            # apply linear interpolation between previous and next value
            delta = timestamp-self.timestamp[idx]
            return self.value[idx] + delta * (self.value[idx+1] - self.value[idx]) / (self.timestamp[idx+1] - self.timestamp[idx])

            
    def get_values(self):
        """
            Returns all collected values
        """
        self.load()
        return self.value
    
    def get_timestamps(self):
        """
            Returns all timestamp
        """
        self.load()
        return self.timestamp