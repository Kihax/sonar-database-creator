import numpy as np
from .timetag_to_timestamp import timetag_to_timestamp
from xarray import DataTree
from typing import List, Optional, Union

class Measurement:
    def __init__(
        self, 
        path: str, 
        dts: List[DataTree], 
        timestamp_key: str, 
        extra_key: Optional[str] = None, 
        approximate_method: str = "linear_interpolation", 
        correct_offset: bool = False, 
        offset_path: Optional[str] = None, 
        offset_index: int = 0,
        max_gap : float = np.inf,
        is_bit64 : bool = False
    ):
        """
            A class to store, extract and exploits data from DataTree
        """
        self.path: str = path
        self.dts: List[DataTree] = dts
        self.timestamp_key: str = timestamp_key
        self.extra_key: Optional[str] = extra_key
                
        
        self.loaded: bool = False
        self.approximate_method: str = approximate_method
        self.correct_offset: bool = correct_offset
        self.offset_path: Optional[str] = offset_path
        self.offset_index: int = offset_index
        self.max_gap = max_gap;
        self.target_dtype = np.float64 if is_bit64 else np.float32

        self.timestamp: np.ndarray = np.array([], dtype=np.int64)
        self.value: np.ndarray = np.array([], dtype=self.target_dtype)
        
        self.quality: np.ndarray = np.array([], dtype=np.int8)

    def load(self) -> None:
        """
            Loads and normalizes data from all DataTrees efficiently.
        """
        if self.loaded:
            return

        ts_accum: List[np.ndarray] = []
        val_accum: List[np.ndarray] = []

        for dt in self.dts:
            offset = 0.0
            if self.correct_offset and self.offset_path is not None:
                offset = float(dt[self.offset_path].deltaXYZ[self.offset_index].values)
            
            # Extract timestamp and convert it
            raw_ts = dt[self.path][self.timestamp_key].values
            ts_accum.append(timetag_to_timestamp(raw_ts))

            # Extract values depending if there is a extra key or not and the format 64 bit or not we want
            if self.extra_key is not None:
                raw_vals = dt[self.path][self.extra_key].values.astype(self.target_dtype)
            else:
                raw_vals = dt[self.path].values.astype(self.target_dtype)
                
            val_accum.append(raw_vals - offset)

        # Concatenate values with previous one
        if ts_accum:
            self.timestamp = np.concatenate(ts_accum)
            self.value = np.concatenate(val_accum)
        else:
            self.timestamp = np.array([], dtype=np.int64)
            self.value = np.array([], dtype=np.float64)

        # Sort with timestamp
        sorted_indices = np.argsort(self.timestamp)
        self.timestamp = self.timestamp[sorted_indices]
        self.value = self.value[sorted_indices]

        # Create a quality list
        quality_list = []
        if len(self.value) > 0:
            quality_list.append(1) # First point is  is supposed to be valid
            last_valid_value = self.value[0]
            
            for i in range(1, len(self.value)):
                # If it's heading we manage to take into account the 360° step
                if "heading" in self.path.lower() or self.max_gap == 3.0:
                    gap = abs((self.value[i] - last_valid_value + 180) % 360 - 180)
                else:
                    gap = abs(self.value[i] - last_valid_value)
                
                if gap > self.max_gap:
                    quality_list.append(0) # Rejected measure
                else:
                    quality_list.append(1) # Valid measure
                    last_valid_value = self.value[i] # become the next reference
                    
        self.quality = np.array(quality_list, dtype=np.int8)

        self.loaded = True

    def _linear_interpolation_angle(self, target_ts: Union[int, np.ndarray], ts_array: np.ndarray, val_array: np.ndarray) -> Union[float, np.ndarray]:
        """
        Performs a circular linear interpolation for angles (degrees) to avoid 0/360 discontinuities.
        Works with both single int timestamp and numpy arrays.
        """
        # Convert angle using degree to radians
        rad_vals = np.radians(val_array)
        
        # Linear interpolation using sin and cos
        interp_sin = np.interp(target_ts, ts_array, np.sin(rad_vals))
        interp_cos = np.interp(target_ts, ts_array, np.cos(rad_vals))
        
        # Reconstuct the angle
        heading_interp = np.degrees(np.arctan2(interp_sin, interp_cos))
        
        # Convert it into 360° range
        return heading_interp % 360
    
    def get_value_from_timestamp(self, timestamp: int, with_quality: bool = False) -> Optional[float]:
        """
            Returns interpolated value for a single timestamp using numpy's robust built-ins.
        """
        self.load()

        if self.value.size == 0:
            return None
        
        # Can filtrate with quality
        if with_quality and self.quality.size > 0:
            mask = (self.quality == 1)
            ts_filtered = self.timestamp[mask]
            val_filtered = self.value[mask]
        else:
            ts_filtered = self.timestamp
            val_filtered = self.value

        # Security if table is empty
        if val_filtered.size == 0:
            return None
        if val_filtered.size == 1:
            return float(val_filtered[0])
        
        if self.approximate_method == "linear_interpolation":
            return float(np.interp(timestamp, ts_filtered, val_filtered, left=val_filtered[0]))
        elif(self.approximate_method == "linear_interpolation_angle"):
            return self._linear_interpolation_angle(timestamp, ts_filtered, val_filtered)
        
        return None

    def get_values_from_timestamps(self, timestamps: np.ndarray, with_quality: bool = False) -> np.ndarray:
        """
        Vectorized method to get values for multiple timestamps at once.
        """
        self.load()
        
        if self.value.size == 0:
            return np.full_like(timestamps, np.nan, dtype=float)
            
        if with_quality and self.quality.size > 0:
            mask = (self.quality == 1)
            ts_filtered = self.timestamp[mask]
            val_filtered = self.value[mask]
        else:
            ts_filtered = self.timestamp
            val_filtered = self.value

        if val_filtered.size == 0:
            return np.full_like(timestamps, np.nan, dtype=float)
        elif val_filtered.size == 1:
            return np.full_like(timestamps, val_filtered[0], dtype=float)
        
        if self.approximate_method == "linear_interpolation_angle":
            return self._linear_interpolation_angle(timestamps, ts_filtered, val_filtered)
        elif self.approximate_method == "linear_interpolation":
            return np.interp(timestamps, ts_filtered, val_filtered, left=val_filtered[0])
        
        return np.full_like(timestamps, np.nan, dtype=float)
            
    def get_values(self) -> np.ndarray:
        """
            Returns all values in measurement
        """
        self.load()
        return self.value
    
    def get_timestamps(self) -> np.ndarray:
        """
            Returns all values in measurement
        """
        self.load()
        return self.timestamp