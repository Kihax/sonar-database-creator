# some basic functions to get data from netcdf files and to get the list of files in the data folder (netcdf file)

import os
import xarray as xr

def get_files(folder_path):
    """
        Returns all file in the given folder

        Args:
            - folder_path (string) : folder path
        
        Returns :
            - file_list (List<String>) : all files in the path
    """
    file_list = []

    for e in os.scandir(folder_path):
        if e.is_file():
            if e.name.endswith("nc"):
                file_list.append(os.path.basename(e.path))
                #print("Content of", e.name, ":")
    return file_list

def get_tree_from_index(file_index):
    """
        Open DataTree from given netcdf file
        Args:
            - file_index (int) : number of the file from get_files('/../../Dataset Metric/')
        Returns:
            - dt : DataTree from netCDF file
            - filename : of the file loaded
    """
    # load data
    file_path = os.path.abspath(os.path.dirname(__file__)) + '/../../Dataset Metric/' # we can run this file from any directory in terminal
    
    files_name = get_files(file_path)

    # open data tree
    if(files_name[file_index].startswith("._")):
        return xr.open_datatree(file_path + files_name[file_index][2:]), files_name[file_index][2:]
    dt = xr.open_datatree(file_path + files_name[file_index])

    return dt, files_name[file_index]

def get_tree_from_file(filename):
    """
        Open DataTree from given netcdf file
        Args:
            - file_index (int) : number of the file from get_files('/../../Dataset Metric/')
        Returns:
            - dt : DataTree from netCDF file
            - filename : of the file loaded
    """
    # load data
    file_path = os.path.abspath(os.path.dirname(__file__)) + '/../../Dataset Metric/' # we can run this file from any directory in terminal

    dt = xr.open_datatree(file_path + filename)

    return dt

def get_all_latitude_longitude():
    """
        Get latitude and longitude values from all file in Data folder

        Returns : 
            latitude (List<String>): list of all latitudes
            longitude (List<String>): List of all longitudes
    """
    # load data
    file_path = os.path.abspath(os.path.dirname(__file__)) + '/../../Data/' # we can run this file from any directory in terminal
    
    files_name = get_files(file_path)

    latitude = []
    longitude = []

    for i in range(len(files_name)):
        if(files_name[i].startswith("._")):
            continue
        print("Processing file:", files_name[i])
        dt = xr.open_datatree(file_path + files_name[i])
        dt_latitude = dt["/Platform/Position/latitude"].values
        dt_longitude = dt["/Platform/Position/longitude"].values

        latitude = latitude + list(dt_latitude)
        longitude = longitude + list(dt_longitude)

    return latitude, longitude