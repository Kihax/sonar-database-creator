from lib import DataManagement, Sonar
from lib.file_management import get_all_latitude_longitude, get_files, get_tree_from_index, get_tree_from_file
from lib.DatabaseCreator import DatabaseCreator
from lib.timetag_to_timestamp import timetag_to_timestamp
import numpy as np
from matplotlib import pyplot as plt


dt = get_tree_from_file("2024__0930834_Binned.nc")

dts = [dt]

data = DataManagement(dts)

bathymetry_port = dts[0]["/Sonar/Bathymetry/Port/Block_0"]
bathymetry_starboard = dts[0]["/Sonar/Bathymetry/Starboard/Block_0"]

print(bathymetry_port)

angle_port = bathymetry_port["angle"].values
angle_starboard = bathymetry_starboard["angle"].values

timetag_port = bathymetry_port["timetag"].values
timetag_starboard = bathymetry_starboard["timetag"].values

quality_port = bathymetry_port["quality"].values
quality_sideboard = bathymetry_starboard["quality"].values

time_starboard = bathymetry_starboard["time"].values
time_port = bathymetry_port["time"].values

def compute_profile(ping_number):
    """
    Compute corrected bottom profile from one ping.
    Adapted from script_base.ipynb
    """
    
    # Load the bathymetry time data for this ping (directly from xarray dataset)
    dt = dts[0]  # Use first dataset
    
    ping_timetag = timetag_to_timestamp(dt['/Sonar/Bathymetry/Port/Block_0/timetag'][ping_number].values)
    
    bathy_port_0_time = dt['/Sonar/Bathymetry/Port/Block_0/time'][ping_number, :].values
    bathy_stbd_0_time = dt['/Sonar/Bathymetry/Starboard/Block_0/time'][ping_number, :].values
    
    # Quality data for each sample (0 = bad, >0 = good)
    bathy_port_0_quality = dt['/Sonar/Bathymetry/Port/Block_0/quality'][ping_number, :].values
    bathy_stbd_0_quality = dt['/Sonar/Bathymetry/Starboard/Block_0/quality'][ping_number, :].values
    
    # Compute absolute timetag for every sample
    bathy_port_0_timetag = ping_timetag + bathy_port_0_time * 1e9
    bathy_stbd_0_timetag = ping_timetag + bathy_stbd_0_time * 1e9
    
    n_samples = len(bathy_port_0_time)
    
    # Get roll for every sample using DataManagement interpolation
    roll_smp_port = np.array([
        data.roll.get_value_from_timestamp(bathy_port_0_timetag[k])
        for k in range(n_samples)
    ])
    roll_smp_stbd = np.array([
        data.roll.get_value_from_timestamp(bathy_stbd_0_timetag[k])
        for k in range(n_samples)
    ])
    
    # Correct beam roll with attitude roll at reception time
    bathy_port_0_angle = dt['/Sonar/Bathymetry/Port/Block_0/angle'][ping_number, :].values
    bathy_port_0_angle_roll = bathy_port_0_angle - roll_smp_port - 0.18
    
    bathy_stbd_0_angle = dt['/Sonar/Bathymetry/Starboard/Block_0/angle'][ping_number, :].values
    bathy_stbd_0_angle_roll = bathy_stbd_0_angle - roll_smp_stbd + 0.28
    
    # Compute bottom profile coordinates
    x_port_roll = 1475. * bathy_port_0_time / 2.0 * np.sin(bathy_port_0_angle_roll * np.pi / 180.)
    z_port_roll = -1475. * bathy_port_0_time / 2.0 * np.cos(bathy_port_0_angle_roll * np.pi / 180.)
    
    x_stbd_roll = 1475. * bathy_stbd_0_time / 2.0 * np.sin(bathy_stbd_0_angle_roll * np.pi / 180.)
    z_stbd_roll = -1475. * bathy_stbd_0_time / 2.0 * np.cos(bathy_stbd_0_angle_roll * np.pi / 180.)
    
    # Heave correction using DataManagement
    heave_smp_port = np.array([
        data.heave.get_value_from_timestamp(bathy_port_0_timetag[k])
        for k in range(n_samples)
    ])
    heave_smp_stbd = np.array([
        data.heave.get_value_from_timestamp(bathy_stbd_0_timetag[k])
        for k in range(n_samples)
    ])
    
    # Filter bad samples using boolean indexing
    bad_samples_port = bathy_port_0_quality == 0
    x_port = x_port_roll[~bad_samples_port]
    z_port = (z_port_roll - heave_smp_port)[~bad_samples_port]
    
    bad_samples_stbd = bathy_stbd_0_quality == 0
    x_stbd = x_stbd_roll[~bad_samples_stbd]
    z_stbd = (z_stbd_roll - heave_smp_stbd)[~bad_samples_stbd]
    
    return x_port, z_port, x_stbd, z_stbd

fig, ax = plt.subplots()
ax.set_title("Bottom profiles")
ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.grid(True)

# Plot multiple pings like script_base does
ping_numbers = [100, 150]
for ping in ping_numbers:
    x_port, z_port, x_stbd, z_stbd = compute_profile(ping)
    ax.plot(x_port, z_port, label=f"Ping {ping} Port")
    ax.plot(x_stbd, z_stbd, label=f"Ping {ping} Stbd")

ax.legend()
plt.show()
