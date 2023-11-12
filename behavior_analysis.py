"""
Version: 05/10/2023
@author: jpinho, modified by mcanela
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import pandas as pd
from settings import upaths
from tqdm import tqdm
import cv2
import pickle as pck
from pims import Video
import datetime
# Note: Camera induces a lot of warpping

def deleting_previous_data(lengths_path=upaths['lengths_path'], dlcpath=upaths['dlcpath']):
    # Convert the folder path to a Path object
    lengths_path = Path(lengths_path)
    dlcpath = Path(dlcpath)

    # Get a list of file paths in the folder
    csv_files = list(lengths_path.glob("*.csv"))
    pck_files = list(dlcpath.glob("*.pck"))

    # Delete each  file
    for file in csv_files:
        file.unlink()
    for file in pck_files:
        file.unlink()

deleting_previous_data()


def find_video(dlc_filepath):
    folder = upaths['video_path']
    all_avi_files = folder.glob('*.avi')
    for movie in all_avi_files:
        movie_name = movie.stem
        if movie_name in dlc_filepath.stem:
            return movie


def find_dat_file(dlc_filename, poly_folder):
    all_dat_files = poly_folder.rglob('*.dat')
    for dat_file in all_dat_files:
        dat_name = dat_file.stem
        if dat_name in dlc_filename:
            return dat_file


def get_events(dat_path):
    with open(dat_path, 'r') as fp:
        n_lines_header = 0
        for line in fp:
            n_lines_header += 1
            line = line.strip()
            if line == '':
                break
    events = np.genfromtxt(dat_path, delimiter='\t', skip_header=n_lines_header, dtype=int)
    return events


def read_rec_duration(events, event_num=11):
    eleven = events[events[:, 1] == event_num]
    last_ts = eleven[-2:, 0]
    is_ttl = np.any(events[:, 1] == 15)  # TTL event should be type 15
    return last_ts, is_ttl


def open_data(data_path, poly_folder, bodypart='center', likelihood_th=0.98):
    """
    Open data from DLC, polyboxes as well as the corresponding video
     
    Parameters
    ----------
    data_path: Path or str
        Path to the DLC file to open
    poly_folder: Path or str
        Path to the main folder containing dat files from polyboxes
    bodypart: str
        Which bodypart to load from the DLC file
    likelihood_th: float
        Threshold value on the likelihood given by DLC

    Returns
    -------
    last_ts: np.ndarray
        2 elements, int, with the total duration of polybox recording.
        Not sure about which one to use, but using the first one for now
    frame_rate: int
        Frame rate of the video (vary between 20 and 30)
    missing_frames: int
        How many frames are missing from the beginning (we assume) of the video
        Those frames are also missing rows from the DLC data
    x: np.ndarray
        X coordinates of the bodypart. Starts with a buch of nans, as many as missing frames
    y: np.ndarray
        Y coordinates of the bodypart. Starts with a buch of nans, as many as missing frames
    """
    data_path = Path(data_path)
    poly_folder = Path(poly_folder)
    filename = data_path.stem
    dat_path = find_dat_file(filename, poly_folder)
    events = get_events(dat_path)
    last_ts, is_ttl = read_rec_duration(events)
    video_path = find_video(data_path)
    v = Video(video_path)
    frame_rate = int(v.frame_rate)
    n_frames = len(v)
    v.close()
    dlc_data = open_dlc_data(data_path, bodypart, likelihood_th)
    assert n_frames == dlc_data[0].shape[0]
    missing_frames = int((last_ts[0] / 1000) * frame_rate - n_frames)
    nans = np.zeros(missing_frames) + np.nan
    dlc_data = tuple([np.hstack((nans, c)) for c in dlc_data])
    return last_ts, frame_rate, missing_frames, dlc_data[0], dlc_data[1]


# habituation/box ab/20211103_ERC_julia_AStrocytesHPC_Habituation_box ab_01_01_1DLC_resnet50_FearDetectionJun17shuffle1_100000.csv
# habituation/20211103_ERC_julia_AStrocytesHPC_Habituation_box ab_01_01.dat
def open_dlc_data(data_path, bodypart='center', likelihood_th=0.98):
    """
    Open a file
    how to run: 
        open_data(data_path, bodypart='center')
        
    Parameters:
    ----------
    data_path
    path of the folder where the output data of Deeplabcut are
    
        

    Returns
    -------
    It will open the file to be analysed

    """
    # Done: Given the option to chose a body part and return only the data related to it
    with open(data_path, 'r') as fp:
        _ = fp.readline()
        parts = fp.readline().split(',')
    parts = [p.strip() for p in parts]
    if bodypart not in parts:
        raise ValueError(f'{bodypart} is not a valid bodypart')
    cols = [ix for ix, p in enumerate(parts) if p == bodypart]
    # Open a file in data_path
    dlc_data = np.genfromtxt(data_path, delimiter=',', skip_header=3)
    dlc_data = dlc_data[:, cols]
    coords = likelihoodtreshold(dlc_data, False, likelihood_th)

    return coords


def nan_removal(sig: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Perform a spline interpolation on a signal to remove missing values

    Parameters
    ----------
    sig: np.ndarray
        Signal with missing values
    t: np.ndarray
        Time vector

    Returns
    -------
    sig: np.ndarray
        Signal with interpolated values and no more nans
    """
    sig = sig.copy()
    is_nan = np.isnan(sig)
    if not np.any(is_nan):
        return sig
    is_not_nan = np.logical_not(is_nan)
    sig_correct = sig[is_not_nan]
    t_correct = t[is_not_nan]
    t_nan = t[is_nan]
    spline = splrep(t_correct, sig_correct, s=0)
    filling = splev(t_nan, spline)
    sig[is_nan] = filling
    return sig


def likelihoodtreshold(dlc_data, show=False, likelihood_th=0.98):
    """
    treshold data using likelihood
    how to run:
        likelihoodtreshold(dlc_data, show=False, likelihood_th=0.98)
    
    Parameters:
    ----------
    dlc_data
    The file to be analised
    

    Returns
    -------
   The x and y coordenates treshold using the likelood lower than 0.5

    """
    # Done: Diagnostic plot
    # treshold according likelihood the outliers track points in xx coordinate
    # Done: Simplify using masking with boolean arrays
    data = dlc_data.copy()
    # data below the likehood treshold replaced by nan
    data[dlc_data[:, 2] < likelihood_th, :2] = np.nan
    nan_frames = np.any(np.isnan(data[:, :2]), axis=1)
    t = np.arange(data.shape[0])
    data[:, 0] = nan_removal(data[:, 0], t)  # xx data being interpolated by nan removal function
    data[:, 1] = nan_removal(data[:, 1], t)  # xx data being interpolated by nan removal function
    if show:
        plt.plot(data[:, 0], data[:, 1])
        if np.sum(nan_frames) > 0:
            plt.plot(data[nan_frames, 0], data[nan_frames, 1], 'rx')
    return data[:, 0], data[:, 1]


def convert_cm(xcoordinate, ycoordinate, xscale=20 / 600, yscale=15 / 450):
    """
    convert pixels in cm
    how to run:
           convert_cm(xcoordinate, ycoordinate, xscale=20 / 600, yscale=15 / 450)
    
    Parameters:
    ----------
   xcoordinate
       coordinate in x axis in pixels
   ycoordinate
       coordinate in y axis in pixels
       

    Returns
    -------
   The x and y coordenates in centimeters

    """
    x_cm = xcoordinate * xscale
    y_cm = ycoordinate * yscale
    return x_cm, y_cm


# Done: Merge both functions into one
# Done: Parametrize conversion
# TODO: Show a frame for user to click on and measure


def euclidean_distance(x_cm, y_cm):
    """
    calculate eucidean distance
    how to run:
        euclidean_distance(x_cm, y_cm)

    Parameters:
    ----------
   x_cm
       coordinate in x axis in centimeters
   y_cm
       coordinate in y axis in centimeters
    
    
    Returns
    -------
   e_distance
       The euclidean distance computed using square root of x elevated to 2 +  y elevated to 2 ,
       where x an y are the difference between pairs of frames in x_cm and y_cm,

    """
    # calculate the euclidean distance between each pairs of frames
    x = np.diff(x_cm, axis=0)
    y = np.diff(y_cm, axis=0)
    e_distance = np.sqrt((x * x) + (y * y))
    return e_distance


def speed(e_distance, framerate=25):
    """
    calculate speed
    how to run:
        speed(e_distance, framerate=25)
    
    Parameters
    ----------
    e_distance
        the euclidean distance
    framerate
    

    Returns
    -------
    velocity: np.ndarray
        the velocity for frame in mm/s

    -------

    """
    convertcm_mm = 10
    # Done: Parameter
    # calculate mean velocity
    velocity = e_distance / (1 / framerate) * convertcm_mm
    # Done: Maybe this should be velocity = Edistance / (1 / framerate)
    return velocity

def distance_function(e_distance):
    """
    Parameters
    ----------
    e_distance

    Returns
    -------
    velocity: np.ndarray
        the distance in mm

    -------

    """
    convertcm_mm = 10
    distance = e_distance * convertcm_mm
    return distance


def is_freezing(e_distance, dist_th=0.02):
    """
    Freezing detection based on thresholding the e_distance travelled during a pair of frames
    how to run:
        is_freezing(e_distance, dist_th=0.02)

    Parameters
    ----------
    e_distance: np.array
    dist_th: float
        Distance threshold in cm

    Returns
    -------
    freezing: boolean np.ndarray
        Indicates if the animal is freezing

    """
    return e_distance < dist_th


def sliding_freezing(measure, framerate=25, win_size=1, min_duration=2):
    """
    Compute the moving sum of a signal given in input, over a window
    of size win_size

    Parameters
    ----------
    measure: numpy boolean array
        Boolean array to sum over a sliding window
    win_size: int
        Width of the window, in seconds
        Default to 5
    min_duration: int
        Minimum duration during which there must be True's to get a True over a window
    """
    win_pts = win_size * framerate
    min_pts = min_duration * framerate
    if win_pts > len(measure):
        # This a special case when input array is shorter than the window
        # We could: return None, raise an error
        return np.zeros(measure.shape, dtype=bool)
    # Summing kernel, here we don't normalize it to get the sum
    kernel = np.ones(win_pts)
    # Convolution
    meas_sum = np.convolve(measure, kernel, 'same')
    output = (meas_sum > min_pts)

    return output


def binning(measure, bin_duration=60, win_duration=20, framerate=25, agg_func=np.sum):
    """
    Bin a variable with a bin duration of 60 seconds, for a window duration of 20 minutes,
    framerate of 20 and function default sum
    how to run:
        binning(velocity, bin_duration=60, win_duration=20, framerate=25, agg_func=np.sum)


    Parameters
    ----------
    measure: it could be any variable to be analysed by bins - np.ndarray
    bin_duration :  bin duration of 60 seconds
    win_duration : duration of the experiment (20 minutes)
    framerate : frame rate of the videos (25 fps)
    agg_func : default function

    Returns
    -------
    arr_meas: bins np.array

    """
    bin_size = bin_duration * framerate
    frame_per_min = 60 * framerate
    bins = np.arange(0, win_duration * frame_per_min, bin_size)
    l_meas = []
    for c_bin in bins:
        binned_meas = measure[c_bin:c_bin + bin_size]
        if len(binned_meas) > 0:
            binned_measure = agg_func(binned_meas)
        else:
            binned_measure = np.nan
        l_meas.append(binned_measure)
    arr_meas = np.array(l_meas) 
    return arr_meas


def freezing_periods(is_freezed, min_duration=2, framerate=25):
    w, = np.nonzero(np.diff(is_freezed))
    n_tr = len(w)
    if n_tr == 0:
        return np.array([])
    if n_tr % 2:
        if is_freezed[-1]:
            w = np.hstack((w, len(is_freezed)))
        else:
            w = np.hstack((0, w))
    periods = w.reshape((-1, 2))
    durations = np.squeeze(np.diff(periods, axis=1))
    gi = durations > min_duration * framerate
    periods = periods[gi]
    return periods


def freezing_speed_quantif(e_distance, velocity, distance_var, dist_th=0.02, framerate=25,
                           bin_duration=60, win_duration=20, min_duration=0.5, show=False,
                           MFD=0.1, MAD=0.5, mouse_id='mouse_id'):
    """
    Freezing and speed quantification
    How to run:
        freezing_speed_quantif(e_distance, velocity, dist_th=0.02, framerate=25,
                           bin_duration=60, win_duration=20, show=False)

    Parameters
    ----------
    e_distance : instantant eucleadian distance - np. array
    velocity : instant velocity - np.array
    dist_th : freezing threshold (0.02)
    framerate : frame rate of the videos (25 fps)
    bin_duration : bin duration of 60 seconds
    win_duration : duration of the experiment (20 minutes)
    min_duration (MFD): float
        Minimum duration of immobility, in seconds, before it is considered freezing
    show: boolean
        Plot or not? Default to False
    MFD: Minimum Freeze Duration
    MAD: Minimum Activity Duration

    Returns
    -------
    res: dict
        dur_freezing: Total duration of freezing in seconds (float)
        freezing_bin : freezing by bins - np.array
        speed: Velocity - np.ndarray
        speed_bin : speed by bins - np.array
        speed_freezing : speed during freezing
        speed_moving : speed out of the freezing period
        perc_freezing: Percentage of total time spent freezing - float

    """
    
 
    n_frames = len(e_distance)
    freezing = is_freezing(e_distance, dist_th)
    # freezing = freezing_blocks(freez, framerate)
    # freezing = sliding_freezing(freez, framerate, win_size=5, min_duration=1)
    
    # =============================================================================
    # A function useful for both MFD and MAD
    # =============================================================================   
    
    def moving_window(lst, window_size):
        window = []
        result = []
        for i in range(len(lst)):
            window.append(lst[i])
            if len(window) == window_size:
                result.append(tuple(window))
                window.pop(0)
        return result
    
    # =============================================================================
    # Applying a MFD to the freezing
    # =============================================================================   
    
    window_MFD = round(MFD * framerate)
    
    list_of_windows = moving_window(freezing, window_MFD)
    boolean_list_of_windows = []
    for value in list_of_windows:
        boolean_list_of_windows.append(all(value))
    
    list_of_indexes = [x for x in range(len(freezing))]
    list_of_windows_indexes = moving_window(list_of_indexes, window_MFD)

    dictionary = dict(zip(list_of_windows_indexes, boolean_list_of_windows))

    true_index_list = []
    for key, value in dictionary.items():
        if value == True:
            true_index_list.append(key)

    new_true_index_list = [item for sublist in true_index_list for item in sublist]
    new_true_index_list = list(set(new_true_index_list))

    new_freezing = []
    for new_index in range(len(freezing)):
        if new_index in new_true_index_list:
            new_freezing.append(True)
        if new_index not in new_true_index_list:
            new_freezing.append(False)
    
    # freezing = np.array(new_freezing)
    
    # =============================================================================
    # Applying a MAD to the freezing
    # =============================================================================   
    
    window_MAD = round(MAD * framerate)
    
    list_of_windows_MAD = moving_window(freezing, window_MAD)
    boolean_list_of_windows_MAD = []
    
    def all_false(lst):
        for elem in lst:
            if elem:
                return False
        return True
    
    for value in list_of_windows_MAD:
        boolean_list_of_windows_MAD.append(all_false(value))
    
    list_of_indexes_MAD = [x for x in range(len(freezing))]
    list_of_windows_indexes_MAD = moving_window(list_of_indexes_MAD, window_MAD)

    dictionary_MAD = dict(zip(list_of_windows_indexes_MAD, boolean_list_of_windows_MAD))

    true_index_list_MAD = []
    for key, value in dictionary_MAD.items():
        if value == True:
            true_index_list_MAD.append(key)

    new_true_index_list_MAD = [item for sublist in true_index_list_MAD for item in sublist]
    new_true_index_list_MAD = list(set(new_true_index_list_MAD))

    new_freezing = []
    for new_index in range(len(freezing)):
        if new_index in new_true_index_list_MAD:
            new_freezing.append(False)
        if new_index not in new_true_index_list_MAD:
            new_freezing.append(True)
    
    # freezing = np.array(new_freezing)

    # =============================================================================
    # Calculating the duration of the freezing events
    # =============================================================================
    
    freezing_list = list(freezing)
    
    # To count the freezing events of a particular period
    def slicer_in_periods(lst, begining_min=0, ending_min=3, framerate=25):
        b = round(begining_min * 60 * framerate)
        e = round(ending_min * 60 * framerate)
        new_lst = lst[b:e]
        return new_lst
    
    master_lst = {}
    
        # OFF PERIOD
    # off = slicer_in_periods(freezing_list, 0, 3, framerate)
    # if not off:
    #     master_lst['lengths_off'] = [False, False]
    # else:
    #     master_lst['lengths_off'] = off
        
        # ON PERIOD
    # on = slicer_in_periods(freezing_list, 3, 8, framerate)
    # if not on:
    #     master_lst['lengths_on'] = [False, False]
    # else:
    #     master_lst['lengths_on'] = on
        
        # OFF PERIOD (1st min)
    # first_off = slicer_in_periods(freezing_list, 0, 1, framerate)
    # if not first_off:
    #     master_lst['lengths_first_off'] = [False, False]
    # else:
    #     master_lst['lengths_first_off'] = first_off
    
        # OFF PERIOD (last min)
    last_off = slicer_in_periods(freezing_list, 2, 3, framerate)
    if not last_off:
        master_lst['lengths_last_off'] = [False, False]
    else:
        master_lst['lengths_last_off'] = last_off
    
        # ON PERIOD (1st min)
    first_on = slicer_in_periods(freezing_list, 3, 4, framerate)
    if not first_on:
        master_lst['lengths_first_on'] = [False, False]
    else:
        master_lst['lengths_first_on'] = first_on
    

    # ATTENTION: From this point, the functions become compiled and iterated in the end
    
    # To extract the amount of freezing
    def extract_sequences(lst):    
        """
        This function takes a list lst as an argument and returns a list of
        tuples representing the start and end indices of non-empty sequences
        in the input list.
        """
        
        sequences = []
        start = None
    
        for i in range(len(lst)):
            if lst[i]:
                if start is None:
                    start = i
            else:
                if start is not None:
                    sequences.append((start, i))
                    start = None
                
        if start is not None:
            sequences.append((start, len(lst)))
      
        return sequences
    
    def get_lengths(sequences):
        """
        This function takes a list of tuples sequences as an argument, where
        each tuple represents the start and end indices of a non-empty sequence
        in a list. The function returns a list of lengths of the non-empty
        sequences in sequences.
        """
        
        lengths = []
        for start, end in sequences:
            length = end - start
            lengths.append(length)
        return lengths
    
    def divide_list(lst, value):
        """
        This function takes a list lst and a number value as inputs and
        returns a new list where each element of the input list is divided by
        the input value.
        """
        
        result = []
        for item in lst:
            result.append(item / value)
        return result
    
    # def count_numbers(numbers):
    #     """
    #     This function takes a list of numbers numbers as an argument and
    #     returns a dictionary where each key is a unique number from the input
    #     list and its corresponding value is the count of that number in the
    #     input list.
    #     """
        
    #     counts = {}
    #     for num in numbers:
    #         if num in counts:
    #             counts[num] += 1
    #         else:
    #             counts[num] = 1
    #     return counts
    
    # def sum_values_interval(counts, interval):
    #     """
    #     This function takes two arguments: a dictionary counts where each key
    #     is a number and its corresponding value is the count of that number,
    #     and an interval represented as a tuple of two numbers. The function
    #     returns the sum of the counts of all the numbers in the input
    #     dictionary that fall within the input interval.
    #     """
        
    #     total = 0
    #     for key in counts:
    #         if interval[0] <= key < interval[1]:
    #             total += counts[key]
    #     return total    
    
    # def dictionary_sum_values_interval(lst, tag):
    #     """
    #     This function takes two arguments: a list lst of numbers and a string tag.
    #     The function divides the range of numbers in the input list into ? bins
    #     of width ?, and for each bin, it calculates the sum of the numbers
    #     that fall within that bin using the sum_values_interval function.
    #     The function then returns a dictionary where each key is a string
    #     representing a bin interval and the corresponding value is the sum of
    #     the numbers in that interval.
    #     """
        
    #     results_dict = {}
    #     length_of_each_bin = 0.1 # Change this number to modify the number of bins
    #     total_length = 7
    #     number_of_bins = int(total_length / length_of_each_bin)
        
    #     for i in range(number_of_bins): 
    #         interval_start = i * length_of_each_bin
    #         interval_end = interval_start + length_of_each_bin
    #         interval = (interval_start, interval_end)
    #         result = sum_values_interval(lst, interval)
    #         results_dict[tag + ' ' + str(interval_start) + '-' + str(interval_end)] = result
    #     return results_dict
        
    
    # AFTER DEFINING THE FUNTIONS: Create a function to iterate through all the
    # functions created to cound duration of freezing
    def iterate_freeze_duration(lst_of_lsts, framerate):
        dicts = {}
        
        for tag, lst in lst_of_lsts.items():
            lst_2 = extract_sequences(lst)
            lst_3 = get_lengths(lst_2)
            lst_4 = divide_list(lst_3, framerate)
            # lst_5 = count_numbers(lst_4)
            
            # lst_6 = dictionary_sum_values_interval(lst_5, tag)

            # dicts.append(lst_6)
            
            dicts[tag] = [lst_4]
                    
        return dicts
    
    
    # Create the list of dictionaries based on the previous functions
    my_dicts = iterate_freeze_duration(master_lst, framerate)

    
    # =============================================================================           
    
    empty_bins = binning(np.ones(n_frames), bin_duration, win_duration, framerate, np.sum) == 0
    n_freezing_frames = binning(freezing, bin_duration, win_duration, framerate, np.sum)
    sec_freezing = n_freezing_frames / framerate  # Freezing duration, binned
    sec_freezing[empty_bins] = np.nan
    speed_bin = binning(velocity, bin_duration, win_duration, framerate, np.mean)  # Speed binned
    distance_bin = binning(distance_var, bin_duration, win_duration, framerate)  # Distance binned
    speed_bin[empty_bins] = np.nan
    distance_bin[empty_bins] = np.nan
    speed_freezing = np.mean(velocity[freezing])
    speed_moving = np.mean(velocity[np.logical_not(freezing)])
    perc_freezing = 100 * np.nansum(n_freezing_frames) / n_frames
    no_freezing_velocity = velocity.copy()
    no_freezing_velocity[freezing] = np.nan
    speed_moving_bin = binning(no_freezing_velocity, bin_duration, win_duration, framerate, np.nanmean) #speed moving binned
    speed_moving_bin[empty_bins] = np.nan 
    if show:
        speed_color = 'darkred'
        freezing_color = 'darkblue'
        fig, ax_fr = plt.subplots()
        t = np.arange(0, win_duration * 60, bin_duration)
        ax_sp = ax_fr.twinx()
        ax_fr.plot(t, sec_freezing, c=freezing_color)
        ax_fr.set_ylabel('Binned time (s)', color=freezing_color)
        ax_fr.tick_params(axis='y', labelcolor=freezing_color)
        ax_sp.plot(t, speed_bin, c=speed_color)
        ax_sp.set_ylabel('Animal velocity (mm/s)', color=speed_color)
        ax_sp.tick_params(axis='y', labelcolor=speed_color)
        fig.set_tight_layout(True)
    res = {'dur_freezing': np.nansum(sec_freezing), 'freezing_bin': sec_freezing*100/bin_duration,
           'velocity': velocity, 'speed_bin': speed_bin, 'distance_bin': distance_bin, 'speed_freezing': speed_freezing,
           'speed_moving': speed_moving, 'perc_freezing': perc_freezing, 'speed': np.mean(velocity),
           'distance': np.sum(e_distance), 'freezing': freezing, 'dist': e_distance,
           'freeze_periods': freezing_periods(freezing, min_duration=min_duration, framerate=framerate),
           'speedmoving_bin': speed_moving_bin, } # attention dont call binned variables like speed_moving_bin
    # being that the function of analyse all data is searching for _bin wil skip this variable because have a _ before_bin
    
    # res.update(my_dicts)
    
    mouse_id.update(my_dicts)
    lengths_df = pd.DataFrame.from_dict(mouse_id)
    
    directory_path = 'C:/Users/mcanela/Desktop/Python/Freezing periods'
    filename = 'lengths_' + mouse_id['file'] + '.csv'
    filepath = directory_path + '/' + filename
    lengths_df.to_csv(filepath, index=False)
    
    return res


def create_velocity_map(results, bin_size=.5):
    """
    velocity map, color coded by the color in the x and y map
    how to run:
        create_velocity_map(results, bin_size=.5)

    Parameters
    ----------
    results: dict
    bin_size: float
        In cm (this bin size is the spatial window that we selected to average the speed)

    Returns
    -------
    v_binned:  np.ndarray
        2D array with average speed for each bin

    """
    x, y = results['x'][:-1], results['y'][:-1]
    v = results['velocity']
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xbins = np.arange(xmin, xmax, bin_size)  # bins in xx dimension
    ybins = np.arange(ymin, ymax, bin_size)  # bins in yy dimension
    v_binned = np.zeros((len(ybins), len(xbins))) + np.nan  # v_binned empty with the proper size

    for ix_xb, xb in enumerate(xbins[:-1]):
        xn = xbins[ix_xb + 1]
        for ix_yb, yb in enumerate(ybins[:-1]):
            yn = ybins[ix_yb + 1]
            x_cond = np.logical_and(x >= xb, x < xn)
            y_cond = np.logical_and(y >= yb, y < yn)
            mask = np.logical_and(x_cond, y_cond)
            if np.sum(mask) == 0:
                continue
            avg_speed = np.mean(v[mask])  # compute the average speed in each spatial bin
            v_binned[ix_yb, ix_xb] = avg_speed
    return v_binned


def split_by_freezing(measurement, freezing):
    """
    this function split a variable during freezing and moving periods

    Parameters
    ----------
    measurement: np.ndarray
        variable to be splitted
    freezing: np.ndarray
        Boolean variable that contain the information if the animal is in freezing for each frame

    Returns
    --------
    averages: dict
        the average of the variable (in measurement) during freezing and moving
        {'avg': np.ndarray, 'freezing': np.ndarray, 'moving': np.ndarray}

    -------

    """
    if len(measurement) == len(freezing) + 1:
        measurement = measurement[1:]
    in_freezing = measurement[freezing]  # filter the variable to the moments in freezing
    out_freezing = measurement[
        np.logical_not(freezing)]  # filter the variable to the moments moving
    averages = {'avg': np.nanmean(measurement),  # average of the variable without filters
                'freezing': np.nanmean(in_freezing),  # average of the variable filtered by freezing
                'moving': np.nanmean(out_freezing)}  # average of the variable filtered by moving
    return averages


def pose(data_path, poly_folder, dist_th=0.02):
    """
    Calculate the pose of the animal (each bodyparts distance or all body) during all time
    or splited by freezing or moving time
    how to run:
        create_velocity_map(results, bin_size=.5)

    Parameters
    ----------
    data_path: variable to be splitted
    dist_th : freezing treshold (0.02)

    Returns
    --------
    pose_results: dict.

    """
    data_path = Path(data_path)
    bodyparts = ('nose', 'head', 'center', 'tail')  # bodyparts to be extracted
    part_pairs = list(zip(bodyparts, bodyparts[1:]))  # iterate 2 list to do the pairs of bodyparts
    bp_pos = {}
    # for each bodypart open the file, interpolate by likelihood and convert coordinate pixels in cm
    for bp in bodyparts:
        _, _, _, x, y = open_data(data_path, poly_folder, bp, likelihood_th=0.99)
        # x, y = likelihoodtreshold(bp_data, show=False, likelihood_th=0.99)
        x, y = convert_cm(x, y)
        bp_pos[bp] = (x, y)

    bp_dist = {}
    total_pose = np.zeros(
        len(bp_pos['nose'][0]) - 1)  # create the total pose with the proper size filled of zeros
    for pair in part_pairs:  # calculate the euclidean distance between each pair of body parts
        p1 = bp_pos[pair[0]]
        p2 = bp_pos[pair[1]]
        dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        bp_dist[f'{pair[0]}-{pair[1]}'] = dist[1:]  # Removing first frame, to match freezing length
        total_pose = total_pose + dist[1:]  # compute the total distance adding all the pairs
    e_distance = euclidean_distance(bp_pos['center'][0], bp_pos['center'][1])
    freezing = is_freezing(e_distance, dist_th)
    total_pose_avg = split_by_freezing(total_pose, freezing)
    bp_len_freezing = {}
    # compute the average for each bodypart length splitting by freezing and moving
    for pair, length in bp_dist.items():
        avgs = split_by_freezing(length, freezing)
        bp_len_freezing[pair] = avgs
    bp_len_freezing['total'] = total_pose_avg
    pose_results = {'bodyparts_lengths': bp_dist, 'animal_length': total_pose,
                    'pose_freezing': bp_len_freezing}
    return pose_results


def save_data(results, data_file):
    """

    Parameters
    ----------
    results: dict
        Dictionary with computed data as returned from analyze_mouse
    data_file: Path

    Returns
    -------
    outputfile with all this data
    format .npz

    -------

    """
    # Done: Change name according to raw data file name
    # create a output file with the data

    output_file = get_saved_filepath(data_file)
    with open(output_file, 'wb') as pf:
        pck.dump(results, pf)


# TODO: Update the docstring
def load_freezing_data(data_file):
    """
    Reload computed parameters from NPZ file as saved by the save_data function

    Parameters
    ----------
    data_file: str or Path
        PAth to the NPZ file

    Returns
    -------
    res: dict
        dur_freezing: Total duration of freezing in seconds (float)
        freezing_bin : freezing by bins - np.array
        speed: Velocity - np.ndarray
        speed_bin : speed by bins - np.array
        speed_freezing : speed during freezing
        speed_moving : speed out of the freezing period
        perc_freezing: Percentage of total time spent freezing - float
        x xx coordinates
        y yy cooordinates
        time timecourse
        pose length of body
        v_map velocity map
    """
    results = {}
    with np.load(data_file) as data:
        for k, v in data.items():
            results[k] = v
    return results


def get_saved_filepath(data_path):
    output_file = data_path.parent / f'{data_path.stem}.pck'
    return output_file


def validate_saved(data_path, force, parameters):
    output_file = get_saved_filepath(data_path)
    saved_ok = False
    res = {}
    if not force and output_file.exists():
        with open(output_file, 'rb') as pf:
            res = pck.load(pf)
        saved_ok = True
        if 'params' in res.keys():
            for prm_name, value in parameters.items():
                if res['params'][prm_name] != value:
                    saved_ok = False
                    break
        else:
            saved_ok = False
    return saved_ok, res


def analyze_mouse(data_path, poly_folder, bodypart='center', likelihood_th=0.98,
                  dist_th=0.02, bin_duration=10, win_duration=20, min_duration=2, force=False,
                  mouse_id='mouse_id'):
    data_path = Path(data_path)
    parameters = {'dist_th': dist_th, 'bin_duration': bin_duration,
                  'win_duration': win_duration, 'min_duration': min_duration}
    saved_ok, res = validate_saved(data_path, force, parameters)
    if saved_ok:
        return res
    if not data_path.exists():
        return {}
    _, framerate, _, x, y = open_data(data_path, poly_folder, bodypart, likelihood_th)
    dt = 1 / framerate
    time = np.arange(0, x.shape[0] * dt, dt)
    x, y = convert_cm(x, y)
    dist = euclidean_distance(x, y)
    velocity = speed(dist, framerate)
    distance_var = distance_function(dist)
    res = freezing_speed_quantif(dist, velocity, distance_var, dist_th, framerate, bin_duration, win_duration,
                                 min_duration, mouse_id=mouse_id)
    # Add coordinates into result dictionary
    res['x'] = x
    res['y'] = y
    # Also, time
    res['time'] = time
    #add a slice of freezing # meas variable in percentage # time variable duration of the slice
        
    
    # to analyse probe test in SOC or SPC 
    slice1 = slice_measurement(res['time'], res['freezing'], [180,])
    res['time_slices1'] = len(slice1[0][0])/framerate
    res['time_slices2'] = len(slice1[0][1])/framerate
    res['meas_slices1'] = (((slice1[1][0].sum())/framerate)/res['time_slices1'])*100
    res['meas_slices2'] = (((slice1[1][1].sum())/framerate)/res['time_slices2'])*100 
    
    
    # to measure boredom, we calculate the freezing in the last 5 min of the habituation phase
    slice_boredom = slice_measurement(res['time'], res['freezing'], [900,])
    res['time_slices_boredom'] = len(slice_boredom[0][0])/framerate
    res['meas_slices_boredom'] = (((slice_boredom[1][0].sum())/framerate)/res['time_slices_boredom'])*100

    # to analyse probe test last 10 seconds OFF, and first 10 seconds ON 
    slice_1min = slice_measurement(res['time'], res['freezing'], [170,180,190,])
    res['time_last_10s_off'] = len(slice_1min[0][1])/framerate
    res['time_first_10s_on'] = len(slice_1min[0][2])/framerate
    res['meas_last_10s_off'] = (((slice_1min[1][1].sum())/framerate)/res['time_last_10s_off'])*100
    res['meas_first_10s_on'] = (((slice_1min[1][2].sum())/framerate)/res['time_first_10s_on'])*100

    # to analyse probe test 1st minute OFF, last minute OFF, and 1st minute ON 
    slice_1min = slice_measurement(res['time'], res['freezing'], [60,120,180,240,300,360])
    res['time_off_1min'] = len(slice_1min[0][0])/framerate
    res['time_off_last'] = len(slice_1min[0][2])/framerate
    res['time_on_1min'] = len(slice_1min[0][3])/framerate
    res['time_on_last'] = len(slice_1min[0][5])/framerate
    res['meas_off_1min'] = (((slice_1min[1][0].sum())/framerate)/res['time_off_1min'])*100
    res['meas_off_last'] = (((slice_1min[1][2].sum())/framerate)/res['time_off_last'])*100
    res['meas_on_1min'] = (((slice_1min[1][3].sum())/framerate)/res['time_on_1min'])*100
    res['meas_on_last'] = (((slice_1min[1][5].sum())/framerate)/res['time_on_last'])*100

    # to analyse probe test 2 minute OFF vs 2 minute ON 
    slice_2min = slice_measurement(res['time'], res['freezing'], [120,180,300,])
    res['time_off_2min'] = len(slice_2min[0][0])/framerate
    res['time_on_2min'] = len(slice_2min[0][2])/framerate
    res['meas_off_2min'] = (((slice_2min[1][0].sum())/framerate)/res['time_off_2min'])*100
    res['meas_on_2min'] = (((slice_2min[1][2].sum())/framerate)/res['time_on_2min'])*100  
    
    # to analyse soc phase (soc1 G1G2 May-2023)
    slice2 = slice_measurement(res['time'], res['freezing'], [180,210,240,300,330,360,480,510,540,630,660,690,])
    # to analyse soc phase (soc2 G1G2 May-2023)
    # slice2 = slice_measurement(res['time'], res['freezing'], [180,210,240,330,360,390,450,480,510,630,660,690,])
    res['time_off1'] = len(slice2[0][0])/framerate
    res['time_tone1'] = len(slice2[0][1])/framerate
    res['time_light1'] = len(slice2[0][2])/framerate
    res['time_off2'] = len(slice2[0][3])/framerate
    res['time_tone2'] = len(slice2[0][4])/framerate
    res['time_light2'] = len(slice2[0][5])/framerate
    res['time_off3'] = len(slice2[0][6])/framerate
    res['time_tone3'] = len(slice2[0][7])/framerate
    res['time_light3'] = len(slice2[0][8])/framerate
    res['time_off4'] = len(slice2[0][9])/framerate
    res['time_tone4'] = len(slice2[0][10])/framerate
    res['time_light4'] = len(slice2[0][11])/framerate
    # res['time_off5'] = len(slice2[0][12])/framerate
    
    res['meas_off1'] = (((slice2[1][0].sum())/framerate)/res['time_off1'])*100
    res['meas_tone1'] = (((slice2[1][1].sum())/framerate)/res['time_tone1'])*100
    res['meas_light1'] = (((slice2[1][2].sum())/framerate)/res['time_light1'])*100
    res['meas_off2'] = (((slice2[1][3].sum())/framerate)/res['time_off2'])*100
    res['meas_tone2'] = (((slice2[1][4].sum())/framerate)/res['time_tone2'])*100
    res['meas_light2'] = (((slice2[1][5].sum())/framerate)/res['time_light2'])*100
    res['meas_off3'] = (((slice2[1][6].sum())/framerate)/res['time_off3'])*100
    res['meas_tone3'] = (((slice2[1][7].sum())/framerate)/res['time_tone3'])*100
    res['meas_light3'] = (((slice2[1][8].sum())/framerate)/res['time_light3'])*100
    res['meas_off4'] = (((slice2[1][9].sum())/framerate)/res['time_off4'])*100
    res['meas_tone4'] = (((slice2[1][10].sum())/framerate)/res['time_tone4'])*100
    res['meas_light4'] = (((slice2[1][11].sum())/framerate)/res['time_light4'])*100
    # res['meas_off5'] = (((slice2[1][12].sum())/framerate)/res['time_off5'])*100
    
    res['time_off_all'] =  res['time_off1'] + res['time_off2'] + res['time_off3'] + res['time_off4']
    res['meas_off_all'] = ( ((slice2[1][0].sum())/framerate) + ((slice2[1][3].sum())/framerate) + 
                           ((slice2[1][6].sum())/framerate) + ((slice2[1][9].sum())/framerate) ) / res['time_off_all'] * 100
    
    
    res['meas_1st_tone'] = res['meas_tone1']
    
    res['time_tone_all'] =  res['time_tone2'] + res['time_tone3'] + res['time_tone4']
    res['meas_tone_all'] = ( ((slice2[1][4].sum())/framerate) + ((slice2[1][7].sum())/framerate) + 
                            ((slice2[1][10].sum())/framerate)) / res['time_tone_all'] *100
    
    res['time_light_all'] =   res['time_light1'] + res['time_light2'] + res['time_light3'] + res['time_light4']
    res['meas_light_all'] = ( ((slice2[1][2].sum())/framerate) + ((slice2[1][5].sum())/framerate) + 
                             ((slice2[1][8].sum())/framerate) + ((slice2[1][11].sum())/framerate))/ res['time_light_all']  * 100
    
    
    # res['time_in_ROI'] = in_ROi(data_path, mice_path, base_path=upaths['datapath'], bp='nose', framerate=25)
    # Pose analysis
    pose_res = pose(data_path, poly_folder, dist_th)
    res['pose'] = pose_res
    fused_pose = merge_pose_freezing(pose_res['pose_freezing'])
    for k, v in fused_pose.items():
        res[k] = v
    # v_map = create_velocity_map(res)
    # res['v_map'] = v_map

    res['params'] = parameters
    save_data(res, data_path)
    return res


def merge_pose_freezing(pose_freezing):
    """
    dictionary fusing the pose data, reduction of hierarchy
    (Pose-> head -> avg   will be avg_head_pose)
    Parameters
    ----------
    pose_freezing:
        avg length to each bodypart during freezing and moving

    Returns
    -------
    fused: dict
        the same variables in a dictionary with a reduction of the hierarchy
    """
    fused = {}
    for bodyparts, avgs in pose_freezing.items():
        for key, value in avgs.items():
            fused[f'{bodyparts}_{key}'] = value
    return fused


def in_ROi(data_path, mice_path, base_path=upaths['dlcpath'], bp='nose', framerate=25):
    base_path = Path(base_path)
    mice_path = Path(mice_path)
    df = pd.read_csv(mice_path)
    for ix_row, row in tqdm(df.iterrows()):
        img_path = base_path / (row['image'] + '.png') #pick the path of each snapshot
        img = cv2.imread(img_path)
        xx,yy,ww,hh= cv2.selectROI(img) # extract the xy position of the center of the roi and the weight and height from the center
        top_left_x = (xx - ww) # top left coordinate in x axis of the roi
        top_left_y = (yy - hh) # top left coordinate in y axis of the roi
        top_right_x = (xx + ww) # top right coordinate in x axis of the roi
        bottom_left_y = (yy + hh) # botton left coordinate in y axis of the roi
    data_path = Path(data_path) 
    data = open_data(data_path)
    x, y = likelihoodtreshold(data)
    in_roi=[]
    for ix in x:
        if ix > top_left_x and ix< top_right_x: #if the coordinate of the animal is higher than top_left_x and lower than top_right_x
            for iy in y:
                if iy > top_left_y and iy < bottom_left_y: #also if the coordinate of the animal is higher than top_left_y and lower than bottom_left_y
                    in_roi.append(1) # I will save the value in the variable in_roi
                else:
                    continue
        else:
            continue
    time_in_roi = (len(in_roi))/framerate # time in ROI is the number of frames that the animals was in ROI/framerate
    return time_in_roi


def analyse_all_data(mice_path, poly_folder=upaths['poly'], base_path=upaths['dlcpath'], force=False):
    """
    Search in a directory csv files and run all the functions inside the for cycle in those files
    export a output_file with FreezingTime,FreezingBinsMin,Velocity,output_file

    Parameters
    ----------
    mice_path: str
        Path to a CSV (eventually XLS later) containing info on experiments and files
    poly_folder: str or Path
        Path to the directory containing all polyboxes files
    base_path: str or Path
        Path to the directory where data are to be found
    force: bool
        Recompute or load from disk?

    Returns
    -------
    df: pandas.DataFrame
        Contains input data from the excel file + computed parameters
    -------

    """
    # TODO: Add parameters from analyse_mouse here
    # Done: Have a per file framerate (add it to the mice_path file)
    base_path = Path(base_path)
    df = pd.read_csv(mice_path)
    for ix_row, row in tqdm(df.iterrows()):
        # data_path = base_path / (row['file'] + '.csv') 
        data_path = base_path / (str(row['file']) + '.csv')
        column_names = df.columns
        row_values = row.values
        mouse_id = dict(zip(column_names, row_values))
        
        r = analyze_mouse(data_path, poly_folder, force=force, mouse_id=mouse_id)
        fill_table(df, ix_row, r)
    mice_path = Path(mice_path)
    df.to_excel(mice_path.parent / 'all_computed_data.xlsx')
    return df


def slice_measurement(time, meas, event_times):
    """
    slice a variable according with a specific time
    how to call:
       slice_measurement(results['time'], results['perc_freezing'], [60,])

    Parameters
    ----------
    time: np.ndarray
        Time vector of the entire recording
    meas: np.ndarray
        variable to be sliced
    event_times: list, tuple or np.ndarray
        time of events. We will add the begining and the end of the recording if not
        already present time in seconds where you want to slice

    Returns
    -------
    time_slices: list
        duration of the slice
    meas_slices: list
        the value of the variable sliced


    """
    event_times = np.sort(event_times)
    if event_times[0] != time[0]:
        event_times = np.hstack(([0], event_times))
    if event_times[-1] != time[-1]:
        event_times = np.hstack((event_times, time[-1]))
    ix_evt = np.searchsorted(time, event_times)
    meas_slices = []
    time_slices = []
    for start, stop in zip(ix_evt[:-1], ix_evt[1:]):
        # iterate through all the events List 1 = all excluding last list 2 = all excluding first
        m_slice = meas[start:stop]  # compute the variable sliced for each slice
        t_slice = time[start:stop]  # compute the time sliced for each slice
        meas_slices.append(m_slice)
        time_slices.append(t_slice)
    return time_slices, meas_slices



def fill_table(df, ix_row, results):
    """
    Given the results from the analysis of one file, fill them in the global table
    
    Parameters
    ----------
    df
    ix_row
    results

    Returns
    -------

    """
    for key, value in results.items():
        if isinstance(value, np.ndarray) and '_bin' in key:
            # Deal with arrays by adding a column in the DF for each column in the array
            column_base = key.split('_')[0]
            for bin_num in range(len(value)):
                col_name = f'{column_base}_{bin_num + 1}'
                if col_name not in df.columns:
                    df[col_name] = np.nan
                df.loc[ix_row, col_name] = value[bin_num]
        elif not isinstance(value, np.ndarray) and not isinstance(value, dict):
            if key not in df.columns:
                df[key] = np.nan
            df.loc[ix_row, key] = value


if __name__ == '__main__' and True:
    df = analyse_all_data(upaths['table_path'])
    # df['group']= df['brain']+df['drug']


def create_lengths_df(df=df, lengths_path=upaths['lengths_path']):
    '''
    This function reads a collection of CSV files from a specified folder and
    concatenates them vertically into a single dataframe. It also sets the
    column names of the concatenated dataframe based on the column names in
    the first CSV file.
    '''

    lengths_path = Path(lengths_path)  # Convert the folder path to a Path object

    files = [file.name for file in lengths_path.iterdir() if file.is_file()]  # Get a list of file names in the folder

    # create an empty dataframe with the same columns as the CSV files
    df_lengths = pd.read_csv(lengths_path / files[0])
    df_lengths = df_lengths.iloc[0:0]

    # loop through each CSV file and concatenate the values
    for file in files:
        file_path = lengths_path / file  # Create the file path using Path object
        temp_df = pd.read_csv(file_path, header=0)
        df_lengths = pd.concat([df_lengths, temp_df], axis=0)

    # reset the index and set the column names
    df_lengths.reset_index(drop=True, inplace=True)
    
    # Select the columns starting by "lengths_"
    columns_to_select = [col for col in df_lengths.columns if col.startswith('lengths_')]
    df_lengths = df_lengths[columns_to_select]
    
    # Merge both dataframes
    merged_df = pd.concat([df, df_lengths], axis=1)    
    
    return merged_df

df = create_lengths_df()

def download_dataframes(df, basepath=upaths['basepath']):
    directory = str(basepath)
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%Y%m%d")
    original_date = df.file[0].split('_')[0]

    df.to_csv(directory + '_' + formatted_date + '_' + original_date + '.csv')


download_dataframes(df)

    # manually analyzing one file
    # d = open_data(upaths['datapath'] / '20210808 _ERC project_JP_tone context_habituation ab 2_01_01_1DLC_resnet50_FearDetectionJun17shuffle1_100000.csv')
    # x, y = likelihoodtreshold(d)
    # x, y = convert_cm(x, y)
    # dist = euclidean_distance(x, y)
    # velocity = speed(dist)
    # freezing = is_freezing(dist)
    # results = freezing_speed_quantif(dist, velocity)
    # Can be replaced with:
    # results = analyze_mouse(upaths[
    #                             'datapath'] / '20210808 _ERC project_JP_tone context_habituation ab 2_01_01_1DLC_resnet50_FearDetectionJun17shuffle1_100000.csv')

    # How to analyze all files
    #  df = analyse_all_data(upaths['table_path'])
