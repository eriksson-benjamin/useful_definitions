#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 07:40:14 2019

@author: beriksso
"""

'''
A lot of messy but useful code.
'''

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
import matplotlib
import pickle
import scipy.special as special
try:
    import ppf
except:
    pass
import sys
sys.path.insert(
    0, '/home/beriksso/TOFu/analysis/benjamin/github/TOFu/functions/')
sys.path.insert(0, 'C:/python/TOFu/functions/')
import tofu_functions as dfs
import shelve
import os
from matplotlib.widgets import SpanSelector
import string
try:
    import xlrd
except:
    pass
sys.path.insert(
    0, '/home/beriksso/TOFu/analysis/benjamin/other/return_shot_numbers/')
try:
    from all_shots import check_all_shots
except:
    pass
try:
    import matplotlib as mpl
except:
    pass
try:
    import KM11data
except:
    pass
try:
    import getdat as gd
except:
    pass
import scipy.constants as constants
import inspect
import time
import json


def select_range(fig, ax, x, y):
    '''
    Allows you to visually choose a range of the histogram and returns the 
    xMax and xMin values of the chosen range.
    x: x-data
    y: y-data
    '''
    global onselectMin, onselectMax
    onselectMax = 0

    def onselect(xmin, xmax):
        global onselectMin, onselectMax

        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)

        fig.canvas.draw_idle()
        print('Min: ' + str(xmin))
        print('Max: ' + str(xmax))
        onselectMin = xmin
        onselectMax = xmax

    # set useblit True on gtkagg for enhanced performance
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))
    plt.pause(0.01)
    plt.show()
    selected = 0
    while selected == 0:
        #        print(len(onselectMax))
        if onselectMax != 0:
            plt.close('all')
            selected = 1
            locMin = onselectMin
            locMax = onselectMax
            onselectMin = 0
            onselectMax = 0
            return locMin, locMax
        plt.pause(0.01)


def multipage(filename, figs=None, dpi=200, check=True, tight_layout=True):
    '''
    Saves all open figures to a PDF.
    '''
    if check:
        if os.path.exists(filename):
            inp = input(f'Overwrite {filename}? [y/n] ')
            if inp != 'y':
                return 0
            else:
                print('Overwriting file.')
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        if tight_layout:
            fig.savefig(pp, format='pdf', bbox_inches='tight', pad_inches=0)
        else:
            fig.savefig(pp, format='pdf')
    pp.close()


def multifig(filename, check=True, tight_layout=True, ext='pdf', combine_pdf=False):
    save_pdf = False
    figs = [plt.figure(n) for n in plt.get_fignums()]
    if ext == 'pdf':
        save_pdf = True
        ext = 'svg'
    for i, fig in enumerate(figs):
        if check:
            if os.path.exists(f'{filename}_{i}.{ext}'):
                inp = input(f'Overwrite {filename}_{i}.{ext}? [y/n] ')
                if inp != 'y':
                    return 0
                else:
                    print('Overwriting file.')
        if tight_layout:
            fig.savefig(f'{filename}_{i}.{ext}', format=ext,
                        bbox_inches='tight', pad_inches=0)
        else:
            fig.savefig(f'{filename}_{i}.{ext}', format=ext)

        if save_pdf:
            command = f'inkscape {filename}_{i}.{ext} --export-pdf={filename}_{i}.pdf'
            os.system(command)
            os.remove(f'{filename}_{i}.{ext}')

    if combine_pdf:
        created_files = [f'{filename}_{j}.pdf' for j in range(i + 1)]

        cmd = 'pdfunite '
        for file_name in created_files:
            cmd += f'{file_name} '
        cmd += f'{filename}_combined.pdf'
        os.system(cmd)


# Plot 1D histogram, allows looping several plots into same window
def hist_1D_s(x_data, title='', label='data set', log=True, bins=0,
              ax=-1, normed=0, x_label='time [ns]', y_label='',
              return_hist=False, histtype='step'):
    '''
    Example of how to use legend:
        fig = plt.figure('Time differences')
        ax = fig.add_subplot(111)
        bins = np.linspace(0, 1.15 * np.max(dt_ADQ14), 1000)
        DFS.hist_1D_s(dt_ADQ412, label = 'ADQ412', log = True, bins = bins, x_label = 'Time difference [ms]', ax = ax)
        DFS.hist_1D_s(dt_ADQ14,  label = 'ADQ14',  log = True, bins = bins, x_label = 'Time difference [ms]', ax = ax)
    '''
    if bins is 0:
        bins = np.linspace(np.min(x_data), np.max(x_data), 100)
    hist = plt.hist(x_data, label=label, bins=bins, log=log,
                    histtype=histtype, density=normed)
    plt.title(title)
    plt.xlim([bins[0], bins[-1]])

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Include legend
    if ax != -1:
        handles, labels = ax.get_legend_handles_labels()
        new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
        plt.legend(handles=new_handles, labels=labels, loc='upper right')
    if return_hist:
        return hist

# Plot 1D histogram using bin values (one value per bin)


def hist_1D_weights(x_data, title='', legend_label='data set', log=True,
                    bins=np.arange(0, 1, 0.1), ax=-1, normed=0,
                    x_label='', y_label='', color='k', return_hist=False):

    bin_centres = bins[0:-1] + np.diff(bins) / 2.

    hist = plt.hist(bin_centres, weights=x_data, label=legend_label,
                    bins=bins, log=log, histtype='step',
                    density=normed, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Include legend
    if ax != -1:
        handles, labels = ax.get_legend_handles_labels()
        new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
        plt.legend(handles=new_handles, labels=labels, loc='upper right')
    if return_hist:
        return hist

    # Plot 2D histogram


def hist_2D(x_data, y_data, title='2D histogram', log=True,
            x_range='none', y_range='none', bins=[1000, 1000],
            x_label='Time [ns]', y_label='Pulse amplitude [a.u.]',
            cmap=None):
    if x_range == 'none':
        x_range = [1.1 * np.min(x_data), 1.1 * np.max(x_data)]
    if y_range == 'none':
        y_range = [1.1 * np.min(y_data), 1.1 * np.max(y_data)]

    my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)
    if log == True:
        plt.hist2d(x_data, y_data, bins, range=[
                   x_range, y_range], norm=LogNorm(), cmap=my_cmap, vmin=1)
    else:
        plt.hist2d(x_data, y_data, bins, range=[
                   x_range, y_range], cmap=my_cmap, vmin=1)

    cb = plt.colorbar(orientation='horizontal')
#    cb.remove()
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_matrix(matrix, bin_centres=None, log=False, xlabel=None, ylabel=None):
    plt.figure()
    if bin_centres == None:
        x_bins = np.arange(0, np.shape(matrix)[0])
        y_bins = np.arange(0, np.shape(matrix)[1])
    else:
        x_bins = bin_centres[0]
        y_bins = bin_centres[1]

    # Create fill for x and y
    x_repeated = np.tile(x_bins, len(y_bins))
    y_repeated = np.repeat(y_bins, len(x_bins))
    weights = np.ndarray.flatten(np.transpose(matrix))

    # Set white background
    my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)

    if log:
        normed = matplotlib.colors.LogNorm(vmin=1)
    else:
        normed = None
    # Create 2D histogram using weights
    hist2d = plt.hist2d(x_repeated, y_repeated, bins=(x_bins, y_bins),
                        weights=weights, cmap=my_cmap, norm=normed)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    return hist2d


def plot_fission_chamber(shot_number, plot=True):
    f_chamber = ppf.ppfget(shot_number, dda="TIN", dtyp="RNT")
    f_data = f_chamber[2]
    f_times = f_chamber[4]
    tot_yield = np.sum(f_data) * (f_times[1] - f_times[0])

    if plot:
        plot_string = 'Fission chamber\n' + 'shot number: ' + \
            str(shot_number) + '\ntotal yield: ' + str(tot_yield)
        plt.figure(plot_string)
        plt.plot(f_times, f_data)
        plt.title(plot_string)
        plt.xlabel('Time [s]')
        plt.ylabel('Events')
    return f_data, f_times


def get_neutron_yield(shot, t0, t1):
    # Import fission chamber information
    f_chamber = ppf.ppfget(shot, dda="TIN", dtyp="RNT")
    f_data = f_chamber[2]
    f_times = f_chamber[4]

    if len(f_times) == 0:
        print('WARNING: Fission chamber data unaviailable.')
        return None

    arg0 = np.searchsorted(f_times, t0)
    arg1 = np.searchsorted(f_times, t1)

    n_yield = f_data[arg0:arg1].sum() * np.diff(f_times).mean()
    return n_yield


def pickler(file_name, to_pickle, check=True):
    inp = ' '
    if check:
        if os.path.exists(file_name):
            inp = input(
                f'{file_name} already exists.\nDo you want to overwrite it? [y/n] ')
            if inp != 'y':
                print('File was not overwritten.')
                return 0

    with open(file_name, 'wb') as handle:
        pickle.dump(to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if inp == 'y':
        print('File was overwritten.')


def json_write_dictionary(file_name, to_save, check=True):
    inp = ' '
    if check:
        if os.path.exists(file_name):
            inp = input(
                f'{file_name} already exists.\nDo you want to overwrite it? [y/n] ')
            if inp != 'y':
                print('File was not overwritten.')
                return 0
    with open(file_name, 'w') as handle:
        json.dump(to_save, handle)


def json_read_dictionary(file_name):
    with open(file_name, 'r') as handle:
        j = json.load(handle)
    return j


def unpickle(file_name):
    with open(file_name, 'rb') as handle:
        A = pickle.load(handle)
        return A


def make_gaussian(area, mu, sigma, bins, upscale=np.array([0, 0, 0])):
    ''' 
    Returns a gaussian shape for an x-vector. Uses the error function to evaluate
    bins = x-vector of edges on N bins (length = N+1)
    '''
    if (upscale == np.array([0, 0, 0])).all():
        s_erf = special.erf((bins - mu) / sigma)
        gaus = area * np.diff(s_erf) / 2
    else:
        bin_low = upscale[0]
        bin_high = upscale[1]
        bin_width = upscale[2]
        bins_new = np.arange(bin_low, bin_high, bin_width)
        s_erf = special.erf((bins_new - mu) / sigma)
        gaus = np.diff(bins)[0] / np.diff(bins_new)[0] * \
            area * np.diff(s_erf) / 2

    return gaus


def fit_function(parameters, bins, data):
    '''
    Fit function for fitting a Gaussian
    '''
    area = parameters[0]
    mu = parameters[1]
    sigma = parameters[2]

    # Make gaussian
    A = make_gaussian(area, mu, sigma, bins)
    diff = A - data
    return diff


def return_shots(n_of_shots=0, n_max=np.inf, n_min=0, shot_min=0,
                 shot_max=0, TOFu_complete=False, save_path='N/A', slow=False):
    '''
    Returns a list of shots which fulfil the constraints given in the argument
    n_of_shots: number of shots to return in list, leave 0 for "as many as possible"
    n_max: maximum number of neutrons for the shot
    n_min: minimum number of neutrons for the shot
    shot_min: start looking from this shot number
    shot_max: look highest up to this shot number
    '''

    list_of_shots = np.array([], dtype='int')

    if shot_max == 0:
        shot_max = ppf.pdmsht()
    if n_max == 0 and n_min == 0:
        fission_chambers = False
    else:
        fission_chambers = True

    delta_shots = shot_max - shot_min
    for i, shot_number in enumerate(range(shot_min, shot_max + 1)):
        if slow:
            time.sleep(1)
        # Print progress
        if n_of_shots == 0:
            print(f'Progress: {i+1}/{delta_shots} JPN#{shot_number}')
        else:
            print(f'Progress: {i}/{n_of_shots} JPN#{shot_number}')
        # Check length of the list
        if n_of_shots != 0 and len(list_of_shots) == n_of_shots:
            break

        # Check number of neutrons
        if fission_chambers:
            # Import fission chamber information
            f_chamber = ppf.ppfget(shot_number, dda="TIN", dtyp="RNT")
            f_data = f_chamber[2]
            f_time = f_chamber[4]
            # Find number of neutrons
            n_neutrons = np.sum(f_data * np.median(np.diff(f_time)))

            if n_neutrons < n_min or n_neutrons > n_max:
                continue

        # Check if TOFu has data on all channels
        if TOFu_complete:
            for s1 in dfs.get_dictionaries('S1').keys():
                if np.shape(dfs.get_times(detector_name=s1, shot_number=shot_number)) == ():
                    continue

        # Add shot number to list if passed all requirements
        list_of_shots = np.append(list_of_shots, shot_number)

    # Save list to file
    if save_path != 'N/A':
        to_pickle = {'shot_numbers': list_of_shots}
        pickler(file_name=save_path, to_pickle=to_pickle)

    return list_of_shots


def save_namespace(file_name):
    my_shelf = shelve.open(file_name, 'n')  # 'n' for new
    for key in dir():
        if key == 'file_name':
            continue
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()


def load_namespace(file_name):
    my_shelf = shelve.open(file_name)
    for key in my_shelf:
        globals()[key] = my_shelf[key]
    my_shelf.close()


def import_pulses(board='', channel='', path='', record_length=64, full_path=0):
    '''
    Imports pulse waveforms.
    board: string containing board number ('01', '02', ..., '10')
    channel: string containing channel name ('A', 'B', 'C', 'D')
    path: string containing path to data ('C:\\my_data\\')
    record_length: integer containing length of one record
    full_path: string containing full path to data (board, channel and path can then be ignored)
    '''
    # Import the pulses
    if full_path == 0:
        file_name = path + 'M11D-B' + board + '_DT' + channel
    else:
        file_name = full_path
    D = np.fromfile(file_name, dtype='int16')

    # Reshape pulse data
    if len(D) % record_length != 0:
        print('Error: Number of records could not be calculated using this record length.')
        return 0
    nOfRecords = len(D) / record_length

    return np.reshape(D, (int(nOfRecords), int(record_length)))


def import_times(board='', channel='', path='', full_path=0):
    '''
    board: string containing board number ('01', '02', ..., '10')
    channel: string containing channel name ('A', 'B', 'C', 'D')
    path: string containing path to data ('C:\\my_data\\')
    full_path: string containing the full path to data set (board, channel and path can then be ignored)
    returns times in nanoseconds
    '''
    if full_path == 0:
        fname = path + 'M11D-B' + board + '_TM' + channel
    else:
        fname = full_path
    if board in ('06', '07', '08', '09', '10'):
        # 0.5 = conversion factor to ns for ADQ412's
        T = np.fromfile(fname, dtype='uint64') * 0.5
        return T
    elif board in ('01', '02', '03', '04', '05'):
        # The appropritate column must be chosen for the ADQ14's
        T = np.fromfile(fname, dtype='uint64')
        nrecords = int(len(T) / 5)
        T = np.reshape(T, (nrecords, 5)) / \
            8.  # 0.125 = conversion factor to ns for ADQ14
        return T[:, 2]


def import_offset(board='', path='', full_path=0):
    '''
    board: string containing board number ('01', '02', ..., '10')
    path: string containing path to data ('C:\\my_data\\')
    full_path: string containing the full path to data set (board and path can then be ignored)
    '''
    if full_path == 0:
        fname = path + 'M11D-B' + board + '_OFF'
    else:
        fname = full_path

    offset = np.fromfile(fname, dtype='uint64')

    return offset


def make_phs(shot_number, detector_name, return_data=False, ly_function='gatu'):
    # Get pulses
    board, channel = dfs.get_board_name(detector_name)
    pulses = dfs.get_pulses(board=board, channel=channel,
                            shot_number=shot_number)

    # Baseline reduction
    pulses = dfs.baseline_reduction(pulses)

    # Cleanup
    bias_level = dfs.get_bias_level(board, shot_number)
    pulses, _ = dfs.cleanup(pulses,
                            dx=1,
                            detector_name=detector_name,
                            bias_level=bias_level)

    # Set bins
    if int(detector_name[3:]) < 16:
        bins = np.arange(-3000000, 10000, 1000)
        rec_len = 64
    else:
        bins = np.arange(-150000, 500, 250)
        rec_len = 56

    # Get area under pulse
    ux_values = np.arange(0, rec_len, 0.1)
    x_values = np.arange(0, rec_len)
    pulse_data_sinc = dfs.sinc_interpolation(pulses, x_values, ux_values)
    pulse_area = dfs.get_pulse_area(pulse_data_sinc, 10)

    # Create histogram
    bin_centres = bins[1:] - np.diff(bins)[0] / 2
    hist_vals, _ = np.histogram(pulse_area, bins=bins)

    # Flip
    bin_centres = -bin_centres

    plt.figure(f'Pulse height spectrum - {detector_name}')
    plt.semilogy(bin_centres / 1E6, hist_vals, 'k.')
    plt.errorbar(bin_centres / 1E6,
                 hist_vals,
                 yerr=np.sqrt(hist_vals),
                 linestyle='None',
                 capsize=1.5,
                 color='k')
    plt.xlabel('Pulse area [a.u.]')
    plt.ylabel('Counts')
    plt.xlim(right=bin_centres[np.nonzero(hist_vals)[0][0]] / 1E6, left=0)
    plt.title(f'TOFu {detector_name}, JPN {shot_number}')
    plt.annotate(f'Counts: {np.sum(hist_vals)}', xy=(
        0.7, 0.95), xycoords='axes fraction')

    # Convert to light yield [MeVee]
    light_yield = dfs.get_energy_calibration(-pulse_area, detector_name)

    # Convert to proton recoil [MeV]
    proton_recoil = dfs.inverted_light_yield(light_yield, function=ly_function)

    # Plot light yield
    plt.figure(f'Light yield spectrum - {detector_name}')
    ly_bins = np.arange(-5, 20, 0.01)
    ly_bin_centres = ly_bins[1:] - np.diff(ly_bins)[0] / 2
    ly_vals, _ = np.histogram(light_yield, bins=ly_bins)
    plt.semilogy(ly_bin_centres, ly_vals, 'k.')
    plt.errorbar(ly_bin_centres,
                 ly_vals,
                 yerr=np.sqrt(ly_vals),
                 linestyle='None',
                 capsize=1.5,
                 color='k')
    plt.xlabel('Light yield [MeVee]')
    plt.ylabel('Counts')
    plt.xlim(right=ly_bin_centres[np.nonzero(ly_vals)[0][-1]], left=0)
    plt.title(f'TOFu {detector_name}, JPN {shot_number}')
    plt.annotate(f'Counts: {np.sum(ly_vals)}', xy=(
        0.7, 0.95), xycoords='axes fraction')

    # Plot proton recoil energy
    plt.figure(f'Proton recoil energy spectrum - {detector_name}')
    pr_bins = np.arange(-5, 20, 0.01)
    pr_bin_centres = pr_bins[1:] - np.diff(pr_bins)[0] / 2
    pr_vals, _ = np.histogram(proton_recoil, bins=pr_bins)
    plt.semilogy(pr_bin_centres, pr_vals, 'k.')
    plt.errorbar(pr_bin_centres,
                 pr_vals,
                 yerr=np.sqrt(pr_vals),
                 linestyle='None',
                 capsize=1.5,
                 color='k')
    plt.xlabel('Proton recoil [MeV]')
    plt.ylabel('Counts')
    plt.xlim(right=pr_bin_centres[np.nonzero(pr_vals)[0][-1]], left=0)
    plt.annotate(f'Counts: {np.sum(pr_vals)}', xy=(
        0.7, 0.95), xycoords='axes fraction')
    plt.title(f'TOFu {detector_name}, JPN {shot_number}')
    plt.axvline(x=2.5, linestyle='--', color='k')
    plt.axvline(x=14.1, linestyle='--', color='k')
    plt.show()
    to_return = {'info': 'Returns histogram information in tuple (bin centres, histogram values)',
                 'pulse height spectrum': (bin_centres, hist_vals),
                 'light yield spectrum': (ly_bin_centres, ly_vals),
                 'energy spectrum': (pr_bin_centres, pr_vals)}
    if return_data:
        to_return['pulse height data'] = pulse_area
        to_return['light yield data'] = light_yield
        to_return['energy data'] = proton_recoil
    return to_return


def grab_from_excel(file_name, rows=[0, 0], columns=['A', 'A'], sheet_name=None):
    '''
    Returns values from excel spread sheet.
    Assumes only numbers are to be returned.
    Example: array = grab_from_excel(file_name, rows = [1, 5], columns = ['B', 'AA'])
    returns rows 1 to 5, columns B to AA from the first sheet in the excel spreadsheet
    given by file_name. Sheet name can be specified if several sheets are available.
    '''
    def col2num(col):
        '''
        Translate letter to number
        '''
        num = 0
        for c in col:
            if c in string.ascii_letters:
                num = num * 26 + (ord(c.upper()) - ord('A')) + 1
        return num

    # Select columns and rows
    # Translate letter -> column number
    cols = [col2num(columns[0]) - 1, col2num(columns[1])]
    select_rows = np.arange(rows[0] - 1, rows[1])

    # Load spread sheet
    workbook = xlrd.open_workbook(file_name)

    # Select first sheet or sheet given by user
    if workbook.nsheets > 1:
        sheet = workbook.sheet_by_name(sheet_name)
    else:
        sheet = workbook.sheet_by_index(0)

    # Initialize list
    new_list = np.zeros(
        [select_rows[-1] - select_rows[0] + 1, cols[-1] - cols[0]])

    # Read values from spread sheet and save to new_list
    for i, row in enumerate(select_rows):
        row_vals = sheet.row_slice(
            rowx=row, start_colx=cols[0], end_colx=cols[1])
        for j, r in enumerate(row_vals):
            new_list[i][j] = r.value

    return new_list


def good_shots(shots):
    '''
    Takes array of shots and checks if there is LED/missing data in the shot,
    returns array of good shots.
    '''
    # Update list of good shots
    check_all_shots()

    # Compare list of shots to array of shots
    good_shots = np.loadtxt(
        '/home/beriksso/TOFu/analysis/benjamin/other/return_shot_numbers/all_shots.txt')

    return_array = [s for s in shots if s in good_shots]

    return return_array


def get_CXRS_Ti(shot_number, time_range=None):
    '''
    Returns ion temperature (keV) and time (s) for given shot number measured by KS5
    charge exchange recombination spectrometer.
    '''

    ks5 = ppf.ppfget(shot_number, dda='XCS', dtyp='TI')
    Ti = ks5[2] / 1000.
    time = ks5[4]

    # Find time slice
    if time_range:
        start = np.searchsorted(time, time_range[0], side='right')
        stop = np.searchsorted(time, time_range[1], side='left')
        Ti = Ti[start:stop]
        time = time[start:stop]

    return Ti, time


def reset_matplotlib():
    mpl.rcParams.update(mpl.rcParamsDefault)


def set_nes_plot_style():
    matplotlib.rcParams['interactive'] = True
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'nes_plots.mplstyle')
    plt.style.use(filename)


def plot_nes_pickle_full(file_name, show_background=True, title=None, fig_name='tof spectrum', log2d=False):
    '''
    Plots the TOF spectrum for a given NES pickle.
    '''
    set_nes_plot_style()

    # Import data
    P = unpickle(file_name)
    tof_counts = P['counts']
    bins = P['bins']
    bgr = P['bgr_level']
    ucrt = np.sqrt(tof_counts)
    erg_S1 = P['erg_S1']
    erg_S2 = P['erg_S2']
    hist2d_S1 = P['hist2d_S1']
    hist2d_S2 = P['hist2d_S2']
    S1_info = P['S1_info']
    S2_info = P['S2_info']

    # Plot data with background
    fig = plt.figure(f'{fig_name} 1')
    color = 'k'
    if show_background:
        plt.plot(bins, tof_counts, 'k.', markersize=1.5)
        plt.errorbar(bins, tof_counts, ucrt, linestyle='None', color=color)
        color = 'r'
    plt.xlabel('t$_{TOF}$ (ns)')
    plt.ylabel('counts/bin')

    # Remove background
    bg = np.zeros(len(tof_counts))
    bg[len(bgr) - 1:] = bgr
    bg[:len(bgr) - 1] = np.flip(bgr)[1:]
    plt.plot(bins, bg, 'r--')

    tof = tof_counts - bg

    plt.plot(bins, tof, color=color, marker='.',
             markersize=1.5, linestyle='None')
    plt.errorbar(bins, tof, ucrt, linestyle='None', color=color)
    plt.yscale('log')
    plt.xlim([20, 100])
    plt.ylim(bottom=1)

    TOF_hist, S1_E_hist, S2_E_hist, hist2d_S1, hist2d_S2 = dfs.plot_2D(tof_counts, erg_S1, erg_S2, weights=True,
                                                                       tof_bg_component=bg, hist2D_S1=hist2d_S1, hist2D_S2=hist2d_S2,
                                                                       times_of_flight_cut=tof_counts, energy_S1_cut=erg_S1,
                                                                       energy_S2_cut=erg_S2, S1_info=S1_info, S2_info=S2_info, title=f'{fig_name} 2')

    if title:
        plt.title(title)
    return fig, plt.gca()


def nes_pickle_to_txt(input_name, output_name):
    '''
    Read NES pickle and write (positive tof) bins, counts and background to textfile
    '''
    p = unpickle(input_name)
    bins = p['bins']
    pos_bins = bins >= 0
    counts = p['counts']
    bgr = p['bgr_level']
    to_txt = np.array([bins[pos_bins], counts[pos_bins], bgr]).T
    np.savetxt(output_name, to_txt, header='tof, counts, bg', delimiter=',')


def get_tofor_times(shot_number, time_slice=[40, 70]):

    # Get TOFOR data between seconds
    data = KM11data.KM11get.GetLPFData(shot_number)
    pre = data[6]
    times_S1 = (data[0].astype(np.int64) - pre) * 0.4E-9
    times_S2 = (data[1].astype(np.int64) - pre) * 0.4E-9
    n_all_S1 = data[2]
    n_all_S2 = data[3]
    n_cs_S1 = np.cumsum(n_all_S1)
    n_cs_S2 = np.cumsum(n_all_S2)
    n_S1 = data[4]
    n_S2 = data[5]
    time_cal = 0

    # Sort time stamps into dictionaries
    s1_dict, s2_dict = dfs.get_dictionaries()

    sx_times = dict()

    '''
    S1
    '''
    previous = 0
    for i, s1 in enumerate(s1_dict.keys()):
        temp = times_S1[previous:previous + n_S1[i]]
        # Apply time cal
        temp -= time_cal
        temp *= 1E+9
        # Find time slice
        start = np.searchsorted(temp, time_slice[0] * 1E+9)
        stop = np.searchsorted(temp, time_slice[1] * 1E+9)

        sx_times[s1] = temp[start:stop]
        previous = n_cs_S1[i]

    '''
    S2 
    '''
    previous = 0
    for i, s2 in enumerate(s2_dict.keys()):
        temp = times_S2[previous:previous + n_S2[i]]

        # Apply time cal
        temp -= time_cal
        temp *= 1E+9
        # Find time slice
        start = np.searchsorted(temp, time_slice[0] * 1E+9)
        stop = np.searchsorted(temp, time_slice[1] * 1E+9)

        sx_times[s2] = temp[start:stop]
        previous = n_cs_S2[i]

    return sx_times


def plot_tofor(shot_number, time_range):
    tofor = KM11data.KM11ToF(shot_number, time_range)
    tofor.plot()

    return tofor.axis, tofor.tof


def random_colors(size=1, seed=0):
    if seed:
        np.random.seed(seed)
    if size == 1:
        colors = [np.random.uniform(0, 1), np.random.uniform(
            0, 1), np.random.uniform(0, 1)]
    else:
        colors = [[np.random.uniform(0, 1), np.random.uniform(
            0, 1), np.random.uniform(0, 1)] for i in range(size)]

    return colors


def get_colors(n_colors):
    colors_list = ['k']
    for c in mcolors.TABLEAU_COLORS:
        colors_list.append(c)

    for c in mcolors.CSS4_COLORS:
        colors_list.append(c)

    # Remove whites
    white = ['whitesmoke', 'white', 'snow', 'mistyrose', 'seashell', 'linen',
             'oldlace', 'floralwhite', 'ivory', 'lightyellow', 'honeydew',
             'mintcream', 'azure', 'aliceblue', 'ghostwhite', 'lavenderblush']
    for w in white:
        colors_list.remove(w)

    if n_colors > len(colors_list):
        print('Not enough colors in list, returning random colors.')
        return random_colors(n_colors)
    return colors_list[:n_colors]


def get_collimator_size(shot_number):
    '''
    Returns the pre collimator jaws settings for given shot
    '''
    if shot_number == 0:
        # Get latest shot
        shot_number = np.array([str(ppf.pdmsht())])
    positions = ['M5-POS<D1', 'M5-POS<D2', 'M5-POS<D3', 'M5-POS<D4']
    pos_list = []
    for position in positions:
        pos_list.append(gd.getsca(position, shot_number))

    return pos_list


def codas_gas_code(code):
    '''
    Returns gas composition for given Codas gas code.
    '''
    if code == 30:
        return None
    elif code == 31:
        return 'H2'
    elif code == 32:
        return 'D2'
    elif code == 33:
        return 'Tritium'
    elif code == 34:
        return 'He3'
    elif code == 35:
        return 'He4'
    elif code == 36:
        return 'CD4'
    elif code == 37:
        return 'CH4'
    elif code == 38:
        return 'O2'
    elif code == 39:
        return 'Ar'
    elif code == 45:
        return 'Xe'
    elif code == 22:
        return 'Kr'
    elif code == 23:
        return 'N2'
    elif code == 24:
        return 'SiH4'
    elif code == 25:
        return 'Ne'
    elif code == 27:
        return 'C2H4'
    elif code == 28:
        return 'C2H6'
    elif code == 29:
        return 'C13H4'
    elif code == 40:
        return 'C3H8'
    elif code == 41:
        return 'TMB'
    else:
        print('Unknown code.')
        return 0


def plot_discharge(shot_number, t=[None, None]):
    set_nes_plot_style()
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True)
    fig.set_figheight(8)
    fig.set_figwidth(10)
    ax[0, 0].xaxis.set_major_locator(plt.MultipleLocator(5))
    for axis in ax.flatten():
        axis.tick_params(which='major', length=4, width=1)

    colors = ['k', 'C0', 'C1', 'C2', 'C3']
    ls = ['-', '--', '-.', 'dotted', (0, (3, 5, 1, 5, 1, 5))]
    lw = [1, 2, 2, 2, 2]
    if t[0] != None and t[1] != None:
        ax[0, 0].set_xlim(t[0], t[1])

    '''
    Gas puffing
    '''
    gas_codes = ppf.ppfget(dda='GASH', dtyp='CGAS', pulse=shot_number)[
        2].astype('int')
    for i, gas_code in enumerate(gas_codes):
        gas_type = codas_gas_code(gas_code)
        gas = ppf.ppfget(dda='GASH', dtyp=f'B{gas_code}R', pulse=shot_number)
        ax[0, 0].plot(gas[4], gas[2], color=colors[i],
                      linestyle=ls[i], label=gas_type, linewidth=lw[i])
    ax[0, 0].legend()
    ax[0, 0].set_ylabel(r'R$_{gas}$ $\left(\frac{bar\cdot l}{s}\right)$')

    '''
    Neutron rate
    '''
    nrate = ppf.ppfget(dda='TIN', dtyp='RNT', pulse=shot_number)
    ax[0, 1].plot(nrate[4], nrate[2], color=colors[0], linestyle=ls[0])
    ax[0, 1].set_ylabel(r'$R_n$ $\left( s^{-1} \right)$')

    '''
    NBI power
    '''
    nbi = ppf.ppfget(dda='NBI', dtyp='NBLM', pulse=shot_number)
    icrh = ppf.ppfget(dda='ICRH', dtyp='PTOT', pulse=shot_number)
    lhcd = ppf.ppfget(dda='LHCD', dtyp='PTOT', pulse=shot_number)
    if len(nbi[2] > 0):
        # Which ion was used in beams?
        beam_gas = ppf.ppfget(dda='NBI', dtyp='GAS', pulse=shot_number)[0]
        bg = beam_gas[::-1]
        gas_type = bg[0:bg.find(' ')][::-1]

        # Plot NBI
        ax[1, 0].plot(nbi[4], nbi[2] / 1E6, color=colors[0],
                      linestyle=ls[0], label=f'{gas_type[0]} NBI')

    if len(icrh[2] > 0):
        ax[1, 0].plot(icrh[4], icrh[2] / 1E6, color=colors[1],
                      linestyle=ls[1], label='ICRH', linewidth=lw[1])
    if len(lhcd[2] > 0):
        ax[1, 0].plot(lhcd[4], lhcd[2] / 1E6, color=colors[2],
                      linestyle=ls[2], label='LHCD', linewidth=lw[2])
    ax[1, 0].legend()
    ax[1, 0].set_ylabel('$P$ (MW)')

    '''
    Electron density
    '''
    hrts_ne = ppf.ppfget(dda='HRTX', dtyp='NE0', pulse=shot_number)
    lidx_ne = ppf.ppfget(dda='LIDX', dtyp='NE0', pulse=shot_number)
    if len(hrts_ne[2]) > 0:
        ax[1, 1].plot(hrts_ne[4], hrts_ne[2], color=colors[0],
                      linestyle=ls[0], label='HRTS')
    if len(lidx_ne[2]) > 0:
        ax[1, 1].plot(lidx_ne[4], lidx_ne[2], color=colors[1],
                      linestyle=ls[1], label='Lidar')
    ax[1, 1].set_ylabel('$n_e$ ($m^{-3}$)')
    ax[1, 1].legend()

    '''
    Electron temperature
    '''
    hrts_te = ppf.ppfget(dda='HRTX', dtyp='TE0', pulse=shot_number)
    kk3_te = ppf.ppfget(dda='KK3', dtyp='TEnn', pulse=shot_number)
    lidar_te = ppf.ppfget(dda='LIDX', dtyp='TE0', pulse=shot_number)
    lid2_te = ppf.ppfget(dda='LID2', dtyp='TEAV', pulse=shot_number)
    ecm1_te = ppf.ppfget(dda='ECM1', dtyp='TCOM', pulse=shot_number)
    if len(hrts_te[2] > 0):
        ax[2, 1].plot(hrts_te[4], hrts_te[2] / 1E3,
                      color=colors[0], linestyle=ls[0], label='HRTX')
    if len(kk3_te[2] > 0):
        ax[2, 1].plot(kk3_te[4], kk3_te[2] / 1E3,
                      color=colors[1], linestyle=ls[1], label='KK3')
    if len(lidar_te[2] > 0):
        ax[2, 1].plot(lidar_te[4], lidar_te[2] / 1E3,
                      color=colors[2], linestyle=ls[2], label='LIDX')
    if len(lid2_te[2] > 0):
        ax[2, 1].plot(lid2_te[4], lid2_te[2] / 1E3,
                      color=colors[3], linestyle=ls[3], label='LID2')
    if len(ecm1_te[2] > 0):
        ax[2, 1].plot(ecm1_te[4], ecm1_te[2] / 1E3,
                      color=colors[4], linestyle=ls[4], label='ECM1')

    ax[2, 1].set_ylabel('$T_e$ (keV)')
    ax[2, 1].legend()

    '''
    Z-effective
    '''
    zeff = ppf.ppfget(dda='KS3', dtyp='ZEFV', pulse=shot_number)
    ax[2, 0].plot(zeff[4], zeff[2], color=colors[0], linestyle=ls[0])
    ax[2, 0].set_ylabel('$Z_{eff}$')
    fig.subplots_adjust(hspace=0.2)
    ax[0, 0].set_title(f'JPN {shot_number}', loc='left')


def plot_rf_power(shot_number, t_range=[None, None]):
    # Get ICRH data
    prfa = ppf.ppfget(dda='ICRH', dtyp='PRFA', pulse=shot_number)
    prfb = ppf.ppfget(dda='ICRH', dtyp='PRFB', pulse=shot_number)
    prfc = ppf.ppfget(dda='ICRH', dtyp='PRFC', pulse=shot_number)
    prfd = ppf.ppfget(dda='ICRH', dtyp='PRFD', pulse=shot_number)
    prftot = ppf.ppfget(dda='ICRH', dtyp='PTOT', pulse=shot_number)

    # Plot
    plt.figure(shot_number)
    plt.title(f'JPN {shot_number}', loc='left')
    plt.plot(prftot[4], prftot[2] / 1E6, label='total')
    plt.plot(prfa[4], prfa[2] / 1E6, label='antenna A')
    plt.plot(prfb[4], prfb[2] / 1E6, label='antenna B')
    plt.plot(prfc[4], prfc[2] / 1E6, label='antenna C')
    plt.plot(prfd[4], prfd[2] / 1E6, label='antenna D')

    plt.xlabel('Time (s)')
    plt.ylabel('RF power (MW)')
    plt.legend()


def tofu_background_component(bgr):
    new_bgr = np.append(np.flip(bgr[1:]), bgr)

    return new_bgr


def plot_kt5p(shot_number, t_range=[None, None]):
    set_nes_plot_style()
    t = ppf.ppfget(dda='KT5P', dtyp='TTOT', pulse=shot_number)
    d = ppf.ppfget(dda='KT5P', dtyp='DTOT', pulse=shot_number)
    h = ppf.ppfget(dda='KT5P', dtyp='HTOT', pulse=shot_number)

    plt.figure(f'{shot_number} KT5P')

    # Calculate T fractions
    t_f1 = t[2] / (t[2] + h[2] + d[2]) * 100
    t_f2 = t[2] / (t[2] + d[2]) * 100
    t_f3 = d[2] / (t[2] + d[2] + h[2]) * 100

    # Remove NaNs
    not_nans = np.invert(np.isnan(t_f1) * np.isnan(t_f2) * np.isnan(t_f3))
    t_f1 = t_f1[not_nans]
    t_f2 = t_f2[not_nans]
    t_f3 = t_f3[not_nans]
    times = t[4][not_nans]

    # Plot
    plt.plot(times, t_f1, label='$n_T/(n_T+n_D+n_H)$')
    plt.plot(times, t_f2, label='$n_T/(n_D+n_T)$')
    plt.plot(times, t_f3, label='$n_D/(n_D+n_T+n_H$')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (%)')
    plt.legend()
    plt.title(f'JPN {shot_number}')

    try:
        # Set x-y lims
        if t_range[0] != None and t_range[1] != None:
            plt.xlim(t_range)
            # Grab subsets
            t_s1 = t_f1[(times > t_range[0]) & (times < t_range[1])]
            t_s2 = t_f2[(times > t_range[0]) & (times < t_range[1])]
            y_max = np.max([t_s1.max(), t_s2.max()])
            y_min = np.min([t_s1.min(), t_s2.min()])
            if y_max > 90:
                y_max = 110
            else:
                y_max *= 1.2
            if y_min < 10:
                y_min = -5
            else:
                y_min *= 0.8
            plt.ylim(y_min, y_max)
    except:
        breakpoint()
    return h, d, t


def get_t_fraction(t_low, t_high):
    """Return a list of shots with T fraction between t_low and t_high."""
    path = '/home/beriksso/TOFu/analysis/benjamin/other/gas_fraction/t_gas_fraction.dat'
    data = np.loadtxt(path)
    shot_numbers = data[:, 0]
    fractions = data[:, 1]
    mask = ((fractions >= t_low) & (fractions <= t_high))

    return shot_numbers[mask], fractions[mask]


def get_bin_centres(bins):
    return bins[1:] - np.diff(bins)[0] / 2


def get_count_rate(shot_number, detector):
    time_stamps = dfs.get_times(shot_number, detector_name=detector)

    board, channel = dfs.get_board_name(detector)
    if int(board) > 5:
        offset = dfs.get_offset(board, shot_number)
        time_stamps -= offset

    width = 0.1
    time_steps = np.arange(40, 80, width)
    args = np.searchsorted(time_stamps * 1E-9, time_steps)
    count_rate = np.diff(args) / width
    time_centres = get_bin_centres(time_steps)

    return count_rate, time_centres


def plot_count_rate(shot_number, detector):
    count_rate, time_centres = get_count_rate(shot_number, detector)

    plt.figure(f'{shot_number} {detector} count rate')
    plt.plot(time_centres, count_rate / 1E6)
    plt.xlabel('Time (s)')
    plt.ylabel('Count rate (MHz)')
    plt.title(f'{shot_number} {detector}')


def pile_up_estimate(shot_number, detector):
    count_rate, time_centres = get_count_rate(shot_number, detector)
    board, channel = dfs.get_board_name(detector)
    if int(board) > 5:
        record_length = 56 * 1E-9
    else:
        record_length = 64 * 1E-9
    pile_up = 1 - np.exp(-record_length * count_rate)
    plt.figure(f'Pile-up {shot_number} {detector}')
    plt.plot(time_centres, pile_up * 100)
    plt.xlabel('Time (s)')
    plt.ylabel('Pile up probability (%)')
    plt.title(f'Pile up {shot_number} {detector}')
    plot_count_rate(shot_number, detector)

    return pile_up, time_centres


def interpolate_new_axis(new_x_axis, x, y):
    new_y_axis = np.interp(new_x_axis, x, y)
    return new_y_axis


def nbi_beam_energy(shot_number):
    # Total power from all NBI sources
    ptot = ppf.ppfget(shot_number, 'NBI', 'PTOT')

    # Total electron flux provided by NBI sources
    etot = ppf.ppfget(shot_number, 'NBI', 'EFLX')

    # Gas type
    gas = ppf.ppfget(shot_number, 'NBI', 'GAS')
    if (gas[2] == 2).all():
        m = constants.physical_constants['deuteron mass']
    elif (gas[2] == 3).all():
        m = constants.physical_constants['triton mass']

    # Interpolate to same axis
    etot_interp = interpolate_new_axis(ptot[4], etot[4], etot[2])

    # Calculate average energy per NBI particle [J/s]/[particles/s] = [J/particle]
    ek_j = ptot[2] / etot_interp

    # Translate to keV
    ek_kev = ek_j * constants.elementary_charge

    return ek_kev


def get_function_path(function):
    print(os.path.abspath(inspect.getfile(function)))


def ohmic_heating(shot_number, t0=40, t1=120):
    # Get electron density [1/m^3]
    ne = ppf.ppfget(dda='HRTX', dtyp='NE0', pulse=shot_number)
    ne_t = ne[4]
    n_e = ne[2]
    if len(n_e) == 0:
        raise Exception(
            f'Electron density (HRTX/NE0) data not available for JPN {shot_number}.')

    # Get electron temperature [eV]
    ecm1_te = ppf.ppfget(dda='ECM1', dtyp='TCOM', pulse=shot_number)
    Te_t = ecm1_te[4]
    T_e = ecm1_te[2]
    if len(T_e) == 0:
        raise Exception(
            f'Electron temperature (ECM1/TCOM) not available for JPN {shot_number}.')

    # Get plasma current [A]
    I = ppf.ppfget(dda='MAGN', dtyp='IPLA', pulse=shot_number)
    I_p = -I[2]
    Ip_t = I[4]
    if len(Ip_t) == 0:
        raise Exception(
            f'Plasma current (MAGN/IPLA) not available for JPN {shot_number}.')

    y_vals = np.array([n_e, T_e, I_p])
    x_vals = np.array([ne_t, Te_t, Ip_t])
    new_y_vals = [[], [], []]
    lengths = np.array([len(n_e), len(T_e), len(I_p)])
    argsorted = np.flip(np.argsort(lengths))

    if np.array_equal(ne_t, Te_t) and np.array_equal(Te_t, Ip_t):
        time_axis = np.copy(ne_t)
    else:
        for arg in argsorted[:-1]:
            new_y_vals[arg] = interpolate_new_axis(x_vals[argsorted[-1]],
                                                   x_vals[arg],
                                                   y_vals[arg])
        new_y_vals[argsorted[-1]] = y_vals[argsorted[-1]]
        time_axis = x_vals[argsorted[-1]]

    # Select time range
    t_bool = (time_axis > t0) & (time_axis < t1)
    n_e = new_y_vals[0][t_bool]
    T_e = new_y_vals[1][t_bool]
    I_p = new_y_vals[2][t_bool]
    time_axis = time_axis[t_bool]

    # Convert temperature to Kelvin
    k_convert = constants.physical_constants['Boltzmann constant in eV/K'][0]
    T_e /= k_convert

    # Calculate Debye length [m]
    lambda_d = np.sqrt(constants.epsilon_0 * constants.k *
                       T_e / (n_e * constants.e**2))

    # Calculate coulomb logarithm
    ln_L = np.log(12 * np.pi * n_e * lambda_d**3)

    # Calculate resisitivity [Ohm m]
    eta = np.pi * constants.e**2 * np.sqrt(constants.electron_mass) / (
        (4 * np.pi * constants.epsilon_0)**2 * (constants.k * T_e)**(3 / 2)) * ln_L

    # Calculate Ohmic power
    volume = 100  # m^3
    poloidal_area = np.pi * 1 * 0.5  # (this is a guess) m^2
    P_ohm = volume * eta * (I_p / poloidal_area)**2

    return time_axis, P_ohm


def get_linestyle(ls):
    if ls == 'dash-dot-dotted':
        return (0, (3, 5, 1, 5, 1, 5))
    elif ls == 'loosely dotted':
        return (0, (1, 10))
    elif ls == 'dotted':
        return (0, (1, 1))
    elif ls == 'densely dotted':
        return (0, (1, 1))
    elif ls == 'long dash with offset':
        return (5, (10, 3))
    elif ls == 'loosely dashed':
        return (0, (5, 10))
    elif ls == 'dashed':
        return (0, (5, 5))
    elif ls == 'densely dashed':
        return (0, (5, 1))
    elif ls == 'loosely dashdotted':
        return (0, (3, 10, 1, 10))
    elif ls == 'dashdotted':
        return (0, (3, 5, 1, 5))
    elif ls == 'densely dashdotted':
        return (0, (3, 1, 1, 1))
    elif ls == 'dashdotdotted':
        return (0, (3, 5, 1, 5, 1, 5))
    elif ls == 'loosely dashdotdotted':
        return (0, (3, 10, 1, 10, 1, 10))
    elif ls == 'densely dashdotdotted':
        return (0, (3, 1, 1, 1, 1, 1))

    else:
        print('Available linestyles:')
        print('loosely dotted')
        print('dotted')
        print('densely dotted')
        print('long dash with offset')
        print('loosely dashed')
        print('dashed')
        print('densely dashed')
        print('loosely dashdotted')
        print('dashdotted')
        print('densely dashdotted')
        print('dashdotdotted')
        print('loosely dashdotdotted')
        print('densely dashdotdotted')

        raise Exception(f'Unknown linestyle: {ls}')


def get_input_argument(tofu_info, key):
    arg = np.where(tofu_info == key)[0]
    if key in ['--JPN']:
        return int(tofu_info[arg + 1][0])
    elif key in ['--time-range']:
        return tofu_info[arg + 1], tofu_info[arg + 2]


def setup_matrix(matrix, x_bins, y_bins):
    """Create fill for x and y."""
    x_repeated = np.tile(x_bins, len(y_bins))
    y_repeated = np.repeat(y_bins, len(x_bins))
    weights = np.ndarray.flatten(np.transpose(matrix))

    return x_repeated, y_repeated, weights


def get_kinematic_cuts(input_arguments, tof):
    """Import equation for kinematic cuts."""
    if '--apply-cut-factors' in input_arguments:
        arg = np.argwhere(input_arguments == '--apply-cut-factors')[0][0]
        c1 = float(input_arguments[arg + 1])
        c2 = float(input_arguments[arg + 2])
        c3 = float(input_arguments[arg + 3])
    else:
        c1 = 1
        c2 = 1
        c3 = 1

    S1_min, S1_max, S2_max = dfs.get_kincut_function(tof, (c1, c2, c3))
    return S1_min, S1_max, S2_max


def nes_pickle_plotter(t_counts, t_bins, t_bgr, e_bins_S1, e_bins_S2, matrix_S1,
                       matrix_S2, inp_args):
    """Plot for technical TOFu paper."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(4, 12)

    # Set colorbar min/max
    vmin = 1
    vmax = (matrix_S1.max() if matrix_S1.max() > matrix_S2.max() else
            matrix_S2.max())
    normed = matplotlib.colors.LogNorm(vmin, vmax)

    # Set white background
    try:
        my_cmap = matplotlib.cm.get_cmap('jet').copy()
    except:
        my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)

    # Plot S1 2D histogram
    # --------------------
    x_r, y_r, weights = setup_matrix(matrix_S1, t_bins, e_bins_S1)
    ax1.hist2d(x_r, y_r, bins=(t_bins, e_bins_S1), weights=weights,
               cmap=my_cmap, norm=normed)

    # Plot S2 2D histogram
    # --------------------
    x_r, y_r, weights = setup_matrix(matrix_S2, t_bins, e_bins_S2, )
    ax2.hist2d(x_r, y_r, bins=(t_bins, e_bins_S1), weights=weights,
               cmap=my_cmap, norm=normed)

    # Plot TOF projection
    # -------------------
    ax3.plot(t_bins, t_counts, 'k.', markersize=1)
    ax3.errorbar(t_bins, t_counts, np.sqrt(t_counts), color='k',
                 linestyle='None')

    # Plot background component
    ax3.plot(t_bins, t_bgr, 'C0--')

    # Add lines for kinematic cuts
    if '--disable-cuts' not in inp_args:
        S1_min, S1_max, S2_max = get_kinematic_cuts(inp_args, t_bins)
        ax1.plot(t_bins, np.array([S1_min, S1_max]).T, 'r')
        ax2.plot(t_bins, S2_max, 'r')

    # Configure plot
    # --------------
    ax1.set_ylabel('$E_{ee}^{S1}$ $(MeV_{ee})$')
    ax2.set_ylabel('$E_{ee}^{S2}$ $(MeV_{ee})$')
    ax3.set_ylabel('counts')
    ax3.set_xlabel('$t_{TOF}$ (ns)')

    y1 = (0, 2.3)
    y2 = (0, 6)
    x3 = (-100, 100)
    y3 = (10, 2E3)
    ax1.set_ylim(y1)
    ax2.set_ylim(y2)
    ax3.set_yscale('log')
    ax3.set_yticks([1, 10, 100, 1000])
    ax3.set_xlim(x3)
    ax3.set_ylim(y3)

    bbox = dict(facecolor='white', edgecolor='black')
    ax1.text(0.89, 0.88, '(a)', transform=ax1.transAxes, bbox=bbox)
    ax2.text(0.89, 0.88, '(b)', transform=ax2.transAxes, bbox=bbox)
    ax3.text(0.89, 0.88, '(c)', transform=ax3.transAxes, bbox=bbox)

    # Add colorbar
    # ------------
    fig.subplots_adjust(top=0.8)
    cbar_ax = fig.add_axes([0.2, 0.84, 0.73, 0.02])
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=normed)
    try:
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    except:
        print('Colorbar feature not available on Galactica.')

    plt.subplots_adjust(hspace=0.1)


def nes_background(bgr):
    # Reshape background
    return np.append(np.flip(bgr[1:]), bgr)


def import_data(file_name):
    """Import 2D histogram data from given file."""
    p = unpickle(file_name)
    input_args = p['input_arguments']

    # Projection on t_tof axis
    t_counts = p['counts']

    # Reshape background
    bgr = p['bgr_level']
    t_bgr = np.append(np.flip(bgr[1:]), bgr)

    # 2D matrices
    m_S1 = p['hist2d_S1']
    m_S2 = p['hist2d_S2']

    # Bins
    t_bins = p['bins']
    e_bins_S1 = p['S1_info']['energy bins']
    e_bins_S2 = p['S2_info']['energy bins']

    # Calculate bin centres
    e_S1 = e_bins_S1[1:] - np.diff(e_bins_S1) / 2
    e_S2 = e_bins_S2[1:] - np.diff(e_bins_S2) / 2

    return t_counts, t_bins, t_bgr, e_S1, e_S2, m_S1, m_S2, input_args


def plot_nes_pickle(f_name):
    set_nes_plot_style()
    dat = import_data(f_name)

    # Plot 2d histogram
    nes_pickle_plotter(*dat)


def sum_nes_pickles(f_names, f_out=None):
    """Sum events of NES pickles to single file."""
    f0 = unpickle(f_names[0])
    erg_S1 = f0['erg_S1']
    erg_S2 = f0['erg_S2']
    counts = f0['counts']
    bgr_level = f0['bgr_level']
    hist2d_S1 = f0['hist2d_S1']
    hist2d_S2 = f0['hist2d_S2']
    shot_number = f0['shot_number']
    time_ranges = {shot_number: f0['time_range']}
    input_arguments = {shot_number: f0['input_arguments']}

    # Theses should be identical for all shots
    S1_info = f0['S1_info']
    S2_info = f0['S2_info']
    bins = f0['bins']

    # Boolean to check for identical input arguments
    identical_args = True
    for f_name in f_names[1:]:
        f0 = unpickle(f_name)
        erg_S1 += f0['erg_S1']
        erg_S2 += f0['erg_S2']
        counts += f0['counts']
        bgr_level += f0['bgr_level']
        hist2d_S1 += f0['hist2d_S1']
        hist2d_S2 += f0['hist2d_S2']

        # Check if input arguments are the same as previous
        if not np.all(input_arguments[shot_number] == f0['input_arguments']):
            identical_args = False

        shot_number = f0['shot_number']
        time_ranges[shot_number] = f0['time_range']
        input_arguments[shot_number] = f0['input_arguments']

    # If all input arguments are identical, just save one version of them
    if identical_args:
        input_arguments = f0['input_arguments']

    to_write = {'erg_S1': erg_S1, 'erg_S2': erg_S2, 'bins': bins,
                'counts': counts, 'bgr_level': bgr_level,
                'hist2d_S1': hist2d_S1, 'hist2d_S2': hist2d_S2,
                'time_ranges': time_ranges, 'input_arguments': input_arguments,
                'S1_info': S1_info, 'S2_info': S2_info}

    if f_out:
        pickler(f_out, to_write)

    return to_write


if __name__ == '__main__':
    # path = '/common/scratch/beriksso/TOFu/data/model_inadequacy'
    # files = os.listdir(path)
    # fnames = [f'{path}/{file}' for file in files]
    # summed = sum_nes_pickles(fnames, '/home/beriksso/model_inadequacy.pickle')
    pass
