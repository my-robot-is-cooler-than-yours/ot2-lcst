import matplotlib as mpl
import seaborn as sns
import os
from cycler import cycler
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from sklearn import linear_model
import time
import socket
import pyDOE2
import paramiko
import subprocess
from tkinter import Tk
from tkinter import filedialog
import json
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator
from tpot import TPOTRegressor
import shutil

# Set plotting parameters globally
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['font.family'] = 'Times New Roman'

# Define a custom color palette
custom_colors = [
    '#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
    '#DDCC77', '#CC6677', '#882255', '#AA4499',  # Original palette
    '#661100', '#6699CC', '#AA4466', '#4477AA', '#228833',
    '#66CCEE', '#EEDD88', '#EE6677', '#AA3377', '#BBBBBB',
    '#333333', '#FFDD44', '#9988CC', '#66AA77', '#117755'
]

# Set the color cycle using plt.rc
plt.rc('axes', prop_cycle=cycler('color', custom_colors))

current_directory = os.getcwd()


def log_msg(message):
    """Log a message with a timestamp."""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] {message}")


def get_output_path():
    """
    Prompt the user to select an output folder or create a new folder to save experiment results.

    :return: Full path of the selected folder.
    """
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    while True:
        # Prompt the user to select a folder
        output_path = filedialog.askdirectory(title="Select Output Folder")

        if not output_path:
            log_msg("No folder selected, please select a folder.")
        else:
            break

    root.quit()  # Close the Tkinter root window
    return output_path


def get_file_path():
    """
    Prompt the user to select a file.

    :return: Full path of the selected file.
    """
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    while True:
        # Prompt the user to select a file
        file_name = filedialog.askopenfilename(title="Select a File")

        if not file_name:
            log_msg("No file selected, please select a file.")
        else:
            break

    root.quit()  # Close the Tkinter root window
    return file_name


def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        log_msg(f"Processing time of {func.__qualname__}(): {time.time() - start_time:.2f} seconds.")
        return result

    return measure_time


def load_data(path_input: str) -> pd.DataFrame:
    """
    Load CSV data into a Pandas DataFrame.
    Does not modify column headings or perform any data cleaning.

    :param path_input: The file path of the CSV to load.
    :return pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    try:
        return pd.read_csv(path_input)
    except FileNotFoundError as e:
        log_msg(f"Error: File not found - {e}")
    except pd.errors.EmptyDataError as e:
        log_msg(f"Error: Empty file - {e}")
    except Exception as e:
        log_msg(f"Error loading file: {e}")
    return pd.DataFrame()  # Return an empty DataFrame on failure


def generate_lhs_design(num_samples, total_volume, step_size, num_factors):
    # Generate LHS design
    lhs = pyDOE2.lhs(num_factors, samples=num_samples)

    # Scale LHS design to the volume constraints
    scaled_lhs = lhs * (total_volume / num_factors)

    # Round to the nearest step_size and ensure minimum volume constraint
    scaled_lhs = np.round(scaled_lhs / step_size, 2) * step_size
    scaled_lhs = np.round(np.clip(scaled_lhs, step_size, None), 2)

    # Calculate the solvent volume as total_volume minus the sum of the components
    solvent_volumes = total_volume - np.sum(scaled_lhs, axis=1)

    # Combine the component volumes with the solvent volume as the last column
    scaled_lhs = np.column_stack((scaled_lhs, solvent_volumes))

    return scaled_lhs


def gen_volumes_csv(out_path, num_samples=46, total_volume=300, step_size=20, num_factors=2):
    # Generate and verify the LHS design
    while True:
        scaled_lhs = generate_lhs_design(num_samples, total_volume, step_size, num_factors)
        if np.all(np.sum(scaled_lhs, axis=1) == total_volume) and len(np.unique(scaled_lhs, axis=0)) == num_samples:
            log_msg("VERIFIED: All samples are unique and sum to 300 uL.")
            break
        log_msg("ERROR: Some samples are not unique or do not sum to 300 uL. Retrying...")

    # Define column names
    columns = [f'Component {i + 1}' for i in range(num_factors)] + ['Solvent']

    # Create DataFrame and save to CSV
    volumes = pd.DataFrame(scaled_lhs, columns=columns)
    current_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    volumes_file_path = out_path + f"\\Volumes_{current_time}.csv"
    volumes.to_csv(volumes_file_path, index=False)

    # Prepare duplicated volumes and additional solvent rows
    duplicated_volumes = np.repeat(scaled_lhs, 2, axis=0)
    duplicated_df = pd.DataFrame(duplicated_volumes, columns=columns)

    # Add four new rows with zeros in component columns and total_volume in the final column
    extra_rows = pd.DataFrame([[0] * num_factors + [total_volume]] * 4, columns=columns)
    duplicated_df = pd.concat([extra_rows, duplicated_df], ignore_index=True)

    # Save duplicated DataFrame to CSV
    duplicated_out_path = out_path + f"\\Duplicated_Volumes_{current_time}.csv"
    duplicated_df.to_csv(duplicated_out_path, index=False)

    log_msg("\n" + duplicated_df.round(2).to_csv(index=False))

    return duplicated_df, duplicated_out_path


def load_data_new(path: str, start_wavelength: int = 220, end_wavelength: int = 1000) -> pd.DataFrame:
    """
    Loads a CSV file without headers and assumes the first column contains identifiers.
    Renames the columns to include 'Row/Col' and a range of wavelength values.

    :param path: The file path of the CSV to load.
    :param start_wavelength: The starting wavelength for column renaming.
    :param end_wavelength: The ending wavelength for column renaming.
    :return pd.DataFrame: A pandas DataFrame with updated column names.
    """
    try:
        df = pd.read_csv(path, header=None)
        df.columns = ['Row/Col'] + list(range(start_wavelength, end_wavelength + 1))
        return df
    except FileNotFoundError as e:
        log_msg(f"Error: File not found - {e}")
    except pd.errors.EmptyDataError as e:
        log_msg(f"Error: Empty file - {e}")
    except Exception as e:
        log_msg(f"Error loading file: {e}")
    return pd.DataFrame()


def separate_columns(df: pd.DataFrame) -> tuple:
    """
    Separate numeric and non-numeric columns from a DataFrame.

    :param df: The DataFrame to separate.
    :return tuple: Three items - numeric columns (DataFrame), non-numeric columns (DataFrame), and the original column order (list).
    """
    numeric_cols = df.select_dtypes(include='number')
    non_numeric_cols = df.select_dtypes(exclude='number')
    original_columns = df.columns.tolist()
    return numeric_cols, non_numeric_cols, original_columns


def recombine_data(numeric_data: pd.DataFrame, non_numeric_data: pd.DataFrame, original_columns: list) -> pd.DataFrame:
    """
    Recombine numeric and non-numeric columns back into their original order.

    :param numeric_data: The numeric data.
    :param non_numeric_data: The non-numeric data.
    :param original_columns: The original column order.

    :return pd.DataFrame: The recombined DataFrame with columns in the original order.
    """
    combined_df = pd.concat([non_numeric_data, numeric_data], axis=1)
    combined_df = combined_df.reindex(columns=original_columns)
    return combined_df


def separate_subtract_and_recombine(raw_df: pd.DataFrame, plate_data: pd.DataFrame,
                                    blank_index: int = 0) -> pd.DataFrame:
    """
    Separate numeric and non-numeric columns, subtract the plate background and blank row,
    and recombine all columns into their original order.

    :param raw_df: The raw data containing both numeric and non-numeric columns.
    :param plate_data: The plate background data.
    :param blank_index: The index of the row to use as the blank for correction.

    :return pd.DataFrame: The fully corrected DataFrame.
    """
    # Separate numeric and non-numeric columns
    numeric_cols_raw, non_numeric_cols_raw, original_columns_raw = separate_columns(raw_df)
    numeric_cols_plate, non_numeric_cols_plate, _ = separate_columns(plate_data)

    # Subtract plate background and blank row
    plate_corrected_data = numeric_cols_raw - numeric_cols_plate

    # Blank correction (optimized with vectorized subtraction)
    blank_row = plate_corrected_data.iloc[blank_index]
    blank_corrected_data = plate_corrected_data.subtract(blank_row, axis=1)

    # Recombine columns, maintaining original order
    final_plate = recombine_data(blank_corrected_data, non_numeric_cols_raw, original_columns_raw)

    return final_plate


def plot_heatmap(df, value_col, title, ax, cmap='coolwarm', annot=True, fmt=".3f", cbar=True) -> None:
    """
    Plot a heatmap from a DataFrame.
    Extracts row/col labels and pivots the DataFrame to a heatmap format.

    :param df: DataFrame containing the data.
    :param value_col: Column to use for heatmap values.
    :param title: Title of the plot.
    :param ax: Axis object to plot on.
    :param cmap: Color map for the heatmap (default 'coolwarm').
    :param annot: Annotate the heatmap cells with values (default True).
    :param fmt: String formatting for annotations (default ".3f").
    :param cbar: Show color bar (default True).
    """
    try:
        df['Row'] = df['Row/Col'].str[0]  # Extract row letter
        df['Col'] = df['Row/Col'].str[1:].astype(int)  # Extract column number

        # Pivot the DataFrame to a format suitable for a heatmap
        heatmap_data = df.pivot(index='Row', columns='Col', values=value_col)

        sns.heatmap(heatmap_data, annot=annot, fmt=fmt, cmap=cmap, cbar=cbar, ax=ax)
        ax.set_title(title)

    except KeyError as e:
        log_msg(f"Error: The DataFrame does not have the expected columns. {e}")
    except Exception as e:
        log_msg(f"An error occurred while plotting the heatmap: {e}")


def plot_line(df, x_col_start, x_col_end, ax, title="Absorbance Spectra", samples_start=0, samples_end=1,
              wavelength_range=(220, 1000), ylim: tuple = False, legend=True) -> None:
    """
    Plot absorbance spectra for selected samples.

    :param df: DataFrame containing the data.
    :param x_col_start: Column index for the start of wavelength data.
    :param x_col_end: Column index for the end of wavelength data.
    :param ax: Axis object to plot on.
    :param title: Title of the plot.
    :param samples_start: Starting sample index (default 0).
    :param samples_end: Ending sample index (default 4).
    :param wavelength_range: Tuple representing the wavelength range (default (220, 1000)).
    :param ylim: Tuple representing the y-axis limits (default (-1.0, 2)).
    :param legend: Boolean to display legend (default True).
    """
    try:
        x = [int(i) for i in df.columns[x_col_start:x_col_end].values]  # Wavelength values

        for i in range(samples_start, samples_end):
            ax.plot(x, df.iloc[i, x_col_start:x_col_end].values, label=f'{df.iloc[i, 0]}')  # Label by sample index

        ax.set_xlim(wavelength_range)

        if ylim:
            ax.set_ylim(ylim)

        # Customize plot appearance
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in', pad=15)

        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance (AU)")
        ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')

        if legend:
            ax.legend(loc='best', fontsize=8)

    except KeyError as e:
        log_msg(f"Error: The DataFrame does not have the expected columns. {e}")
    except Exception as e:
        log_msg(f"An error occurred while plotting the spectra: {e}")


def least_squares_deconvolution(sample_spectrum, styrene_spectrum, polystyrene_spectrum) -> tuple:
    """
    Perform least-squares deconvolution to find the best coefficients
    for combining styrene and polystyrene spectra to fit the sample spectrum.

    :param sample_spectrum: Sample spectrum to fit.
    :param styrene_spectrum: Styrene reference spectrum.
    :param polystyrene_spectrum: Polystyrene reference spectrum.
    :return result.x: Coefficients for styrene and polystyrene.
    """

    def residuals(coeffs):
        c_styrene, c_polystyrene = coeffs
        combined_spectrum = c_styrene * styrene_spectrum + c_polystyrene * polystyrene_spectrum
        return np.sum((sample_spectrum - combined_spectrum) ** 2)

    result = minimize(residuals, [0.5, 0.5], bounds=[(0, None), (0, None)])
    return result.x


def scipy_curve_fit(sample_spectrum, styrene_spectrum, polystyrene_spectrum):
    """
    Perform curve fitting using scipy's curve_fit function.

    :param sample_spectrum: Sample spectrum to fit.
    :param styrene_spectrum: Styrene reference spectrum.
    :param polystyrene_spectrum: Polystyrene reference spectrum.
    :return: Fitted coefficients for styrene and polystyrene.
    """

    def model_func(wavelengths, c_styrene, c_polystyrene):
        return c_styrene * styrene_spectrum + c_polystyrene * polystyrene_spectrum

    wavelengths = np.arange(len(sample_spectrum))  # Assuming wavelength range equals spectrum length
    try:
        params, _ = curve_fit(model_func, wavelengths, sample_spectrum, p0=[0.5, 0.5])
        return params
    except Exception as e:
        log_msg(f"Error in curve fitting: {e}")
        return None


def prepare_spectra(styrene_spectrum_path, polystyrene_spectrum_path, range_start=0, range_end=None):
    """
    Load and prepare styrene and polystyrene spectra from files, selecting a wavelength range.

    :param styrene_spectrum_path: File path for styrene spectrum.
    :param polystyrene_spectrum_path: File path for polystyrene spectrum.
    :param range_start: Starting index for wavelength range (default 0).
    :param range_end: Ending index for wavelength range (default None, i.e., full range).
    :return: NumPy arrays of styrene and polystyrene spectra.
    """
    try:
        styrene_spectrum = load_data_new(styrene_spectrum_path)
        polystyrene_spectrum = load_data_new(polystyrene_spectrum_path)

        num_styrene, _, _ = separate_columns(styrene_spectrum)
        num_polystyrene, _, _ = separate_columns(polystyrene_spectrum)

        return num_styrene.values[0][range_start:range_end], num_polystyrene.values[0][range_start:range_end]

    except FileNotFoundError as e:
        log_msg(f"Error: File not found. {e}")
        return None, None
    except Exception as e:
        log_msg(f"Error in preparing spectra: {e}")
        return None, None


def fit_spectra(sample_spectrum, styrene_spectrum, polystyrene_spectrum,
                deconvolution_method=least_squares_deconvolution):
    """
    Fit the sample spectrum using a linear combination of styrene and polystyrene spectra
    via the selected deconvolution method.

    :param sample_spectrum: Sample spectrum to fit.
    :param styrene_spectrum: Styrene reference spectrum.
    :param polystyrene_spectrum: Polystyrene reference spectrum.
    :param deconvolution_method: Deconvolution method to use (default is least_squares_deconvolution).
    :return: Fitted coefficients for styrene and polystyrene.
    """
    try:
        return deconvolution_method(sample_spectrum, styrene_spectrum, polystyrene_spectrum)
    except Exception as e:
        log_msg(f"Error during spectrum fitting: {e}")
        return None


def calculate_r_squared(sample_spectrum, fitted_spectrum):
    """
    Calculate the R-squared value between the observed sample spectrum and the fitted spectrum.

    :param sample_spectrum: array-like, The observed spectrum of the sample.
    :param fitted_spectrum: array-like, The fitted spectrum based on component spectra.

    :return: float, The R-squared value representing the fit quality.
    """
    SS_res = np.sum((sample_spectrum - fitted_spectrum) ** 2)
    SS_tot = np.sum((sample_spectrum - np.mean(sample_spectrum)) ** 2)
    return 1 - (SS_res / SS_tot)


def process_samples(data_df, volumes_df, styrene_spectrum, polystyrene_spectrum, range_start, range_end,
                    deconvolution_method, plot_spectra=False, out_path=None):
    """
    Process each sample spectrum by fitting it to known styrene and polystyrene component spectra,
    and optionally plot the results.

    :param data_df: pd.DataFrame, DataFrame containing the sample absorbance spectra.
    :param volumes_df: pd.DataFrame, DataFrame containing the actual volumes of styrene and polystyrene.
    :param styrene_spectrum: array-like, Known absorbance spectrum for styrene.
    :param polystyrene_spectrum: array-like, Known absorbance spectrum for polystyrene.
    :param range_start: int, Starting index for the wavelength range to fit.
    :param range_end: int, Ending index for the wavelength range to fit.
    :param deconvolution_method: callable, Method used to fit the spectra to the components.
    :param plot_spectra: bool, optional, Whether to plot the fitted spectra (default is False).
    :param out_path: str, optional, Path to save the plots, if plotting is enabled (default is None).

    :return: tuple, Predicted and actual styrene and polystyrene components as lists.
    """
    styrene_components_pred, styrene_components_actual = [], []
    ps_components_pred, ps_components_actual = [], []

    for i in range(data_df.shape[0]):
        unknown_spectrum = data_df.select_dtypes(include='number').iloc[i, :].values[range_start:range_end]

        c_styrene_opt, c_polystyrene_opt = fit_spectra(unknown_spectrum, styrene_spectrum, polystyrene_spectrum,
                                                       deconvolution_method)

        fitted_spectrum = c_styrene_opt * styrene_spectrum + c_polystyrene_opt * polystyrene_spectrum

        styrene_components_pred.append(c_styrene_opt)
        styrene_components_actual.append(volumes_df.iloc[i, 0] * 0.025 / 300)
        ps_components_pred.append(c_polystyrene_opt)
        ps_components_actual.append(volumes_df.iloc[i, 1] * 0.25 / 300)

        if plot_spectra:
            # Plot the observed and fitted spectra
            fig, ax = plt.subplots(figsize=(8, 5))
            wavelengths = data_df.select_dtypes(include='number').columns.astype(float)[range_start:range_end]

            plt.plot(wavelengths, unknown_spectrum, label='Observed Mixture Spectrum', color='black')
            plt.plot(wavelengths, styrene_spectrum * c_styrene_opt, label='Predicted Styrene Component',
                     color='red', linestyle="-.")
            plt.plot(wavelengths, polystyrene_spectrum * c_polystyrene_opt, label='Predicted Polystyrene Component',
                     color='green', linestyle="-.")
            plt.plot(wavelengths, fitted_spectrum, label='Fitted Spectrum', color='blue', linestyle="--")

            # Customize plot appearance
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(0.5)

            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', pad=10)

            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Absorbance")

            ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')
            ax.legend(loc='best', fontsize=8)

            if out_path:
                plt.savefig(f"{out_path}/index_{i}.png")

        else:
            pass

    return styrene_components_pred, styrene_components_actual, ps_components_pred, ps_components_actual


def linear_regression(x_train, y_train, x_test, y_test):
    """
    Perform linear regression and evaluate the model on test data.

    :param x_train: array-like, Training data for the independent variable.
    :param y_train: array-like, Training data for the dependent variable.
    :param x_test: array-like, Test data for the independent variable.
    :param y_test: array-like, Test data for the dependent variable.

    :return: tuple, The fitted regression model and the predicted test values.
    """
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    log_msg(f"Equation: y = {regr.coef_[0]:.4f}x + {regr.intercept_:.4f}")
    log_msg(f"Mean squared error: {mean_squared_error(y_test, y_pred):.4f}")
    log_msg(f"R^2: {r2_score(y_test, y_pred):.4f}")

    return regr, y_pred


def plot_results(x_test, y_test, y_pred, regr, output_path, title, y_axis_label):
    """
    Plot the linear regression results including the expected fit and test data.

    :param x_test: array-like, Test data for the independent variable (predicted values).
    :param y_test: array-like, Test data for the dependent variable (actual values).
    :param y_pred: array-like, Predicted values from the regression model.
    :param regr: LinearRegression, The fitted regression model.
    :param output_path: str, Path to save the plot.
    :param title: str, Title of the plot.
    :param y_axis_label: str, Label for the y-axis.

    :return: None, Saves the plot to the specified output path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    plt.plot(x_test, y_pred, color='red', label=f"Expected Fit y = {regr.coef_[0]: .4f}x + {regr.intercept_: .4f}",
             linewidth=1, zorder=0)
    plt.scatter(x_test, y_test, color='black', label='Test Data', s=25)

    # Customize plot appearance
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_title(title)
    ax.set_xlabel("Predicted Spectral Fraction")
    ax.set_ylabel(y_axis_label)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', pad=10)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')
    ax.legend(loc='best', fontsize=8)

    plt.savefig(output_path)


def spectra_pca(df: pd.DataFrame, num_components: int, volumes: np.ndarray, plot_data: bool = False,
                x_bounds: tuple = False, out_path: str = current_directory):
    """
    Performs principal component analysis on array of spectra. Optionally plots PC plot and wavelength contribution plot
    colour coded with concentration.

    Returns pca_scores, pca_components, explained_variance

    :param df: pd.DataFrame containing sample spectra as rows.
    :param num_components: int, number of PCs to retain.
    :param volumes: np.ndarray, columns with volumes corresponding to each spectrum.
    :param plot_data: bool.
    :param x_bounds: tuple, determines x bounds on output plot.
    :param out_path: str, output path of plot.
    :return pca_scores: numpy.ndarray, contains PCA scores as array for corresponding num_components PCs.
    :return pca_components: numpy.ndarray, contains PCA components as array for corresponding num_component PCs.
    :return explained_variance: numpy.ndarray, contains the variance explained by each of the PCs.
    """
    # Convert volumes to concentrations
    num_analytes = volumes.shape[1]-1  # Number of analytes from the volume DataFrame
    for i in range(num_analytes):
        volumes[:, i] *= [0.025, 0.25, 0.5][i] / 300  # Replace with correct factors as needed

    # Perform PCA
    pca = PCA(n_components=num_components)  # Choose the number of components to retain
    pca_scores = pca.fit_transform(df)  # Get the scores (projections of data)
    pca_components = pca.components_  # Get the PCs (eigenvectors)
    explained_variance = pca.explained_variance_ratio_  # Variance explained by each PC

    if plot_data:
        # Plot the first two principal components (scores)
        plt.figure()
        scatter = plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=volumes[:, 1], cmap="viridis")
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA for Styrene/Toluene/Polystyrene Mixtures')

        # Add color bar to show concentration scale
        cbar = plt.colorbar(scatter)
        cbar.set_label('Concentration (mg/mL')

        plt.savefig(os.path.join(out_path, f"PCA Scores"))

        # Plot the loading of PC1 (contribution of each wavelength to PC1)
        plt.figure()
        plt.plot(np.arange(df.shape[1]) + 220, pca_components[0])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Loading on PC1')
        plt.title('Wavelength Loading on PC1')
        if x_bounds:
            plt.xlim(x_bounds)
        else:
            plt.xlim()

        plt.savefig(os.path.join(out_path, f"PCA Wavelength Weighting"))
    else:
        pass

    return pca_scores, pca_components, explained_variance


@timeit
def curve_fitting_lin_reg(plate_path, data_path, volumes_path, out_path):
    # Paths
    # plate_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\Plate 2a.csv"
    # data_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\240919_1305.csv"
    styrene_spectrum_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Styrene & PS Cuvette Specs\PRD Plate Reader Specs\styrene 0.025 mgmL.csv"
    polystyrene_spectrum_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\Styrene & PS Cuvette Specs\PRD Plate Reader Specs\polystyrene 0.250 mgmL.csv"
    # volumes_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\18-Sep-2024\Volumes 18-Sep Duplicated.csv"
    # out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\Test Folders\02-Oct-2024 expanded script figures"

    range_start = 40
    range_end = 120

    # Load data
    data_corrected = separate_subtract_and_recombine(load_data_new(data_path), load_data_new(plate_path))
    styrene_spectrum, polystyrene_spectrum = prepare_spectra(styrene_spectrum_path, polystyrene_spectrum_path,
                                                             range_start, range_end)
    volumes_df = load_data(volumes_path)

    # Choose deconvolution method here
    deconvolution_method = least_squares_deconvolution  # Or deconvolution_method_2, etc.

    styrene_pred, styrene_actual, ps_pred, ps_actual = process_samples(
        data_corrected, volumes_df, styrene_spectrum, polystyrene_spectrum, range_start, range_end,
        deconvolution_method,
        plot_spectra=True, out_path=out_path)

    # Styrene
    # Split into training and test data
    x_train, x_test = np.array(styrene_pred[:-20]).reshape(-1, 1), np.array(styrene_pred[-20:]).reshape(-1, 1)
    y_train, y_test = styrene_actual[:-20], styrene_actual[-20:]

    # Perform linear regression
    regr_styrene, y_pred = linear_regression(x_train, y_train, x_test, y_test)

    # Plot and save results
    plot_results(x_test, y_test, y_pred, regr_styrene, rf"{out_path}\linear model styrene LSR.png",
                 "Predicted Spectral Fraction vs Actual Concentration of Styrene", "Styrene Concentration (mg/mL)")

    # Polystyrene
    # Split into training and test data
    x_train, x_test = np.array(ps_pred[:-20]).reshape(-1, 1), np.array(ps_pred[-20:]).reshape(-1, 1)
    y_train, y_test = ps_actual[:-20], ps_actual[-20:]

    # Perform linear regression
    regr_polystyrene, y_pred = linear_regression(x_train, y_train, x_test, y_test)

    # Plot and save results
    plot_results(x_test, y_test, y_pred, regr_polystyrene, rf"{out_path}\linear model polystyrene LSR.png",
                 "Predicted Spectral Fraction vs Actual Concentration of Polystyrene",
                 "Polystyrene Concentration (mg/mL)")

    # # Pass in new data from 23-Sep Expt
    # new_data_raw_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\23-Sep-2024\240923_1512.csv"
    # plate_2c_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\23-Sep-2024\plate 2c.csv"
    # new_volumes_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\23-Sep-2024\Volumes No Solvent 23-Sep Duplicated.csv"
    #
    # new_data_raw = load_data_new(new_data_raw_path)
    # plate_2c = load_data_new(plate_2c_path)
    # new_volumes_df = load_data(new_volumes_path)
    #
    # # Process background etc
    # new_data_processed = separate_subtract_and_recombine(new_data_raw, plate_2c)
    #
    # # Deconvolute using cuvette peaks and get coefficients
    # styrene_components_pred, styrene_components_actual, ps_components_pred, ps_components_actual = process_samples(
    #     new_data_processed, new_volumes_df, styrene_spectrum, polystyrene_spectrum, range_start, range_end,
    #     deconvolution_method)
    #
    # styrene_concs_pred = regr_styrene.predict(np.array(styrene_components_pred).reshape(-1, 1))
    # ps_concs_pred = regr_polystyrene.predict(np.array(ps_components_pred).reshape(-1, 1))
    #
    # # Plot and save results
    # plot_results(styrene_components_pred, styrene_components_actual, styrene_concs_pred, regr_styrene,
    #              rf"{out_path}\new data with existing model for styrene LSR.png",
    #              "Predicted Spectral Fraction vs Actual Concentration of Styrene",
    #              "Stryene Concentration (mg/mL)")
    #
    # plot_results(ps_components_pred, ps_components_actual, ps_concs_pred, regr_polystyrene,
    #              rf"{out_path}\new data with existing model for polystyrene LSR.png",
    #              "Predicted Spectral Fraction vs Actual Concentration of Polystyrene",
    #              "Polystyrene Concentration (mg/mL)")
    #
    # log_msg(f"Mean squared error: {mean_squared_error(styrene_components_actual, styrene_concs_pred):.4f}")
    # log_msg(f"R^2: {r2_score(styrene_components_actual, styrene_concs_pred):.4f}")
    # log_msg(f"Mean squared error: {mean_squared_error(ps_components_actual, ps_concs_pred):.4f}")
    # log_msg(f"R^2: {r2_score(ps_components_actual, ps_concs_pred):.4f}")

    # crude = load_data_new(
    #     r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\crude df 80.csv")
    #
    # # Deconvolute using cuvette peaks and get coefficients
    # styrene_components_pred, styrene_components_actual, ps_components_pred, ps_components_actual = process_samples(
    #     crude, new_volumes_df, styrene_spectrum, polystyrene_spectrum, range_start, range_end, deconvolution_method,
    #     plot_spectra=True,
    #     out_path=r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\04-Oct-2024 Crude")
    #
    # styrene_concs_pred = regr_styrene.predict(np.array(styrene_components_pred).reshape(-1, 1))
    # ps_concs_pred = regr_polystyrene.predict(np.array(ps_components_pred).reshape(-1, 1))
    #
    # log_msg(styrene_concs_pred)
    # log_msg(ps_concs_pred)
    # log_msg(styrene_concs_pred / ps_concs_pred)


@timeit
def ml_screening(plate_path, data_path, volumes_df, out_path):
    # Correct data
    data_corrected = separate_subtract_and_recombine(load_data_new(data_path), load_data_new(plate_path))

    # Load in volumes
    volumes = volumes_df
    volumes_abs = pd.concat([volumes, data_corrected.iloc[:, 1:]], axis=1).to_numpy()

    # Correct from volume to concentration
    volumes_abs[:, 0] *= 0.025 / 300
    volumes_abs[:, 1] *= 0.25 / 300

    # Define range of wavelengths to search
    start_index = 40
    end_index = 120

    # Extract features (absorbance spectra) and targets (concentrations)
    X = volumes_abs[:, start_index:end_index]  # Absorbance spectra
    y = volumes_abs[:, :2]  # Concentrations of styrene and polystyrene

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Random Forest': RandomForestRegressor(),
    }

    # List of model names and predicted values
    # model_names = []
    # y_preds = []

    # List to store the metrics for each model
    metrics = {
        'Model': [],
        'R² Styrene': [],
        'MSE Styrene': [],
        'R² Polystyrene': [],
        'MSE Polystyrene': [],
        'Coefficients': [],
        'Intercepts': []
    }

    # Train models and store predictions
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # model_names.append(name)
        # y_preds.append(y_pred)

        # Calculate metrics for Styrene and Polystyrene
        r2_styrene = r2_score(y_test[:, 0], y_pred[:, 0])
        mse_styrene = mean_squared_error(y_test[:, 0], y_pred[:, 0])

        r2_polystyrene = r2_score(y_test[:, 1], y_pred[:, 1])
        mse_polystyrene = mean_squared_error(y_test[:, 1], y_pred[:, 1])

        log_msg(
            f"Model: {name} - R^2 = {r2_styrene: .4f}/{r2_polystyrene: .4f} - MSE: {mse_styrene: .4f}/{mse_polystyrene: .4f}")

        # Append metrics to the dictionary
        metrics['Model'].append(name)
        metrics['R² Styrene'].append(r2_styrene)
        metrics['MSE Styrene'].append(mse_styrene)
        metrics['R² Polystyrene'].append(r2_polystyrene)
        metrics['MSE Polystyrene'].append(mse_polystyrene)

        # Check and save model coefficients and intercepts if they exist
        if hasattr(model, 'coef_'):
            # Convert coefficients to list for JSON serialization
            metrics['Coefficients'].append(model.coef_.tolist())
        else:
            metrics['Coefficients'].append(None)

        if hasattr(model, 'intercept_'):
            # Convert intercepts to list for JSON serialization
            # Intercept can be scalar or array, so convert accordingly
            if isinstance(model.intercept_, np.ndarray):
                metrics['Intercepts'].append(model.intercept_.tolist())
            else:
                metrics['Intercepts'].append(model.intercept_)
        else:
            metrics['Intercepts'].append(None)

    # Create subplots: one row for each model, two columns for styrene and polystyrene
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models)))
    fig.suptitle('Model Predictions vs Actual Concentrations', fontsize=16)

    # Plot each model's predictions and add R² and MSE values
    for i, (name, model) in enumerate(models.items()):
        # Predict for the current model
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics for Styrene and Polystyrene
        r2_styrene = r2_score(y_test[:, 0], y_pred[:, 0])
        mse_styrene = mean_squared_error(y_test[:, 0], y_pred[:, 0])

        r2_polystyrene = r2_score(y_test[:, 1], y_pred[:, 1])
        mse_polystyrene = mean_squared_error(y_test[:, 1], y_pred[:, 1])

        # Styrene (first column of y)
        axes[i, 0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.7)
        axes[i, 0].plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'k--', lw=2)
        axes[i, 0].set_xlabel('Actual Styrene Concentration')
        axes[i, 0].set_ylabel('Predicted Styrene Concentration')
        axes[i, 0].set_title(f'{name} - Styrene')

        # Add R² and MSE as text annotations
        axes[i, 0].text(0.05, 0.9, f'R² = {r2_styrene:.4f}\nMSE = {mse_styrene:.4f}',
                        transform=axes[i, 0].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.5))

        # Polystyrene (second column of y)
        axes[i, 1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.7)
        axes[i, 1].plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'k--', lw=2)
        axes[i, 1].set_xlabel('Actual Polystyrene Concentration')
        axes[i, 1].set_ylabel('Predicted Polystyrene Concentration')
        axes[i, 1].set_title(f'{name} - Polystyrene')

        # Add R² and MSE as text annotations
        axes[i, 1].text(0.05, 0.9, f'R² = {r2_polystyrene:.4f}\nMSE = {mse_polystyrene:.4f}',
                        transform=axes[i, 1].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.5))

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path + r"\model_screening_concs_corrected.png")

    # Convert the metrics dictionary to a DataFrame for easy manipulation
    metrics_df = pd.DataFrame(metrics)

    # End training #

    return models, metrics_df, scaler


@timeit
def ml_screening_multi(plate_path, data_path, volumes_df, out_path, plot_spectra=False, start_index=20, end_index=200):
    # Correct data for background and blank
    data_corrected = separate_subtract_and_recombine(load_data_new(data_path), load_data_new(plate_path))

    if plot_spectra:
        for i in range(data_corrected.shape[0]):
            # Plot the observed and fitted spectra
            fig, ax = plt.subplots(figsize=(8, 5))
            wavelengths = data_corrected.select_dtypes(include='number').columns.astype(float)[start_index:end_index]

            plt.plot(wavelengths, data_corrected.iloc[i, start_index:end_index],
                     label=f'Observed Mixture Spectrum {float(volumes_df.iloc[i, 0]), float(volumes_df.iloc[i, 1]), float(volumes_df.iloc[i, 2])}',
                     color='black')

            # Customize plot appearance
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(0.5)

            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', pad=10)

            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Absorbance")

            ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')
            ax.legend(loc='best', fontsize=8)

            if out_path:
                # Create the spectra folder if it doesn’t exist
                spectra_path = os.path.join(out_path, "spectra")
                os.makedirs(spectra_path, exist_ok=True)

                # Save the plot
                plt.savefig(os.path.join(spectra_path, f"index_{i}.png"))
                plt.close()

    # Load in volumes
    volumes = volumes_df
    volumes_abs = pd.concat([volumes, data_corrected.iloc[:, 1:]], axis=1).to_numpy()

    # Correct from volume to concentration
    num_analytes = volumes.shape[1]-1  # Number of analytes from the volume DataFrame
    for i in range(num_analytes):
        volumes_abs[:, i] *= [0.025, 0.25, 0.5][i] / 300  # Replace with correct factors as needed

    # Extract features (absorbance spectra) and targets (concentrations)
    X = volumes_abs[:, start_index:end_index]  # Absorbance spectra
    y = volumes_abs[:, :num_analytes]  # Concentrations for all analytes

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Random Forest': RandomForestRegressor(),
    }

    # List to store the metrics for each model
    metrics = {'Model': []}
    for i in range(num_analytes):
        metrics[f'R² Analyte {i + 1}'] = []
        metrics[f'MSE Analyte {i + 1}'] = []

    metrics['Coefficients'] = []
    metrics['Intercepts'] = []

    # Train models and store predictions
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Append model name
        metrics['Model'].append(name)

        # Calculate metrics for each analyte
        for i in range(num_analytes):
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            metrics[f'R² Analyte {i + 1}'].append(r2)
            metrics[f'MSE Analyte {i + 1}'].append(mse)

        # Check and save model coefficients and intercepts if they exist
        metrics['Coefficients'].append(getattr(model, 'coef_', None).tolist() if hasattr(model, 'coef_') else None)
        metrics['Intercepts'].append(
            getattr(model, 'intercept_', None).tolist() if hasattr(model, 'intercept_') else None)

    # Create subplots dynamically based on analytes
    fig, axes = plt.subplots(len(models), num_analytes, figsize=(6 * num_analytes, 4 * len(models)))
    fig.suptitle('Model Predictions vs Actual Concentrations', fontsize=16)

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test_scaled)

        for j in range(num_analytes):
            ax = axes[i, j] if num_analytes > 1 else axes[i]
            ax.scatter(y_test[:, j], y_pred[:, j], alpha=0.7)
            ax.plot([y_test[:, j].min(), y_test[:, j].max()], [y_test[:, j].min(), y_test[:, j].max()], 'k--', lw=2)
            ax.set_xlabel(f'Actual Analyte {j + 1} Concentration')
            ax.set_ylabel(f'Predicted Analyte {j + 1} Concentration')
            ax.set_title(f'{name} - Analyte {j + 1}')
            r2 = r2_score(y_test[:, j], y_pred[:, j])
            mse = mean_squared_error(y_test[:, j], y_pred[:, j])
            ax.text(0.05, 0.9, f'R² = {r2:.4f}\nMSE = {mse:.4f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.5))

    # Adjust layout and save plot
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path + r"\model_screening_concs_corrected.png")

    # Convert metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics)

    return models, metrics_df, scaler


@timeit
def verify_models(plate_path, data_path, volumes_df, out_path, models, scaler, start_index=40, end_index=120):
    # Load and correct data for background and blank
    data_corrected = separate_subtract_and_recombine(load_data_new(data_path), load_data_new(plate_path))
    volumes = volumes_df
    volumes_abs = pd.concat([volumes, data_corrected.iloc[:, 1:]], axis=1).to_numpy()

    # Adjust concentrations by volume
    num_analytes = volumes.shape[1] - 1  # Number of analytes based on volumes_df columns
    for i in range(num_analytes):
        volumes_abs[:, i] *= [0.025, 0.25, 0.5][i] / 300  # Adjust as needed

    # Extract features (absorbance spectra) and target concentrations
    X = volumes_abs[:, start_index:end_index]  # Absorbance spectra
    y_test = volumes_abs[:, :num_analytes]  # Target concentrations

    # Normalize features using the previously fitted scaler
    X_scaled_new = scaler.transform(X)

    # Initialize list to store model predictions for the new dataset
    y_pred_new = []

    # Generate predictions from models on new data
    for name, model in models.items():
        y_pred = model.predict(X_scaled_new)
        y_pred_new.append(y_pred)

    # Plot model predictions vs actual values
    fig, axes = plt.subplots(len(models), num_analytes, figsize=(6 * num_analytes, 4 * len(models)))
    fig.suptitle('Model Predictions vs Actual Concentrations (Validation)', fontsize=16)

    for i, (name, y_pred) in enumerate(zip(models.keys(), y_pred_new)):
        for j in range(num_analytes):
            ax = axes[i, j] if num_analytes > 1 else axes[i]
            ax.scatter(y_test[:, j], y_pred[:, j], alpha=0.7)
            ax.plot([y_test[:, j].min(), y_test[:, j].max()], [y_test[:, j].min(), y_test[:, j].max()], 'k--', lw=2)
            ax.set_xlabel(f'Actual Analyte {j + 1} Concentration')
            ax.set_ylabel(f'Predicted Analyte {j + 1} Concentration')
            ax.set_title(f'{name} - Analyte {j + 1}')

            # Calculate and display R² and MSE
            r2 = r2_score(y_test[:, j], y_pred[:, j])
            mse = mean_squared_error(y_test[:, j], y_pred[:, j])
            ax.text(0.05, 0.9, f'R² = {r2:.4f}\nMSE = {mse:.4f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.5))

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_path, "model_screening_concs_validation.png"))

    return y_pred_new


def send_message(conn, message_type, message_data=""):
    """Send a message to the client, prefixed with a message type."""
    message = f"{message_type}|{message_data}"
    conn.sendall(message.encode())


def receive_message(conn):
    """Receive a message from the client."""
    data = conn.recv(1024).decode()
    return data.split("|", 1)


def run_subprocess(protocol_path):
    """
    Uses the subprocess module to transfer a protocol file to the OT-2 using legacy SCP (Secure Copy Protocol).
    This function is useful for uploading files from your local machine to the OT-2 before executing the protocol.

    :return: None
    """
    # Define the SCP command with the -O flag
    scp_command = [
        "scp",
        "-i",
        r"C:\Users\PRD-OT2\ot2_ssh_key",  # Path to your SSH key
        "-O",  # Force the legacy SCP protocol
        rf"{protocol_path}",
        # Local file path
        "root@169.254.21.69:/data/user_storage/prd_protocols"  # Destination on OT-2
    ]

    # Run the command using subprocess
    try:
        result = subprocess.run(scp_command, check=True, text=True, capture_output=True)
        log_msg("File transferred successfully!")

    except subprocess.CalledProcessError as e:
        log_msg(f"An error occurred: {e}")
        log_msg(f"Error output: {e.stderr}")


def run_ssh_command(protocol_name):
    """
    Establish an SSH connection to the Opentrons OT-2 and execute a protocol via SSH.
    This function uses Paramiko to communicate with the OT-2, executes the given protocol, and then
    processes the output to check for the "Protocol Finished" message.

    :return: None
    """
    # Variable to determine if protocol finished
    complete = False

    try:
        # Replace these with your own details
        hostname = "169.254.21.69"  # Replace with your OT-2's IP address
        username = "root"  # OT-2 default username is 'root'
        key_path = r"C:\Users\PRD-OT2\ot2_ssh_key"  # Path to your SSH private key
        protocol_path = f"'/data/user_storage/prd_protocols/{protocol_name}'"  # Path to your protocol on the OT-2

        # If using a passphrase with your SSH key
        key_passphrase = ""  # Replace with your SSH key passphrase or None if no passphrase

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the private key
        # private_key = paramiko.RSAKey.from_private_key_file(key_path, password=key_passphrase)

        ssh.connect(hostname, username=username, key_filename=key_path)

        stdin, stdout, stderr = ssh.exec_command(f'sh -l -c "opentrons_execute {protocol_path}"')

        # List to store the entire output
        full_output = []

        # Real-time output processing versus all-in-one at the end
        while True:
            line = stdout.readline()  # Read each line as it's received
            if not line:  # Break the loop when there's no more output
                break
            print(line, end='')  # log_msg the output line by line without extra newlines

            # Store the line in the full output list
            full_output.append(line)

            # Check for "Protocol Finished" in the real-time output
            if "Protocol Finished" in line:
                log_msg("Protocol end detected")
                complete = True
                break

            # Small delay to avoid overloading the loop
            time.sleep(0.1)

        # Read the output from stdout
        output = ''.join(stdout.readlines())

        log_msg(full_output)  # Optionally still log_msg the entire output for debugging or completion detection

        # Check for the phrase "Protocol Finished"
        if ' Protocol Finished\n' in full_output:
            log_msg("Protocol end detected")

        else:
            log_msg("Protocol end not detected")

    except Exception as e:
        log_msg(f"An error occurred: {e}")

    finally:
        # Close the SSH connection
        stdin.close()
        ssh.close()

    return complete


def run_simulation(protocol_name):
    """
    Establish an SSH connection to the Opentrons OT-2 and execute a protocol via SSH.
    This function uses Paramiko to communicate with the OT-2, executes the given protocol, and then
    processes the output to check for the "Protocol Finished" message.

    :return: None
    """
    # Variable to determine if protocol finished
    complete = False

    try:
        # Replace these with your own details
        hostname = "169.254.21.69"  # Replace with your OT-2's IP address
        username = "root"  # OT-2 default username is 'root'
        key_path = r"C:\Users\PRD-OT2\ot2_ssh_key"  # Path to your SSH private key
        protocol_path = f"'/data/user_storage/prd_protocols/{protocol_name}'"  # Path to your protocol on the OT-2

        # If using a passphrase with your SSH key
        key_passphrase = ""  # Replace with your SSH key passphrase or None if no passphrase

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the private key
        # private_key = paramiko.RSAKey.from_private_key_file(key_path, password=key_passphrase)

        ssh.connect(hostname, username=username, key_filename=key_path)

        stdin, stdout, stderr = ssh.exec_command(f'sh -l -c "opentrons_simulate {protocol_path}"')

        # List to store the entire output
        full_output = []

        # Real-time output processing versus all-in-one at the end
        while True:
            line = stdout.readline()  # Read each line as it's received
            if not line:  # Break the loop when there's no more output
                break
            print(line, end='')  # log_msg the output line by line without extra newlines

            # Store the line in the full output list
            full_output.append(line)

            # Check for "Protocol Finished" in the real-time output
            if "Protocol Finished" in line:
                log_msg("Protocol end detected")
                complete = True
                break

            # Small delay to avoid overloading the loop
            time.sleep(0.1)

        # Read the output from stdout
        output = ''.join(stdout.readlines())

        log_msg(full_output)  # Optionally still log_msg the entire output for debugging or completion detection

        # Check for the phrase "Protocol Finished"
        if ' Protocol Finished\n' in full_output:
            log_msg("Protocol end detected")

        else:
            log_msg("Protocol end not detected")

    except Exception as e:
        log_msg(f"An error occurred: {e}")

    finally:
        # Close the SSH connection
        stdin.close()
        ssh.close()

    return complete


def conc_model(conn, user_name: str = "Lachlan"):
    """
    For use in handle_client() function. Takes background & sample CSVs & generates ML model from corrected data.

    :param conn: Socket object to facilitate connection to 32-bit client.
    :param user_name: Name of the user running the experiment, obtained when handle_client() is run.
    :return:
    """
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Define paths for output data and protocol to be uploaded
    # out_path = r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\Automated Testing\22-Oct Almost Full Auto w. Offsets"
    log_msg("Select path for data output:")
    out_path = get_output_path()

    # protocol_name = "Mixtures Expt - SSH"
    log_msg("Select path for protocol upload:")
    protocol_path = get_output_path()
    protocol_name = protocol_path.split("/")[-1]

    verification = False

    # Dictionary for experiment metadata
    experiment_metadata = {
        "user": user_name,
        "start_time": start_time,
        "output_path": out_path
    }

    while True:
        # Ask plate reader to take blank reading of plate. Wait for user confirmation to proceed
        log_msg("Plate background is required to be taken before proceeding")

        while True:
            user_input = input(">>> Has empty plate been prepared and ready to be inserted? (yes/no): \n>>> ")
            if user_input.lower() == "yes":
                break
            else:
                log_msg("Waiting for plate preparation...")

        log_msg("Requesting plate background from reader")
        send_message(conn, "PLATE_BACKGROUND", "Empty Plate Reading")

        log_msg("Awaiting message from client")
        msg_type, msg_data = receive_message(conn)

        if msg_type == "PLATE_BACKGROUND":
            log_msg("Plate background data received")

            # Save plate bg to variable
            plate_background_path = msg_data
            log_msg("Plate background path saved")

        volumes_df, volumes_path = gen_volumes_csv()  # Generate volumes for first experiment

        # Upload duplicated volumes CSV to OT-2 with retry logic to handle connection errors
        upload_success = False
        while not upload_success:
            try:
                log_msg("Uploading protocol to OT-2")
                run_subprocess(volumes_path)
                log_msg("Upload complete")
                upload_success = True
            except Exception as e:
                log_msg(f"Error uploading protocol: {e}")
                user_input = input(">>> Retry upload? (yes/no): \n>>> ")
                if user_input.lower() != "yes":
                    log_msg("Upload cancelled by user")
                    break

        # # Remove this block once auto protocol gen has been sorted
        # while True:
        #     user_input = input(">>> Has protocol been updated with new dispense volumes? (yes/no): \n>>> ")
        #     if user_input.lower() == "yes":
        #         break
        #     else:
        #         log_msg("Waiting for protocol preparation...")

        # Upload updated script to OT-2
        # run_subprocess(r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Honours Python Main\OT-2 Protocols\DoE + Monomers Experiment\Mixtures Expt - SSH.py")

        upload_success = False
        while not upload_success:
            try:
                log_msg("Uploading protocol to OT-2")
                run_subprocess(protocol_path)
                log_msg("Upload complete")
                upload_success = True
            except Exception as e:
                log_msg(f"Error uploading protocol: {e}")
                user_input = input(">>> Retry upload? (yes/no): \n>>> ")
                if user_input.lower() != "yes":
                    log_msg("Upload cancelled by user")
                    break

        # Make robot prepare samples
        log_msg("Please allow robot to prepare samples before proceeding")

        # SSH into OT-2 and run opentrons_execute command
        log_msg(f"SSh'ing into OT-2 and running opentrons_execute command on protocol {protocol_name}")
        output = run_ssh_command(protocol_name)
        # output=True

        # Wait for robot confirmation that run is complete
        while True:
            # Check for the phrase "Protocol Finished" in shell output
            if output:
                log_msg("OT-2 protocol finished, proceeding")
                break

            else:
                log_msg("OT-2 either not finished protocol or error has occurred")
                user_input = input(">>> Proceed manually? \n>>> ")
                if user_input.lower() == "yes":
                    break
                else:
                    pass

        # # Remove this block once labware definition for plate reader tray has been sorted
        # while True:
        #     user_input = input(">>> Has full plate been prepared and ready to be inserted? (yes/no): \n>>> ")
        #     if user_input.lower() == "yes":
        #         break
        #     else:
        #         log_msg("Waiting for plate preparation...")

        log_msg("Requesting to run measurement protocol")
        send_message(conn, "RUN_PROTOCOL", "Empty Plate Reading")

        log_msg("Awaiting message from client")
        msg_type, msg_data = receive_message(conn)

        if msg_type == "CSV_FILE":
            if verification is False:
                log_msg(f"Measurement complete")
                log_msg(f"Received CSV file with path: {msg_data}")

                data_path = msg_data
                log_msg("Data path saved")

                # Run machine learning screening
                log_msg("Doing ML stuff")
                models, metrics, scaler = ml_screening(plate_background_path, data_path, volumes_df, out_path)

                log_msg(f"\n{metrics}")

            elif verification is True:
                log_msg("!!!Verification Step!!!")
                log_msg(f"Measurement complete")
                log_msg(f"Received CSV file with path: {msg_data}")

                data_path = msg_data
                log_msg("Data path saved")

                # Run machine learning screening
                log_msg("Doing ML stuff")
                verify_models(plate_background_path, data_path, volumes_df, out_path, models, scaler)
                log_msg("Verification step complete")

                break  # should break from outer loop

        # Test certain conditions being met after analysis completed

        # Decision based on metrics for verification
        styrene_valid = all(r2 >= 0.90 and mse < 0.001 for r2, mse in zip(metrics['R² Styrene'], metrics['MSE Styrene']))
        polystyrene_valid = all(r2 >= 0.90 and mse < 0.001 for r2, mse in zip(metrics['R² Polystyrene'], metrics['MSE Polystyrene']))

        if styrene_valid and polystyrene_valid:
            log_msg("Initial model parameters OK - verification step authorised.")
            test = "yes"
        else:
            log_msg("Model parameters poor. Closing loop.")
            test = "no"

        # test = input(">>> Condition met? \n>>> ")  # Imagine this is some condition being met from the ML quality parameters

        if test.lower() == "yes":
            pass
            verification = True
            log_msg("Decision - verification requested")
            # loop repeats to take set of verification samples

        else:
            # send_message(conn, "ANALYSIS_COMPLETE")
            verification = False  # this should break loop
            break

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    experiment_metadata["end_time"] = end_time
    log_msg(f"Experiment ended at {end_time}")

    # Save experiment metadata
    with open(os.path.join(out_path, 'experiment_metadata.json'), 'w') as f:
        json.dump(experiment_metadata, f, indent=4)

    log_msg("Metadata saved")


def conc_model_for_testing(conn, user_name: str = "Lachlan"):
    """
    For use in handle_client() function. Takes background & sample CSVs & generates ML model from corrected data.

    :param conn: Socket object to facilitate connection to 32-bit client.
    :param user_name: Name of the user running the experiment, obtained when handle_client() is run.
    :return:
    """
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Define paths for output data and protocol to be uploaded
    log_msg("Select path for data output:")
    out_path = get_output_path()

    # protocol_name = "Mixtures Expt - SSH"
    log_msg("Select path for protocol upload:")
    protocol_path = get_file_path()
    protocol_name = protocol_path.split("/")[-1]

    verification = False

    # Dictionary for experiment metadata
    experiment_metadata = {
        "user": user_name,
        "start_time": start_time,
        "output_path": out_path
    }

    while True:
        # Ask plate reader to take blank reading of plate. Wait for user confirmation to proceed
        log_msg("Plate background is required to be taken before proceeding")

        while True:
            user_input = input(">>> Has empty plate been prepared and ready to be inserted? (yes/no): \n>>> ")
            if user_input.lower() == "yes":
                break
            else:
                log_msg("Waiting for plate preparation...")

        log_msg("Requesting plate background from reader")
        send_message(conn, "PLATE_BACKGROUND", "Empty Plate Reading")

        log_msg("Awaiting message from client")
        msg_type, msg_data = receive_message(conn)

        if msg_type == "PLATE_BACKGROUND":
            log_msg("Plate background data received")

            # Save plate bg to variable
            plate_background_path = msg_data
            log_msg("Plate background path saved")

        volumes_df, volumes_path = gen_volumes_csv()  # Generate volumes for first experiment

        # Upload duplicated volumes CSV to OT-2 with retry logic to handle connection errors
        upload_success = False
        while not upload_success:
            try:
                log_msg("Uploading volumes data to OT-2")
                run_subprocess(volumes_path)
                log_msg("Upload complete")
                upload_success = True
            except Exception as e:
                log_msg(f"Error uploading protocol: {e}")
                user_input = input(">>> Retry upload? (yes/no): \n>>> ")
                if user_input.lower() != "yes":
                    log_msg("Upload cancelled by user")
                    break

        upload_success = False
        while not upload_success:
            try:
                log_msg("Uploading protocol to OT-2")
                run_subprocess(protocol_path)
                log_msg("Upload complete")
                upload_success = True
            except Exception as e:
                log_msg(f"Error uploading protocol: {e}")
                user_input = input(">>> Retry upload? (yes/no): \n>>> ")
                if user_input.lower() != "yes":
                    log_msg("Upload cancelled by user")
                    break

        # Make robot prepare samples
        log_msg("Please allow robot to prepare samples before proceeding")

        # SSH into OT-2 and run opentrons_execute command
        log_msg(f"SSh'ing into OT-2 and running opentrons_execute command on protocol {protocol_name}")
        output = run_ssh_command(protocol_name)
        # output = True

        # Wait for robot confirmation that run is complete
        while True:
            # Check for the phrase "Protocol Finished" in shell output
            if output:
                log_msg("OT-2 protocol finished, proceeding")
                break

            else:
                log_msg("OT-2 either not finished protocol or error has occurred")
                user_input = input(">>> Proceed manually? \n>>> ")
                if user_input.lower() == "yes":
                    break
                else:
                    pass

        log_msg("Requesting to run measurement protocol")
        send_message(conn, "RUN_PROTOCOL", "Empty Plate Reading")

        log_msg("Awaiting message from client")
        msg_type, msg_data = receive_message(conn)

        if msg_type == "CSV_FILE":
            if verification is False:
                log_msg(f"Measurement complete")
                log_msg(f"Received CSV file with path: {msg_data}")

                data_path = msg_data
                log_msg("Data path saved")

                # Run machine learning screening
                log_msg("Doing ML stuff")
                models, metrics, scaler = ml_screening(plate_background_path, data_path, volumes_df, out_path)

                log_msg(f"\n{metrics}")

                # try:
                #     curve_fitting_lin_reg(plate_background_path, data_path, volumes_path, out_path)
                # except Exception as e:
                #     log_msg(f"An error occured during curve fitting linear regression: {e}")

            elif verification is True:
                log_msg("!!!Verification Step!!!")
                log_msg(f"Measurement complete")
                log_msg(f"Received CSV file with path: {msg_data}")

                data_path = msg_data
                log_msg("Data path saved")

                # Run machine learning screening
                log_msg("Doing ML stuff")
                verify_models(plate_background_path, data_path, volumes_df, out_path, models, scaler)
                log_msg("Verification step complete")

                break  # should break from outer loop

        # Decision based on metrics for verification
        styrene_valid = all(r2 >= 0.80 and mse < 0.005 for r2, mse in zip(metrics['R² Styrene'], metrics['MSE Styrene']))
        polystyrene_valid = all(r2 >= 0.80 and mse < 0.005 for r2, mse in zip(metrics['R² Polystyrene'], metrics['MSE Polystyrene']))

        if styrene_valid or polystyrene_valid:
            log_msg("Initial model parameters OK - verification step authorised.")
            test = "yes"
        else:
            log_msg("Model parameters poor. Closing loop.")
            test = "no"

        if test.lower() == "yes":
            pass
            verification = True
            log_msg("Decision - verification requested")
            # loop repeats to take set of verification samples

        else:
            # send_message(conn, "ANALYSIS_COMPLETE")
            verification = False  # this should break loop
            break

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    experiment_metadata["end_time"] = end_time
    log_msg(f"Experiment ended at {end_time}")

    # Convert metrics df to dict
    experiment_metadata["Metrics"] = metrics.to_dict()

    # Save experiment metadata ensuring correct encoding
    with open(os.path.join(
            r"C:\Users\Lachlan Alexander\Desktop\Uni\2024 - Honours\Experiments\DOE + Monomer + Polymer Mixtures\Automated Testing\28-Oct Full Auto",
            'experiment_metadata_test.json'), 'w', encoding='utf-8') as f:
        json.dump(experiment_metadata, f, indent=4, ensure_ascii=False)

    log_msg("Metadata saved")


def check_stable_temp(conn,
                      goal_temp: str,
                      stabilization_time: float = 60,
                      check_interval: float = 5,
                      range_tolerance: float = 0.2,
                      temps1=[],
                      temps2=[],
                      time_stamps=[],
                      *args):
    """
    Checks that the plate reader heating plate is at a stable goal temperature for a period of time.

    :param conn: Socket object used for plate reader comms
    :param goal_temp: The goal temperature to be reached for a certain period of time
    :param stabilization_time: The amount of time the temp is required to be stable for before proceeding
    :param check_interval: The amount of time between temperature checks
    :param range_tolerance: The tolerance (+/-) on the range that the temperature must be between
    :param args:
    :return:
    """

    # Calculate acceptable temperature range
    target_min = float(goal_temp) - range_tolerance
    target_max = float(goal_temp) + range_tolerance

    # Initialize a counter for stable time
    stable_time = 0

    # Loop until current_temp reaches and stays within the specified range for the required period
    while stable_time < stabilization_time:
        log_msg("Requesting current temperature.")
        send_message(conn, "GET_TEMP")

        log_msg("Awaiting message from client.")
        msg_type, msg_data = receive_message(conn)

        if msg_type == "TEMPS":
            log_msg(f"Temperature measurement complete.")

            time_stamps.append(datetime.now())  # Add current timestamp for each measurement
            log_msg("Timestamp saved")

            log_msg(f"Temp1, Temp2: {msg_data}")

            temp1 = int(msg_data.split(",")[0].strip()) / 10
            temp2 = int(msg_data.split(",")[1].strip()) / 10

            temps1.append(temp1)
            temps2.append(temp2)
            log_msg("Temperatures saved.")

            current_temp = temp1  # Set to lower plate temp since upper plate runs hot

            # Check if current_temp is within the target range
            if target_min <= current_temp <= target_max:
                stable_time += check_interval  # Increment stable time by check interval
                log_msg(f"Temperature within range ({target_min} to {target_max}) for {stable_time} seconds.")
            else:
                stable_time = 0  # Reset stable time if out of range
                log_msg(f"Temperature out of range; resetting stable time counter.")

            # Wait before taking the next measurement
            time.sleep(check_interval)

    log_msg(f"Goal temperature of {goal_temp} has been reached and has been stable for {stable_time} seconds.")

    return None


def measurements_over_time(conn, user_name: str = "Lachlan"):
    """
    Takes measurements over time and plots absorbance at given wavelength as function of time between measurements.

    :param conn: Socket object to facilitate connection to 32-bit client.
    :param user_name: Name of the user running the experiment, obtained when handle_client() is run.
    :return:
    """
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Define paths for output data and protocol to be uploaded
    log_msg("Select path for data output:")
    out_path = get_output_path()

    # protocol_name = "Mixtures Expt - SSH"
    log_msg("Select path for protocol upload:")
    protocol_path = get_file_path()
    protocol_name = protocol_path.split("/")[-1]

    # Dictionary for experiment metadata
    experiment_metadata = {
        "user": user_name,
        "start_time": start_time,
        "output_path": out_path
    }

    while True:
        # Ask plate reader to take blank reading of plate. Wait for user confirmation to proceed
        log_msg("Plate background is required to be taken before proceeding")

        while True:
            user_input = input(">>> Has empty plate been prepared and ready to be inserted? (yes/no): \n>>> ")
            if user_input.lower() == "yes":
                break
            else:
                log_msg("Waiting for plate preparation...")

        log_msg("Requesting plate background from reader")
        send_message(conn, "PLATE_BACKGROUND", "Empty Plate Reading")

        log_msg("Awaiting message from client")
        msg_type, msg_data = receive_message(conn)

        if msg_type == "PLATE_BACKGROUND":
            log_msg("Plate background data received")

            # Save plate bg to variable
            plate_background_path = msg_data
            log_msg("Plate background path saved")

        upload_success = False
        while not upload_success:
            try:
                log_msg("Uploading protocol to OT-2")
                run_subprocess(protocol_path)
                log_msg("Upload complete")
                upload_success = True
            except Exception as e:
                log_msg(f"Error uploading protocol: {e}")
                user_input = input(">>> Retry upload? (yes/no): \n>>> ")
                if user_input.lower() != "yes":
                    log_msg("Upload cancelled by user")
                    break

        # Make robot prepare samples
        log_msg("Please allow robot to prepare samples before proceeding")

        # SSH into OT-2 and run opentrons_execute command
        log_msg(f"SSh'ing into OT-2 and running opentrons_execute command on protocol {protocol_name}")
        output = run_ssh_command(protocol_name)
        # output = True

        # Wait for robot confirmation that run is complete
        while True:
            # Check for the phrase "Protocol Finished" in shell output
            if output:
                log_msg("OT-2 protocol finished, proceeding")
                break

            else:
                log_msg("OT-2 either not finished protocol or error has occurred")
                user_input = input(">>> Proceed manually? \n>>> ")
                if user_input.lower() == "yes":
                    break
                else:
                    pass

        data_paths = []
        time_stamps = []  # Store time stamps for each measurement

        for i in range(6):
            log_msg("Requesting to run measurement protocol")
            send_message(conn, "RUN_PROTOCOL", "Empty Plate Reading")

            log_msg("Awaiting message from client")
            msg_type, msg_data = receive_message(conn)

            if msg_type == "CSV_FILE":
                log_msg(f"Measurement complete")
                log_msg(f"Received CSV file with path: {msg_data}")

                data_paths.append(msg_data)
                time_stamps.append(datetime.now())  # Add current timestamp for each measurement
                log_msg("Data path saved")

                if i < 6:  # Only wait if there are more measurements to be taken
                    log_msg("Sleeping for one hour")
                    time.sleep(600)  # sleep for 10 mins before measuring again

        # Calculate averages and standard deviations
        abs_std_dev = []

        for path in data_paths:
            plate = load_data_new(plate_background_path)
            data = load_data_new(path)

            corrected_array_260 = separate_subtract_and_recombine(data, plate, 0)[260][1:25].to_numpy()
            corrected_array_282 = separate_subtract_and_recombine(data, plate, 0)[282][1:25].to_numpy()

            average = np.average(corrected_array_260)
            std = np.std(corrected_array_260)

            abs_std_dev.append((average, std))

        # Extract averages, standard deviations, and times for plotting
        averages = [item[0] for item in abs_std_dev]
        std_devs = [item[1] for item in abs_std_dev]
        times = [(t - time_stamps[0]).total_seconds() / 60 for t in time_stamps]  # Minutes from start

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data with error bars for standard deviation
        ax.errorbar(times, averages, yerr=std_devs, fmt='-o', ecolor='gray', capsize=5)
        ax.plot(times, averages, 'o-', label='Average Absorbance')

        # Add labels and titles
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Average Absorbance')
        ax.set_title('Absorbance of Styrene in BuOAc Over Time')

        # # Add annotations for the first and last points
        # ax.text(times[0], averages[0], f'Avg: {averages[0]:.4f}\nStd: {std_devs[0]:.4f}',
        #         ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        # ax.text(times[-1], averages[-1], f'Avg: {averages[-1]:.4f}\nStd: {std_devs[-1]:.4f}',
        #         ha='left', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        # Show grid, legend, and layout adjustments
        ax.grid(True)
        # plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.tight_layout()
        plt.savefig(out_path + r"\absorbance_over_time.png")
        plt.show()

        break

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_msg(f"Experiment ended at {end_time}")

    experiment_metadata["end_time"] = end_time
    experiment_metadata["data_paths"] = data_paths
    experiment_metadata["timestamps"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in time_stamps]
    experiment_metadata["averages"] = averages
    experiment_metadata["std_devs"] = std_devs

    # Save experiment metadata ensuring correct encoding
    with open(os.path.join(
            out_path,
            'experiment_metadata_test.json'), 'w', encoding='utf-8') as f:
        json.dump(experiment_metadata, f, indent=4, ensure_ascii=False)

    log_msg("Metadata saved")


def temperature_over_time(conn, user_name: str = "Lachlan"):
    """
    Takes measurements over time and plots absorbance at a given wavelength as a function of time between measurements.

    :param conn: Socket object to facilitate connection to 32-bit client.
    :param user_name: Name of the user running the experiment, obtained when handle_client() is run.
    :return:
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    start_temp = "25.0"
    target_temp = "27.0"
    step_size = 0.5
    current_temp = 0

    # Initialize lists for timestamps and temperatures
    time_stamps = []
    measurement_times = []
    temps1 = []
    temps1_plotting = []
    temps2 = []
    temps2_plotting = []
    data_paths=[]

    # Define paths for output data and protocol to be uploaded
    log_msg("Select path for data output:")
    out_path = get_output_path()

    # Define protocol name for sample dispensing
    log_msg("Select path for protocol upload:")
    protocol_path = get_file_path()
    protocol_name = protocol_path.split("/")[-1]

    # Dictionary for experiment metadata
    experiment_metadata = {
        "user": user_name,
        "start_time": start_time,
        "output_path": out_path,
        "temperature_steps": [],
    }

    # Plate background and sample preparation steps

    log_msg("Plate background is required to be taken before proceeding.")

    while True:
        user_input = input(">>> Has empty plate been prepared and ready to be inserted? (yes/no): \n>>> ")
        if user_input.lower() == "yes":
            break
        else:
            log_msg("Waiting for plate preparation...")

    log_msg("Collecting plate background from reader.")
    send_message(conn, "PLATE_BACKGROUND", "Empty Plate Reading")

    log_msg("Awaiting update from client.")
    msg_type, msg_data = receive_message(conn)

    if msg_type == "PLATE_BACKGROUND":
        log_msg("Plate background data received.")

        # Save plate bg to variable
        plate_background_path = msg_data
        log_msg(f"Plate background saved to {plate_background_path}.")

    upload_success = False
    while not upload_success:
        try:
            log_msg("Uploading protocol to OT-2.")
            run_subprocess(protocol_path)
            log_msg("Upload complete.")
            upload_success = True
        except Exception as e:
            log_msg(f"Error uploading protocol: {e}")
            user_input = input(">>> Retry upload? (yes/no): \n>>> ")
            if user_input.lower() != "yes":
                log_msg("Upload cancelled by user.")
                break

    # Make robot prepare samples
    # SSH into OT-2 and run opentrons_execute command
    log_msg(f"SSH'ing into OT-2 and running opentrons_execute command on protocol {protocol_name}.")
    log_msg(f"Please allow OT-2 to finalise protocol before making any changes.")
    output = run_ssh_command(protocol_name)

    # Wait for robot confirmation that run is complete
    while True:
        # Check for the phrase "Protocol Finished" in shell output
        if output:
            log_msg("OT-2 protocol complete.")
            break

        else:
            log_msg("OT-2 has either not finished protocol or an error has occurred.")
            user_input = input(">>> Proceed manually? (yes/no) \n>>> ")
            if user_input.lower() == "yes":
                break
            else:
                pass

    # Start temperature setting and stabilization process
    log_msg(f"Setting goal temperature to {start_temp}.")
    send_message(conn, message_type="SET_TEMP", message_data=start_temp)

    # Wait for stabilization at the starting temperature
    check_stable_temp(conn, start_temp, stabilization_time=60, check_interval=3, range_tolerance=0.3,
                      temps1=temps1, temps2=temps2, time_stamps=time_stamps)

    # Take first reading for start_temp
    log_msg("Collecting absorbance data from plate reader.")
    send_message(conn, "RUN_PROTOCOL", "Empty Plate Reading")

    log_msg("Awaiting update from plate reader.")
    while True:
        msg_type, msg_data = receive_message(conn)
        if msg_type == "CSV_FILE":
            log_msg("Data received.")
            break
        else:
            log_msg("Yet to receive data.")
            time.sleep(5)

    log_msg(f"Measurement complete. Received CSV file with path: {msg_data}.")

    data_paths.append(msg_data)
    measurement_times.append(datetime.now())
    log_msg("Data path saved.")

    temps1_plotting.append(25.0)
    temps2_plotting.append(25.0)
    log_msg("Temperature data saved.")

    # Set new temp
    current_temp = float(start_temp) + step_size

    # Loop through stabilisation, temp setting, and measurement steps to obtain abs/temp/time data until target_temp reached
    while current_temp <= float(target_temp):
        # Set and stabilize at each incremental temperature
        log_msg(f"Setting goal temperature to {str(current_temp)}.")
        send_message(conn, message_type="SET_TEMP", message_data=str(current_temp))

        check_stable_temp(conn, current_temp, stabilization_time=30, check_interval=5, range_tolerance=0.2,
                          temps1=temps1, temps2=temps2, time_stamps=time_stamps)

        experiment_metadata["temperature_steps"].append({
            "target_temperature": current_temp,
            "temps1": temps1[-1],  # Last recorded value at this step
            "temps2": temps2[-1],
            "timestamp": time_stamps[-1].strftime("%Y-%m-%d %H:%M:%S")
        })

        # Allow five minutes for plate to reach current temp
        # time.sleep(secs=5 * 60)
        log_msg("plate reaching temp...")
        log_msg("OK")

        # Take reading for current temp
        log_msg("Requesting to run measurement protocol")
        send_message(conn, "RUN_PROTOCOL", "Empty Plate Reading")

        log_msg("Awaiting message from client")

        while True:
            msg_type, msg_data = receive_message(conn)
            if msg_type == "CSV_FILE":
                log_msg("Data received.")
                break
            else:
                log_msg("Yet to receive data.")
                time.sleep(5)

        log_msg(f"Measurement complete")
        log_msg(f"Received CSV file with path: {msg_data}")

        data_paths.append(msg_data)
        time_stamps.append(datetime.now())  # Add current timestamp for each measurement
        log_msg("Data path saved")

        temps1_plotting.append(temps1[-1])
        temps2_plotting.append(temps2[-1])
        log_msg("Temperature data saved")

        current_temp += step_size  # Increment temperature

    # Calculate averages and standard deviations
    abs_std_dev = []

    transmittance_dir = os.path.join(out_path, 'transmittance spectra')
    os.makedirs(transmittance_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for path in data_paths:
        plate = load_data_new(plate_background_path)
        data = load_data_new(path)

        corrected_array = separate_subtract_and_recombine(data, plate, 0)[600][
                          1:13].to_numpy()  # only 600 nm from wells 2-12
        try:
            # Convert absorbance to percentage transmittance
            transmittance_array = 10 ** (-corrected_array) * 100

            # Save the transmittance data to CSV
            transmittance_df = pd.DataFrame(transmittance_array)
            transmittance_filename = os.path.join(transmittance_dir, f"transmittance_{os.path.basename(path)}")
            transmittance_df.to_csv(transmittance_filename, index=False)

            # Calculate average and standard deviation for transmittance data
            average = np.average(transmittance_array)
            std = np.std(transmittance_array)

            abs_std_dev.append((average, std))
        except Exception as e:
            log_msg(f"Error occurred: {e}")

    # Extract averages, standard deviations, and times for plotting
    averages = [item[0] for item in abs_std_dev]
    std_devs = [item[1] for item in abs_std_dev]
    # times = [(t - time_stamps[0]).total_seconds() / 60 for t in time_stamps]  # Minutes from start

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data with error bars for standard deviation
    ax.errorbar(temps1_plotting, averages, yerr=std_devs, fmt='-o', ecolor='gray', capsize=5)
    ax.plot(temps1_plotting, averages, 'o-', label='Average Absorbance')

    # Add labels and titles
    ax.set_xlabel("Temperature of Lower Heating Plate " + "("+u'\u2103'+")")
    ax.set_ylabel('Average % Transmittance')
    ax.set_title('% Transmittance at 500 nm of p(NIPAM) in Water versus Temperature')

    # Show grid, legend, and layout adjustments
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_path + r"\abs_versus_temp.png")
    # plt.show()
    plt.close()

    # Plotting
    times = [(t - time_stamps[0]).total_seconds() / 60 for t in time_stamps]  # Minutes from start
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, temps1, label='Heating Plate 1 Temperature')
    ax.plot(times, temps2, label='Heating Plate 2 Temperature')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Temperature')
    ax.set_title('Heating Plate Temperature Over Time')
    ax.legend(loc='best')
    ax.grid(True, linestyle='-', linewidth=0.2, which='major', axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "plate_temp_over_time.png"))
    # plt.show()
    plt.close()

    # Save data to JSON with structured metadata
    json_file_path = os.path.join(out_path, f'temperature_data_{start_time.replace(":", "_")}.json')
    experiment_metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_metadata["timestamps"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in time_stamps]
    experiment_metadata["measurement_timestamps"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in measurement_times]
    experiment_metadata["temps1"] = temps1
    experiment_metadata["temps1_plotted"] = temps1_plotting
    experiment_metadata["temps2"] = temps2
    experiment_metadata["temps2_plotted"] = temps2_plotting
    experiment_metadata["averages"] = averages
    experiment_metadata["std_devs"] = std_devs

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(experiment_metadata, json_file, indent=4)

    log_msg(f"Data saved to {json_file_path}")

    log_msg("Experiment ended and plot saved.")


def temperature_over_time_ref(conn, user_name: str = "Lachlan"):
    """
    Takes measurements over time and plots absorbance at a given wavelength as a function of time between measurements.

    :param conn: Socket object to facilitate connection to 32-bit client.
    :param user_name: Name of the user running the experiment, obtained when handle_client() is run.
    :return:
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    start_temp = 25.0
    target_temp = 45.0
    step_size = 0.5
    current_temp = start_temp
    pause_time = 5 * 60  # Length of time that the samples are allowed to reach the current set temperature

    # Initialize lists for timestamps and temperatures
    time_stamps = []
    measurement_times = []
    temps1 = []
    temps1_plotting = []
    temps2 = []
    temps2_plotting = []
    data_paths = []

    # volumes_csv = r"C:\Users\Public\UV_VIS_DATA\LA\LCST\Duplicated_Volumes.csv"
    volumes_csv = r"C:\Users\Public\ot-2-plate-reader-python\copolymerisation\Copolymerisation-Volumes.csv"

    log_msg("Select path for data output:")
    out_path = get_output_path()

    # Define new directory for abs measurements and ensure it exists
    abs_spectra_path = os.path.join(out_path, "abs_spectra")
    os.makedirs(abs_spectra_path, exist_ok=True)

    log_msg("Select path for protocol upload:")
    protocol_path = get_file_path()
    protocol_name = protocol_path.split("/")[-1]

    experiment_metadata = {
        "user": user_name,
        "start_time": start_time,
        "output_path": out_path,
        "temperature_steps": [],
    }
    #
    log_msg("Plate background is required to be taken before proceeding.")

    while True:
        user_input = input(">>> Has empty plate been prepared and ready to be inserted? (yes/no): \n>>> ")
        if user_input.lower() == "yes":
            break
        else:
            log_msg("Waiting for plate preparation...")

    log_msg("Collecting plate background from reader.")
    send_message(conn, "PLATE_BACKGROUND", "600nm")

    try:
        msg_type, msg_data = receive_message(conn)
        if msg_type == "PLATE_BACKGROUND":
            log_msg("Plate background data received.")

            # Move file to new directory
            new_path = os.path.join(abs_spectra_path, os.path.basename(msg_data))
            plate_background_path = new_path # to prevent erroring during curve fitting this variable is assigned to the new path
            shutil.move(msg_data, new_path)

            log_msg(f"Plate background saved to {plate_background_path}.")
    except Exception as e:
        log_msg(f"Error receiving plate background: {e}")
        return

    upload_success = False
    while not upload_success:
        try:
            log_msg("Uploading protocol and volumes to OT-2.")
            run_subprocess(protocol_path)
            run_subprocess(volumes_csv)
            log_msg("Uploads complete.")
            upload_success = True
        except Exception as e:
            log_msg(f"Error uploading protocol: {e}")
            user_input = input(">>> Retry upload? (yes/no): \n>>> ")
            if user_input.lower() != "yes":
                log_msg("Uploads cancelled by user.")
                return

    # Make robot prepare samples
    # SSH into OT-2 and run opentrons_execute command
    log_msg(f"SSH'ing into OT-2 and running opentrons_execute command on protocol {protocol_name}.")
    log_msg(f"Please allow OT-2 to finalise protocol before making any changes.")

    protocol_success = False
    while not protocol_success:
        try:
            output = run_ssh_command(protocol_name)

            # Check for the phrase "Protocol Finished" in shell output
            if output:
                log_msg("OT-2 protocol complete.")
                protocol_success = True
            else:
                log_msg("OT-2 has either not finished protocol or an error has occurred.")
                user_input = input(">>> Retry running the protocol? (yes/no)\n>>> ")
                if user_input.lower() == "yes":
                    log_msg("Retrying the OT-2 protocol...")
                    run_subprocess(protocol_path)
                    run_subprocess(volumes_csv)
                    continue  # Retry the run_ssh_command
                elif user_input.lower() == "no":
                    log_msg("Proceeding through the rest of the program.")
                    break  # Exit the loop and proceed manually
                else:
                    log_msg("Invalid input. Please enter 'yes' or 'no'.")
        except Exception as e:
            log_msg(f"An error occurred while running the OT-2 protocol: {e}")
            user_input = input(">>> Retry running the protocol? (yes/no)\n>>> ")
            if user_input.lower() == "yes":
                log_msg("Retrying the OT-2 protocol...")
                run_subprocess(protocol_path)
                run_subprocess(volumes_csv)
                continue  # Retry the run_ssh_command
            elif user_input.lower() == "no":
                log_msg("Proceeding through the rest of the program.")
                break  # Exit the loop and proceed manually
            else:
                log_msg("Invalid input. Please enter 'yes' or 'no'.")

    log_msg("Starting temperature adjustment protocol.")

    def stabilize_and_measure(temp):
        """Stabilizes at the specified temperature and collects data."""
        try:
            log_msg(f"Setting goal temperature to {temp}.")
            send_message(conn, message_type="SET_TEMP", message_data=str(temp))

            check_stable_temp(conn, temp, stabilization_time=45, check_interval=5, range_tolerance=0.2,
                              temps1=temps1, temps2=temps2, time_stamps=time_stamps)

            experiment_metadata["temperature_steps"].append({
                "target_temperature": temp,
                "temps1": temps1[-1],
                "temps2": temps2[-1],
                "timestamp": time_stamps[-1].strftime("%Y-%m-%d %H:%M:%S")
            })

            # Allowing ten minutes for plate to reach current temperature
            log_msg(f"Allowing {pause_time//60} minutes for plate to reach current temperature.")
            time.sleep(pause_time)

            log_msg("Requesting to run measurement protocol.")
            send_message(conn, "RUN_PROTOCOL", "600nm")

            while True:
                msg_type, msg_data = receive_message(conn)
                if msg_type == "CSV_FILE":
                    log_msg("Data received.")

                    # Move file to new directory
                    new_path = os.path.join(abs_spectra_path, os.path.basename(msg_data))
                    shutil.move(msg_data, new_path)

                    # Update data_paths with the new path
                    data_paths.append(new_path)

                    measurement_times.append(datetime.now())
                    temps1_plotting.append(temps1[-1])
                    temps2_plotting.append(temps2[-1])
                    log_msg("Temperature and data path saved.")
                    break
                else:
                    log_msg("Yet to receive data.")
                    time.sleep(5)
        except Exception as e:
            log_msg(e)

    # Step through temperatures up to target_temp
    try:
        while current_temp <= target_temp:
            stabilize_and_measure(current_temp)
            current_temp += step_size

        # Step through temperatures down to start_temp
        current_temp = target_temp - step_size
        while current_temp >= start_temp:
            stabilize_and_measure(current_temp)
            current_temp -= step_size

    except Exception as e:
        log_msg(f"Error during temperature measurement steps: {e}")
        return

    try:
        json_file_path = os.path.join(out_path, f'temperature_data_{start_time.replace(":", "_")}.json')
        experiment_metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_metadata["timestamps"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in time_stamps]
        experiment_metadata["measurement_timestamps"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in measurement_times]
        experiment_metadata["temps1"] = temps1
        experiment_metadata["temps1_plotted"] = temps1_plotting
        experiment_metadata["temps2"] = temps2
        experiment_metadata["temps2_plotted"] = temps2_plotting

        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(experiment_metadata, json_file, indent=4)
    except Exception as e:
        log_msg(e)

    log_msg(f"Data saved to {json_file_path}")
    log_msg("Experiment ended and plots saved.")


def temperature_over_time_sim(conn, user_name: str = "Lachlan"):
    """
    Takes measurements over time and plots absorbance at a given wavelength as a function of time between measurements.

    :param conn: Socket object to facilitate connection to 32-bit client.
    :param user_name: Name of the user running the experiment, obtained when handle_client() is run.
    :return:
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    start_temp = 25.0
    target_temp = 25.5
    step_size = 0.5
    current_temp = start_temp
    pause_time = 0  # Length of time that the samples are allowed to reach the current set temperature

    # Initialize lists for timestamps and temperatures
    time_stamps = []
    measurement_times = []
    temps1 = []
    temps1_plotting = []
    temps2 = []
    temps2_plotting = []
    data_paths = []

    volumes_csv = r"C:\Users\Public\UV_VIS_DATA\LA\LCST\Duplicated_Volumes.csv"

    log_msg("Select path for data output:")
    out_path = get_output_path()

    # Define new directory for abs measurements and ensure it exists
    abs_spectra_path = os.path.join(out_path, "abs_spectra")
    os.makedirs(abs_spectra_path, exist_ok=True)

    log_msg("Select path for protocol upload:")
    protocol_path = get_file_path()
    protocol_name = protocol_path.split("/")[-1]

    experiment_metadata = {
        "user": user_name,
        "start_time": start_time,
        "output_path": out_path,
        "temperature_steps": [],
    }

    log_msg("Plate background is required to be taken before proceeding.")

    while True:
        user_input = input(">>> Has empty plate been prepared and ready to be inserted? (yes/no): \n>>> ")
        if user_input.lower() == "yes":
            break
        else:
            log_msg("Waiting for plate preparation...")

    log_msg("Collecting plate background from reader.")
    send_message(conn, "PLATE_BACKGROUND", "600nm")

    try:
        msg_type, msg_data = receive_message(conn)
        if msg_type == "PLATE_BACKGROUND":
            log_msg("Plate background data received.")

            # Move file to new directory
            new_path = os.path.join(abs_spectra_path, os.path.basename(msg_data))
            plate_background_path = new_path # to prevent erroring during curve fitting this variable is assigned to the new path
            shutil.move(msg_data, new_path)

            log_msg(f"Plate background saved to {plate_background_path}.")
    except Exception as e:
        log_msg(f"Error receiving plate background: {e}")
        return

    upload_success = False
    while not upload_success:
        try:
            log_msg("Uploading protocol and volumes to OT-2.")
            run_subprocess(protocol_path)
            run_subprocess(volumes_csv)
            log_msg("Uploads complete.")
            upload_success = True
        except Exception as e:
            log_msg(f"Error uploading protocol: {e}")
            user_input = input(">>> Retry upload? (yes/no): \n>>> ")
            if user_input.lower() != "yes":
                log_msg("Uploads cancelled by user.")
                return

    # Make robot prepare samples
    # SSH into OT-2 and run opentrons_execute command
    log_msg(f"SSH'ing into OT-2 and running opentrons_execute command on protocol {protocol_name}.")
    log_msg(f"Please allow OT-2 to finalise protocol before making any changes.")

    protocol_success = False
    while not protocol_success:
        try:
            output = run_simulation(protocol_name)

            # Check for the phrase "Protocol Finished" in shell output
            if output:
                log_msg("OT-2 protocol complete.")
                protocol_success = True
            else:
                log_msg("OT-2 has either not finished protocol or an error has occurred.")
                user_input = input(">>> Retry running the protocol? (yes/no)\n>>> ")
                if user_input.lower() == "yes":
                    log_msg("Retrying the OT-2 protocol...")
                    run_subprocess(protocol_path)
                    run_subprocess(volumes_csv)
                    continue  # Retry the run_ssh_command
                elif user_input.lower() == "no":
                    log_msg("Proceeding through the rest of the program.")
                    break  # Exit the loop and proceed manually
                else:
                    log_msg("Invalid input. Please enter 'yes' or 'no'.")
        except Exception as e:
            log_msg(f"An error occurred while running the OT-2 protocol: {e}")
            user_input = input(">>> Retry running the protocol? (yes/no)\n>>> ")
            if user_input.lower() == "yes":
                log_msg("Retrying the OT-2 protocol...")
                run_subprocess(protocol_path)
                run_subprocess(volumes_csv)
                continue  # Retry the run_ssh_command
            elif user_input.lower() == "no":
                log_msg("Proceeding through the rest of the program.")
                break  # Exit the loop and proceed manually
            else:
                log_msg("Invalid input. Please enter 'yes' or 'no'.")

    log_msg("Starting temperature adjustment protocol.")

    def stabilize_and_measure(temp):
        """Stabilizes at the specified temperature and collects data."""
        try:
            log_msg(f"Setting goal temperature to {temp}.")
            send_message(conn, message_type="SET_TEMP", message_data=str(temp))

            check_stable_temp(conn, temp, stabilization_time=45, check_interval=5, range_tolerance=0.2,
                              temps1=temps1, temps2=temps2, time_stamps=time_stamps)

            experiment_metadata["temperature_steps"].append({
                "target_temperature": temp,
                "temps1": temps1[-1],
                "temps2": temps2[-1],
                "timestamp": time_stamps[-1].strftime("%Y-%m-%d %H:%M:%S")
            })

            # Allowing ten minutes for plate to reach current temperature
            log_msg(f"Allowing {pause_time//60} minutes for plate to reach current temperature.")
            time.sleep(pause_time)

            log_msg("Requesting to run measurement protocol.")
            send_message(conn, "RUN_PROTOCOL", "600nm")

            while True:
                msg_type, msg_data = receive_message(conn)
                if msg_type == "CSV_FILE":
                    log_msg("Data received.")

                    # Move file to new directory
                    new_path = os.path.join(abs_spectra_path, os.path.basename(msg_data))
                    shutil.move(msg_data, new_path)

                    # Update data_paths with the new path
                    data_paths.append(new_path)

                    measurement_times.append(datetime.now())
                    temps1_plotting.append(temps1[-1])
                    temps2_plotting.append(temps2[-1])
                    log_msg("Temperature and data path saved.")
                    break
                else:
                    log_msg("Yet to receive data.")
                    time.sleep(5)
        except Exception as e:
            log_msg(e)

    # Step through temperatures up to target_temp
    try:
        while current_temp <= target_temp:
            stabilize_and_measure(current_temp)
            current_temp += step_size

        # Step through temperatures down to start_temp
        # current_temp = target_temp - step_size
        # while current_temp >= start_temp:
        #     stabilize_and_measure(current_temp)
        #     current_temp -= step_size

    except Exception as e:
        log_msg(f"Error during temperature measurement steps: {e}")
        return

    try:
        json_file_path = os.path.join(out_path, f'temperature_data_{start_time.replace(":", "_")}.json')
        experiment_metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_metadata["timestamps"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in time_stamps]
        experiment_metadata["measurement_timestamps"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in measurement_times]
        experiment_metadata["temps1"] = temps1
        experiment_metadata["temps1_plotted"] = temps1_plotting
        experiment_metadata["temps2"] = temps2
        experiment_metadata["temps2_plotted"] = temps2_plotting

        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(experiment_metadata, json_file, indent=4)
    except Exception as e:
        log_msg(e)

    log_msg(f"Data saved to {json_file_path}")
    log_msg("Experiment ended and plots saved.")


def dummy(conn, user_name: str = "Lachlan"):
    """
    Test function/protocol to be used with closed-loop system.
    Runs dummy protocol on OT-2, takes measurement on plate reader, saves .json data to specified directory.

    :param conn: Socket object to facilitate connection to 32-bit client.
    :param user_name: Name of the user running the experiment, obtained when handle_client() is run.
    :return:
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_msg("Select path for data output:")
    out_path = get_output_path()

    # Define the new directory for raw absorbance spectra and ensure it exists
    abs_spectra_path = os.path.join(out_path, "abs_spectra")
    os.makedirs(abs_spectra_path, exist_ok=True)

    log_msg("Select path for protocol upload:")
    protocol_path = get_file_path()
    protocol_name = protocol_path.split("/")[-1]

    volumes_csv = r"C:\Users\Public\64 Bit\DoE + Monomers Experiment\Duplicated_Volumes.csv"

    experiment_metadata = {
        "user": user_name,
        "start_time": start_time,
        "output_path": out_path,
        "temperature_steps": [],
    }

    log_msg("Plate background is required to be taken before proceeding.")

    while True:
        user_input = input(">>> Has empty plate been prepared and ready to be inserted? (yes/no): \n>>> ")
        if user_input.lower() == "yes":
            break
        else:
            log_msg("Waiting for plate preparation...")

    log_msg("Collecting plate background from reader.")
    send_message(conn, "PLATE_BACKGROUND", "600nm")

    data_paths = []

    try:
        msg_type, msg_data = receive_message(conn)
        if msg_type == "PLATE_BACKGROUND":
            log_msg("Plate background data received.")
            plate_background_path = msg_data

            # Move the file to abs_spectra directory
            new_path = os.path.join(abs_spectra_path, os.path.basename(msg_data))
            shutil.move(msg_data, new_path)

            # Update data_paths with the new path
            data_paths.append(new_path)

            log_msg(f"Plate background saved to {new_path}.")
    except Exception as e:
        log_msg(f"Error receiving plate background: {e}")
        return

    for path in data_paths:
        data = load_data_new(path)
        print(data.head(20))

    upload_success = False
    while not upload_success:
        try:
            log_msg("Uploading dummy protocol to OT-2.")
            run_subprocess(protocol_path)
            log_msg("Uploads complete.")
            upload_success = True
        except Exception as e:
            log_msg(f"Error uploading protocol: {e}")
            user_input = input(">>> Retry upload? (yes/no): \n>>> ")
            if user_input.lower() != "yes":
                log_msg("Uploads cancelled by user.")
                return

    # Make robot prepare samples
    # SSH into OT-2 and run opentrons_execute command
    log_msg(f"SSH'ing into OT-2 and running opentrons_execute command on protocol {protocol_name}.")
    log_msg(f"Please allow OT-2 to finalise protocol before making any changes.")

    protocol_success = False
    while not protocol_success:
        try:
            output = run_ssh_command(protocol_name)

            # Check for the phrase "Protocol Finished" in shell output
            if output:
                log_msg("OT-2 protocol complete.")
                protocol_success = True
            else:
                log_msg("OT-2 has either not finished protocol or an error has occurred.")
                user_input = input(">>> Retry running the protocol? (yes/no)\n>>> ")
                if user_input.lower() == "yes":
                    log_msg("Retrying the OT-2 protocol...")
                    run_subprocess(protocol_path)
                    run_subprocess(volumes_csv)
                    continue  # Retry the run_ssh_command
                elif user_input.lower() == "no":
                    log_msg("Proceeding through the rest of the program.")
                    break  # Exit the loop and proceed manually
                else:
                    log_msg("Invalid input. Please enter 'yes' or 'no'.")
        except Exception as e:
            log_msg(f"An error occurred while running the OT-2 protocol: {e}")
            user_input = input(">>> Retry running the protocol? (yes/no)\n>>> ")
            if user_input.lower() == "yes":
                log_msg("Retrying the OT-2 protocol...")
                run_subprocess(protocol_path)
                run_subprocess(volumes_csv)
                continue  # Retry the run_ssh_command
            elif user_input.lower() == "no":
                log_msg("Proceeding through the rest of the program.")
                break  # Exit the loop and proceed manually
            else:
                log_msg("Invalid input. Please enter 'yes' or 'no'.")

    try:
        json_file_path = os.path.join(out_path, f'dummy_data_{start_time.replace(":", "_")}.json')
        experiment_metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(experiment_metadata, json_file, indent=4)
    except Exception as e:
        log_msg(e)

    log_msg(f"Data saved to {json_file_path}")
    log_msg("Dummy test complete.")


@timeit
def handle_client(conn):
    """Handle multiple messages from the client.
    :param conn: socket.socket object, connection to client socket.
    """
    try:
        user_name = input(">>> Enter your name: \n>>> ")

        while True:
            choice = input(">>> Enter workflow number: "
                           "\n1. Conc Model "
                           "\n2. Simulate Conc Model "
                           "\n3. Measure Temp Over Time "
                           "\n4. LCST Protocol"
                           "\n5. Simulate LCST Protocol  "
                           "\n6. Shutdown "
                           "\n>>> "
                           )

            if choice == "1":
                conc_model(conn, user_name)

            if choice == "2":
                conc_model_for_testing(conn, user_name)

            if choice == "3":
                measurements_over_time(conn, user_name)

            if choice == "4":
                temperature_over_time_ref(conn, user_name)

            if choice == "5":
                temperature_over_time_sim(conn, user_name)

            if choice == "6":
                send_message(conn, "SHUTDOWN")
                break

            else:
                log_msg("Invalid entry, please enter a number only.")

    except Exception as e:
        log_msg(f"Error handling client: {e}")


def server_main():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 65432))  # Start the server
            s.listen()
            log_msg("Waiting for connection from 32-bit script...")

            # while True:
            conn, addr = s.accept()
            with conn:
                log_msg(f"Connected by {addr}")

                handle_client(conn)
                log_msg("Shutting down.")
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()

    except Exception as e:
        log_msg(f"An error occurred: {e}")


if __name__ == "__main__":
    server_main()