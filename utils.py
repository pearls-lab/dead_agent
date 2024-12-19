from collections import Counter

def flatten_and_count_unique(array_2d):
    """
    Flattens a 2D array and counts unique elements.

    Parameters:
        array_2d (list of list of int/float): The 2D array to process.

    Returns:
        dict: A dictionary with unique elements as keys and their counts as values.
    """
    # Flatten the 2D array and count elements in one pass
    element_counts = Counter(element for row in array_2d for element in row)
    return dict(element_counts)


def exponential_moving_average(data, window = 10):
    # ChatGPT generated
    """
    Calculate the Exponential Moving Average (EMA) of an array of numbers.

    Parameters:
        data (list or array-like): The input array of numbers.
        window (int): The smoothing window (period). Determines how much weight is given to recent values.

    Returns:
        list: The EMA values.
    """
    if not data or window <= 0:
        raise ValueError("Data must be a non-empty array and window must be a positive integer.")

    ema = []
    alpha = 2 / (window + 1)  # Smoothing factor

    # Initialize the first EMA value with the first data point
    ema.append(data[0])

    # Calculate the EMA for the rest of the data
    for i in range(1, len(data)):
        ema_value = (data[i] * alpha) + (ema[-1] * (1 - alpha))
        ema.append(ema_value)

    return np.array(ema)

import matplotlib.pyplot as plt
import numpy as np

def plot_array_and_save(array, output_path, title = "title: placeholder", x_label = "x_label: placeholder", y_label = "y_label: placeholder", y_max = 10):
    """ From ChatGPT
    Plots a 1D array and saves it as an image.

    Parameters:
    - array (ndarray): A 1D numpy array to plot.
    - output_path (str): Path where the image will be saved.
    """
    if not isinstance(array, np.ndarray):
        print("Wrapping array in numpy array...")
        array = np.array(array)
    if array.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")

    plt.figure()
    plt.plot(array, marker='o')  # Line plot with markers
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()