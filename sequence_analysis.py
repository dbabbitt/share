#!/usr/bin/env python
# Utility Functions to run Jupyter notebooks.
# Dave Babbitt <dave.babbitt@gmail.com>
# Author: Dave Babbitt, Machine Learning Engineer
# coding: utf-8

# Soli Deo gloria

"""
Run this in a Git Bash terminal if you push anything:
    cd ~/OneDrive/Documents/GitHub/notebooks/sh
    ./update_share_submodules.sh
"""

from base_config import BaseConfig
from os import (
    makedirs as makedirs, path as osp
)
from pandas import (
    Series
)
from re import (
    sub
)
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Check for presence of 'get_ipython' function (exists in Jupyter)
try:
    get_ipython()
    from IPython.display import display
except NameError:

    def display(message):
        """
        Display a message. If IPython's display is unavailable, fall back to
        printing.

        Parameters:
            message (str): The message to display.
        """
        print(message)


class SequenceAnalysis(BaseConfig):
    def __init__(
        self, data_folder_path=None, saves_folder_path=None
    ):

        # Assume the data folder exists
        if data_folder_path is None:
            self.data_folder = osp.join(os.pardir, 'data')
        else:
            self.data_folder = data_folder_path

        # Assume the saves folder exists
        if saves_folder_path is None:
            self.saves_folder = osp.join(os.pardir, 'saves')
        else:
            self.saves_folder = saves_folder_path

        super().__init__()  # Inherit shared attributes
        self.saves_png_folder = osp.join(self.saves_folder, 'png')
        makedirs(name=self.saves_png_folder, exist_ok=True)

        try:
            from pysan.elements import get_alphabet
            self.get_alphabet = get_alphabet
        except Exception:
            self.get_alphabet = lambda sequence: set(sequence)

    # -------------------
    # Numeric Functions
    # -------------------

    # -------------------
    # String Functions
    # -------------------

    # -------------------
    # List Functions
    # -------------------

    def get_jitter_list(self, ages_list):
        """
        Generate a list of jitter values for plotting age data points with a
        scattered plot.

        Parameters:
            ages_list (list): A list of ages for which jitter values are
            generated.

        Returns:
            list of float
                A list of jitter values corresponding to the input ages.
        """

        # Initialize an empty list to store jitter values
        jitter_list = []

        # Iterate over the list of age groups
        for splits_list in self.split_list_by_gap(ages_list):

            # If multiple ages in group, calculate jitter values for each age
            if len(splits_list) > 1:

                # Generate jitter values using cut and extend
                jitter_list.extend(
                    pd.cut(
                        np.array(
                            [min(splits_list) - 0.99, max(splits_list) + 0.99]
                        ),
                        len(splits_list) - 1,
                        retbins=True
                    )[1]
                )

            # If only one age in a group, add that age as the jitter value
            else:
                jitter_list.extend(splits_list)

        # Return the list of jitter values
        return jitter_list

    @staticmethod
    def split_list_by_gap(ages_list, value_difference=1, verbose=False):
        """
        Divide a list of ages into sublists based on gaps in the age sequence.

        Parameters:
            ages_list (list of int or float): A list of ages to be split into
            sublists.

        Returns:
            list of lists of int or float
                A list of sublists, each containing consecutive ages.
        """

        # List to store sublists of consecutive ages
        splits_list = []

        # Temporary list to store the current consecutive ages
        current_list = []

        # Initialize with a value lower than the first age
        previous_age = ages_list[0] - value_difference

        # Iterate over the list of ages
        for age in ages_list:

            # Check if there is a gap between current age and previous age
            if age - previous_age > value_difference:

                # Append the current_list to splits_list
                splits_list.append(current_list)

                # Reset the current_list
                current_list = []

            # Add the current age to the current_list
            current_list.append(age)

            # Update the previous_age
            previous_age = age

        # Append the last current_list to splits_list
        splits_list.append(current_list)

        # Return the list of sublists of ages
        return splits_list

    @staticmethod
    def count_ngrams(actions_list, highlighted_ngrams):
        """
        Count the occurrences of a sequence of elements (n-grams) in a list.

        This static method traverses through the `actions_list` and counts
        how many times the sequence `highlighted_ngrams` appears. It is
        useful for analyzing the frequency of specific patterns within a
        list of actions or tokens.

        Parameters:
            actions_list (list):
                A list of elements in which to count the occurrences of the
                n-gram.
            highlighted_ngrams (list):
                The sequence of elements (n-gram) to count occurrences of.

        Returns:
            int:
                The count of how many times `highlighted_ngrams` occurs in
                `actions_list`.

        Examples:
            actions = ['jump', 'run', 'jump', 'run', 'jump']
            ngrams = ['jump', 'run']
            nu.count_ngrams(actions, ngrams)  # 2
        """

        # Initialize the count of n-gram occurrences
        count = 0

        # Calculate the range for the loop to avoid IndexErrors
        range_limit = len(actions_list) - len(highlighted_ngrams) + 1

        # Loop over the actions_list at that window size to find occurrences
        for i in range(range_limit):

            # Check if the current slice matches the highlighted_ngrams
            if actions_list[
                i:i + len(highlighted_ngrams)
            ] == highlighted_ngrams:

                # Increment the count if a match is found
                count += 1

        return count

    @staticmethod
    def get_sequences_by_count(tg_dict, count=4):
        """
        Get sequences from the input dictionary based on a specific sequence
        length.

        Parameters:
            tg_dict (dict): Dictionary containing sequences.
            count (int, optional): Desired length of sequences to filter.
            Default is 4.

        Returns:
            list: List of sequences with the specified length.

        Raises:
            AssertionError: If no sequences of the specified length are found
            in the dictionary.
        """

        # Convert the lengths in the dictionary values into value counts
        value_counts = Series(
            [len(actions_list) for actions_list in tg_dict.values()]
        ).value_counts()

        # Get desired sequence length of exactly count sequences
        value_counts_list = value_counts[value_counts == count].index.tolist()
        s = f"You don't have exactly {count} sequences of the same length in"
        s += " the dictionary"
        assert value_counts_list, s
        sequences = [
            actions_list
            for actions_list in tg_dict.values()
            if (len(actions_list) == value_counts_list[0])
        ]

        return sequences

    @staticmethod
    def split_list_by_exclusion(
        splitting_indices_list, excluded_indices_list=[]
    ):
        """
        Split a list of row indices into a list of lists, where each inner list
        contains a contiguous sequence of indices that are not in the excluded
        indices list.

        Parameters:
            splitting_indices_list: A list of row indices to split.
            excluded_indices_list:
                A list of row indices that should be considered excluded.
                Empty by default.

        Returns:
            A list of lists, where each inner list contains a contiguous
            sequence of indices that are not in the excluded indices.
        """

        # Initialize the output list
        split_list = []

        # Initialize the current list
        current_list = []

        # Iterate over the splitting indices list
        for current_idx in range(
            int(min(splitting_indices_list)),
            int(max(splitting_indices_list)) + 1
        ):

            # Check that the current index is in the splitting indices list
            if (
                current_idx in splitting_indices_list
            ):

                # Check that the current index is not in the excluded list
                if (
                    current_idx not in excluded_indices_list
                ):

                    # If so, add it to the current list
                    current_list.append(current_idx)

            # Otherwise
            else:

                # If the current list is not empty
                if current_list:

                    # Add it to the split list
                    split_list.append(current_list)

                # And start a new current list
                current_list = []

        # If the current list is not empty
        if current_list:

            # Add it to the split list
            split_list.append(current_list)

        # Return the split list
        return split_list

    @staticmethod
    def convert_strings_to_integers(sequence, alphabet_list=None):
        """
        Convert a sequence of strings into a sequence of integers and a
        mapping dictionary.

        This method converts each string in the input sequence to an
        integer based on its position in an alphabet list. If the alphabet
        list is not provided, it is generated from the unique elements of
        the sequence. The method returns a new sequence where each string
        is replaced by its corresponding integer, and a dictionary mapping
        strings to integers.

        Parameters:
            sequence (iterable):
                A sequence of strings to be converted.
            alphabet_list (list, optional):
                A list of the unique elements of sequence, passed in to
                stabilize the order. If None (default), the alphabet is
                derived from the `sequence`.

        Returns:
            tuple:
                A tuple containing two elements:
                - new_sequence (numpy.ndarray): An array of integers
                  representing the converted sequence.
                - string_to_integer_map (dict): A dictionary mapping the
                  original strings to their corresponding integer codes.

        Note:
            Strings not present in the alphabet are mapped to -1 in the
            dictionary.

        Examples:
            sequence = ['apple', 'banana', 'apple', 'cherry']
            new_sequence, mapping = nu.convert_strings_to_integers(sequence)
            display(new_sequence)  # array([0, 1, 0, 2])
            display(mapping)  # {'apple': 0, 'banana': 1, 'cherry': 2}
        """
        assert (
            hasattr(sequence, '__iter__') and not isinstance(sequence, str)
        ), f"sequence needs to be an iterable, not a {type(sequence)}."

        # Create an alphabet from the sequence if not provided
        if alphabet_list is None:
            alphabet_list = sorted(set(sequence))

        # Initialize the map with an enumeration of the alphabet
        string_to_integer_map = {
            string: index for index, string in enumerate(alphabet_list)
        }

        # Convert seq of strs to seq of ints, assigning -1 for unknown strs
        new_sequence = np.array(
            [string_to_integer_map.get(string, -1) for string in sequence]
        )

        return (new_sequence, string_to_integer_map)

    def get_ndistinct_subsequences(self, sequence, verbose=False):
        """
        Calculate the number of distinct subsequences in a given sequence.

        This function implements an algorithm to count the number of
        distinct subsequences (non-contiguous substrings) that can be
        formed from a given input sequence (`sequence`). The current
        implementation assumes the sequence consists of characters.

        Parameters:
            sequence (str or list):
                The input sequence to analyze. If not a string, it will be
                converted to a string.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            int
                The number of distinct subsequences in the input sequence.

        Note:
            This function replaces the functionality of
            `get_ndistinct_subsequences` from the `pysan` package.
        """

        # Handle non-string inputs and convert them to strings (if possible)
        if (
            not isinstance(sequence, str)
            or not all([len(str(e)) == 1 for e in sequence])
        ):

            # Convert non-string sequences to strings
            ns, string_to_integer_map = self.convert_strings_to_integers(
                sequence, alphabet_list=None
            )
            sequence = []
            for e in ns:
                sequence.append(e)

        if verbose:
            print(f'sequence: {sequence}')

        # Create an array to store index of last occurrences
        last = [-1 for i in range(256 + 1)]

        # Get length of input string
        sequence_length = len(sequence)

        # Create an array to store counts of distinct subsequences
        dp = [-2] * (sequence_length + 1)

        # Create the base case: the empty substring has one subsequence
        dp[0] = 1

        # Traverse through all lengths from 1 to sequence_length
        for i in range(1, sequence_length + 1):

            # Set the number of subsequences for the current substring length
            dp[i] = 2 * dp[i - 1]

            # Has the current character appeared before?
            if last[sequence[i - 1]] != -1:

                # Remove all subsequences ending with previous occurrence
                dp[i] = dp[i] - dp[last[sequence[i - 1]]]

            # Update the last occurrence of the current character
            last[sequence[i - 1]] = i - 1

        # Return the count of distinct subsequences for the entire sequence
        return dp[sequence_length]

    def get_turbulence(self, sequence, verbose=False):
        """
        Compute turbulence for a given sequence.

        This function computes the turbulence of a sequence based on the
        definition provided by Elzinga & Liefbroer (2007). Turbulence is a
        measure of variability in sequences, often used in sequence
        analysis.

        Parameters:
            sequence (list):
                The input sequence for which turbulence is to be computed.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            float
                The computed turbulence value for the given sequence.

        Note:
            This function replaces `get_turbulence` from the `pysan` package.

        References:
            - Elzinga, C. H., & Liefbroer, A. C. (2007). De-standardization of
              Family-Life Trajectories of Young Adults: A Cross-National
              Comparison Using Sequence Analysis.
              https://www.researchgate.net/publication/225402919
            - TraMineR sequence analysis library.
              http://traminer.unige.ch/doc/seqST.html
        """

        # Import necessary modules
        import statistics
        from pysan.statistics import get_spells
        import math

        # Compute the number of distinct subsequences in the sequence (phi)
        phi = self.get_ndistinct_subsequences(sequence, verbose=verbose)
        if verbose:
            print('phi:', phi)

        # Compute the durations of each state in the sequence
        state_durations = [value for key, value in get_spells(sequence)]
        if verbose:
            print('durations:', state_durations)
            print('mean duration:', statistics.mean(state_durations))

        # Compute the variance of state durations
        try:
            variance_of_state_durations = statistics.variance(state_durations)

        # Has variance computation failed (eg, due to insufficient data)?
        except Exception:

            # Set variance to 0
            variance_of_state_durations = 0.0

        if verbose:
            print('variance:', variance_of_state_durations)

        # Compute the mean of state durations (tbar)
        tbar = statistics.mean(state_durations)

        # Compute the maximum state duration variance (smax)
        maximum_state_duration_variance = (
            (len(sequence) - 1) * (1 - tbar) ** 2
        )
        if verbose:
            print('smax:', maximum_state_duration_variance)

        # Compute the top right part of the formula
        top_right = maximum_state_duration_variance + 1

        # Compute the bottom right part of the formula
        bot_right = variance_of_state_durations + 1

        # Compute the turbulence value using the provided formula
        turbulence = math.log2(phi * (top_right / bot_right))
        if verbose:
            print('turbulence:', turbulence)

        # Return the computed turbulence value
        return turbulence

    @staticmethod
    def replace_consecutive_elements(actions_list, element):
        """
        Replace consecutive elements in a list with a count of how many
        there are in a row.

        This function iterates through a list (`actions_list`) and
        replaces consecutive occurrences of a specified element
        (`element`) with a string representation indicating the element
        and its count. For example, if the list contains ['a', 'a', 'b',
        'a', 'a'], the function would return ['a x2', 'b', 'a x2'].

        Parameters:
            actions_list (list):
                A list of elements.
            element (any):
                The element to replace consecutive occurrences of.

        Returns:
            list
                A new list with consecutive elements replaced by count strings
                (e.g., 'element xN').
        """

        # Initialize an empty result list to store the modified elements
        result = []

        # Initialize a count to keep track of consecutive occurrences
        count = 0

        # Loop through each element in the input list
        for i in range(len(actions_list)):

            # If the current element is the target element, increment count
            if actions_list[i] == element:

                # Increment the count if it's the target element
                count += 1

            # If a new element is encountered (not the target element)
            else:

                # Check if there were consecutive elements before
                if count > 0:

                    # Append representation of previous element and its count
                    result.append(f'{element} x{str(count)}')

                # Add the current element to the result list
                result.append(actions_list[i])

                # Reset the count for the new element
                count = 0

        # Handle the last element if there was a sequence at the end
        if count > 0:

            # Append the last counted element with its count to the result
            result.append(f'{element} x{str(count)}')

        # Return the modified list with counts replacing consecutive elements
        return result

    @staticmethod
    def count_swaps_to_perfect_order(
        ideal_list, compared_list, verbose=False
    ):
        """
        Count the number of swaps required to make a compared list
        identical to the ideal list without penalizing lists with repeated
        elements.

        Parameters:
            ideal_list (list): The list representing the ideal order.
            compared_list (list): The list to be compared and modified.

        Returns:
            int: The number of swaps required.

        Raises:
            ValueError: If the lengths of 'ideal_list' and 'compared_list'
            are not equal.
        """

        # Check if lengths of lists are equal
        n = len(ideal_list)
        if n != len(compared_list):
            raise ValueError('Lists must be of equal length')
        swaps = 0

        # Initialize a dictionary for the indices of ideal_list
        ideal_indices = {element: i for i, element in enumerate(ideal_list)}

        # Iterate through the compared list
        for i in range(n):

            # If the element is not in its correct position
            if compared_list[i] != ideal_list[i]:

                # Find the correct position of the element in ideal_list
                correct_index = ideal_indices[compared_list[i]]

                # Swap the elements
                compared_list[i], compared_list[correct_index] = (
                    compared_list[correct_index], compared_list[i]
                )

                # Add that swap to the count
                swaps += 1

        return swaps

    # -------------------
    # File Functions
    # -------------------

    # -------------------
    # Path Functions
    # -------------------

    # -------------------
    # Storage Functions
    # -------------------

    # -------------------
    # Module Functions
    # -------------------

    # -------------------
    # URL and Soup Functions
    # -------------------

    # -------------------
    # Pandas Functions
    # -------------------

    # -------------------
    # 3D Point Functions
    # -------------------

    # -------------------
    # Sub-sampling Functions
    # -------------------

    # -------------------
    # Plotting Functions
    # -------------------

    @staticmethod
    def get_color_cycled_list(alphabet_list, color_dict, verbose=False):
        """
        Get a list of colors cycled from the given color dictionary and the
        default color cycle.

        This method matches each alphabet in the input list with a color from
        the input color dictionary. If a color is not specified for an
        alphabet, it assigns the next color from the matplotlib color cycle.

        Parameters:
            alphabet_list (list of str):
                A list of keys for which colors are to be matched.
            color_dict (dict):
                A dictionary mapping elements from the alphabet to desired
                colors.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            list of str
                A list of colors, where the length matches the alphabet list.
                Colors are assigned based on the color dictionary and the
                Matplotlib color cycle for missing entries.
        """

        # Print the input alphabet list and color dictionary if verbose
        if verbose:
            print(f'alphabet_list = {alphabet_list}')
            print(f'color_dict = {color_dict}')

        # Import the cycle iterator from itertools
        from itertools import cycle

        # Get the color cycle from matplotlib's rcParams
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = cycle(prop_cycle.by_key()['color'])

        # Initialize an empty list to store the resulting colors
        colors_list = []

        # Iterate over each key in the alphabet list
        for key in alphabet_list:

            # Assign color from dict, otherwise use next color from the cycle
            value = color_dict.get(key, next(colors))

            # Append the color to the colors list
            colors_list.append(value)

        # Print the resulting colors list if verbose mode is on
        if verbose:
            print(f'colors_list = {colors_list}')

        # Return the final list of colors
        return colors_list

    def plot_sequence(
        self, sequence, highlighted_ngrams=[], color_dict=None, suptitle=None,
        first_element='SESSION_START', last_element='SESSION_END',
        alphabet_list=None, verbose=False
    ):
        """
        Create a standard sequence plot where each element corresponds to a
        position on the y-axis. The optional highlighted_ngrams parameter can
        be one or more n-grams to be outlined in a red box.

        Parameters:
            sequence:
                A list of strings or integers representing the sequence to
                plot.
            highlighted_ngrams:
                A list of n-grams to be outlined in a red box.
            color_dict:
                An optional dictionary whose keys are the alphabet list and
                whose values are a single color format string to allow
                consistent visualization between calls.
            suptitle:
                An optional title for the plot.
            first_element:
                The element in alphabet_list that will be forced to the
                beginning if already in the list. Defaults to SESSION_START.
            last_element:
                The element in alphabet_list that will be forced to the end if
                already in the list. Defaults to SESSION_END.
            alphabet_list:
                A list of strings or integers representing the set of elements
                in sequence.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            A matplotlib figure and axes objects.
        """

        # Convert the sequence to a NumPy array
        import numpy as np
        np_sequence = np.array(sequence)

        # Get the unique characters in the sequence
        if highlighted_ngrams:
            sample_ngram = highlighted_ngrams[0]
            highlighted_type = type(sample_ngram)
        else:
            sample_ngram = None
            highlighted_type = None
        if alphabet_list is None:
            if highlighted_type is list:
                alphabet_list = sorted(self.get_alphabet(sequence + [
                    el
                    for sublist in highlighted_ngrams
                    for el in sublist
                ]))
            else:
                alphabet_list = sorted(
                    self.get_alphabet(sequence + highlighted_ngrams)
                )
        if last_element in alphabet_list:
            alphabet_list.remove(last_element)
            alphabet_list.append(last_element)
        if first_element in alphabet_list:
            alphabet_list.insert(
                0, alphabet_list.pop(alphabet_list.index(first_element))
            )

        # Set up the color dictionary with alphabet_list keys
        color_dict = self.update_color_dict(alphabet_list, color_dict)

        # Get the length of the alphabet
        alphabet_len = len(alphabet_list)

        # Convert the sequence to integers
        INT_SEQUENCE, _ = self.convert_strings_to_integers(
            np_sequence, alphabet_list=alphabet_list
        )

        # Create a string-to-integer map
        if highlighted_type is list:
            _, string_to_integer_map = self.convert_strings_to_integers(
                sequence + [
                    el for sublist in highlighted_ngrams for el in sublist
                ], alphabet_list=alphabet_list
            )
        else:
            _, string_to_integer_map = self.convert_strings_to_integers(
                sequence + highlighted_ngrams, alphabet_list=alphabet_list
            )

        # If the sequence is not already in integer format, convert it
        if verbose:
            print(f'np_sequence.dtype.str = {np_sequence.dtype.str}')
        # if np_sequence.dtype.str not in ['<U21', '<U11']: int_sequence = np_sequence

        # Create a figure and axes
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(
            figsize=[len(sequence) * 0.3, alphabet_len * 0.3]
        )

        # Force the xticks to land on integers only
        xtick_locations = range(len(sequence))
        xtick_labels = [n+1 for n in xtick_locations]
        ax.set_xticks(ticks=xtick_locations)
        ax.set_xticklabels(xtick_labels, minor=False)

        # Extend the edges of the plot
        ax.set_xlim([-0.5, len(sequence) - 0.5])

        # Iterate over the alphabet and plot the points for each character
        if False:  # verbose
            color_cycle = plt.rcParams['axes.prop_cycle']
            print('\nPrinting the colors in the rcParams color cycle:')
            for color in color_cycle:
                print(color)
            print()
        for i, value in enumerate(alphabet_list):

            # Get the positions of the current character in the sequence
            points = np.where(np_sequence == value, i, np.nan)

            # Plot the points on the axes
            ax.scatter(
                x=range(len(np_sequence)), y=points, marker='s', label=value,
                s=35, color=color_dict[value]
            )

        # Set the yticks label values
        plt.yticks(range(alphabet_len), alphabet_list)

        # Match the label colors with the color cycle and color dictionary
        colors_list = self.get_color_cycled_list(
            alphabet_list, color_dict, verbose=verbose
        )

        # Set the yticks label color
        for label, color in zip(plt.gca().get_yticklabels(), colors_list):
            label.set_color(color)

        # Set the y limits on the axes
        ax.set_ylim(-1, alphabet_len)

        # Highlight any of the n-grams given
        if highlighted_ngrams != []:
            if verbose:
                print(f'highlighted_ngrams = {highlighted_ngrams}')

            def highlight_ngram(int_ngram):
                if verbose:
                    print(f'int_ngram in highlight_ngram: {int_ngram}')

                # Get the length of the n-gram
                n = len(int_ngram)

                # Find all matches of the n-gram in the sequence
                match_positions = []
                if verbose:
                    print(f'INT_SEQUENCE in highlight_ngram: {INT_SEQUENCE}')
                for x in range(len(INT_SEQUENCE) - n + 1):
                    this_ngram = list(INT_SEQUENCE[x:x + n])
                    if str(this_ngram) == str(int_ngram):
                        match_positions.append(x)

                # Draw a red box around each match
                if verbose:
                    print(
                        f'int_ngram={int_ngram},'  # noqa E231
                        f' min(int_ngram)={min(int_ngram)},'  # noqa E231
                        f' max(int_ngram)={max(int_ngram)},'  # noqa E231
                        f' match_positions={match_positions}'
                    )
                for position in match_positions:
                    bot = min(int_ngram) - 0.25
                    top = max(int_ngram) + 0.25
                    left = position - 0.25
                    right = left + n - 0.5
                    if verbose:
                        print(
                            f'bot={bot}, top={top},'  # noqa E231
                            f' left={left},'  # noqa E231
                            f' right={right}'
                        )

                    line_width = 1
                    plt.plot(
                        [left, right], [bot, bot], color='grey',
                        linewidth=line_width, alpha=0.5
                    )
                    plt.plot(
                        [left, right], [top, top], color='grey',
                        linewidth=line_width, alpha=0.5
                    )
                    plt.plot(
                        [left, left], [bot, top], color='grey',
                        linewidth=line_width, alpha=0.5
                    )
                    plt.plot(
                        [right, right], [bot, top], color='grey',
                        linewidth=line_width, alpha=0.5
                    )

            # check if only one n-gram has been supplied
            if highlighted_type is str:
                highlight_ngram(
                    [string_to_integer_map[x] for x in highlighted_ngrams]
                )
            elif highlighted_type is int:
                highlight_ngram(highlighted_ngrams)

            # multiple n-gram's found
            elif highlighted_type is list:
                for ngram in highlighted_ngrams:
                    if type(ngram[0]) is str:
                        highlight_ngram(
                            [string_to_integer_map[x] for x in ngram]
                        )
                    elif type(ngram[0]) is int:
                        highlight_ngram(ngram)
                    else:
                        raise Exception('Invalid data format', ngram)

        # Set the suptitle on the figure
        if suptitle is not None:
            if alphabet_len <= 6:
                if verbose:
                    from scipy.optimize import curve_fit
                    import matplotlib.pyplot as plt
                    import numpy as np
                    import os.path as osp
                    from re import sub

                    # The data to predict the y-value of the suptitle
                    x = np.array([1, 4, 6])
                    y = np.array([1.95, 1.08, 1.0])

                    # Create a figure and axis
                    fig, ax = plt.subplots()

                    # Plot data points
                    ax.plot(x, y, 'o', label='Data points')

                    # Define linear function
                    def linear_func(x, m, b):
                        """
                        Compute a linear function: y = m * x + b.

                        Args:
                            x (float): The independent variable.
                            m (float): The slope of the line.
                            b (float): The y-intercept of the line.

                        Returns:
                            float: The value of the linear function at x.
                        """
                        return m * x + b

                    # Fit linear function to data
                    popt_linear, pcov_linear = curve_fit(linear_func, x, y)
                    m, b = popt_linear
                    fitted_equation = (
                        f'y = {m:.2f}*alphabet_len + {b:.2f}'  # noqa E231
                    )
                    # print(fitted_equation)

                    # Plot linear fit
                    ax.plot(
                        x, linear_func(x, *popt_linear), label='Linear line'
                    )

                    # Define exponential decay function
                    def exp_decay_func(x, a, b, c):
                        """
                        Compute an exponential decay function:
                            y = a * exp(-b * x) + c.

                        Args:
                            x (float): The independent variable.
                            a (float): The initial value (amplitude).
                            b (float): The decay rate.
                            c (float):
                                The value the function asymptotes to as x
                                approaches infinity.

                        Returns:
                            float:
                                The value of the exponential decay function at
                                x.
                        """
                        return a * np.exp(-b * x) + c

                    # Fit exponential decay function to data
                    popt_exp, pcov_exp = curve_fit(exp_decay_func, x, y)
                    a, b, c = popt_exp
                    fitted_equation = (
                        f'y = {a:.2f} * np.exp(-{b:.2f} '  # noqa E231
                        f'* alphabet_len) + {c:.2f}'  # noqa E231
                    )
                    # print(fitted_equation)

                    # Plot exponential decay fit
                    ax.plot(
                        x, exp_decay_func(x, *popt_exp),
                        label='Exponential Decay line'
                    )

                    # Set labels and legend
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.legend()

                    # Save figure to PNG
                    file_path = osp.join(
                        self.saves_png_folder,
                        sub(
                            r'\W+', '_', str(suptitle)
                        ).strip('_').lower() + '_verbose.png'
                    )
                    # print(f'Saving verbose to {file_path}')
                    plt.savefig(file_path, bbox_inches='tight')
                    plt.close(fig)
                y = 2.06 * np.exp(-0.75 * alphabet_len) + 0.98
                if verbose:
                    print(f'alphabet_len={alphabet_len}, y={y}')
            else:
                y = 0.95
            fig.suptitle(suptitle, y=y)

            # Save figure to PNG
            from os import path as osp
            from re import sub
            file_path = osp.join(
                self.saves_png_folder,
                sub(r'\W+', '_', str(suptitle)).strip('_').lower() + '.png'
            )
            if verbose:
                print(f'Saving figure to {file_path}')
            plt.savefig(file_path, bbox_inches='tight')

        return (fig, ax)

    def plot_sequences(self, sequences, gap=True, color_dict=None):
        """
        Create a scatter-style sequence plot for a collection of sequences.

        Parameters:
            sequences (list):
                A list of sequences to plot.
            gap (bool, optional):
                Whether to leave a gap between different values in a sequence.
                Defaults to True.

        Returns:
            plt.Figure
                The matplotlib figure object.
        """

        # Determine the maximum sequence length
        max_sequence_length = max([len(s) for s in sequences])

        # Create a figure with appropriate dimensions
        plt.figure(figsize=[max_sequence_length * 0.3, 0.3 * len(sequences)])
        alphabet_cache = {
            sequence: self.get_alphabet(sequence) for sequence in sequences
        }

        for y, sequence in enumerate(sequences):

            # Convert the sequence to a NumPy array
            np_sequence = np.array(sequence)

            # Disable automatic color cycling
            plt.gca().set_prop_cycle(None)

            # Get the unique values in the sequence
            unique_values = alphabet_cache[sequence]

            # Set up the color dictionary with unique_values keys
            color_dict = self.update_color_dict(unique_values, color_dict)

            # Plot the value positions as scatter points with labels
            if gap:

                for i, value in enumerate(unique_values):
                    points = np.where(np_sequence == value, y + 1, np.nan)
                    plt.scatter(
                        x=range(len(np_sequence)),
                        y=points,
                        marker='s',
                        label=value,
                        s=100,
                        color=color_dict[value]
                    )
            else:

                for i, value in enumerate(unique_values):
                    points = np.where(np_sequence == value, 1, np.nan)
                    plt.bar(
                        range(len(points)),
                        points,
                        bottom=[y for x in range(len(points))],
                        width=1,
                        align='edge',
                        label=value,
                        color=color_dict[value]
                    )

        # Set the y-axis limits with or without gaps
        if gap:
            plt.ylim(0.4, len(sequences) + 0.6)
            plt.xlim(-0.6, max_sequence_length - 0.4)
        else:
            plt.ylim(0, len(sequences))
            plt.xlim(0, max_sequence_length)

        # Force x-ticks to land on integers only
        xtick_locations = range(len(sequences[0]))
        xtick_labels = [n+1 for n in xtick_locations]
        plt.xticks(ticks=xtick_locations, labels=xtick_labels, minor=False)

        # Get the legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Convert the legend handles and labels into a dictionary
        by_label = dict(zip(labels, handles))

        # Place the legend in the upper left corner
        plt.legend(
            by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 1.1),
            loc='upper left'
        )

        # Hide the y-axis ticks and labels
        plt.tick_params(axis='y', which='both', left=False, labelleft=False)

        # Return the matplotlib figure object
        return plt

    @staticmethod
    def update_color_dict(alphabet_list, color_dict=None):
        """
        Create or update a dictionary based on the given alphabet list.

        Parameters:
            alphabet_list (list):
                A list of keys to include in the dictionary. color_dict (dict,
                optional): An existing dictionary. Defaults to None.

        Returns:
            dict:
                A dictionary with keys from `alphabet_list`. If `color_dict`
                is supplied, its values are preserved for matching keys;
                otherwise, values are set to None.

        Examples:
            alphabet_list = ['a', 'b', 'c', 'd']
            existing_dict = {'a': 'red', 'b': 'blue'}

            # Case 1: No color dictionary provided
            print(
                update_color_dict(alphabet_list)
            )  # {'a': None, 'b': None, 'c': None, 'd': None}

            # Case 2: An existing color dictionary is provided
            print(
                update_color_dict(alphabet_list, existing_dict)
            )  # {'a': 'red', 'b': 'blue', 'c': None, 'd': None}
        """

        # Was the color dictionary not supplied?
        if color_dict is None:

            # Create it with keys from alphabet_list and values set to None
            color_dict = {a: None for a in alphabet_list}

        # Otherwise
        else:

            # Update a new one with alphabet_list keys and color_dict values
            color_dict = {a: color_dict.get(a) for a in alphabet_list}

        return color_dict


# print('\\b(' + '|'.join(dir()) + ')\\b')
