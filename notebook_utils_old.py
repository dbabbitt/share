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

from bs4 import BeautifulSoup as bs
from datetime import timedelta
from numpy import nan
from os import (
    listdir as listdir, makedirs as makedirs, path as osp, remove as remove,
    walk as walk
)
from pandas import (
    DataFrame, Series, concat, get_dummies, notnull, read_csv, read_pickle,
    to_datetime, read_html
)
from re import (
    IGNORECASE, Pattern, split, sub
)
import humanize
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pkgutil
import re
import seaborn as sns
import subprocess
import sys
import time
import urllib
import warnings
try:
    import dill as pickle
except Exception:
    import pickle

warnings.filterwarnings('ignore')

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

# Check if pandas is installed and import relevant functions
try:
    from pandas.core.arrays.numeric import is_integer_dtype, is_float_dtype

    def is_integer(srs):
        """
        Check if the given pandas Series has an integer data type.

        Parameters:
            srs (pd.Series): A pandas Series to check.

        Returns:
            bool: True if the Series contains integers, False otherwise.
        """
        return is_integer_dtype(srs)

    def is_float(srs):
        """
        Check if the given pandas Series has a float data type.

        Parameters:
            srs (pd.Series): A pandas Series to check.

        Returns:
            bool: True if the Series contains floats, False otherwise.
        """
        return is_float_dtype(srs)

except Exception:

    def is_integer(srs):
        """
        Check if the given list-like object contains integer values.

        Parameters:
            srs (pd.Series or list): A pandas Series or list-like object to
            check.

        Returns:
            bool: True if any of the values are of integer type, False
            otherwise.
        """
        return any(map(
            lambda value: np.issubdtype(type(value), np.integer), srs.tolist()
        ))

    def is_float(srs):
        """
        Check if the given list-like object contains float values.

        Parameters:
            srs (pd.Series or list): A pandas Series or list-like object to
            check.

        Returns:
            bool: True if any of the values are of float type, False otherwise.
        """
        return any(map(lambda value: np.issubdtype(
            type(value), np.floating
        ), srs.tolist()))


class NotebookUtilities(object):
    """
    This class implements the core of the utility
    functions needed to install and run GPTs and
    also what is common to running Jupyter notebooks.

    Example:

        # Add the path to the shared utilities directory
        import os.path as osp, os as os

        # Define the shared folder path using join for better compatibility
        shared_folder = osp.abspath(osp.join(
            osp.dirname(__file__), os.pardir, os.pardir, os.pardir, 'share'
        ))

        # Add the shared folder to system path if it's not already included
        import sys
        if shared_folder not in sys.path:
            sys.path.insert(1, shared_folder)

        # Attempt to import the Storage object
        try:
            from notebook_utils import NotebookUtilities
        except ImportError as e:
            print(f"Error importing NotebookUtilities: {e}")

        # Initialize with data and saves folder paths
        nu = NotebookUtilities(
            data_folder_path=osp.abspath(osp.join(
                osp.dirname(__file__), os.pardir, 'data'
            )),
            saves_folder_path=osp.abspath(osp.join(
                osp.dirname(__file__), os.pardir, 'saves'
            ))
        )
    """

    def __init__(
        self, data_folder_path=None, saves_folder_path=None, verbose=False
    ):

        # Create the data folder if it doesn't exist
        if data_folder_path is None:
            self.data_folder = osp.join(os.pardir, 'data')
        else:
            self.data_folder = data_folder_path
        makedirs(self.data_folder, exist_ok=True)
        if verbose:
            print(
                'data_folder: {}'.format(osp.abspath(self.data_folder)),
                flush=True
            )

        # Create the saves folder if it doesn't exist
        if saves_folder_path is None:
            self.saves_folder = osp.join(os.pardir, 'saves')
        else:
            self.saves_folder = saves_folder_path
        makedirs(self.saves_folder, exist_ok=True)
        if verbose:
            print(
                'saves_folder: {}'.format(osp.abspath(self.saves_folder)),
                flush=True
            )

        self.pip_command_str = f'{sys.executable} -m pip'

        # Assume this is instantiated in a subfolder one below the main
        self.github_folder = osp.dirname(osp.abspath(osp.curdir))

        # Create the assumed directories
        self.data_csv_folder = osp.join(self.data_folder, 'csv')
        makedirs(name=self.data_csv_folder, exist_ok=True)
        self.saves_csv_folder = osp.join(self.saves_folder, 'csv')
        makedirs(name=self.saves_csv_folder, exist_ok=True)
        self.saves_mp3_folder = osp.join(self.saves_folder, 'mp3')
        makedirs(name=self.saves_mp3_folder, exist_ok=True)
        self.saves_pickle_folder = osp.join(self.saves_folder, 'pkl')
        makedirs(name=self.saves_pickle_folder, exist_ok=True)
        self.saves_text_folder = osp.join(self.saves_folder, 'txt')
        makedirs(name=self.saves_text_folder, exist_ok=True)
        self.saves_wav_folder = osp.join(self.saves_folder, 'wav')
        makedirs(name=self.saves_wav_folder, exist_ok=True)
        self.saves_png_folder = osp.join(self.saves_folder, 'png')
        makedirs(name=self.saves_png_folder, exist_ok=True)
        self.txt_folder = osp.join(self.data_folder, 'txt')
        makedirs(self.txt_folder, exist_ok=True)

        # Create the model directories
        self.bin_folder = osp.join(self.data_folder, 'bin')
        makedirs(self.bin_folder, exist_ok=True)
        self.cache_folder = osp.join(self.data_folder, 'cache')
        makedirs(self.cache_folder, exist_ok=True)
        self.data_models_folder = osp.join(self.data_folder, 'models')
        makedirs(name=self.data_models_folder, exist_ok=True)
        self.db_folder = osp.join(self.data_folder, 'db')
        makedirs(self.db_folder, exist_ok=True)
        self.graphs_folder = osp.join(self.saves_folder, 'graphs')
        makedirs(self.graphs_folder, exist_ok=True)
        self.indices_folder = osp.join(self.saves_folder, 'indices')
        makedirs(self.indices_folder, exist_ok=True)

        # Ensure the Scripts folder is in PATH
        self.anaconda_folder = osp.dirname(sys.executable)
        self.scripts_folder = osp.join(self.anaconda_folder, 'Scripts')
        if self.scripts_folder not in sys.path:
            sys.path.insert(1, self.scripts_folder)

        # Handy list of the different types of encodings
        self.encoding_types_list = ['utf-8', 'latin1', 'iso8859-1']
        self.encoding_type = self.encoding_types_list[0]
        self.encoding_errors_list = ['ignore', 'replace', 'xmlcharrefreplace']
        self.encoding_error = self.encoding_errors_list[2]
        self.decoding_types_list = [
            'ascii', 'cp037', 'cp437', 'cp863', 'utf_32', 'utf_32_be',
            'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7',
            'utf_8', 'utf_8_sig', 'latin1', 'iso8859-1'
        ]
        self.decoding_type = self.decoding_types_list[11]
        self.decoding_errors_list = self.encoding_errors_list.copy()
        self.decoding_error = self.decoding_errors_list[2]

        # Regular expressions to determine URL from file path
        s = r'\b(https?|file)://[-A-Z0-9+&@#/%?=~_|$!:,.;]*[A-Z0-9+&@#/%=~_|$]'
        self.url_regex = re.compile(s, IGNORECASE)
        s = r'\b[c-d]:\\(?:[^\\/:*?"<>|\x00-\x1F]{0,254}'
        s += r'[^.\\/:*?"<>|\x00-\x1F]\\)*(?:[^\\/:*?"<>'
        s += r'|\x00-\x1F]{0,254}[^.\\/:*?"<>|\x00-\x1F])'
        self.filepath_regex = re.compile(s, IGNORECASE)

        # Compile the pattern for identifying function definitions
        self.simple_defs_regex = re.compile(r'\bdef ([a-z0-9_]+)\(')

        # Create a pattern to match function definitions
        self.ipynb_defs_regex = re.compile('\\s+"def ([a-z0-9_]+)\\(')

        # Create a regex for finding instance methods and self usage
        s = '^    def ([a-z]+[a-z_]+)\\(\\s*self,\\s+(?:[^\\)]+)\\):'
        self.instance_defs_regex = re.compile(s, re.MULTILINE)

        # Create a regex to search for self references within function bodies
        self.self_regex = re.compile('\\bself\\b')

        # Compile regex to find all unprefixed comments in the source code
        self.comment_regex = re.compile('^( *)# ([^\r\n]+)', re.MULTILINE)

        # Compile a regex pattern to match non-alphanumeric characters
        self.lower_ascii_regex = re.compile('[^a-z0-9]+')

        # Various aspect ratios
        self.facebook_aspect_ratio = 1.91
        self.twitter_aspect_ratio = 16/9

        try:
            from pysan.elements import get_alphabet
            self.get_alphabet = get_alphabet
        except Exception:
            self.get_alphabet = lambda sequence: set(sequence)

        # Module lists
        self.object_evaluators = [
            fn for fn in dir(inspect) if fn.startswith('is')
        ]
        module_paths = sorted([
            path
            for path in sys.path
            if path and not path.startswith(osp.dirname(__file__))
        ])
        self.standard_lib_modules = sorted([
            module_info.name
            for module_info in pkgutil.iter_modules(path=module_paths)
        ])

    # -------------------
    # Numeric Functions
    # -------------------

    @staticmethod
    def float_to_ratio(value, tolerance=0.01):
        """
        Convert a float to a ratio of two integers, approximating the float
        within a specified tolerance.

        Parameters:
            value (float): The float to convert.
            tolerance (float, optional):
                The acceptable difference between the float and the ratio.
                Default is 0.01.

        Returns:
            tuple:
                A tuple of two integers (numerator, denominator) representing
                the ratio that approximates the float.
        """

        # Use helper function to check if the fraction is within tolerance
        from fractions import Fraction

        def is_within_tolerance(numerator, denominator, value, tolerance):
            """
            Check if the ratio (numerator/denominator) is within the given
            tolerance of the target value.
            """
            return abs(numerator / denominator - value) <= tolerance

        # Validate input
        if not isinstance(value, (float, int)):
            raise TypeError('Input value must be a float or an integer.')
        if not isinstance(tolerance, float) or tolerance <= 0:
            raise ValueError('Tolerance must be a positive float.')

        # Use the Fraction module to find an approximate fraction
        fraction = Fraction(value).limit_denominator()
        numerator, denominator = (fraction.numerator, fraction.denominator)

        # Adjust the fraction only if needed
        while not is_within_tolerance(
            numerator, denominator, value, tolerance
        ):
            numerator += 1
            if is_within_tolerance(numerator, denominator, value, tolerance):
                break
            denominator += 1

        return (numerator, denominator)

    # -------------------
    # String Functions
    # -------------------

    @staticmethod
    def compute_similarity(a, b):
        """
        Calculate the similarity between two strings.

        Parameters:
            a (str): The first string.
            b (str): The second string.

        Returns:
            float
                The similarity between the two strings, as a float between 0
                and 1.
        """
        from difflib import SequenceMatcher

        return SequenceMatcher(None, str(a), str(b)).ratio()

    @staticmethod
    def get_first_year_element(x):
        """
        Extract the first year element from a given string, potentially
        containing multiple date or year formats.

        Parameters:
            x (str): The input string containing potential year information.

        Returns:
            int or float
                The extracted first year element, or NaN if no valid year
                element is found.
        """

        # Split the input string using various separators
        stripped_list = split('( |/|\\x96|\\u2009|-|\\[)', str(x), 0)

        # Remove non-numeric characters from each element in the stripped list
        stripped_list = [sub('\\D+', '', x) for x in stripped_list]

        # Filter elements with lengths between 3 and 4, as likely to be years
        stripped_list = [
            x for x in stripped_list if len(x) >= 3 and len(x) <= 4
        ]

        try:

            # Identify the index of the 1st numeric in the stripped list
            numeric_list = [x.isnumeric() for x in stripped_list]

            # If a numeric substring is found, extract the first numeric value
            if True in numeric_list:
                idx = numeric_list.index(True, 0)
                first_numeric = int(stripped_list[idx])

            # If no numeric substring is found, raise an exception
            else:
                raise Exception('No numeric year element found')

        # Handle exceptions and return the first substring if no numeric
        except Exception:

            # If there are any substrings, return the 1st one as the year
            if stripped_list:
                first_numeric = int(stripped_list[0])

            # If there are no substrings, return NaN
            else:
                first_numeric = np.nan

        return first_numeric

    @staticmethod
    def format_timedelta(time_delta):
        """
        Format a time delta object to a string in the
        format '0 sec', '30 sec', '1 min', '1:30', '2 min', etc.

        Parameters:
          time_delta: A time delta object representing a duration.

        Returns:
            A string representing the formatted time delta in a human-readable
            format: '0 sec', '30 sec', '1 min', '1:30', '2 min', etc.
        """

        # Extract total seconds from the time delta object
        seconds = time_delta.total_seconds()

        # Calculate the number of whole minutes
        minutes = int(seconds // 60)

        # Calculate the remaining seconds after accounting for minutes
        seconds = int(seconds % 60)

        # Format the output string for zero minutes, showing only seconds
        if minutes == 0:
            return f'{seconds} sec'

        # If there are minutes and seconds, return in 'min:sec' format
        elif seconds > 0:
            return f'{minutes}:{seconds:02}'  # noqa: E231

        # If there are only minutes, return in 'min' format
        else:
            return f'{minutes} min'

    @staticmethod
    def outline_chars(text_str, verbose=False):
        ord_list = []
        for char in list(text_str):
            i = ord(char)
            if i >= ord('a'):
                i += (ord('𝕒') - ord('a'))
            elif i >= ord('A'):
                i += (ord('𝔸') - ord('A'))
            if verbose:
                print(f'{char} or {ord(char)}: {i} or {chr(i)}')
            ord_list.append(i)

        return ''.join([chr(i) for i in ord_list])

    # -------------------
    # List Functions
    # -------------------

    @staticmethod
    def conjunctify_nouns(noun_list=None, and_or='and', verbose=False):
        """
        Concatenate a list of nouns into a grammatically correct string with
        specified conjunctions.

        Parameters:
            noun_list (list or str): A list of nouns to be concatenated.
            and_or (str, optional): The conjunction used to join the nouns.
            Default is 'and'.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            str
                A string containing the concatenated nouns with appropriate
                conjunctions.

        Example:
            noun_list = ['apples', 'oranges', 'bananas']
            conjunction = 'and'
            result = conjunctify_nouns(noun_list, and_or=conjunction)
            print(result)
            Output: 'apples, oranges, and bananas'
        """

        # Handle special cases where noun_list is None or not a list
        if noun_list is None:
            return ''
        if not isinstance(noun_list, list):
            noun_list = list(noun_list)

        # If there are more than two nouns in the list
        if len(noun_list) > 2:

            # Create a noun string of the last element in the list
            last_noun_str = noun_list[-1]

            # Create comma-delimited but-last string out of the rest
            but_last_nouns_str = ', '.join(noun_list[:-1])

            # Join the but-last string and the last noun string with `and_or`
            list_str = f', {and_or} '.join(
                [but_last_nouns_str, last_noun_str]
            )

        # If just two nouns in the list, join the nouns with `and_or`
        elif len(noun_list) == 2:
            list_str = f' {and_or} '.join(noun_list)

        # If there is just one noun in the list, make that the returned string
        elif len(noun_list) == 1:
            list_str = noun_list[0]

        # Otherwise, make a blank the returned string
        else:
            list_str = ''

        # Print debug output if verbose
        if verbose:
            print(
                f'noun_list="{noun_list}", and_or="{and_or}", '
                f'list_str="{list_str}"'
            )

        # Return the conjuncted noun list
        return list_str

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
    def get_shape(list_of_lists):
        """
        Return the shape of a list of lists, assuming the sublists are all of
        the same length.

        Parameters:
            list_of_lists: A list of lists.

        Returns:
            A tuple representing the shape of the list of lists.
        """

        # Check if the list of lists is empty
        if not list_of_lists:
            return ()

        # Get the length of the first sublist
        num_cols = len(list_of_lists[0])

        # Check if all of the sublists are the same length
        for sublist in list_of_lists:
            if len(sublist) != num_cols:
                raise ValueError(
                    'All of the sublists must be the same length.'
                )

        # Return a tuple representing the shape of the list of lists
        return (len(list_of_lists), num_cols)

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

    def check_4_doubles(self, item_list, verbose=False):
        """
        Find and compare items within a list to identify similar pairs.

        This method compares each item in the list with every other item
        to find the most similar item based on a computed similarity
        metric. The results are returned in a DataFrame containing pairs
        of items and their similarity scores and byte representations.

        Parameters:
            item_list (list):
                The list of items to be compared for similarity.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            pandas.DataFrame:
                A DataFrame containing columns for the first item, second item,
                their byte representations, and the maximum similarity score
                found for each pair.
        """

        # Start the timer if verbose is enabled
        if verbose:
            t0 = time.time()

        rows_list = []
        n = len(item_list)

        # Iterate over each item in the list
        for i in range(n-1):

            # Get the current item to compare with others
            first_item = item_list[i]

            # Initialize the maximum similarity score
            max_similarity = 0.0

            # Initialize the item with the highest similarity to the current
            max_item = first_item

            # Compare the current item with the rest of the items in the list
            for j in range(i+1, n):

                # Get the item to compare against
                second_item = item_list[j]

                # Ensure items are not identical before similarity calculation
                if first_item != second_item:

                    # Compute the similarity between the two items
                    this_similarity = self.compute_similarity(
                        str(first_item), str(second_item)
                    )

                    # Has a higher similarity been found?
                    if this_similarity > max_similarity:

                        # Update max_similarity and max_item
                        max_similarity = this_similarity
                        max_item = second_item

            # Create row dict to store information for each similar item pair
            row_dict = {'first_item': first_item, 'second_item': max_item}

            # Convert items to byte arrays for string representation
            row_dict['first_bytes'] = '-'.join([str(x) for x in bytearray(
                str(first_item), encoding=self.encoding_type, errors='replace'
            )])
            row_dict['second_bytes'] = '-'.join([str(x) for x in bytearray(
                str(max_item), encoding=self.encoding_type, errors='replace'
            )])
            row_dict['max_similarity'] = max_similarity

            # Add the row dictionary to the list of rows
            rows_list.append(row_dict)

        # Define the column names for the resulting DataFrame
        column_list = [
            'first_item', 'second_item', 'first_bytes', 'second_bytes',
            'max_similarity'
        ]

        # Create the DataFrame from the list of row dictionaries
        item_similarities_df = DataFrame(rows_list, columns=column_list)

        # Display end time for performance measurement (if verbose)
        if verbose:
            t1 = time.time()
            print(
                f'Finished in {t1 - t0:.2f} '  # noqa: E231
                f'seconds ({time.ctime(t1)})'
            )

        return item_similarities_df

    def check_for_typos(
        self, left_list, right_list,
        rename_dict={'left_item': 'left_item', 'right_item': 'right_item'},
        verbose=False
    ):
        """
        Check the closest names for typos by comparing items from left_list
        with items from right_list and computing their similarities.

        Parameters:
            left_list (list): List containing items to be compared (left side).
            right_list (list):
                List containing items to be compared (right side).
            rename_dict (dict, optional):
                Dictionary specifying custom column names in the output
                DataFrame. Default is {'left_item': 'left_item', 'right_item':
                'right_item'}.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            pandas.DataFrame: DataFrame containing columns: 'left_item',
            'right_item', and 'max_similarity'.

        Example:
            commonly_misspelled_words = [
                "absence", "consensus", "definitely", "broccoli", "necessary"
            ]
            common_misspellings = [
                "absense", "concensus", "definately", "brocolli", "neccessary"
            ]
            typos_df = nu.check_for_typos(
                commonly_misspelled_words,
                common_misspellings,
                rename_dict={
                    'left_item': 'commonly_misspelled',
                    'right_item': 'common_misspelling'
                }
            ).sort_values(
                [
                    'max_similarity', 'commonly_misspelled',
                    'common_misspelling'
                ],
                ascending=[False, True, True]
            )
            display(typos_df)
        """

        # Initialize the time taken for the computation if verbose is True
        if verbose:
            t0 = time.time()

        # Initialize an empty list to store rows of the output data frame
        rows_list = []

        # Iterate through items in the left list
        for left_item in left_list:
            max_similarity = 0.0
            max_item = left_item

            # Iterate through items in the right list
            for right_item in right_list:
                this_similarity = self.compute_similarity(
                    left_item, right_item
                )

                # Find the most similar item
                if this_similarity > max_similarity:
                    max_similarity = this_similarity
                    max_item = right_item

            # Create a dictionary representing a row in the output data frame
            row_dict = {
                'left_item': left_item,
                'right_item': max_item,
                'max_similarity': max_similarity
            }

            # Add the row dictionary to the list of rows
            rows_list.append(row_dict)

        # Define the column names for the output data frame
        column_list = ['left_item', 'right_item', 'max_similarity']

        # Create a df from the list of rows, rename columns if necessary
        name_similarities_df = DataFrame(
            rows_list, columns=column_list
        ).rename(columns=rename_dict)

        # Print the time taken for the computation if verbose is True
        if verbose:
            t1 = time.time()
            print(t1-t0, time.ctime(t1))

        # Return the resulting data frame
        return name_similarities_df

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

    @staticmethod
    def get_numbered_text(
        numbering_format,
        current_number_dict={'1': 1, 'A': 1, 'a': 1, 'i': 1, 'I': 1}
    ):
        """
        Apply numbering based on a format string and a current number.

        This static method takes a numbering format string
        (`numbering_format`) and a current number dictionary
        (`current_number_dict`) as inputs. It then parses the format
        string to identify numbering placeholders (letters 'A', 'a', '1',
        'I', or 'i') and replaces them with the corresponding numbered
        representation based on the current number of that placeholder.

        The function supports the following numbering formats:
        - '1': Sequential numbering, starting from 1 (e.g., 'Step 1'
        becomes 'Step 2').
        - 'A': Uppercase alphabetic numbering, starting from A (e.g.,
        'Part A' becomes 'Part B').
        - 'a': Lowercase alphabetic numbering, starting from a (e.g.,
        'Section a' becomes 'Section b').
        - 'i': Roman numeral numbering, starting from i (e.g., 'Point i'
        becomes 'Point ii').
        - 'I': Roman numeral numbering, starting from I (e.g., 'Epoch I'
        becomes 'Epoch II').

        Parameters:
            numbering_format (str):
                The numbering format string, which should contain one of the
                following characters: 'A', '1', 'a', 'I', or 'i'. These
                characters represent the place in the string where the
                current number should be inserted, and their case and type
                represent the format that the number should be in.
            current_number_dict (dict with strs as keys and ints as values,
            optional):
                The current number based on the format codes to replace the
                placeholder in the numbering format. Defaults to {'1': 1,
                'A': 1, 'a': 1, 'i': 1, 'I': 1}.
        Returns:
            str
                The formatted string with the current number inserted in
                place of the placeholder.
        Raises:
            AssertionError
                If the `numbering_format` string does not contain any of the
                supported numbering placeholders (A, a, 1, I, or i).
        """
        if isinstance(current_number_dict, int):
            current_number_dict = {
                '1': current_number_dict, 'A': current_number_dict,
                'a': current_number_dict, 'i': current_number_dict,
                'I': current_number_dict
            }

        # Extract the format codes by keeping only 'A', '1', 'a', 'I', or 'i'
        format_codes = re.sub('[^A1aiI]+', '', numbering_format)

        # If the format codes are empty, return the numbering_format as is
        if not format_codes:
            return numbering_format

        # Extract each character as the numbering code
        for format_code in format_codes:

            # Is the format code '1'?
            if format_code == '1':

                # Apply sequential numbering (replace with current number)
                numbering_format = numbering_format.replace(
                    format_code, str(current_number_dict[format_code])
                )

            # Is the format code 'i'?
            elif format_code == 'i':

                # Apply lowercase roman numeral numbering
                import roman
                numbering_format = numbering_format.replace(
                    format_code, roman.toRoman(
                        current_number_dict[format_code]
                    ).lower()
                )

            # Is the format code 'I'?
            elif format_code == 'I':

                # Apply uppercase roman numeral numbering
                import roman
                numbering_format = numbering_format.replace(
                    format_code, roman.toRoman(
                        current_number_dict[format_code]
                    ).upper()
                )

            # Otherwise
            else:

                # Adjust the ASCII code by the offset value of the format
                new_char_code = ord(
                    format_code
                ) + current_number_dict[format_code] - 1

                # Apply alphabetic numbering
                numbering_format = numbering_format.replace(
                    format_code, chr(new_char_code)
                )

        return numbering_format

    def apply_multilevel_numbering(
        self, text_list,
        level_map={0: "", 4: "A. ", 8: "1. ", 12: "a) ", 16: "i) "},
        add_indent_back_in=False, verbose=False
    ):
        """
        Take a list of strings with indentation and infix or prefix them
        with the appropriate numbering text based on a multi-level list
        style.

        Parameters:
            text_list:
                A list of strings, where each string may be prepended with 0,
                4, 8, 12, or 16 spaces representing indentation levels, or
                a list of tuples (int, str), where the int is the number of
                spaces and the str is the left-stripped text.
            level_map:
                A dictionary that maps indentation to numbering format.
                Defaults to {0: "", 4: "A. ", 8: "1. ", 12: "a) ", 16: "i) "}.
            add_indent_back_in:
                Whether to prepend the indention

        Returns:
            A new list of strings with numbering text prepended based on
            indentation.

        Example:
            level_count = 8
            text_list = [
                ' ' * (i*4) + f'This is level {i}'
                for i in range(level_count+1)
            ]
            text_list += [
                ' ' * (i*4) + f'This is level {i} again'
                for i in range(level_count, -1, -1)
            ]
            numbered_list = nu.apply_multilevel_numbering(
                text_list,
                level_map={
                    0: "", 4: "I. ", 8: "A. ", 12: "1. ", 16: "a. ",
                    20: "I) ", 24: "A) ", 28: "1) ", 32: "a) "
                }, add_indent_back_in=True
            )
            for line in numbered_list:
                print(line)
        """

        # Initialize current_level to track current indentation level
        current_level = 0

        # Initialize map to track current number within a level
        current_number_map = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

        numbered_text_list = []

        # Loop through each sentence in text_list
        for text in text_list:

            # Get the level by the indent
            if isinstance(text, tuple):
                indent = text[0]
                text = text[1]
            else:
                indent = len(text) - len(text.lstrip())
            if verbose:
                print(f'indent = {indent}')
            new_level = indent // 4
            if verbose:
                print(f'new_level = {new_level}')

            # Handle level changes and update numbering
            if new_level > current_level:
                current_level = new_level
                current_number_map[current_level] = 1
            elif new_level < current_level:
                current_level = new_level
                current_number_map[current_level] = current_number_map.get(
                    current_level, 0
                ) + 1
            else:
                current_number_map[current_level] = current_number_map.get(
                    current_level, 0
                ) + 1
            if verbose:
                print(
                    'current_number_map[current_level] ='
                    f' current_number_map[{current_level}] ='
                    f' {current_number_map[current_level]}'
                )

            # Generate numbering text and prepend it to the string
            numbered_text = self.get_numbered_text(
                level_map[indent],
                current_number_dict=current_number_map[current_level]
            )
            if verbose:
                print(f'numbered_text = "{numbered_text}"')
            if add_indent_back_in:
                level_str = '    ' * (current_level-1) + numbered_text
                level_str += text.lstrip()
            else:
                level_str = numbered_text + text.lstrip()
            numbered_text_list.append(level_str)

        return numbered_text_list

    # -------------------
    # File Functions
    # -------------------

    @staticmethod
    def get_function_file_path(func):
        """
        Get the relative or absolute file path where a function is stored.

        Parameters:
            func: A Python function.

        Returns:
            A string representing the relative or absolute file path where
            the function is stored.

        Example:
            my_function = lambda: None
            file_path = nu.get_function_file_path(my_function)
            print(osp.abspath(file_path))
        """

        # Work out which source or compiled file an object was defined in
        file_path = inspect.getfile(func)

        # Is the function defined in a Jupyter notebook?
        if file_path.startswith('<stdin>'):

            # Return the absolute file path
            return osp.abspath(file_path)

        # Otherwise, return the relative file path
        else:
            return osp.relpath(file_path)

    def get_utility_file_functions(self, util_path=None, verbose=False):
        """
        Extract a set of function names already defined in the utility file.

        Parameters:
            util_path (str, optional):
                The path to the utility file. Default is
                '../py/notebook_utils.py'.

        Returns:
            set of str: A set containing the names of functions already
            defined in the utility file.
        """

        # Set the utility path if not provided
        if util_path is None:
            util_path = osp.join(os.pardir, 'py', 'notebook_utils.py')

        # Read the utility file and extract function names
        with open(util_path, 'r', encoding='utf-8') as f:

            # Read the file contents line by line
            lines_list = f.readlines()

            # Initialize an empty set to store function names
            utils_set = set()

            # Iterate over each line in the file
            if verbose:
                print(f'Iterating over each line in {util_path}')
            for line in lines_list:

                # Search for function definitions
                match_obj = self.simple_defs_regex.search(line)

                # If function definition is found
                if match_obj:

                    # Extract the function name from the match and add it
                    scraping_util = match_obj.group(1)
                    utils_set.add(scraping_util)

        return utils_set

    @staticmethod
    def open_path_in_notepad(
        path_str, home_key='USERPROFILE', text_editor_path=None,
        continue_execution=True, verbose=True
    ):
        """
        Open a file in Notepad or a specified text editor.

        Parameters:
            path_str (str): The path to the file to be opened.
            home_key (str, optional):
                The environment variable key for the home directory. Default
                is 'USERPROFILE'.
            text_editor_path (str, optional):
                The path to the text editor executable. Default is Notepad++.
            continue_execution (bool, optional):
                If False, interacts with the subprocess and attempts to open
                the parent folder in explorer if it gets a bad return code.
                Default is True.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            None

        Notes:
            The function uses subprocess to run the specified text editor
            with the provided file path.

        Example:
            nu.open_path_in_notepad(r'C:\this_example.txt')
        """

        # Establish the text_editor_path in this operating system, if needed
        if text_editor_path is None:
            if os.name == 'nt':
                text_editor_path = 'C:\\Program'
                text_editor_path += ' Files\\Notepad++\\notepad++.exe'
            else:
                text_editor_path = '/mnt/c/Program'
                text_editor_path += ' Files/Notepad++/notepad++.exe'

        # Expand '~' to the home directory in the file path
        environ_dict = dict(os.environ)
        if '~' in path_str:
            if home_key in environ_dict:
                path_str = path_str.replace('~', environ_dict[home_key])
            else:
                path_str = osp.expanduser(path_str)

        # Get the absolute path to the file
        absolute_path = osp.abspath(path_str)
        if os.name != 'nt':
            absolute_path = absolute_path.replace(
                '/mnt/c/', 'C:\\'
            ).replace('/', '\\')
        if verbose:
            print(f'Attempting to open {absolute_path}')

        # Open the file in Notepad++ or the specified text editor
        cmd = [text_editor_path, absolute_path]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if not continue_execution:
            out, err = proc.communicate()
            if (proc.returncode != 0):
                if verbose:
                    print('Open attempt failed: ' + err.decode('utf8'))
                subprocess.run(['explorer.exe', osp.dirname(absolute_path)])

    @staticmethod
    def modify_inkscape_labels(
        file_path, output_path='C:\\Users\\daveb\\Downloads\\scratchpad.svg',
        verbose=False
    ):
        """
        Modify an SVG file by changing the 'id' attribute to the value of the
        'inkscape:label' and removing the 'inkscape:label' attribute.

        Parameters:
            file_path (str): Path to the SVG file to modify.
            output_path (str): Path to save the modified SVG file.

        Returns:
            None
        """
        if not osp.exists(file_path):
            raise FileNotFoundError(
                f'The file at {file_path} does not exist.'
            )

        # Open and parse the SVG file
        with open(file_path, 'r', encoding='utf-8') as file:
            svg_content = file.read()
        soup = bs(svg_content, 'xml')  # Use 'xml' parser for SVG

        # Find all tags with 'inkscape:label' and 'id' attributes
        for tag in soup.find_all(attrs={'inkscape:label': True, 'id': True}):
            inkscape_label = tag.get('inkscape:label', '').strip()
            tag_id = tag.get('id', '').strip()

            if inkscape_label and tag_id:
                if verbose:
                    print(f"Changing id '{tag_id}' to '{inkscape_label}'")

                # Update 'id' attribute with the value of 'inkscape:label'
                tag['id'] = inkscape_label

                # Remove the 'inkscape:label' attribute
                del tag['inkscape:label']

        # Save the modified SVG content to the output file
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(str(soup))

    def get_notebook_functions_dictionary(self, github_folder=None):
        """
        Get a dictionary of all functions defined within notebooks in a
        GitHub folder and their definition counts.

        This function scans a specified GitHub folder (or the parent
        directory by default) for Jupyter notebooks (`.ipynb` files). It
        then parses each notebook and identifies function definitions
        using a regular expression. The function name is used as the key
        in a dictionary, and the value stores the count of how many times
        that function is defined across all the notebooks.

        Parameters:
            github_folder (str, optional):
                The path of the root folder of the GitHub repository
                containing the notebooks. Defaults to the parent directory of
                the current working directory.

        Returns:
            dict
                The dictionary of function definitions with the count of their
                occurrences.
        """

        # Create a list of directories to exclude during the search
        black_list = ['.ipynb_checkpoints', '$Recycle.Bin']

        # If no github_folder is provided, use the default one
        if github_folder is None:
            github_folder = self.github_folder

        # Initialize empty dict to store function names and their counts
        rogue_fns_dict = {}

        # Walk through the directory structure
        for sub_directory, directories_list, files_list in walk(
            github_folder
        ):

            # Skip blacklisted directories
            if all(map(lambda x: x not in sub_directory, black_list)):

                # Iterate through files in the current directory
                for file_name in files_list:

                    # Process only Jupyter notebook files, excluding 'Attic'
                    if (
                        file_name.endswith('.ipynb')
                        and ('Attic' not in file_name)
                    ):

                        # Construct the full file path
                        file_path = osp.join(sub_directory, file_name)

                        # Open and read the notebook file
                        with open(
                            file_path, 'r', encoding=self.encoding_type
                        ) as f:

                            # Read the file contents line by line
                            lines_list = f.readlines()

                            # Loop through each line in the notebook
                            for line in lines_list:

                                # Search for function definitions
                                match_obj = self.ipynb_defs_regex.search(line)

                                # Has a function definition been found?
                                if match_obj:

                                    # Extract the function name
                                    fn = match_obj.group(1)

                                    # Increment the function count
                                    rogue_fns_dict[fn] = rogue_fns_dict.get(
                                        fn, 0
                                    ) + 1

        # Return the dictionary of function names and their counts
        return rogue_fns_dict

    def get_notebook_functions_set(self, github_folder=None):
        """
        Get a set of all functions defined within notebooks in a GitHub
        folder.

        This function leverages the `get_notebook_functions_dictionary` to
        retrieve a dictionary containing function names and their
        occurrence counts across notebooks in the specified GitHub folder
        (or the parent directory by default). It then extracts a set
        containing only the unique function names, effectively eliminating
        duplicates.

        Parameters:
            github_folder (str, optional):
                The path to the root folder of the GitHub repository
                containing the notebooks. Defaults to the parent directory of
                the current working directory.
        Returns:
            set
                A set containing all unique function names defined within the
                processed notebooks.
        """

        # If no GitHub folder is provided, use the default folder path
        if github_folder is None:
            github_folder = self.github_folder

        # Get the dictionary of function names and their counts
        rogue_fns_dict = self.get_notebook_functions_dictionary(
            github_folder=github_folder
        )

        # Extract a set containing only the unique function names
        rogue_fns_set = set(rogue_fns_dict.keys())

        # Return the set of unique function names
        return rogue_fns_set

    def show_duplicated_util_fns_search_string(
        self, util_path=None, github_folder=None
    ):
        """
        Search for duplicate utility function definitions in Jupyter
        notebooks within a specified GitHub repository folder. The function
        identifies rogue utility function definitions in Jupyter notebooks
        and prints a regular expression pattern to search for instances of
        these definitions. The intention is to   replace these calls with the
        corresponding `nu.` equivalent and remove the duplicates.

        Parameters:
            util_path (str, optional):
                The path to the utilities file to check for existing utility
                function definitions. Defaults to `../py/notebook_utils.py`.
            github_folder (str, optional):
                The path of the root folder of the GitHub repository
                containing the notebooks. Defaults to the parent directory of
                the current working directory.

        Returns:
            None:
                The function prints the regular expression pattern to
                identify rogue utility function definitions.
        """

        # Get a list of rogue functions already in utilities file
        utils_set = self.get_utility_file_functions(util_path=util_path)

        # Make a set of rogue util functions
        if github_folder is None:
            github_folder = self.github_folder
        rogue_fns_list = [
            fn
            for fn in self.get_notebook_functions_dictionary(
                github_folder=github_folder
            ).keys()
            if fn in utils_set
        ]

        if rogue_fns_list:
            print(
                'Search for *.ipynb; file masks in the'  # noqa: E702
                f' {github_folder} folder for this pattern:'  # noqa: E231
            )
            print('\\s+"def (' + '|'.join(rogue_fns_list) + ')\\(')
            print(
                'Replace each of the calls to these definitions with calls'
                ' the the nu. equivalent (and delete the definitions).'
            )

    def list_dfs_in_folder(self, pickle_folder=None):
        """
        List DataFrame names stored as pickles in a specified folder.

        Parameters:
            pickle_folder (str, optional):
                The folder path where pickle files are stored. If None, uses
                the default saves_pickle_folder. Default is None.

        Returns:
            list of str: A list of DataFrame pickle file names.
        """

        # Set the pickle folder if not provided
        if pickle_folder is None:
            pickle_folder = self.saves_pickle_folder

        # Filter the file names to include only pickle files
        pickles_list = [
            file_name.split('.')[0]
            for file_name in listdir(pickle_folder)
            if file_name.split('.')[1] in ['pkl', 'pickle']
        ]

        # Filter the list to include only DataFrame names (ending with '_df')
        dfs_list = [
            pickle_name
            for pickle_name in pickles_list
            if pickle_name.endswith('_df')
        ]

        # Return the list of DataFrame pickle file names
        return dfs_list

    def show_dupl_fn_defs_search_string(
        self, util_path=None, repo_folder=None
    ):
        """
        Identify and report duplicate function definitions in Jupyter
        notebooks and suggest how to consolidate them.

        Parameters:
            util_path (str, optional):
                The path to the utility file where refactored functions will
                be added. Defaults to '../py/notebook_utils.py'.
            repo_folder (str, optional):
                The path to the GitHub repository containing the Jupyter
                notebooks. Default is the parent folder of the current
                directory.

        Returns:
            None

        Notes:
            The function prints a search string pattern that can be used to
            identify duplicate function definitions in Jupyter notebooks. The
            pattern is based on the function names extracted from the
            notebook using `get_notebook_functions_dictionary()`.

        Example:
            nu.show_dupl_fn_defs_search_string()
        """

        # Set the utility path if not provided
        if util_path is None:
            util_path = osp.abspath(osp.join(
                os.pardir, 'share', 'notebook_utils.py'
            ))

        # Set the GitHub folder path if not provided
        if repo_folder is None:
            repo_folder = self.github_folder

        # Get the function definitions dictionary
        function_definitions_dict = self.get_notebook_functions_dictionary()

        # Convert the dictionary to a DataFrame
        df = DataFrame([
            {'function_name': k, 'definition_count': v}
            for k, v in function_definitions_dict.items()
        ])

        # Create a mask to filter functions with more than one definition
        mask_series = df.definition_count > 1
        duplicate_fns_list = df[mask_series].function_name.tolist()

        # If there are duplicate function defs, print a search string
        if duplicate_fns_list:
            print(
                'Search for *.ipynb; file masks in the'  # noqa: E702
                f' {repo_folder} folder for this pattern:'  # noqa: E231
            )
            print('\\s+"def\\s+(' + '|'.join(duplicate_fns_list) + ')')
            print(
                'Consolidate these duplicate definitions and add the'
                f' refactored one to {util_path} (and delete the'
                ' definitions).'
            )

    def delete_ipynb_checkpoint_folders(self, github_folder=None):
        """
        Delete all dot-ipynb_checkpoints folders within the specified GitHub
        folder and its subdirectories.

        Parameters:
            github_folder (str, optional):
                The path to the GitHub folder containing the
                '.ipynb_checkpoints' folders. If not provided, the current
                working directory is used.

        Returns:
            None
        """
        import shutil

        # Set the GitHub folder path if not provided
        if github_folder is None:
            github_folder = self.github_folder

        # Iterate over all subdirectories within the github_folder
        for sub_directory, directories_list, files_list in walk(
            github_folder
        ):

            # Does the .ipynb_checkpoints directory exist?
            if '.ipynb_checkpoints' in directories_list:

                # Construct the full path to the '.ipynb_checkpoints' folder
                folder_path = osp.join(sub_directory, '.ipynb_checkpoints')

                # Remove the folder and its contents
                shutil.rmtree(folder_path)

    @staticmethod
    def get_random_py_file(py_folder, verbose=False):
        """
        Walks through the specified folder and collects all .py files.
        Returns the path of a randomly selected .py file.

        Parameters:
            py_folder (str): Path to the folder to search for .py files.

        Returns:
            str:
                The path to a randomly selected .py file, or None if no .py
                files are found.
        """
        import random

        # List to store paths of .py files
        py_files = []

        # Walk through the folder structure
        black_list = ['.ipynb_checkpoints', '$Recycle.Bin']
        for parent_directory, _, files in os.walk(py_folder):
            if all(map(lambda x: x not in parent_directory, black_list)):
                for file in files:

                    # Check if the file has a .py extension
                    if file.endswith('.py'):
                        py_files.append(osp.join(parent_directory, file))

        # If no .py files are found, return None
        if not py_files:
            return None

        # Randomly select and return one .py file
        return random.choice(py_files)

    # -------------------
    # Path Functions
    # -------------------

    @staticmethod
    def get_top_level_folder_paths(folder_path, verbose=False):
        """
        Get all top-level folder paths within a given directory.

        Parameters:
            folder_path (str): The path to the directory to scan for
            top-level folders.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            list[str]: A list of absolute paths to all top-level folders
            within the provided directory.

        Raises:
            FileNotFoundError: If the provided folder path does not exist.
            NotADirectoryError:
                If the provided folder path points to a file or non-existing
                directory.

        Notes:
            This function does not recursively scan for subfolders within the
            top-level folders. If `verbose` is True, it will print the number
            of discovered top-level folders.
        """

        # Make sure the provided folder exists and is a directory
        if not osp.exists(folder_path):
            raise FileNotFoundError(
                f'Directory {folder_path} does not exist.'
            )
        if not osp.isdir(folder_path):
            raise NotADirectoryError(
                f'Path {folder_path} is not a directory.'
            )

        # Initialize an empty list to store top-level folder paths
        top_level_folders = []

        # Iterate through items in the specified folder
        for item in listdir(folder_path):

            # Construct the full path for each item
            full_item_path = osp.join(folder_path, item)

            # Is the item a directory?
            if osp.isdir(full_item_path):

                # Add its path to the list
                top_level_folders.append(full_item_path)

        # Optionally print information based on the `verbose` flag
        if verbose:
            print(
                f'Found {len(top_level_folders)} top-level folders'
                f' in {folder_path}.'
            )

        # Return the list of top-level folder paths
        return top_level_folders

    @staticmethod
    def print_all_files_ending_starting_with(
        root_dir='D:\\Documents\\GitHub', ends_with='.yml',
        starts_with='install_config_',
        black_list=['$RECYCLE.BIN', '$Recycle.Bin']
    ):
        if isinstance(root_dir, list):
            root_dir_list = root_dir
        else:
            root_dir_list = [root_dir]
        if isinstance(ends_with, list):
            endswith_list = ends_with
        else:
            endswith_list = [ends_with]
        if isinstance(starts_with, list):
            startswith_list = starts_with
        else:
            startswith_list = [starts_with]
        for root_dir in root_dir_list:
            for sub_directory, directories_list, files_list in os.walk(
                root_dir
            ):
                if all(map(lambda x: x not in sub_directory, black_list)):
                    for file_name in files_list:
                        endswith_bool = False
                        for ends_with in endswith_list:
                            endswith_bool = (
                                endswith_bool or file_name.endswith(ends_with)
                            )
                        startswith_bool = False
                        for starts_with in startswith_list:
                            startswith_bool = (
                                startswith_bool
                                or file_name.startswith(starts_with)
                            )
                        if endswith_bool and startswith_bool:
                            file_path = osp.join(sub_directory, file_name)
                            print(file_path)

    @staticmethod
    def print_all_files_starting_with(
        root_dir='D:\\Vagrant_Projects\\local-vagrant', starts_with='host',
        black_list=['$RECYCLE.BIN', '$Recycle.Bin']
    ):
        if isinstance(root_dir, list):
            root_dir_list = root_dir
        else:
            root_dir_list = [root_dir]
        if isinstance(starts_with, list):
            startswith_list = starts_with
        else:
            startswith_list = [starts_with]
        for root_dir in root_dir_list:
            for sub_directory, directories_list, files_list in os.walk(
                root_dir
            ):
                if all(map(lambda x: x not in sub_directory, black_list)):
                    for file_name in files_list:
                        startswith_bool = False
                        for starts_with in startswith_list:
                            startswith_bool = (
                                startswith_bool
                                or file_name.startswith(starts_with)
                            )
                        if startswith_bool:
                            file_path = osp.join(sub_directory, file_name)
                            print(file_path)

    @staticmethod
    def print_all_files_ending_with(
        root_dir='D:\\', ends_with='.box',
        black_list=['$RECYCLE.BIN', '$Recycle.Bin']
    ):
        if isinstance(root_dir, list):
            root_dir_list = root_dir
        else:
            root_dir_list = [root_dir]
        if isinstance(ends_with, list):
            endswith_list = ends_with
        else:
            endswith_list = [ends_with]
        for root_dir in root_dir_list:
            for sub_directory, directories_list, files_list in os.walk(
                root_dir
            ):
                if all(map(lambda x: x not in sub_directory, black_list)):
                    for file_name in files_list:
                        endswith_bool = False
                        for ends_with in endswith_list:
                            endswith_bool = (
                                endswith_bool or file_name.endswith(ends_with)
                            )
                        if endswith_bool:
                            file_path = osp.join(sub_directory, file_name)
                            print(file_path)

    @staticmethod
    def get_git_lfs_track_commands(
        repository_name, repository_dir='D:\\Documents\\GitHub'
    ):
        black_list = [osp.join(repository_dir, repository_name, '.git')]
        file_types_set = set()
        for sub_directory, directories_list, files_list in os.walk(
            osp.join(repository_dir, repository_name)
        ):
            if all(map(lambda x: x not in sub_directory, black_list)):
                for file_name in files_list:
                    file_path = osp.join(sub_directory, file_name)
                    bytes_count = osp.getsize(file_path)
                    if bytes_count > 50_000_000:
                        file_types_set.add(file_name.split('.')[-1])
        print('git lfs install')
        for file_type in file_types_set:
            print('git lfs track "*.{}"'.format(file_type))
        print('git add .gitattributes')

    @staticmethod
    def get_specific_gitignore_files(
        repository_name, repository_dir='D:\\Documents\\GitHub',
        text_editor_path='C:\\Program Files\\Notepad++\\notepad++.exe'
    ):
        print(
            '\n    # Ignore big files (GitHub will warn you when pushing'
            ' files larger than 50 MB. You will not be allowed to\n    #'
            ' push files larger than 100 MB.) Tip: If you regularly push'
            ' large files to GitHub, consider introducing\n    # Git Large'
            ' File Storage (Git LFS) as part of your workflow.'
        )
        repository_path = osp.join(repository_dir, repository_name)
        black_list = [osp.join(repository_path, '.git')]
        for sub_directory, directories_list, files_list in os.walk(
            repository_path
        ):
            if all(map(lambda x: x not in sub_directory, black_list)):
                for file_name in files_list:
                    file_path = osp.join(sub_directory, file_name)
                    bytes_count = osp.getsize(file_path)
                    if bytes_count > 50_000_000:
                        print('/'.join(
                            osp.relpath(file_path, repository_path).split(
                                os.sep
                            )
                        ))
        file_path = osp.join(repository_dir, repository_name, '.gitignore')
        print()
        subprocess.run([text_editor_path, osp.abspath(file_path)])

    def remove_empty_folders(self, folder_path, remove_root=True):
        """
        Function to remove empty folders
        """
        if not osp.isdir(folder_path):

            return

        # Remove empty subfolders
        files = os.listdir(folder_path)
        if len(files):
            for f in files:
                full_path = osp.join(folder_path, f)
                if osp.isdir(full_path):
                    self.remove_empty_folders(full_path)

        # If folder empty, delete it
        files = os.listdir(folder_path)
        if len(files) == 0 and remove_root:
            print('Removing empty folder: {}'.format(folder_path))
            os.rmdir(folder_path)

    @staticmethod
    def get_all_directories_containing(
        root_dir='C:\\', contains_str='activate',
        black_list=['$RECYCLE.BIN', '$Recycle.Bin', '.git']
    ):
        dir_path_list = []
        if type(root_dir) == list:
            root_dir_list = root_dir
        else:
            root_dir_list = [root_dir]
        if type(contains_str) == list:
            contains_list = contains_str
        else:
            contains_list = [contains_str]
        for root_dir in root_dir_list:
            for sub_directory, directories_list, files_list in os.walk(
                root_dir
            ):
                if all(map(lambda x: x not in sub_directory, black_list)):
                    for dir_name in directories_list:
                        contains_bool = False
                        for contains_str in contains_list:
                            contains_bool = (
                                contains_bool or contains_str in dir_name
                            )
                        if contains_bool:
                            dir_path = osp.join(sub_directory, dir_name)
                            dir_path_list.append(dir_path)

        return dir_path_list

    @staticmethod
    def get_all_directories_named(
        root_dir='C:\\', named_str='activate',
        black_list=['$RECYCLE.BIN', '$Recycle.Bin', '.git']
    ):
        dir_path_list = []
        if type(root_dir) == list:
            root_dir_list = root_dir
        else:
            root_dir_list = [root_dir]
        if type(named_str) == list:
            named_list = named_str
        else:
            named_list = [named_str]
        for root_dir in root_dir_list:
            for parent_directory, child_folders, _ in os.walk(root_dir):
                if all(map(lambda x: x not in parent_directory, black_list)):
                    for dir_name in child_folders:
                        named_bool = False
                        for named_str in named_list:
                            named_bool = named_bool or named_str == dir_name
                        if named_bool:
                            dir_path = osp.join(parent_directory, dir_name)
                            dir_path_list.append(dir_path)

        return dir_path_list

    # -------------------
    # Storage Functions
    # -------------------

    @staticmethod
    def attempt_to_pickle(
        df, pickle_path, raise_exception=False, verbose=True
    ):
        """
        Attempt to pickle a DataFrame to a file while handling potential
        exceptions.

        This method attempts to save a given DataFrame to the specified
        file path. It uses a pickle protocol of 4 or lower for broader
        compatibility. If the operation fails, it optionally raises an
        exception and/or prints an error message.

        Parameters:
            df (pandas.DataFrame):
                The DataFrame to pickle.
            pickle_path (str):
                The path to the pickle file.
            raise_exception (bool, optional):
                Whether to raise an exception if the pickle fails. Defaults
                to False.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            None

        Raises:
            Exception
                If `raise_exception` is True and pickling the DataFrame fails.

        Notes:
            This method attempts to pickle a DataFrame using the highest
            possible protocol supported by the installed version of Python,
            up to a maximum of 4. If pickling fails, it will remove the
            partially written file and, if `verbose` is True, print an error
            message. If `raise_exception` is True, it will re-raise the
            exception after cleaning up.
        """

        # Try to compress and store the dataframe with a pickle protocol <= 4
        try:

            # Print the absolute path to the pickle file if verbose
            if verbose:
                print(
                    'Pickling to {}'.format(osp.abspath(pickle_path)),
                    flush=True
                )

            df.to_pickle(
                pickle_path, protocol=min(4, pickle.HIGHEST_PROTOCOL)
            )

        # Catch any exception that occurs during the pickling process
        except Exception as e:

            # Remove the pickle file if it was partially created
            remove(pickle_path)

            # Print the exception message if verbose
            if verbose:
                cell_count = df.shape[0] * df.shape[1]
                print(
                    f"{e}: Couldn't save {cell_count:,} cells"  # noqa E231
                    " as a pickle.", flush=True
                )

            # Re-raise the exception if specified
            if raise_exception:
                raise

    def csv_exists(self, csv_name, folder_path=None, verbose=False):
        """
        Check if a CSV file exists in the specified folder or the default CSV
        folder.

        Parameters:
            csv_name (str):
                The name of the CSV file (with or without the '.csv'
                extension).
            folder_path (str, optional):
                The path to the folder containing the CSV file. If None, uses
                the default saves_csv_folder specified in the class.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            bool: True if the CSV file exists, False otherwise.
        """

        # Set folder path if not provided
        if folder_path is None:
            folder_path = self.saves_csv_folder

        # Construct the full path to the CSV file, including the .csv
        if csv_name.endswith('.csv'):
            csv_path = osp.join(folder_path, csv_name)
        else:
            csv_path = osp.join(folder_path, f'{csv_name}.csv')

        # Optionally print the absolute path to the CSV file (verbose)
        if verbose:
            print(osp.abspath(csv_path), flush=True)

        # Check if the CSV file exists
        return osp.isfile(csv_path)

    def load_csv(self, csv_name=None, folder_path=None):
        """
        Load a CSV file from the specified folder or the default CSV folder,
        returning the data as a pandas DataFrame.

        Parameters:
            csv_name (str, optional):
                The name of the CSV file (with or without the '.csv'
                extension). If None, loads the most recently modified CSV
                file in the specified or default folder.
            folder_path (str, optional):
                The path to the folder containing the CSV file. If None, uses
                the default data_csv_folder specified in the class.

        Returns:
            pandas.DataFrame:
                The data from the CSV file as a pandas DataFrame.
        """

        # Set folder path if not provided
        if folder_path is None:
            csv_folder = self.data_csv_folder
        elif folder_path.endswith('csv'):
            csv_folder = folder_path
        else:
            csv_folder = osp.join(folder_path, 'csv')

        # Is no specific CSV file named?
        if csv_name is None:

            # Use the most recently modified CSV file
            csv_path = max(
                [osp.join(csv_folder, f) for f in listdir(csv_folder)],
                key=osp.getmtime
            )

        # If a specific CSV file is named, construct the full path
        elif csv_name.endswith('.csv'):
            csv_path = osp.join(csv_folder, csv_name)

        else:
            csv_path = osp.join(csv_folder, f'{csv_name}.csv')

        # Load the CSV file as a df using the class-specific encoding
        data_frame = read_csv(
            osp.abspath(csv_path), encoding=self.encoding_type
        )

        return data_frame

    def pickle_exists(self, pickle_name):
        """
        Check if a pickle file exists.

        Parameters:
            pickle_name (str): The name of the pickle file.

        Returns:
            bool: True if the pickle file exists, False otherwise.
        """

        # Construct pickle path using pickle_name and saves_pickle_folder
        pickle_path = osp.join(
            self.saves_pickle_folder, '{}.pkl'.format(pickle_name)
        )

        # Return if the pickle file exists at the specified path
        return osp.isfile(pickle_path)

    def load_object(
        self, obj_name, pickle_path=None, download_url=None, verbose=False
    ):
        """
        Load an object from a pickle file, CSV file, or download it from a
        URL.

        Parameters:
            obj_name (str):
                The name of the object to load.
            pickle_path (str, optional):
                The path to the pickle file containing the object. Defaults
                to None. If None, the function attempts to construct the path
                based on the object name and the `saves_pickle_folder`
                attribute.
            download_url (str, optional):
                The URL to download the object from. Defaults to None. If no
                pickle file is found and a download URL is provided, the
                object will be downloaded and saved as a CSV file.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            object
                The loaded object.

        Raises:
            Exception
                If the object cannot be loaded from any source (pickle, CSV,
                or download).
        """

        # If no pickle path provided, construct default path with object name
        if pickle_path is None:
            pickle_path = osp.join(
                self.saves_pickle_folder, '{}.pkl'.format(obj_name)
            )

        # Check if the pickle file exists at the specified path
        if not osp.isfile(pickle_path):

            # If the pickle file doesn't exist and verbose, print a message
            if verbose:
                pp = osp.abspath(pickle_path)
                print(
                    f'No pickle exists at {pp} - attempting to load as csv.',
                    flush=True
                )

            # If pickle doesn't exist, try loading from CSV
            csv_path = osp.join(
                self.saves_csv_folder, '{}.csv'.format(obj_name)
            )

            # Check if the CSV file exists at the specified path
            if not osp.isfile(csv_path):
                if verbose:
                    cp = osp.abspath(csv_path)
                    print(
                        f'No csv exists at {cp} - attempting to download from'
                        ' URL.',
                        flush=True
                    )

                # Download object as CSV if URL provided and no CSV exists
                object = read_csv(
                    download_url, low_memory=False,
                    encoding=self.encoding_type
                )

            # If the CSV file exists, read the object from the CSV file
            else:

                # Load object from existing CSV file
                object = read_csv(
                    csv_path, low_memory=False, encoding=self.encoding_type
                )

            # If loaded object is a df, attempt to save it as a pickle
            if isinstance(object, DataFrame):
                self.attempt_to_pickle(
                    object, pickle_path, raise_exception=False
                )

            # Otherwise, pickle the object using the appropriate protocol
            else:
                with open(pickle_path, 'wb') as handle:
                    if sys.version_info.major == 2:
                        pickle.dump(object, handle, 2)
                    elif sys.version_info.major == 3:
                        pickle.dump(object, handle, pickle.HIGHEST_PROTOCOL)

        else:

            # If the pickle file exists, try to load the object
            try:
                object = read_pickle(pickle_path)

            # If reading the pickle file fails, fall back to pickle module
            except Exception:
                with open(pickle_path, 'rb') as handle:
                    object = pickle.load(handle)

        # If verbose, print a message indicating the object was loaded
        if verbose:
            print(
                'Loaded object {} from {}'.format(obj_name, pickle_path),
                flush=True
            )

        # Return the loaded object
        return object

    def load_data_frames(self, verbose=True, **kwargs):
        """
        Load Pandas DataFrames from pickle or CSV files, potentially
        switching between folders if necessary.

        Parameters:
            **kwargs (dict):
                Keyword arguments specifying the names of the data frames to
                load. The frame_name is used to construct file paths for
                loading DataFrames.

        Returns:
            dict:
                A dictionary where keys are frame_names and values are Pandas
                DataFrames.
        """
        frame_dict = {}  # Dictionary to store loaded DataFrames

        # Iterate over each frame_name provided in kwargs
        for frame_name in kwargs:
            was_successful = False

            # Attempt to load the data frame from a pickle file
            if not was_successful:
                pickle_path = osp.abspath(
                    osp.join(self.saves_pickle_folder, f'{frame_name}.pkl')
                )

                # If the pickle file exists, load it using load_object
                if osp.isfile(pickle_path):
                    if verbose:
                        print(
                            f'Attempting to load {pickle_path}.', flush=True
                        )
                    try:
                        frame_dict[frame_name] = self.load_object(frame_name)
                        was_successful = True
                    except Exception as e:
                        if verbose:
                            print(str(e).strip())
                        was_successful = False

            # If pickle file doesn't exist, check for CSV file with same name
            if not was_successful:
                csv_name = f'{frame_name}.csv'
                csv_path = osp.abspath(
                    osp.join(self.saves_csv_folder, csv_name)
                )

                # Does the CSV file exist in the saves folder?
                if osp.isfile(csv_path):
                    if verbose:
                        print(
                            f'No pickle exists for {frame_name} - attempting'
                            f' to load {csv_path}.', flush=True
                        )

                    # load it from there
                    try:
                        frame_dict[frame_name] = self.load_csv(
                            csv_name=frame_name,
                            folder_path=self.saves_folder
                        )
                        was_successful = True
                    except Exception as e:
                        if verbose:
                            print(str(e).strip())
                        was_successful = False

            # Does the CSV file not exist in the saves folder?
            if not was_successful:
                csv_path = osp.abspath(
                    osp.join(self.data_csv_folder, csv_name)
                )

                # Does the CSV file exist in the data folder?
                if osp.isfile(csv_path):
                    if verbose:
                        print(
                            f'No csv exists for {frame_name} -'
                            f' trying {csv_path}.', flush=True
                        )

                    # load it from there
                    try:
                        frame_dict[frame_name] = self.load_csv(
                            csv_name=frame_name
                        )
                        was_successful = True
                    except Exception as e:
                        if verbose:
                            print(str(e).strip())
                        was_successful = False

            # Does the CSV file not exist anywhere?
            if not was_successful:
                if verbose:
                    print(
                        f'No csv exists for {frame_name} - just forget it.',
                        flush=True
                    )

                # Skip loading this
                frame_dict[frame_name] = None

        return frame_dict

    def save_data_frames(self, include_index=False, verbose=True, **kwargs):
        """
        Save data frames to CSV files.

        Parameters:
            include_index: Whether to include the index in the CSV files.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
            **kwargs:
                A dictionary of data frames to save. The keys of the
                dictionary are the names of the CSV files to save the data
                frames to.

        Returns:
            None
        """

        # Iterate over dfs in kwargs dictionary and save them to CSV files
        for frame_name in kwargs:

            # Check if it's a dataframe
            if isinstance(kwargs[frame_name], DataFrame):

                # Generate the path to the CSV file
                csv_path = osp.join(
                    self.saves_csv_folder, '{}.csv'.format(frame_name)
                )

                # Print a message about the saved file if verbose is True
                if verbose:
                    print(
                        'Saving to {}'.format(osp.abspath(csv_path)),
                        flush=True
                    )

                # Save the data frame to a CSV file
                kwargs[frame_name].to_csv(
                    csv_path, sep=',', encoding=self.encoding_type,
                    index=include_index
                )

    def store_objects(self, verbose=True, **kwargs):
        """
        Store objects to pickle files.

        This function iterates through keyword arguments (**kwargs) and
        attempts to pickle each provided object to a separate file. The
        keys in the dictionary are used as filenames (with a `.pkl`
        extension) within the `self.saves_pickle_folder` directory.

        Parameters:
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.
            **kwargs (dict):
                The objects to store. The keys of the dictionary are the
                names of the objects, and the values are the objects
                themselves.

        Returns:
            None
        """

        # Iterate over each object in kwargs
        for obj_name in kwargs:

            # Construct the path where the object will be pickled
            pickle_path = osp.join(
                self.saves_pickle_folder, '{}.pkl'.format(obj_name)
            )

            # If the object is a dataframe, use the attempt_to_pickle method
            if isinstance(kwargs[obj_name], DataFrame):
                self.attempt_to_pickle(
                    kwargs[obj_name], pickle_path, raise_exception=False,
                    verbose=verbose
                )

            else:

                # For non-df objects, print a message if verbose
                if verbose:
                    print(
                        'Pickling to {}'.format(osp.abspath(pickle_path)),
                        flush=True
                    )

                # Open the pickle file for writing
                with open(pickle_path, 'wb') as handle:

                    # If the Python version is 2, use protocol 2
                    if sys.version_info.major == 2:
                        pickle.dump(kwargs[obj_name], handle, 2)

                    # If the version is 3, use the highest protocol up to 4
                    elif sys.version_info.major == 3:
                        pickle.dump(
                            kwargs[obj_name], handle,
                            min(4, pickle.HIGHEST_PROTOCOL)
                        )

    # -------------------
    # Module Functions
    # -------------------

    def get_random_function(self, py_folder, verbose=True):
        function_objs_list = []
        while not function_objs_list:
            while True:
                try:
                    random_py_file = self.get_random_py_file(py_folder)
                    library_name = osp.relpath(
                        random_py_file, py_folder
                    ).replace('.py', '')
                    import_call = 'import ' + library_name
                    if verbose:
                        print(import_call)
                    exec(import_call)
                    break
                except (SyntaxError, ImportError, ValueError):
                    # self.open_path_in_notepad(random_py_file)
                    pass
            possible_attributes = [
                f'{library_name}.{fn}'
                for fn in dir(eval(library_name))
                if not fn.startswith('_')
            ]
            if verbose:
                print(possible_attributes)
            utils_list = [
                f'{library_name}.{fn}'
                for fn in self.get_utility_file_functions(
                    util_path=random_py_file
                )
                if not fn.startswith('_')
            ]
            if verbose:
                print(utils_list)
            possible_functions = sorted(
                set(possible_attributes).intersection(set(utils_list))
            )
            if verbose:
                print(possible_functions)
            for possible_function in possible_functions:
                try:
                    if verbose:
                        print(possible_function)
                    function_obj = eval(possible_function)
                    if callable(function_obj):
                        function_objs_list.append(function_obj)
                except Exception:
                    continue
        import random
        random_function = random.choice(function_objs_list)

        return (random_py_file, random_function)

    def get_evaluations(self, obj):
        """
        Evaluate an object using a list of evaluator functions and return a
        list of matching evaluations.

        Args:
            obj: The object to be evaluated.

        Returns:
            list:
                A list of evaluator names (without the 'is' prefix) that
                return True for the given object.
        """

        # Initialize a list of evaluations
        evaluations_list = []

        # Loop through each of inspect's evaluators
        for evaluator in self.object_evaluators:

            # Attempt to evaluate a specific evaluator
            try:

                # Pass 'inspect' explicitly into eval's context
                evaluation = eval(
                    f'inspect.{evaluator}(obj)',
                    {'inspect': inspect, 'obj': obj}
                )

                # Does it evaluate?
                if evaluation:

                    # Remove the 'is' prefix and add it to the list
                    evaluations_list.append(evaluator[2:])

            # Ignore evaluations that don't work
            except Exception:
                continue

        # Return the list of evaluations
        return evaluations_list

    def get_library_names(
        self, module_obj, import_call, verbose=False
    ):
        library_names_list = []
        try:
            exec(import_call)  # Execute the import statement
        except ImportError:
            pass  # Ignore import errors and continue
        if verbose:
            pass

        # Is the module obj just a string?
        dir_list = []
        if isinstance(module_obj, str):

            # Create the dir list using eval
            try:
                dir_list = eval(f'dir({module_obj})')
            except AttributeError:
                pass  # Ignore attribute errors and continue

        # Otherwise, create the dir list from the object
        else:
            dir_list = dir(module_obj)

        # Iterate over the attributes of the module
        for library_name in dir_list:
            if verbose:
                print(f'library_name: {library_name}')

            # Skip standard modules
            if library_name in self.standard_lib_modules:
                if verbose:
                    print(f'{library_name} is in the standard modules')
                continue

            # Skip built-in modules
            if library_name in sys.builtin_module_names:
                if verbose:
                    print(f'{library_name} is in the built-in modules')
                continue

            # Skip double underscore-prefixed attributes
            if library_name.startswith('__'):
                if verbose:
                    print(f'{library_name} has a double underscore-prefix')
                continue

            # Add what's left to the library names list
            library_names_list.append(library_name)

        # Return the list of libraries
        return library_names_list

    def get_dir_tree(
        self, module_name, function_calls=[], contains_str=None,
        not_contains_str=None, recurse_classes=True, recurse_modules=False,
        import_call=None, level=4, verbose=False
    ):
        """
        Introspect a Python module to discover available functions and
        classes programmatically.

        Parameters:
            module_name : str
                The name of the module to inspect.
            function_calls : list, optional
                A list to accumulate found attributes (default is an empty
                list).
            contains_str : str, optional
                If provided, only include attributes containing this
                substring (case-insensitive).
            not_contains_str : str, optional
                If provided, exclude attributes containing this substring
                (case-insensitive).
            recurse_classes : bool, optional
                Whether to recursively inspect classes (default is True).
            recurse_modules : bool, optional
                Whether to recursively inspect modules (default is False).
            import_call : str, optional
                The import statement to execute for the module (default is
                None).
            verbose : bool, optional
                If True, print debug or status messages (default is False).

        Returns:
            list[str]
                A sorted list of attributes in the module that match the
                filtering criteria.

        Example:
            module_name = 'nu'
            import_call = '''
            from notebook_utils import NotebookUtilities
            nu = NotebookUtilities(
                data_folder_path=osp.abspath(osp.join(os.pardir, 'data')),
                saves_folder_path=osp.abspath(osp.join(os.pardir, 'saves'))
            )'''
            nu_functions = nu.get_dir_tree(
                module_name, function_calls=[], contains_str='_regex',
                import_call=import_call, recurse_modules=True, level=3,
                verbose=False
            )
            sorted(nu_functions, key=lambda x: x[::-1])[:6]

        Notes:
            This function dynamically imports the specified module and
            retrieves its attributes, filtering them based on the provided
            criteria. It can also recursively explore classes and modules if
            specified.
        """

        # Base case: Stop recursion when level reaches 0
        if level == 0:
            return []

        # Try to get the module object by first importing it
        if import_call is None:
            import_call = 'import ' + module_name.split('.')[0]
        if verbose:
            print(f'import_call: {import_call}')

        # Dynamically import the module
        try:
            exec(import_call)  # Execute the import statement
        except ImportError:
            pass  # Ignore import errors and continue

        # Filter out skippable attributes of the module
        library_names_list = self.get_library_names(
            module_name, import_call, verbose=False
        )

        # Are there no library names left?
        if not library_names_list:

            # Get a better representation of the module and try again
            try:
                module_obj = inspect.getmodule(eval(module_name))
                if verbose:
                    print(f'module_obj: {module_obj}')
                library_names_list = self.get_library_names(
                    module_obj, import_call, verbose=verbose
                )
            except AttributeError:
                pass  # Ignore attribute errors and continue

        # Iterate over the library names list
        for library_name in library_names_list:

            # Construct the full attribute name
            function_call = f'{module_name}.{library_name}'

            # Evaluate the function or class
            try:
                function_obj = eval(function_call)
            except Exception:
                function_obj = None

            # Get evaluations of the object from the inspect library
            evaluations_list = self.get_evaluations(function_obj)

            # Are there no evaluations?
            if not evaluations_list:

                # Get a better representation of the function and try again
                module_obj = inspect.getmodule(function_obj)
                if verbose:
                    print(f'module_obj: {module_obj}')
                evaluations_list = self.get_evaluations(module_obj)

            if evaluations_list:
                function_calls.append(function_call)

            if verbose:
                print(
                    f'function_call: {function_call},'  # noqa: E231
                    f' evaluations_list: {evaluations_list}'
                )

            # Recursively explore classes if specified
            if recurse_classes and 'class' in evaluations_list:
                function_calls = self.get_dir_tree(
                    module_name=function_call, function_calls=function_calls,
                    recurse_classes=recurse_classes,
                    recurse_modules=recurse_modules,
                    import_call=import_call, level=level - 1, verbose=verbose
                )
                continue

            # Recursively explore modules if specified
            elif recurse_modules and 'module' in evaluations_list:
                function_calls = self.get_dir_tree(
                    module_name=function_call, function_calls=function_calls,
                    recurse_classes=recurse_classes,
                    recurse_modules=recurse_modules,
                    import_call=import_call, level=level - 1, verbose=verbose
                )
                continue

        # Apply filtering criteria if provided
        if contains_str:
            function_calls = [
                fn for fn in function_calls if contains_str in fn.lower()
            ]
        if not_contains_str:
            function_calls = [
                fn
                for fn in function_calls
                if not_contains_str not in fn.lower()
            ]

        # Return a sorted list of unique function calls
        return sorted(set(function_calls))

    def add_staticmethod_decorations(
        self, python_folder=osp.join(os.pardir, 'py'), verbose=True
    ):
        """
        Scan a Python folder structure and automatically add static method
        decorators to non-static method-decorated instance methods.

        This method searches through all Python files in a specified
        folder, excluding certain blacklisted directories, and refactors
        functions that do not reference the 'self' variable to be static
        methods by adding the static method decorator.

        Parameters:
            python_folder (str, optional):
                Relative path to the folder to scan for Python files
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            None

        Note:
            This function modifies files in-place. It's recommended to back
            up the folder structure before running it.
        """

        # Create a list of directories to exclude from the search
        black_list = ['.ipynb_checkpoints', '$Recycle.Bin']

        # Print starting message if verbose
        if verbose:
            print(
                'Scanning Python folder structure to add staticmethod'
                ' decorations'
            )

        # Helper funtion
        def f():

            # Open the file and read its contents
            with open(file_path, 'r', encoding=self.encoding_type) as f:
                file_text = f.read()

            # Split the file text into function parts
            fn_parts_list = self.instance_defs_regex.split(file_text)

            # Iterate over function names and bodies
            for fn_name, fn_body in zip(
                fn_parts_list[1::2], fn_parts_list[2::2]
            ):

                # Check if the function body does not use 'self'
                if not self.self_regex.search(fn_body):

                    # Create a new regex specific to the method name
                    s = f'^    def {fn_name}\\(\\s*self'  # noqa: E272, E222
                    s += ',\\s+(?:[^\\)]+)\\):'  # noqa: E231
                    instance_def_regex = re.compile(s, re.MULTILINE)

                    # Search for the method definition in the file text
                    match_obj = instance_def_regex.search(file_text)

                    # Update file text if method is not decorated
                    if match_obj:
                        replaced_str = match_obj.group()

                        # Prepare str with static method decorator
                        replacing_str = '    @staticmethod\n'
                        replacing_str += replaced_str.replace('self, ', '')

                        # Replace original method def with refactored one
                        file_text = file_text.replace(
                            replaced_str, replacing_str
                        )

            # Write the modified text back to the file
            with open(file_path, 'w', encoding=self.encoding_type) as f:
                print(file_text.rstrip(), file=f)

        # Walk through the directory tree
        for sub_directory, directories_list, files_list in walk(
            python_folder
        ):

            # Skip blacklisted directories
            if all(map(lambda x: x not in sub_directory, black_list)):

                # Process each Python file in the directory
                for file_name in files_list:
                    if file_name.endswith('.py'):

                        # Construct the full path to the Python file
                        file_path = osp.join(sub_directory, file_name)
                        try:
                            f()

                        # Handle any exceptions during file read/write
                        except Exception as e:
                            print(
                                f'{e.__class__.__name__} error trying to'
                                f' read {file_name}: {str(e).strip()}'
                            )

    def update_modules_list(self, modules_list=None, verbose=False):
        """
        Update the list of modules that are installed.

        Parameters:
            modules_list (list of str, optional):
                The list of modules to update. If None, the list of installed
                modules will be used. Defaults to None.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            None
        """

        # Create the modules list with pip if not supplied
        if modules_list is None:
            self.modules_list = [
                o.decode().split(' ')[0]
                for o in subprocess.check_output(
                    f'{self.pip_command_str} list'.split(' ')
                ).splitlines()[2:]
                ]

        # Set the class variable if it is
        else:
            self.modules_list = modules_list

        if verbose:
            print(
                'Updated modules list to {}'.format(self.modules_list),
                flush=True
            )

    def ensure_module_installed(
        self, module_name, upgrade=False, update_first=False, verbose=True
    ):
        """
        Ensure a Python module is installed, upgrading it and/or updating the
        modules list first if specified.

        This method checks if a specified Python module is in the
        `modules_list`. If the module is not found, it installs the module
        using pip, with options to upgrade the module to the latest
        version, update the modules list before checking, and control the
        verbosity of the output during the installation process.

        Parameters:
            module_name (str):
                The name of the Python module to check and potentially
                install.
            upgrade (bool, optional):
                Whether to upgrade the module to the latest version if it is
                already installed. Defaults to False.
            update_first (bool, optional):
                Whether to update the modules list before checking if the
                module is installed. Defaults to False.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to True.

        Returns:
            None
        """

        # Update the internal list of modules if requested
        if update_first:
            self.update_modules_list(verbose=False)

        # Check if the module is not in the current list of installed modules
        if module_name not in self.modules_list:

            # Construct the pip command string for installation/upgrade
            command_str = f'{self.pip_command_str} install {module_name}'

            # Append the upgrade flag to the command if needed
            if upgrade:
                command_str += ' --upgrade'

            # Print the command if verbose, otherwise add quiet flag
            if verbose:
                print(command_str, flush=True)
            else:
                command_str += ' --quiet'

            # Execute the pip command and capture the output
            output_str = subprocess.check_output(command_str.split(' '))

            # Print the output if status messages requested
            if verbose:
                for line_str in output_str.splitlines():
                    print(line_str.decode(), flush=True)

            # Update the internal list of installed modules
            self.update_modules_list(verbose=False)

    def extract_comments(self, function_obj, verbose=False):
        """
        Extract all comments from the source code of a given function along
        with their correct indentation.

        Parameters:
            function_obj (callable):
                The function whose comments need to be extracted.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            list:
                A list of tuples containing (indentation, comment) for each
                comment.
        """

        # Get the source code of the function
        try:
            source_code = inspect.getsource(function_obj)
        except OSError:
            raise ValueError(
                'Cannot retrieve source code for function:'
                f" {function_obj.__name__}"
            )

        # Initialize the comment tuples list
        comment_tuples = []

        # Split code into lines to retrieve the indentation
        code_lines = source_code.splitlines()
        from io import StringIO
        code_io = StringIO(source_code)

        # Tokenize the source code
        import tokenize
        for token in tokenize.generate_tokens(code_io.readline):

            # Is the token a comment?
            if token.type == tokenize.COMMENT:

                # Extract comment text without the pound sign
                comment = token.string.lstrip('#').strip()

                # Get the line number of the comment
                line_number = token.start[0] - 1

                # Get the column number where the comment starts
                column_number = token.start[1]

                # Is the indentation a standalone comment?
                if (
                    column_number == 0
                    or code_lines[line_number].lstrip().startswith('#')
                ):

                    # Indentation is the column number of the comment
                    indentation = column_number

                # Is the indentation an inline comment?
                else:

                    # Indentation matches the code before the comment
                    indentation = (
                        len(code_lines[line_number])
                        - len(code_lines[line_number].lstrip())
                    )

                comment_tuple = (indentation, comment)
                comment_tuples.append(comment_tuple)

        return comment_tuples

    def get_function_in_its_class(
        self, function_obj, verbose=False
    ):
        """
        Retrieve the source code and docstring of a function defined in a
        class.

        Parameters
        ----------
        function_obj : function
            The function object for which the source code and docstring
            are to be retrieved.
        verbose : bool, optional, default=False
            If True, prints additional debug information.

        Returns
        -------
        tuple
            A tuple containing the source code of the function (str) and
            the docstring of the function (str).

        Raises
        ------
        ValueError
            If the function's source code or module cannot be found.

        Notes
        -----
        - This method works by inspecting the module where the function is
          defined.
        - It searches for the function inside all classes within the module.
        - If the function is a static or class method, it extracts the
          underlying function object before retrieving its source code and
          docstring.
        """

        # Initialize the function's source code and docstring variables
        function_source = None
        docstring = None

        # Get the name of the function
        function_name = function_obj.__name__

        # Get the module where the function is defined
        module = inspect.getmodule(function_obj)

        # Retrieve the source code for the entire module
        if module is not None:

            # Loop through the module's classes, methods, objects, et al
            for obj in module.__dict__.values():

                # Is this object a class?
                if inspect.isclass(obj):

                    # Check if the class has the function
                    if hasattr(obj, function_name):
                        func = getattr(obj, function_name)

                        # Handle static methods and class methods
                        if isinstance(func, (staticmethod, classmethod)):

                            # Extract the actual function
                            func = func.__func__

                        # Ensure the object is a function or method
                        if (
                            inspect.isfunction(func)
                            or inspect.ismethod(func)
                        ):

                            # Get the source code for the method
                            function_source = inspect.getsource(func)

                            # Get the docstring for the method
                            docstring = inspect.getdoc(func)

                            # Exit the loop once the function is found
                            break

            # Raise an error if the function's source could not be found
            if function_source is None:
                raise ValueError(
                    'Could not find source for function'
                    f" '{function_name}'."
                )

        # Raise an error if the module could not be retrieved
        else:
            raise ValueError(
                f"Could not retrieve the module for '{function_name}'."
            )

        # Return the source code and docstring as a tuple
        return function_source, docstring

    def describe_procedure(
        self, function_obj, docstring_prefix='The procedure to', verbose=False
    ):
        """
        Generate a step-by-step description of how to perform a function by
        hand from its comments.

        This static method analyzes the source code comments of a provided
        function object `function_obj` to extract a step-by-step description
        of the procedure it implements. It prints a list containing the
        formatted description.

        Parameters:
            function_obj (callable):
                The function object whose procedure needs to be described.
            docstring_prefix (str, optional):
                A prefix to prepend to the procedure description. Default is
                'The procedure to'.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            None

        Notes:
            The function assumes that the docstring starts with a
            one-sentence paragraph where the first word is a verb (in the
            imperative form, as opposed to the third-person singular present
            tense). The function also assumes that comments don't end with
            punctuation and will ignore comments containing the word
            'verbose'.
        """

        # Ensure the function object is some kind of function or method
        assert inspect.isroutine(function_obj), (
            "The function object must be some kind of function or method"
        )

        # Attempt to retrieve the source code directly
        try:

            # Get the source code for the function or method
            function_source = inspect.getsource(function_obj)

            # Get the docstring for the function or method
            docstring = inspect.getdoc(function_obj)

        # Or, fall back to finding the function in its class or module
        except OSError:
            function_source, docstring = self.get_function_in_its_class(
                function_obj, verbose=verbose
            )

        # Start with a description header (including prefix)
        docstring = re.sub(
            '\\s+', ' ', docstring.strip().split('.')[0]
        )
        docstring_suffix = 'is as follows:'  # noqa E231
        comments_list = [(
            0, f'{docstring_prefix} {docstring.lower()} {docstring_suffix}'
        )]

        # Extract the comments
        for comment_tuple in self.extract_comments(
            function_obj, verbose=verbose
        ):

            # Ignore any debug or QA statements
            if any(map(lambda x: x in comment_tuple[1], ['verbose', 'noqa'])):
                continue

            comments_list.append(comment_tuple)

        # If there are any comments in the list
        if len(comments_list) > 1:
            comments_list = self.apply_multilevel_numbering(
                comments_list,
                level_map={
                    0: "", 4: "I. ", 8: "A. ", 12: "1. ", 16: "a. ",
                    20: "I) ", 24: "A) ", 28: "1) ", 32: "a) "
                }, add_indent_back_in=True, verbose=verbose
            )

            # Print its procedure description and comments on their own lines
            print('\n'.join(comments_list))

    # -------------------
    # URL and Soup Functions
    # -------------------

    @staticmethod
    def get_filename_from_url(url, verbose=False):
        """
        Extract the filename from a given URL.

        Parameters:
            url (str): The URL from which to extract the filename.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            str: The extracted filename from the URL.
        """

        # Parse the URL and extract the filename from the path
        file_name = urllib.parse.urlparse(url).path.split('/')[-1]

        # Print verbose information if verbose flag is True
        if verbose:
            print(f"Extracted filename from '{url}': '{file_name}'")

        return file_name

    def get_style_column(self, tag_obj, verbose=False):
        """
        Extract the style column from a given Wikipedia infobox
        BeautifulSoup'd tag object and return the style column tag object.

        Parameters:
            tag_obj (bs4.element.Tag):
                The BeautifulSoup tag object to extract the style column from.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            bs4.element.Tag
                The modified BeautifulSoup tag object representing the style
                column.
        """

        # Display the initial tag object if verbose is True
        if verbose:
            display(tag_obj)

        # Get the parent td tag object (table tag object)
        tag_obj = self.get_td_parent(tag_obj, verbose=verbose)
        if verbose:
            display(tag_obj)

        # Traverse siblings of tag backward until a style column is found
        from bs4.element import NavigableString
        while isinstance(
            tag_obj, NavigableString
        ) or not tag_obj.has_attr('style'):
            tag_obj = tag_obj.previous_sibling
            if verbose:
                display(tag_obj)

        # Display text content of found style column if verbose
        if verbose:
            display(tag_obj.text.strip())

        # Return the style column tag object
        return tag_obj

    @staticmethod
    def get_td_parent(tag_obj, verbose=False):
        """
        Find and return the closest ancestor of the given BeautifulSoup tag
        object that is a 'td' tag.

        Parameters:
            tag_obj (bs4.element.Tag):
                The BeautifulSoup tag object whose 'td' ancestor needs to be
                found.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            bs4.element.Tag: The closest 'td' ancestor tag object.
        """
        if verbose:
            display(tag_obj)

        # Traverse the parent tags upward until a table cell (<td>) is found
        while (tag_obj.name != 'td'):
            tag_obj = tag_obj.parent
            if verbose:
                display(tag_obj)

        # Return the closest 'td' ancestor tag object
        return tag_obj

    def download_file(
        self, url, download_dir=None, exist_ok=False, verbose=False
    ):
        """
        Download a file from the internet.

        Parameters:
            url: The URL of the file to download.
            download_dir:
                The directory to download the file to. If None, the file will
                be downloaded to the `downloads` subdirectory of the data
                folder.
            exist_ok:
                If True, the function will not raise an error if the file
                already exists.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            The path to the downloaded file.
        """

        # Get the file name from the URL
        file_name = self.get_filename_from_url(url, verbose=verbose)

        # Use the downloads subdirectory if download_dir isn't specified
        if download_dir is None:
            download_dir = osp.join(self.data_folder, 'downloads')

        # Create the download directory if it does not exist
        makedirs(download_dir, exist_ok=True)

        # Compute the path to the downloaded file
        file_path = osp.join(download_dir, file_name)

        # If the file does not exist or if exist_ok is True, download the file
        if exist_ok or not osp.isfile(file_path):
            from urllib.request import urlretrieve
            urlretrieve(url, file_path)

        return file_path

    def get_page_soup(self, page_url_or_filepath, driver=None, verbose=False):
        """
        Get the BeautifulSoup soup object for a given page URL or filepath.

        Parameters:
            page_url_or_filepath (str):
                The URL or filepath of the page to get the soup object for.
            driver (selenium.webdriver, optional):
                Whether to get the page source from the Selenium webpage.
                Defaults to None.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            BeautifulSoup: The BeautifulSoup soup object for the given page.
        """

        # If the page URL or filepath is a URL, get the page HTML
        if self.url_regex.fullmatch(page_url_or_filepath):
            if driver is None:
                with urllib.request.urlopen(
                    page_url_or_filepath
                ) as response:
                    page_html = response.read()
            else:
                page_html = driver.page_source

        # If it's a file path, ensure it exists and get the page HTML that way
        elif self.filepath_regex.fullmatch(page_url_or_filepath):
            assert osp.isfile(
                page_url_or_filepath
            ), f"{page_url_or_filepath} doesn't exist"
            with open(
                page_url_or_filepath, 'r', encoding=self.encoding_type
            ) as f:
                page_html = f.read()

        # If the string is already in the format we want, it IS the page HTML
        else:
            page_html = page_url_or_filepath

        # Parse the page HTML using BeautifulSoup
        page_soup = bs(page_html, 'html.parser')

        # If verbose output is enabled, print the page URL or filepath
        if verbose:
            print(f'Getting soup object for: {page_url_or_filepath}')

        # Return the page soup object
        return page_soup

    def get_page_tables(self, tables_url_or_filepath, verbose=True):
        """
        Retrieve tables from a given URL or file path and return a list of
        DataFrames.

        Parameters:
            tables_url_or_filepath (str):
                The URL or file path of the page containing tables.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            List[pandas.DataFrame]:
                A list of DataFrames containing tables from the specified
                source.

        Example:
            tables_url = 'https://en.wikipedia.org/wiki/'
            tables_url += 'Provinces_of_Afghanistan'
            page_tables_list = nu.get_page_tables(tables_url, verbose=True)

        """

        # Check if the input is a URL or a filepath
        if self.url_regex.fullmatch(
            tables_url_or_filepath
        ) or self.filepath_regex.fullmatch(tables_url_or_filepath):

            # If it's a filepath, check if the file exists
            if self.filepath_regex.fullmatch(tables_url_or_filepath):
                assert osp.isfile(
                    tables_url_or_filepath
                ), f"{tables_url_or_filepath} doesn't exist"

            # Read tables from the URL or file path
            tables_df_list = read_html(tables_url_or_filepath)
        else:

            # If it's not a URL or filepath, assume it's a str
            from io import StringIO

            # Create a StringIO object from the string
            f = StringIO(tables_url_or_filepath)

            # Read the tables from the StringIO object
            tables_df_list = read_html(f)

        # Print a summary of the tables if verbose is True
        if verbose:
            print(sorted(
                [(i, df.shape) for (i, df) in enumerate(tables_df_list)],
                key=lambda x: x[1][0]*x[1][1], reverse=True
            ))

        # Return the list of pandas DataFrames containing the tables
        return tables_df_list

    def get_wiki_tables(self, tables_url_or_filepath, verbose=True):
        """
        Get a list of DataFrames from Wikipedia tables.

        Parameters:
            tables_url_or_filepath:
                The URL or filepath to the Wikipedia page containing the
                tables.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to True.

        Returns:
            A list of DataFrames containing the data from the Wikipedia
            tables.

        Raises:
            Exception:
                If there is an error getting the Wikipedia page or the tables
                from the page.
        """
        table_dfs_list = []
        try:

            # Get the BeautifulSoup object for the Wikipedia page
            page_soup = self.get_page_soup(
                tables_url_or_filepath, verbose=verbose
            )

            # Find all the tables on the Wikipedia page
            table_soups_list = page_soup.find_all(
                'table', attrs={'class': 'wikitable'}
            )

            # Recursively get the dfs for all tables on the page
            table_dfs_list = []
            for table_soup in table_soups_list:
                table_dfs_list += self.get_page_tables(
                    str(table_soup), verbose=False
                )

            # Print sorted list of tables by their size if verbose
            if verbose:
                print(sorted([(i, df.shape) for i, df in enumerate(
                    table_dfs_list
                )], key=lambda x: x[1][0] * x[1][1], reverse=True))

        except Exception as e:

            # If verbose, print the error message
            if verbose:
                print(str(e).strip())

            # Recursively get dfs for tables on  page again, but verbose=False
            table_dfs_list = self.get_page_tables(
                tables_url_or_filepath, verbose=False
            )

        # Return the list of DataFrames
        return table_dfs_list

    def get_wiki_infobox_data_frame(self, page_titles_list, verbose=True):
        """
        Get a DataFrame of key/value pairs from the infobox of Wikipedia
        biographical entries.

        This function retrieves the infobox data from Wikipedia pages and
        constructs a DataFrame where each row corresponds to a page and
        each column corresponds to an infobox attribute.

        Parameters:
            page_titles_list (list of str):
                A list of titles of the Wikipedia pages containing the
                infoboxes.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to True.

        Returns:
            pandas.DataFrame
                A DataFrame containing the extracted infobox data with page
                titles as the first column and infobox labels/values as
                separate columns. Returns an empty DataFrame if no data is
                found.

        Note:
            - This function assumes a specific infobox structure and may
            require adjustments for different Wikipedia page formats.
            - It is assumed that the infobox contains no headers which would
            prefix any duplicate labels.
        """

        # Import necessary modules not already imported in the class
        import wikipedia

        # Initialize an empty list to store rows of data
        rows_list = []

        # Define a helper function to clean text
        def clean_text(parent_soup, verbose=False):

            # Initialize a list to store text from child elements
            texts_list = []
            for child_soup in parent_soup.children:
                if '{' not in child_soup.text:

                    # Add stripped text to the list if it doesn't contain '{'
                    texts_list.append(child_soup.text.strip())

            # Join the list of texts into a single string
            parent_text = ' '.join(texts_list)

            # Trim various enclosing parentheses/brackets of space
            for this, with_that in zip(
                [' )', ' ]', '( ', '[ '], [')', ']', '(', '[']
            ):
                parent_text = parent_text.replace(this, with_that)

            # Remove extra whitespace characters and non-breaking spaces
            parent_text = sub('[\\s\\u200b\\xa0]+', ' ', parent_text).strip()

            return parent_text

        # Use a progress bar if verbose is True
        if verbose:
            from tqdm import tqdm_notebook as tqdm
            page_titles_list = tqdm(page_titles_list)

        # Iterate over each page title in the list
        for page_title in page_titles_list:

            # Initialize a dictionary to store the data for the current page
            row_dict = {'page_title': page_title}
            try:

                # Retrieve the Wikipedia page object
                bio_obj = wikipedia.WikipediaPage(title=page_title)

                # Get the HTML content of the page
                bio_html = bio_obj.html()

                # Parse the HTML content using BeautifulSoup
                page_soup = bs(bio_html, 'html.parser')

                # Find all infobox tables with specific class attributes
                infobox_soups_list = page_soup.find_all(
                    'table', attrs={'class': 'infobox'}
                )

                # Initialize a list to store labels
                labels_list = []

                # Iterate over each infobox table
                for infobox_soup in infobox_soups_list:

                    # Find label elements within the infobox
                    label_soups_list = infobox_soup.find_all('th', attrs={
                        'scope': 'row', 'class': 'infobox-label',
                        'colspan': False
                    })

                    # Iterate over each label cell
                    for infobox_label_soup in label_soups_list:

                        # Clean and format label text, ensuring unique keys
                        key = self.lower_ascii_regex.sub(
                            '_', clean_text(infobox_label_soup).lower()
                        ).strip('_')
                        if key and (key not in labels_list):

                            # Add the label if it's not a duplicate
                            labels_list.append(key)

                            # Find the corresponding value cell
                            value_soup = infobox_label_soup.find_next(
                                'td', attrs={'class': 'infobox-data'}
                            )

                            # Add the key-value pair to the dictionary
                            row_dict[key] = clean_text(value_soup)
            except Exception as e:

                # Continue to the next page if an error occurs
                if verbose:
                    print(
                        f'{e.__class__.__name__} error processing'
                        f' {page_title}: {str(e).strip()}'
                    )
                continue

            # Add the dictionary to the list of rows
            rows_list.append(row_dict)

        # Create a dataframe from the list of rows
        df = DataFrame(rows_list)

        # Return the dataframe
        return df

    # -------------------
    # Pandas Functions
    # -------------------

    @staticmethod
    def get_inf_nan_mask(X_train, y_train):
        """
        Return a mask indicating which elements of X_train and y_train are not
        inf or nan.

        Parameters:
            X_train: A NumPy array of numbers.
            y_train: A NumPy array of numbers.

        Returns:
            A numpy array of booleans, where True indicates that the
            corresponding element of X_train and y_train is not inf or nan.
            This combined mask can be used on both X_train and y_train.

        Example:
            inf_nan_mask = nu.get_inf_nan_mask(X_train, y_train)
            X_train_filtered = X_train[inf_nan_mask]
            y_train_filtered = y_train[inf_nan_mask]
        """

        # Check if the input lists are empty
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            return np.array([], dtype=bool)

        # Create a notnull mask across the X_train and y_train columns
        mask_series = concat(
            [DataFrame(y_train), DataFrame(X_train)], axis='columns'
        ).map(notnull).all(axis='columns')

        # Return the mask indicating not inf or nan
        return mask_series

    @staticmethod
    def get_column_descriptions(df, analysis_columns=None, verbose=False):
        """
        Generate a DataFrame containing descriptive statistics for specified
        columns in a given DataFrame.

        Parameters:
            df (pandas.DataFrame):
                The DataFrame to analyze.
            analysis_columns (list of str, optional):
                A list of specific columns to analyze. If None, all columns
                will be analyzed. Defaults to None.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            pandas.DataFrame:
                A DataFrame containing the descriptive statistics of the
                analyzed columns.
        """

        # If analysis_columns not provided, use all columns in the df
        if analysis_columns is None:
            analysis_columns = df.columns

        # Convert the CategoricalDtype instances to strings, then group
        grouped_columns = df.columns.to_series().groupby(
            df.dtypes.astype(str)
        ).groups

        # Initialize an empty list to store the descriptive statistics rows
        rows_list = []

        # Iterate over each data type and its corresponding column group
        for dtype, dtype_column_list in grouped_columns.items():
            for column_name in dtype_column_list:

                # Check if the column is in the analysis columns
                if column_name in analysis_columns:

                    # Create a boolean mask for null values in the column
                    null_mask_series = df[column_name].isnull()
                    filtered_df = df[~null_mask_series]

                    # Create a row dictionary to store the column description
                    row_dict = {
                        'column_name': column_name,
                        'dtype': str(dtype),
                        'count_blanks': df[column_name].isnull().sum()
                    }

                    # Count unique values in the column
                    try:
                        row_dict['count_uniques'] = df[column_name].nunique()

                    # Set count of unique values to NaN if an error occurs
                    except Exception:
                        row_dict['count_uniques'] = np.nan

                    # Count the number of zeros
                    try:
                        row_dict['count_zeroes'] = int((
                            df[column_name] == 0
                        ).sum())

                    # Set count of zeros to NaN if an error occurs
                    except Exception:
                        row_dict['count_zeroes'] = np.nan

                    # Check if the column contains any dates
                    date_series = to_datetime(
                        df[column_name], errors='coerce'
                    )
                    null_series = date_series[~date_series.notnull()]
                    null_count = null_series.shape[0]
                    date_count = date_series.shape[0]
                    row_dict['has_dates'] = (null_count < date_count)

                    # Find the minimum value
                    try:
                        row_dict['min_value'] = filtered_df[column_name].min()

                    # Set minimum value to NaN if an error occurs
                    except Exception:
                        row_dict['min_value'] = np.nan

                    # Find the maximum value
                    try:
                        row_dict['max_value'] = filtered_df[column_name].max()

                    # Set maximum value to NaN if an error occurs
                    except Exception:
                        row_dict['max_value'] = np.nan

                    # Check if the column contains only integers
                    try:
                        is_integer = df[column_name].apply(
                            lambda x: float(x).is_integer()
                        )
                        row_dict['only_integers'] = is_integer.all()

                    # Set only_integers to NaN if an error occurs
                    except Exception:
                        row_dict['only_integers'] = float('nan')

                    # Append the row dictionary to the rows list
                    rows_list.append(row_dict)

        # Define column order for the resulting DataFrame
        columns_list = [
            'column_name', 'dtype', 'count_blanks', 'count_uniques',
            'count_zeroes', 'has_dates', 'min_value', 'max_value',
            'only_integers'
        ]

        # Create a data frame from the list of dictionaries
        blank_ranking_df = DataFrame(rows_list, columns=columns_list)

        # Return the data frame containing the descriptive statistics
        return blank_ranking_df

    @staticmethod
    def get_statistics(describable_df, columns_list, verbose=False):
        """
        Calculate and returns descriptive statistics for a subset of columns
        in a Pandas DataFrame.

        Parameters:
            describable_df (pandas.DataFrame):
                The DataFrame to calculate descriptive statistics for.
            columns_list (list of str):
                A list of specific columns to calculate statistics for.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            pandas.DataFrame:
                A DataFrame containing the descriptive statistics for the
                analyzed columns. The returned DataFrame includes the mean,
                mode, median, standard deviation (SD), minimum, 25th
                percentile, 50th percentile (median), 75th percentile, and
                maximum.
        """

        # Compute basic descriptive statistics for the specified columns
        df = describable_df[columns_list].describe().rename(
            index={'std': 'SD'}
        )

        # If the mode is not already included in the statistics, calculate it
        if 'mode' not in df.index:

            # Create the mode row dictionary
            row_dict = {
                cn: describable_df[cn].mode().iloc[0] for cn in columns_list
            }

            # Convert row dictionary to a df to match the df structure
            row_df = DataFrame([row_dict], index=['mode'])

            # Append the row data frame to the df data frame
            df = concat([df, row_df], axis='index', ignore_index=False)

        # If median is not already included in the statistics, calculate it
        if 'median' not in df.index:

            # Create the median row dictionary
            row_dict = {
                cn: describable_df[cn].median() for cn in columns_list
            }

            # Convert row_dict to a data frame to match the df structure
            row_df = DataFrame([row_dict], index=['median'])

            # Append the row data frame to the df data frame
            df = concat([df, row_df], axis='index', ignore_index=False)

        # Define the desired index order for the resulting DataFrame
        index_list = [
            'mean', 'mode', 'median', 'SD', 'min', '25%', '50%', '75%', 'max'
        ]

        # Create a boolean mask to select rows with desired index values
        mask_series = df.index.isin(index_list)
        df = df[mask_series].reindex(index_list)

        # If verbose is True, print additional information
        if verbose:
            print(f'columns_list: {columns_list}')
            display(describable_df)
            display(df)

        # Return the filtered DataFrame containing the selected statistics
        return df

    @staticmethod
    def modalize_columns(df, columns_list, new_column_name):
        """
        Create a new column in a DataFrame representing the modal value of
        specified columns.

        Parameters:
            df (pandas.DataFrame): The input DataFrame.
            columns_list (list):
                The list of column names from which to calculate the modal
                value.
            new_column_name (str): The name of the new column to create.

        Returns:
            pandas.DataFrame:
                The modified DataFrame with the new column representing the
                modal value.

        Example:
            import numpy as np
            import pandas as pd

            df = pd.DataFrame({
                'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3], 'C': ['a', 'b', 'c']
            })
            df['D'] = pd.Series([np.nan, 2, np.nan])
            df['E'] = pd.Series([1, np.nan, 3])
            df = nu.modalize_columns(df, ['D', 'E'], 'F')
            display(df)
            assert all(df['A'] == df['F'])
        """

        # Ensure that all columns are in the data frame
        columns_list = sorted(set(df.columns).intersection(set(columns_list)))

        # Create a mask series indicating rows with one unique value across
        singular_series = df[columns_list].apply(
            Series.nunique, axis='columns'
        ) == 1

        # Check that there is less than two unique column values for all
        mask_series = singular_series | (df[columns_list].apply(
            Series.nunique, axis='columns'
        ) < 1)
        assert mask_series.all(), (
            f'\n\nYou have more than one {new_column_name} in your'
            f' columns:\n{df[~mask_series][columns_list]}'  # noqa E231
        )

        # Replace non-unique or missing values with NaN
        df.loc[~singular_series, new_column_name] = nan

        # Define a function to extract the first valid value in each row
        def extract_first_valid_value(srs):
            """
            Extract the first valid value from a pandas Series.

            Parameters:
                srs (pd.Series): A pandas Series object.

            Returns:
                The first valid value in the Series, or raises an error if no
                valid index exists.
            """
            return srs[srs.first_valid_index()]

        # For identical columns-values rows, set new column to modal value
        singular_values = df[singular_series][columns_list]
        df.loc[singular_series, new_column_name] = singular_values.apply(
            extract_first_valid_value, axis='columns'
        )

        return df

    @staticmethod
    def get_regexed_columns(df, search_regex, verbose=False):
        """
        Identify columns in a DataFrame that contain references based on a
        specified regex pattern.

        Parameters:
            df (pandas.DataFrame): The input DataFrame.
            search_regex (re.Pattern, optional):
                The compiled regular expression pattern for identifying
                references.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            list:
                A list of column names that contain references based on the
                specified regex pattern.
        """

        # Ensure that the search_regex is a compiled regex object
        assert (
            isinstance(search_regex, Pattern)
        ), "search_regex must be a compiled regular expression."

        # Print the type of the search_regex if verbose mode is enabled
        if verbose:
            print(type(search_regex))

        # Apply the regex to each element and count occurrences per column
        srs = df.map(
            lambda x: bool(search_regex.search(str(x))), na_action='ignore'
        ).sum()

        # Extract column names where the count of occurrences is not zero
        columns_list = srs[srs != 0].index.tolist()

        return columns_list

    @staticmethod
    def get_regexed_dataframe(
        filterable_df, columns_list, search_regex, verbose=False
    ):
        """
        Create a DataFrame that displays an example of what search_regex is
        finding for each column in columns_list.

        Parameters:
            filterable_df (pandas.DataFrame): The input DataFrame to filter.
            columns_list (list of str):
                The list of column names to investigate for matches.
            search_regex (re.Pattern, optional):
                The compiled regular expression pattern for identifying
                matches. If None, the default pattern for detecting references
                will be used.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            pandas.DataFrame:
                A DataFrame containing an example row for each column in
                columns_list that matches the regex pattern.
        """

        # Ensure that all names in columns_list are in there
        assert all(
            map(lambda cn: cn in filterable_df.columns, columns_list)
        ), "Column names in columns_list must be in filterable_df.columns"

        # Print the debug info if verbose is True
        if verbose:
            print(type(search_regex))

        # Create an empty DataFrame to store the filtered rows
        filtered_df = DataFrame([])

        # For each column, filter df and extract first row that matches
        for cn in columns_list:

            # Create a mask to filter rows where column matches pattern
            mask_series = filterable_df[cn].map(
                lambda x: bool(search_regex.search(str(x)))
            )

            # Concatenate the first matching row not already in the result
            df = filterable_df[mask_series]
            mask_series = ~df.index.isin(filtered_df.index)
            if mask_series.any():
                filtered_df = concat(
                    [filtered_df, df[mask_series].iloc[0:1]], axis='index'
                )

        return filtered_df

    @staticmethod
    def convert_to_data_frame(row_index, row_series, verbose=True):
        """
        Convert a row represented as a Pandas Series into a single-row
        DataFrame.

        Parameters:
            row_index (int):
                The index to be assigned to the new DataFrame row.
            row_series (pandas.Series):
                The Pandas Series representing the row's data.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to True.

        Returns:
            pandas.DataFrame:
                A single-row DataFrame containing the data from the input
                Pandas Series.
        """

        # Print type of row_index if verbose and it is not an integer
        if verbose and type(row_index) != int:
            print(type(row_index))

        # Create new df with data from input Series and specified index
        df = DataFrame(data=row_series.to_dict(), index=[row_index])

        return df

    @staticmethod
    def one_hot_encode(df, columns, dummy_na=True):
        """
        One-hot encode the specified columns in the provided DataFrame.

        This function performs one-hot encoding on a subset of columns
        within a DataFrame. One-hot encoding is a technique for
        representing categorical variables as binary features. Each
        category maps to a new column with a value of 1 for that category
        and 0 for all other categories.

        Parameters:
            df (pandas.DataFrame):
                The DataFrame containing the columns to be encoded.
            columns (list of str):
                A list of column names to encode as one-hot features.
            dummy_na (bool, optional):
                Whether to add a column to indicate NaNs. Defaults to True.

        Returns:
            pandas.DataFrame:
                A data frame with the encoded columns minus the original
                columns.
        """

        # Create one-hot encoded representation of the specified columns
        dummies = get_dummies(df[columns], dummy_na=dummy_na)

        # Create a list of extra dummy variable column names
        columns_list = sorted(set(dummies.columns).difference(set(df.columns)))

        # Concatenate the data frame with the dummy variables
        df = concat([df, dummies[columns_list]], axis='columns')

        # Drop the original encoded columns from the DataFrame
        df = df.drop(columns, axis='columns')

        # Return the DataFrame with the one-hot encoded features
        return df

    def get_flattened_dictionary(self, value_obj, row_dict={}, key_prefix=''):
        """
        Take a value_obj (either a dictionary, list or scalar value) and
        create a flattened dictionary from it, where keys are made up of the
        keys/indices of nested dictionaries and lists. The keys are
        constructed with a key_prefix (which is updated as the function
        traverses the value_obj) to ensure uniqueness. The flattened
        dictionary is stored in the row_dict argument, which is updated at
        each step of the function.

        Parameters:
            value_obj (dict, list, scalar value):
                The object to be flattened into a dictionary.
            row_dict (dict, optional):
                The dictionary to store the flattened object.
            key_prefix (str, optional):
                The prefix for constructing the keys in the row_dict.

        Returns:
            row_dict (dict):
                The flattened dictionary representation of the value_obj.
        """

        # Check if the value is a dictionary
        if isinstance(value_obj, dict):

            # Iterate through the dictionary
            for k, v, in value_obj.items():

                # Recursively call function with dictionary key in prefix
                row_dict = self.get_flattened_dictionary(
                    v, row_dict=row_dict,
                    key_prefix=f'{key_prefix}{"_" if key_prefix else ""}{k}'
                )

        # Check if the value is a list
        elif isinstance(value_obj, list):

            # Get the minimum number of digits in the list length
            list_length = len(value_obj)
            digits_count = min(len(str(list_length)), 2)

            # Iterate through the list
            for i, v in enumerate(value_obj):

                # Add leading zeros to the index
                if i == 0 and list_length == 1:
                    i = ''
                else:
                    i = str(i).zfill(digits_count)

                # Recursively call function with the list index in prefix
                row_dict = self.get_flattened_dictionary(
                    v, row_dict=row_dict, key_prefix=f'{key_prefix}{i}'
                )

        # If neither a dictionary nor a list, add value to row dictionary
        else:
            if key_prefix.startswith('_') and key_prefix[1:] not in row_dict:
                key_prefix = key_prefix[1:]
            row_dict[key_prefix] = value_obj

        return row_dict

    def show_time_statistics(self, describable_df, columns_list):
        """
        Display time-related statistics for specified columns in a DataFrame.

        Parameters:
            describable_df (pandas.DataFrame):
                The DataFrame to calculate descriptive statistics for.
            columns_list (list of str):
                A list of specific time-related columns to calculate
                statistics for.

        Returns:
            pandas.DataFrame:
                A DataFrame containing the descriptive statistics for the
                analyzed time-related columns.
        """

        # Get time-related statistics using the get_statistics method
        df = self.get_statistics(describable_df, columns_list)

        # Apply a formatting function to convert milliseconds to timedelta
        df = df.map(lambda x: self.format_timedelta(
            timedelta(milliseconds=int(x))
        ), na_action='ignore').T

        # Format the standard deviation (SD) column to include the ± symbol
        df.SD = df.SD.map(lambda x: '\xB1' + str(x))
        # df.SD = df.SD.map(lambda x: '±' + str(x))

        # Display the resulting DataFrame
        display(df)

    @staticmethod
    def clean_numerics(df, columns_list=None, verbose=False):
        if columns_list is None:
            columns_list = df.columns
        for cn in columns_list:
            df[cn] = df[cn].map(lambda x: re.sub(r'[^0-9\.]+', '', str(x)))
            df[cn] = pd.to_numeric(df[cn], errors='coerce', downcast='integer')

        return df

    @staticmethod
    def get_numeric_columns(df, is_na_dropped=True):
        """
        Identify numeric columns in a DataFrame.

        Parameters:
            df (pandas.DataFrame):
                The DataFrame to search for numeric columns.
            is_na_dropped (bool, optional):
                Whether to drop columns with all NaN values. Default is True.

        Returns:
            list
                A list of column names containing numeric values.

        Notes:
            This function identifies numeric columns by checking if the data
            in each column can be interpreted as numeric. It checks for
            integer, floating-point, and numeric-like objects.

        Examples:
            import pandas as pd
            df = pd.DataFrame({
                'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3], 'C': ['a', 'b', 'c']
            })
            nu.get_numeric_columns(df)  # ['A', 'B']
        """

        # Initialize an empty list to store numeric column names
        numeric_columns = []

        # Iterate over DataFrame columns to identify numeric columns
        for cn in df.columns:

            # Are they are integers or floats?
            if is_integer(df[cn]) or is_float(df[cn]):

                # Append element to the list
                numeric_columns.append(cn)

        # Optionally drop columns with all NaN values
        if is_na_dropped:
            numeric_columns = df[numeric_columns].dropna(
                axis='columns', how='all'
            ).columns

        # Sort and return the list of numeric column names
        return sorted(numeric_columns)

    def convert_to_df(self, row_index, row_series):
        """
        Convert a given pandas.Index and pandas.Series into a DataFrame.

        Parameters:
            row_index (pandas.Index): The index to be used for the DataFrame.
            row_series (pandas.Series): The series containing the row data.

        Returns:
            pandas.DataFrame:
                A DataFrame with the given index and row data as a column.

        Raises:
            ValueError:
                If the length of the row_index and row_series do not match.
            TypeError:
                If row_index is not a pandas.Index or row_series is not a
                pandas.Series.
        """

        # Validate input types
        if not isinstance(row_index, pd.Index):
            raise TypeError("row_index must be of type pandas.Index.")
        if not isinstance(row_series, pd.Series):
            raise TypeError("row_series must be of type pandas.Series.")

        # Validate input lengths
        if len(row_index) != len(row_series):
            raise ValueError(
                "The length of row_index and row_series must be the same."
            )

        # Create the DataFrame
        df = pd.DataFrame(
            {row_series.name or 'Data': row_series}, index=row_index
        )

        return df

    def split_df_by_indices(self, df, indices_list, verbose=False):
        """
        Split a DataFrame into a list of smaller DataFrames based on specified
        row indices.

        It iterates over the rows of the input DataFrame and accumulates them
        into sub-DataFrames whenever it encounters a row index that is in the
        `indices_list`. Once an index is found, the accumulated rows are
        appended as a separate DataFrame to a list which is returned at the
        end. The process then continues, accumulating rows again until the
        next index or the end of the DataFrame is reached.

        Parameters:
            df (pandas.DataFrame):
                The DataFrame to be split.
            indices_list (pandas.index or list):
                A list of row indices where the DataFrame should be split.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            list of pandas.DataFrame:
                A list of DataFrames, each corresponding to a index.
        """
        split_dfs = []
        current_df = DataFrame()

        # Iterate over the rows of the dataframe
        for row_index, row_series in df.iterrows():

            # Check if the current row index is in the indices_list
            if row_index in indices_list:

                # Append the current dataframe to the list if it has rows
                if current_df.shape[0] > 0:
                    split_dfs.append(current_df)

                # Reset the current dataframe for the next split
                current_df = DataFrame()

            # Print verbose output if enabled
            if verbose:
                print(f'Row Index: {row_index}')
                display(row_series)
                display(self.convert_to_df(row_index, row_series))
                raise Exception('Verbose debugging')

            # Append the current row to the current DataFrame
            current_df = concat(
                [current_df, self.convert_to_df(row_index, row_series)],
                axis='index'
            )

        # Append the final dataframe chunk if it has rows
        if current_df.shape[0] > 0:
            split_dfs.append(current_df)

        # Return the list of split DataFrames
        return split_dfs

    @staticmethod
    def split_df_by_iloc(df, indices_list, verbose=False):
        """
        Split a DataFrame into a list of smaller DataFrames based on specified
        iloc-indexer start integers.

        This static method takes a DataFrame (`df`), a list of indices
        (`indices_list`), and an optional `verbose` flag. It splits the
        DataFrame into sub-DataFrames based on the provided indices. The split
        points are defined as the elements of `indices_list` incremented by 1
        to include the rows up to (but not including) the next index.

        Parameters:
            df (pandas.DataFrame):
                The DataFrame to be split.
            indices_list (list or array of integers):
                the iloc indexer start integers to split on.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            List[DataFrame]:
                A list of DataFrames, the indices of each starting with one of
                the elements of indices_list and ending right before the next
                one.
        """

        # Calculate split indices based on changes before splittable rows
        split_indices = [0] + list(indices_list[:-1] + 1) + [len(df)]

        # Gather the sub-dataframes in a list
        split_dfs = []
        for i in range(len(split_indices) - 1):
            start_iloc = split_indices[i]
            end_iloc = split_indices[i + 1]
            split_dfs.append(df.iloc[start_iloc:end_iloc])

        # Return the list of split DataFrames
        return split_dfs

    @staticmethod
    def replace_consecutive_rows(
        df, element_column, element_value, time_diff_column='time_diff',
        consecutive_cutoff=500
    ):
        """
        Replace consecutive rows in a DataFrame with a single row where the
        element_value is appended with a count of consecutive occurrences.

        This function iterates through a DataFrame and identifies consecutive
        occurrences of a specific element based on a designated column value
        and a time difference threshold. If consecutive occurrences are found
        within the time threshold, the function combines those rows into a
        single row where the element_value is modified to include a count of
        the consecutive elements.

        Parameters:
            df (pandas.DataFrame):
                The DataFrame containing the data to be processed.
            element_column (str):
                The name of the column containing the elements to check for
                consecutive occurrences.
            element_value (str):
                The value to identify and count consecutive occurrences of.
            time_diff_column (str, optional):
                The name of the column containing the time difference between
                rows. Defaults to 'time_diff'. Values in this column are used
                to determine if rows are considered consecutive based on the
                `consecutive_cutoff` parameter.
            consecutive_cutoff (int, optional):
                The maximum time difference (in time units) to consider rows
                consecutive. Defaults to 500. Rows with a time difference less
                than or equal to this value will be considered consecutive.

        Returns:
            pandas.DataFrame
                A new DataFrame with the rows of consecutive elements replaced
                with a single row. The replaced row's element_value will be
                appended with " x<count>" where "<count>" is the number of
                consecutive occurrences.
        """

        # Create an empty copy of the df to avoid modifying the original
        result_df = DataFrame([], columns=df.columns)

        # Initialize variables to keep track of row index and current row
        row_index = 0
        row_series = Series([])

        # Initialize a counter for consecutive occurrences
        count = 0

        # Iterate over each row in the input DataFrame
        for row_index, row_series in df.iterrows():

            # Get the value of the element column for the current row
            column_value = row_series[element_column]

            # Get the value of the time_diff column for the current row
            time_diff = row_series[time_diff_column]

            # Check if current element is target element and within cutoff
            if (
                column_value == element_value
                and time_diff <= consecutive_cutoff
            ):

                # If consecutive element found, increment count
                count += 1

                # Keep track of the previous row's index and data
                previous_row_index = row_index
                previous_row_series = row_series

            # Does the element column value or time difference not match?
            else:

                # Add the current row to the result data frame
                result_df.loc[row_index] = row_series

                # Are there consecutive elements?
                if count > 0:

                    # Replace the last consecutive element with a count
                    result_df.loc[previous_row_index] = previous_row_series
                    result_df.loc[
                        previous_row_index, element_column
                    ] = f'{element_value} x{str(count)}'

                # Reset the count of consecutive elements
                count = 0

        # Handle last element by adding last row to result data frame
        result_df.loc[row_index] = row_series

        # Was the last element part of a consecutive series?
        if count > 0:

            # Replace it with a count of how many there were
            result_df.loc[
                row_index, element_column
            ] = f'{element_value} x{count}'

        return result_df

    def rebalance_data(
        self,
        unbalanced_df,
        name_column,
        value_column,
        sampling_strategy_limit,
        verbose=False
    ):
        """
        Rebalance the given unbalanced DataFrame by under-sampling the
        majority class(es) to the specified sampling_strategy_limit.

        Parameters:
            unbalanced_df (pandas.DataFrame):
                The unbalanced DataFrame to rebalance.
            name_column (str):
                The name of the column containing the class labels.
            value_column (str):
                The name of the column containing the values associated with
                each class label.
            sampling_strategy_limit (int):
                The maximum number of samples to keep for each class label.
            verbose (bool, optional): Whether to print debug output during the
                rebalancing process. Defaults to False.

        Returns:
            pandas.DataFrame:
                A rebalanced DataFrame with an undersampled majority class.
        """

        # Count name_column occurrences for each unique value in value_column
        if verbose:
            print('Creating the random under-sampler')
        counts_dict = unbalanced_df.groupby(value_column).count()[
            name_column
        ].to_dict()

        # Limit each class count to sampling_strategy_limit in counts_dict
        sampling_strategy = {
            k: min(sampling_strategy_limit, v)
            for k, v in counts_dict.items()
        }

        # Initialize RandomUnderSampler with the defined sampling strategy
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy)

        # Apply under-sampling to rebalance based on the sampling strategy
        if verbose:
            print('Resampling the data')
        X_res, y_res = rus.fit_resample(
            unbalanced_df[name_column].values.reshape(-1, 1),
            unbalanced_df[value_column].values.reshape(-1, 1),
        )

        # Create a rebalanced df with the resampled name and value columns
        if verbose:
            print('Converting data to Pandas DataFrame')
        rebalanced_df = DataFrame(X_res, columns=[name_column])
        rebalanced_df[value_column] = y_res

        # Return the rebalanced data frame
        return rebalanced_df

    # -------------------
    # 3D Point Functions
    # -------------------

    @staticmethod
    def get_coordinates(second_point, first_point=None):
        """
        Extract and return the 3D coordinates of two points from provided
        strings.

        This static method parses two strings representing 3D point
        coordinates (`second_point` and `first_point`). If `first_point`
        is not provided, it assumes the origin (0, 0, 0) as the first
        point.

        Parameters:
            second_point (str):
                A string containing the x, y, and z coordinates of the second
                point separated by commas (e.g., "(-1.2,3.4,5.6)").
            first_point (str, optional):
                A string containing the x, y, and z coordinates of the first
                point separated by commas (e.g., "(-1.2,3.4,5.6)"). Defaults
                to None (origin).

        Returns:
            tuple of float
                A tuple containing six floating-point values representing the
                x, y, and z coordinates of the first point followed by the x,
                y, and z coordinates of the second point.
        """

        # Handle the case where the first point is not provided (use origin)
        if first_point is None:
            x1 = 0.0  # The x-coordinate of the first point (origin)
            y1 = 0.0  # The y-coordinate of the first point (origin)
            z1 = 0.0  # The z-coordinate of the first point (origin)

        # Or, if provided, parse the coordinates from the string
        else:
            location_tuple = eval(first_point)
            x1 = location_tuple[0]  # The x-coordinate of the first point
            y1 = location_tuple[1]  # The y-coordinate of the first point
            z1 = location_tuple[2]  # The z-coordinate of the first point

        # Parse the coordinates of the second point from the string
        location_tuple = eval(second_point)
        x2 = location_tuple[0]  # The x-coordinate of the second point
        y2 = location_tuple[1]  # The y-coordinate of the second point
        z2 = location_tuple[2]  # The z-coordinate of the second point

        # Return the coordinates of the two points
        return x1, x2, y1, y2, z1, z2

    @staticmethod
    def get_euclidean_distance(first_point, second_point):
        """
        Calculate the Euclidean distance between two 2D or 3D points.

        This static method calculates the Euclidean distance between two
        points (`first_point` and `second_point`). It supports both 2D (x,
        y) and 3D (x, y, z) coordinates.

        Parameters:
            first_point (tuple):
                A tuple containing the coordinates of the first point.
            second_point (tuple):
                A tuple containing the coordinates of the second point.

        Returns:
            float
                The Euclidean distance between the two points, or numpy.nan if
                the points have mismatched dimensions.
        """

        # Initialize the Euclidean distance to NaN
        euclidean_distance = nan

        # Check if both points have the same dimensions (2D or 3D)
        assert len(first_point) != len(second_point), (
            f'Mismatched dimensions: {len(first_point)}'
            f' != {len(second_point)}'
        )

        # Check if the points are in 3D
        if len(first_point) == 3 and len(second_point) == 3:

            # Unpack the coordinates of the first and second points
            x1, y1, z1 = first_point
            x2, y2, z2 = second_point

            # Calculate the Euclidean distance for 3D points
            euclidean_distance = math.sqrt(
                (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
            )

        # Check if both points are 2D
        elif len(first_point) == 2 and len(second_point) == 2:

            # Unpack the coordinates of the first and second points
            x1, z1 = first_point
            x2, z2 = second_point

            # Calculate the Euclidean distance for 2D points
            euclidean_distance = math.sqrt((x1 - x2)**2 + (z1 - z2)**2)

        # Return the calculated Euclidean distance
        return euclidean_distance

    def get_relative_position(self, second_point, first_point=None):
        """
        Calculate the position of a point relative to another point.

        This static method calculates the relative position of a second
        point (`second_point`) based on a reference point (`first_point`).
        If `first_point` is not provided, it assumes the origin (0, 0, 0)
        by calling a separate function `get_coordinates` (assumed to be
        implemented elsewhere).

        Parameters:
            second_point (tuple):
                A tuple containing the x, y, and z coordinates of the second
                point.
            first_point (tuple, optional):
                A tuple containing the x, y, and z coordinates of the reference
                point. If not specified, the origin is retrieved from
                get_coordinates.

        Returns:
            tuple:
                A tuple containing the x, y, and z coordinates of the second
                point relative to the reference point.
        """

        # Retrieve the coordinates for both points, defaulting to origin
        x1, x2, y1, y2, z1, z2 = self.get_coordinates(
            second_point, first_point=first_point
        )

        # Calculate the relative position by adding corresponding coordinates
        relative_position = (
            round(x1 + x2, 1), round(y1 + y2, 1), round(z1 + z2, 1)
        )

        # Return the calculated relative position as a tuple
        return relative_position

    def get_nearest_neighbor(self, base_point, neighbors_list):
        """
        Get the point nearest in Euclidean distance between two 2D or 3D
        points, the base_point and an item from the neighbors_list.

        This function finds the neighbor in `neighbors_list` that is
        closest (minimum Euclidean distance) to the `base_point`.

        Parameters:
            base_point (tuple):
                A tuple containing the coordinates of the base point.
            neighbors_list (list of tuples):
                A list of tuples representing the coordinates of neighboring
                points.

        Returns:
            tuple
                The coordinates of the nearest neighbor in the
                `neighbors_list`, or None if the list is empty.
        """

        # Initialize the minimum distance to infinity
        min_distance = math.inf

        # Initialize the nearest neighbor to None
        nearest_neighbor = None

        # Iterate over each point in the neighbors list
        for neighbor_point in neighbors_list:

            # Calculate distance between base point and current neighbor
            distance = self.get_euclidean_distance(base_point, neighbor_point)

            # Update nearest neighbor/minimum distance if closer one found
            if distance < min_distance:

                # Update the minimum distance to the calculated distance
                min_distance = distance

                # Update the nearest neighbor to the current neighbor point
                nearest_neighbor = neighbor_point

        # Return the nearest neighbor (or None if the list is empty)
        return nearest_neighbor

    # -------------------
    # Sub-sampling Functions
    # -------------------

    @staticmethod
    def get_minority_combinations(sample_df, groupby_columns):
        """
        Get the minority combinations of a DataFrame.

        Parameters:
            sample_df: A Pandas DataFrame.
            groupby_columns: A list of column names to group by.

        Returns:
            A Pandas DataFrame containing a single sample row of each of the
            four smallest groups.
        """

        # Create an empty data frame with sample_df columns
        df = DataFrame([], columns=sample_df.columns)

        # Loop through 4 smallest dfs by their groupby key/values pairs
        for bool_tuple in sample_df.groupby(
            groupby_columns
        ).size().sort_values().index.tolist()[:4]:

            # Filter name in column to corresponding value of tuple
            mask_series = True
            for cn, cv in zip(groupby_columns, bool_tuple):
                mask_series &= (sample_df[cn] == cv)

            # Append a single record from the filtered data frame
            if mask_series.any():
                df = concat(
                    [df, sample_df[mask_series].sample(1)], axis='index'
                )

        # Return the data frame with the four sample rows
        return df

    @staticmethod
    def get_random_subdictionary(super_dict, n=5):
        """
        Extract a random subdictionary with a specified number of key-value
        pairs from a given superdictionary.

        Parameters:
            super_dict (dict):
                The dictionary from which to extract a random subdictionary.
            n (int, optional):
                The number of key-value pairs to include in the
                sub-dictionary. Defaults to 5.

        Returns:
            dict:
                A random subdictionary with n key-value pairs from the
                superdictionary.
        """

        # Convert the dictionary's keys into a list
        keys = list(super_dict.keys())

        # Import the random module
        import random

        # Select a random sample of n keys from the list of keys
        random_keys = random.sample(keys, n)

        # Create an empty dictionary to store the sub-dictionary
        sub_dict = {}

        # Iterate over the randomly selected keys
        for key in random_keys:

            # Add their corresponding values to the sub-dictionary
            sub_dict[key] = super_dict[key]

        return sub_dict

    # -------------------
    # Plotting Functions
    # -------------------

    @staticmethod
    def color_distance_from(from_color, to_rgb_tuple):
        """
        Calculate the Euclidean distance between two colors in RGB space.

        This function computes the color distance between the RGB
        values of a specified color and another RGB tuple. It supports
        colors specified as 'white', 'black', or a hexadecimal string.

        Parameters:
            from_color (str):
                The starting color, which can be 'white', 'black', or a
                hexadecimal string.
            to_rgb_tuple (tuple):
                The target RGB tuple (length 3) representing the color to
                compare to.

        Returns:
            float:
                The Euclidean distance between the from_color and the
                to_rgb_tuple in RGB space.

        Raises:
            ValueError:
                If the from_color is not 'white', 'black', or a valid
                hexadecimal color string.

        Examples:
            nu.color_distance_from('white', (255, 0, 0))  # 360.62445840513925
            nu.color_distance_from(
                '#0000FF', (255, 0, 0)
            )  # 360.62445840513925
        """
        from math import sqrt

        # Is the from_color 'white'?
        if from_color == 'white':

            # Compute the Euclidean distance from (255, 255, 255)
            green_diff = 255 - to_rgb_tuple[0]
            blue_diff = 255 - to_rgb_tuple[1]
            red_diff = 255 - to_rgb_tuple[2]
            color_distance = sqrt(green_diff**2 + blue_diff**2 + red_diff**2)

        # Is the from_color 'black'?
        elif from_color == 'black':

            # Compute the Euclidean distance from (0, 0, 0)
            color_distance = sqrt(
                to_rgb_tuple[0]**2 + to_rgb_tuple[1]**2 + to_rgb_tuple[2]**2
            )

        # Otherwise, treat from_color as a hexadecimal string
        else:
            import webcolors
            try:
                rgb_tuple = tuple(webcolors.hex_to_rgb(from_color))
                green_diff = rgb_tuple[0] - to_rgb_tuple[0]
                blue_diff = rgb_tuple[1] - to_rgb_tuple[1]
                red_diff = rgb_tuple[2] - to_rgb_tuple[2]

                # And compute the Euclidean distance from its RGB conversion
                color_distance = sqrt(
                    green_diff**2 + blue_diff**2 + red_diff**2
                )

            except ValueError as e:
                raise ValueError(f'Invalid color value: {from_color}') from e

        return color_distance

    def get_text_color(
        self, text_color='white', bar_color_rgb=(0, 0, 0), verbose=False
    ):
        """
        Determine an appropriate text color based on the background color
        for improved readability.

        This function calculates the most suitable text color to be used
        on a given background color (`bar_color_rgb`). It compares the
        distance of the background color to predefined text colors
        ('white', '#404040', 'black') and selects the most distinct color.
        The default text color is 'white'.

        Parameters:
            text_color (str, optional):
                The default text color to be used if the background color is
                black. Defaults to 'white'.
            bar_color_rgb (tuple, optional):
                A tuple representing the RGB values of the background color.
                Defaults to (0, 0, 0), which is black.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            str
                The chosen text color as a valid HTML/CSS color string (e.g.,
                'white', '#404040', '#000000').

        Note:
            This function uses the `color_distance_from` method to compute the
            distance between colors and the `webcolors` library to convert
            color names to hex codes.
        """

        # Check if a non-black background color is provided
        if bar_color_rgb != (0, 0, 0):

            # Initialize the list to store the distances for each color
            text_colors_list = []

            # Iterate through predefined readable colors
            for color in ['white', '#404040', 'black']:

                # Calculate distance between current color and background
                color_distance = self.color_distance_from(
                    color, bar_color_rgb
                )
                color_tuple = (color_distance, color)

                # Append the color and its distance to the list
                text_colors_list.append(color_tuple)

            # Print the list of color distances if verbose is True
            if verbose:
                print(text_colors_list)

            # Select color with maximum distance from background color
            sorted_list = sorted(text_colors_list, key=lambda x: x[0])
            text_color = sorted_list[-1][1]

            # Attempt to convert the text color to a valid HTML/CSS hex code
            try:

                # Import the webcolors module
                import webcolors

                # Try to convert the color name to hex format
                text_color = webcolors.name_to_hex(text_color)

            # If the color name is not recognized, pass
            except Exception:
                pass

        # Return the selected or default text color
        return text_color

    @staticmethod
    def get_color_cycler(n):
        """
        Generate a color cycler for plotting with a specified number of
        colors.

        This static method creates a color cycler object (`cycler.Cycler`)
        suitable for Matplotlib plotting. The color cycler provides a
        series of colors to be used for lines, markers, or other plot
        elements. The function selects a colormap based on the requested
        number of colors (`n`).

        Parameters:
            n (int):
                The number of colors to include in the cycler.

        Returns:
            cycler.Cycler
                A color cycler object containing the specified number of
                colors.

        Example:
            color_cycler = nu.get_color_cycler(len(possible_cause_list))
            for possible_cause, face_color_dict in zip(
                possible_cause_list, color_cycler()
            ):
                face_color = face_color_dict['color']
        """

        # Initialize an empty color cycler object
        color_cycler = None

        # Import the `cycler` module from matplotlib
        from cycler import cycler

        # Use the Accent color map for less than 9 colors
        if n < 9:
            color_cycler = cycler('color', plt.cm.Accent(np.linspace(
                0, 1, n
            )))

        # Use tab10 colormap for 9 or 10 colors
        elif n < 11:
            color_cycler = cycler('color', plt.cm.tab10(np.linspace(0, 1, n)))

        # Use Paired colormap for 11 or 12 colors
        elif n < 13:
            color_cycler = cycler('color', plt.cm.Paired(np.linspace(0, 1, n)))

        # Use the tab20 color map for 13 or more colors
        else:
            color_cycler = cycler('color', plt.cm.tab20(np.linspace(0, 1, n)))

        return color_cycler

    @staticmethod
    def plot_line_with_error_bars(
        df, xname, xlabel, xtick_text_fn, yname, ylabel, ytick_text_fn, title
    ):
        """
        Create a line plot with error bars to visualize the mean and standard
        deviation of a numerical variable grouped by another categorical
        variable.

        Parameters:
            df (pandas.DataFrame):
                The input DataFrame containing the data to plot.
            xname (str):
                The name of the categorical variable to group by and for the
                x-axis values.
            xlabel (str): The label for the x-axis.
            xtick_text_fn (function):
                A function to humanize x-axis tick labels.
            yname (str): The column name for the y-axis values.
            ylabel (str): The label for the y-axis.
            ytick_text_fn (function):
                A function to humanize y-axis tick labels.
            title (str): The title of the plot.

        Returns:
            None
                The function plots the graph directly using matplotlib.
        """

        # Drop rows with NaN values, group by xname, calculate mean & std
        groupby_list = [xname]
        columns_list = [xname, yname]
        aggs_list = ['mean', 'std']
        df = df.dropna(subset=columns_list).groupby(groupby_list)[yname].agg(
            aggs_list
        ).reset_index()

        # Create the figure and subplot
        fig, ax = plt.subplots(figsize=(18, 9))

        # Plot the line with error bars
        ax.errorbar(
            x=df[xname],
            y=df['mean'],
            yerr=df['std'],
            label=ylabel,
            fmt='-o',  # Line style with markers
        )

        # Set plot title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Humanize x-axis tick labels
        xticklabels_list = []
        for text_obj in ax.get_xticklabels():
            text_obj.set_text(xtick_text_fn(text_obj))
            xticklabels_list.append(text_obj)
        ax.set_xticklabels(xticklabels_list)

        # Humanize y-axis tick labels
        yticklabels_list = []
        for text_obj in ax.get_yticklabels():
            text_obj.set_text(ytick_text_fn(text_obj))
            yticklabels_list.append(text_obj)
        ax.set_yticklabels(yticklabels_list)

    @staticmethod
    def plot_histogram(
        df, xname, xlabel, title, xtick_text_fn=None, ylabel=None,
        xticks_are_temporal=False, ax=None, color=None, bins=100
    ):
        """
        Plot a histogram of a DataFrame column.

        Parameters:
            df:
                A Pandas DataFrame.
            xname:
                The name of the column to plot the histogram of.
            xlabel:
                The label for the x-axis.
            title:
                The title of the plot.
            xtick_text_fn:
                A function that takes a text object as input and returns a new
                text object to be used as the tick label. Defaults to None.
            ylabel:
                The label for the y-axis.
            ax:
                A matplotlib axis object. If None, a new figure and axis will
                be created.

        Returns:
            A matplotlib axis object.
        """

        # Create the figure and subplot
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 9))

        # Plot the histogram with centered bars
        df[xname].hist(
            ax=ax, bins=bins, align='mid', edgecolor='black', color=color
        )

        # Set the grid, title and labels
        plt.grid(False)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ylabel is None:
            ylabel = 'Count of Instances in Bin'
        ax.set_ylabel(ylabel)

        # Check if the xticks are temporal
        if xticks_are_temporal:

            # Set the minor x-axis tick labels to every 30 seconds
            thirty_seconds = 1_000 * 30
            minor_ticks = np.arange(
                0, df[xname].max() + thirty_seconds, thirty_seconds
            )
            ax.set_xticks(minor_ticks, minor=True)

            # Are there are more than 84 minor ticks?
            if len(minor_ticks) > 84:

                # Set major x-axis tick labels to every 5 minutes
                five_minutes = 1_000 * 60 * 5
                major_ticks = np.arange(
                    0, df[xname].max() + five_minutes, five_minutes
                )
                ax.set_xticks(major_ticks)

            # Otherwise, set the major x-axis tick labels to every 60 seconds
            else:
                sixty_seconds = 1_000 * 60
                major_ticks = np.arange(
                    0, df[xname].max() + sixty_seconds, sixty_seconds
                )
                ax.set_xticks(major_ticks)

        # Humanize the x tick labels if there is an xtick text funtion
        if xtick_text_fn is not None:
            xticklabels_list = []
            for text_obj in ax.get_xticklabels():

                # Convert numerical values to minutes+seconds format
                text_obj.set_text(xtick_text_fn(text_obj))

                xticklabels_list.append(text_obj)
            if len(xticklabels_list) > 17:
                ax.set_xticklabels(xticklabels_list, rotation=90)
            else:
                ax.set_xticklabels(xticklabels_list)

        # Humanize the y tick labels
        yticklabels_list = []
        for text_obj in ax.get_yticklabels():
            text_obj.set_text(
                humanize.intword(int(text_obj.get_position()[1]))
            )
            yticklabels_list.append(text_obj)
        ax.set_yticklabels(yticklabels_list)

        return ax

    @staticmethod
    def plot_grouped_box_and_whiskers(
        transformable_df,
        x_column_name,
        y_column_name,
        x_label,
        y_label,
        transformer_name='min',
        is_y_temporal=True
    ):
        """
        Create a grouped box plot visualization to compare the distribution of
        a numerical variable across different groups.

        Parameters:
            transformable_df (pandas.DataFrame):
                DataFrame containing the data to be plotted.
            x_column_name (str):
                The name of the categorical variable to group by and column
                name for the x-axis.
            y_column_name (str): Column name for the y-axis.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            transformer_name (str, optional):
                Name of the transformation applied to the y-axis values before
                plotting (default: 'min').
            is_y_temporal (bool, optional):
                If True, y-axis labels will be formatted as temporal values
                (default: True).

        Returns:
            None
                The function plots the graph directly using seaborn and
                matplotlib.
        """

        # Get the transformed data frame
        if transformer_name is None:
            transformed_df = transformable_df
        else:
            groupby_columns = ['session_uuid', 'scene_id']
            transformed_df = (
                transformable_df.groupby(groupby_columns)
                .filter(lambda df: not df[y_column_name].isnull().any())
                .groupby(groupby_columns)
                .transform(transformer_name)
                .reset_index(drop=False)
                .sort_values(y_column_name)
            )

        # Create a figure and subplots
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Create a box plot of the y column grouped by the x column
        sns.boxplot(
            x=x_column_name,
            y=y_column_name,
            showmeans=True,
            data=transformed_df,
            ax=ax
        )

        # Rotate the x-axis labels to prevent overlapping
        plt.xticks(rotation=45)

        # Label the x- and y-axis
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Humanize y tick labels
        if is_y_temporal:
            yticklabels_list = []
            for text_obj in ax.get_yticklabels():
                text_obj.set_text(
                    humanize.precisedelta(
                        timedelta(milliseconds=text_obj.get_position()[1])
                    )
                    .replace(', ', ',\n')
                    .replace(' and ', ' and\n')
                )
                yticklabels_list.append(text_obj)
            ax.set_yticklabels(yticklabels_list)

    def get_r_squared_value_latex(self, xdata, ydata):
        """
        Calculate the R-squared value and its p-value, and format them in
        LaTeX.

        This function computes the Pearson correlation coefficient between
        two input data arrays (`xdata` and `ydata`). It then calculates
        the R-squared value (coefficient of determination) as the square
        of the Pearson coefficient. The function also retrieves the
        p-value associated with the correlation test.

        The results are formatted into a LaTeX string suitable for
        scientific reports. The string includes the R-squared value
        (rounded to two decimal places), a comma, and the p-value
        (formatted according to significance level: '<0.0001' for p-value
        less than 0.0001, otherwise displayed with four decimal places).

        Parameters:
            xdata (array-like):
                The explanatory variable data.
            ydata (array-like):
                The response variable data.

        Returns:
            str
                A LaTeX string representing the R-squared value and its
                corresponding p-value.
        """

        # Get a mask to filter out infinite and NaN values from the data
        inf_nan_mask = self.get_inf_nan_mask(xdata, ydata)

        # Import the Pearson correlation function from scipy.stats
        from scipy.stats import pearsonr

        # Calculate Pearson's r and the associated p-value
        pearson_r, p_value = pearsonr(
            xdata[inf_nan_mask], ydata[inf_nan_mask]
        )

        # Format coefficient of determination to 2 decimal places
        cods = str('%.2f' % pearson_r**2)

        # Format p-value based on significance level
        if p_value < 0.0001:
            pvalue_statement = '<0.0001'
        else:
            pvalue_statement = '=' + str('%.4f' % p_value)

        # Construct the LaTeX string for the R-squared value and p-value
        s_str = '$r^2=' + cods + ',\\ p' + pvalue_statement + '$'

        # Return the formatted LaTeX string
        return s_str

    @staticmethod
    def get_spearman_rho_value_latex(xdata, ydata):
        """
        Calculate the Spearman's rank correlation coefficient and its
        p-value, and format them in LaTeX.

        This function computes the Spearman's rank correlation coefficient
        between two input data arrays (`xdata` and `ydata`). Spearman's
        rank correlation measures the monotonic relationship between two
        variables, unlike Pearson's correlation which assumes a linear
        relationship. The function also retrieves the p-value associated
        with the correlation test.

        The results are formatted into a LaTeX string suitable for
        scientific reports. The string includes the Spearman's rank
        correlation coefficient (rounded to two decimal places), a comma,
        and the p-value (formatted according to significance level:
        '<0.0001' for p-value less than 0.0001, otherwise displayed with
        four decimal places).

        Parameters:
            xdata (array-like):
                The explanatory variable data.
            ydata (array-like):
                The response variable data.

        Returns:
            str
                A LaTeX string representing the Spearman's rank correlation
                coefficient and its corresponding p-value.
        """

        # Import the Spearman's rank correlation function from scipy.stats
        from scipy.stats import spearmanr

        # Calculate Spearman's rank correlation and the associated p-value
        spearman_corr, p_value = spearmanr(xdata, ydata)

        # Format Spearman's rank correlation coefficient to 2 decimal places
        rccs = str('%.2f' % spearman_corr)

        # Format p-value based on significance level
        if p_value < 0.0001:
            pvalue_statement = '<0.0001'
        else:
            pvalue_statement = '=' + str('%.4f' % p_value)

        # Construct the LaTeX str for the correlation and p-value
        s_str = '$\\rho=' + rccs + ',\\ p' + pvalue_statement + '$'

        # Return the formatted LaTeX string
        return s_str

    def first_order_linear_scatterplot(
        self, df, xname, yname,
        xlabel_str='Overall Capitalism (explanatory variable)',
        ylabel_str='World Bank Gini % (response variable)',
        x_adj='capitalist', y_adj='unequal',
        title='"Wealth inequality is huge in the capitalist societies"',
        idx_reference='United States', annot_reference='most evil',
        aspect_ratio=None, least_x_xytext=(40, -10),
        most_x_xytext=(-150, 55), least_y_xytext=(-200, -10),
        most_y_xytext=(45, 0), reference_xytext=(-75, 25),
        color_list=None, verbose=False
    ):
        """
        Create a first-order linear scatterplot with annotations.

        This function generates a scatterplot for a provided DataFrame
        (`df`) using specified column names (`xname` and `yname`) for the
        x and y axes. It also adds custom labels, annotations for extreme
        data points, and an R-squared value.

        Parameters:
            df (pandas.DataFrame):
                The DataFrame containing the data for the scatterplot.
            xname (str):
                The name of the column to be used for the x-axis.
            yname (str):
                The name of the column to be used for the y-axis.
            xlabel_str (str, optional):
                The label for the x-axis. Defaults to "Overall Capitalism
                (explanatory variable)".
            ylabel_str (str, optional):
                The label for the y-axis. Defaults to "World Bank Gini %
                (response variable)".
            x_adj (str, optional):
                Adjective to describe the x-axis variable (used in
                annotations). Defaults to "capitalist".
            y_adj (str, optional):
                Adjective to describe the y-axis variable (used in
                annotations). Defaults to "unequal".
            title (str, optional):
                The title for the plot. Defaults to '"Wealth inequality is
                huge in the capitalist societies"'.
            idx_reference (str, optional):
                The index label to use for a reference annotation. Defaults to
                "United States".
            annot_reference (str, optional):
                The reference text for the reference annotation. Defaults to
                "most evil".
            aspect_ratio (float, optional):
                The aspect ratio of the figure. Defaults to using a pre-defined
                value from self.facebook_aspect_ratio.
            least_x_xytext (tuple[float, float], optional):
                The position for the "least x" annotation relative to the data
                point. Defaults to (40, -10).
            most_x_xytext (tuple[float, float], optional):
                The position for the "most x" annotation relative to the data
                point. Defaults to (-150, 55).
            least_y_xytext (tuple[float, float], optional):
                The position for the "least y" annotation relative to the data
                point. Defaults to (-200, -10).
            most_y_xytext (tuple[float, float], optional):
                The position for the "most y" annotation relative to the data
                point. Defaults to (45, 0).
            reference_xytext (tuple[float, float], optional):
                The position for the reference point annotation relative to the
                data point. Defaults to (-75, 25).
            color_list (list[str], optional):
                A list of colors to use for the data points. Defaults to None
                (using default scatter plot colors).
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            tuple
                The figure and axis object for the generated scatter plot.
        """

        # Ensure the dataframe index is string-typed for annotations
        assert pd.api.types.is_string_dtype(df.index) and all(
            isinstance(x, str) for x in df.index
        ), "df must have an index labeled with strings"

        # Set the aspect ratio if not provided
        if aspect_ratio is None:
            aspect_ratio = self.facebook_aspect_ratio

        # Calculate figure dimensions based on aspect ratio
        fig_width = 18
        fig_height = fig_width / aspect_ratio

        # Create figure and axis objects for the plot
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111, autoscale_on=True)

        # Define line properties for the regression line
        line_kws = dict(color='k', zorder=1, alpha=.25)

        # Set scatter plot properties, including color list if provided
        if color_list is None:
            scatter_kws = dict(s=30, linewidths=.5, edgecolors='k', zorder=2)
        else:
            scatter_kws = dict(
                s=30, linewidths=.5, edgecolors='k', zorder=2, color=color_list
            )

        # Create the scatter plot with regression line
        sns.regplot(
            x=xname, y=yname, scatter=True, data=df, ax=ax,
            scatter_kws=scatter_kws, line_kws=line_kws
        )

        # Append explanatory variable text to x-axis label if not present
        if not xlabel_str.endswith(' (explanatory variable)'):
            xlabel_str = f'{xlabel_str} (explanatory variable)'

        # Set the x-axis label
        plt.xlabel(xlabel_str)

        # Append response variable text to y-axis label if not present
        if not ylabel_str.endswith(' (response variable)'):
            ylabel_str = f'{ylabel_str} (response variable)'

        # Set the y-axis label
        plt.ylabel(ylabel_str)

        # Define common annotation properties
        kwargs = dict(
            textcoords='offset points', ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        # Extract x and y data for annotations
        xdata = df[xname].values
        least_x = xdata.min()
        if verbose:
            print(f'least_x = {least_x}')
        most_x = xdata.max()
        if verbose:
            print(f'most_x = {most_x}')

        ydata = df[yname].values
        most_y = ydata.max()
        if verbose:
            print(f'most_y = {most_y}')
        least_y = ydata.min()
        if verbose:
            print(f'least_y = {least_y}')

        # Initialize flags to ensure each annotation is only added once
        least_x_tried = False
        most_x_tried = False
        least_y_tried = False
        most_y_tried = False

        # Annotate specific data points based on their values
        for label, x, y in zip(df.index, xdata, ydata):
            if x == least_x and (not least_x_tried):
                plt.annotate(
                    '{} (least {})'.format(label, x_adj), xy=(x, y),
                    xytext=least_x_xytext, **kwargs
                )
                least_x_tried = True
            elif x == most_x and (not most_x_tried):
                plt.annotate(
                    '{} (most {})'.format(label, x_adj), xy=(x, y),
                    xytext=most_x_xytext, **kwargs
                )
                most_x_tried = True
            elif y == least_y and (not least_y_tried):
                plt.annotate(
                    '{} (least {})'.format(label, y_adj), xy=(x, y),
                    xytext=least_y_xytext, **kwargs
                )
                least_y_tried = True
            elif y == most_y and (not most_y_tried):
                plt.annotate(
                    '{} (most {})'.format(label, y_adj), xy=(x, y),
                    xytext=most_y_xytext, **kwargs
                )
                most_y_tried = True
            elif label == idx_reference:
                plt.annotate(
                    '{} ({})'.format(label, annot_reference), xy=(x, y),
                    xytext=reference_xytext, **kwargs
                )

        # Set the plot title with specified title string
        fig.suptitle(t=title, x=0.5, y=0.91)

        # Annotate the r-squared value on the plot
        s_str = self.get_r_squared_value_latex(df[xname], df[yname])
        ax.text(
            0.75, 0.9, s_str, alpha=0.5, transform=ax.transAxes,
            fontsize='x-large'
        )

        return (fig, ax)

    def plot_inauguration_age(
        self,
        inauguration_df,
        groupby_column_name,
        xname,
        leader_designation,
        label_infix,
        label_suffix,
        info_df,
        title_prefix,
        inaugruation_verb='Inauguration',
        legend_tuple=None,
        verbose=False
    ):
        """
        Plot a scatter plot of leaders' ages at inauguration over time, with
        optional groupings and background shading.

        Parameters:
            inauguration_df (pandas.DataFrame):
                DataFrame containing leadership inauguration data.
            groupby_column_name (str):
                Column name for grouping leaders (e.g., country, party).
            xname (str):
                The name of the x-axis variable, representing the year of
                inauguration.
            leader_designation (str):
                The designation of the leaders, such as "President" or
                "Governor".
            label_infix (str):
                Text to be inserted in the label between leader designation
                and groupby_column.
            label_suffix (str):
                Text to be appended to the label.
            info_df (pandas.DataFrame):
                DataFrame containing additional information about turning
                years.
            title_prefix (str):
                A prefix to add to the plot title.
            inaugruation_verb (str, optional):
                The verb to use for inauguration, such as "inauguration" or
                "swearing-in". Defaults to "Inauguration".
            legend_tuple (tuple, optional):
                A tuple specifying the location of the legend, such as (0.02,
                0.76). Defaults to None.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            None
                The function plots the graph directly using matplotlib.
        """

        # Configure the color dictionary
        color_cycler = self.get_color_cycler(
            info_df[groupby_column_name].unique().shape[0]
        )
        face_color_dict = {}
        for groupby_column, fc_dict in zip(
            info_df[groupby_column_name].unique(), color_cycler()
        ):
            face_color_dict[groupby_column] = fc_dict['color']

        # Plot and annotate the figure
        figwidth = 18
        fig, ax = plt.subplots(
            figsize=(figwidth, figwidth/self.twitter_aspect_ratio)
        )
        used_list = []
        import textwrap
        for groupby_column, df in inauguration_df.sort_values(
            'office_rank'
        ).groupby(groupby_column_name):
            if groupby_column[0] in ['A', 'U']:
                ana = 'an'
            else:
                ana = 'a'
            label = (
                f'{leader_designation.title()} {label_infix} {ana}'
                f' {groupby_column} {label_suffix}'
            ).strip()

            # Convert the array to a 2-D array with a single row
            reshape_tuple = (1, -1)
            color = face_color_dict[groupby_column].reshape(reshape_tuple)

            # Plot and annotate all points from the index
            for leader_name, row_series in df.iterrows():
                if groupby_column not in used_list:
                    used_list.append(groupby_column)
                    df.plot(
                        x=xname,
                        y='age_at_inauguration',
                        kind='scatter',
                        ax=ax,
                        label=label,
                        color=color
                    )
                else:
                    df.plot(
                        x=xname,
                        y='age_at_inauguration',
                        kind='scatter',
                        ax=ax,
                        color=color
                    )
                plt.annotate(
                    textwrap.fill(leader_name, width=10),
                    (row_series[xname], row_series.age_at_inauguration),
                    textcoords='offset points',
                    xytext=(0, -4),
                    ha='center',
                    va='top',
                    fontsize=6
                )

        # Add 5 years to the height
        bottom, top = ax.get_ylim()
        height_tuple = (bottom, top+5)
        ax.set_ylim(height_tuple)
        bottom, top = ax.get_ylim()
        height = top - bottom

        # Get the background shading wrap width
        left, right = ax.get_xlim()
        min_shading_width = 9999
        min_turning_name = ''
        wrap_width = info_df.turning_name.map(lambda x: len(x)).min()
        for row_index, row_series in info_df.iterrows():
            turning_year_begin = max(row_series.turning_year_begin, left)
            turning_year_end = min(row_series.turning_year_end, right)
            width = turning_year_end - turning_year_begin
            if width > 0 and width < min_shading_width:
                min_shading_width = width
                min_turning_name = row_series.turning_name
                wrap_width = len(min_turning_name)

        # Add the turning names as background shading
        from matplotlib.patches import Rectangle
        for row_index, row_series in info_df.iterrows():
            turning_year_begin = max(row_series.turning_year_begin, left)
            turning_year_end = min(row_series.turning_year_end, right)
            width = turning_year_end - turning_year_begin
            if width > 0:
                groupby_column = row_series[groupby_column_name]
                turning_name = row_series.turning_name
                rect = Rectangle(
                    (turning_year_begin, bottom), width, height,
                    color=face_color_dict[groupby_column],
                    fill=True, edgecolor=None, alpha=0.1
                )
                ax.add_patch(rect)
                plt.annotate(
                    textwrap.fill(
                        turning_name, width=wrap_width,
                        break_long_words=False
                    ),
                    (turning_year_begin+(width/2), top),
                    textcoords='offset points', xytext=(0, -6),
                    ha='center', fontsize=7, va='top', rotation=-90
                )

        # Set legend
        if legend_tuple is None:
            legend_tuple = (0.02, 0.76)
        legend_obj = ax.legend(loc=legend_tuple)
        if verbose:

            # Get the bounding box of the legend relative to the anchor point
            bbox_to_anchor = legend_obj.get_bbox_to_anchor()

            # Print the size and position of the bounding box
            print(
                bbox_to_anchor.width, bbox_to_anchor.height,
                bbox_to_anchor.xmin, bbox_to_anchor.ymin,
                bbox_to_anchor.xmax, bbox_to_anchor.ymax
            )

            # Get the bounding box of the legend
            bounding_box = legend_obj.get_tightbbox()

            # Print the size and position of the bounding box
            print(
                bounding_box.width, bounding_box.height,
                bounding_box.xmin, bounding_box.ymin,
                bounding_box.xmax, bounding_box.ymax
            )

        # Set labels
        ax.set_xlabel(f'Year of {inaugruation_verb}')
        ax.set_ylabel(f'Age at {inaugruation_verb}')
        ax.set_title(
            f'{title_prefix} {inaugruation_verb} Age vs Year'
        )

    @staticmethod
    def save_fig_as_various(
        fig,
        chart_name,
        dir_names_list=['pgf', 'png', 'svg'],
        verbose=False
    ):
        """
        Save a matplotlib figure to multiple formats in specified directories.

        Parameters:
            fig (Figure): The matplotlib figure to save.
            chart_name (str): The base name for the saved files.
            dir_names_list (list[str]):
                A list of directory names (and file extensions) to save the
                figure in.
            verbose (bool): If True, prints the file paths of the saved plots.

        Returns:
            None
        """
        for dir_name in dir_names_list:
            try:

                # Create the directory path
                dir_path = os.path.join(os.pardir, 'saves', dir_name)

                # Create the directory if it doesn't exist
                os.makedirs(dir_path, exist_ok=True)

                # Construct the file path
                file_path = os.path.join(dir_path, f'{chart_name}.{dir_name}')

                # Remove the file if it already exists
                if os.path.exists(file_path):
                    os.remove(file_path)

                # Save the figure to the file
                if verbose:
                    print(f'Saving plot to {os.path.abspath(file_path)}')
                fig.savefig(file_path, bbox_inches='tight')

            except Exception as e:
                # Handle exceptions and print a clean error message
                print(f"Error saving plot to {dir_name}: {str(e).strip()}")

    def ball_and_chain(self, ax, index, values, face_color=None, label=None):
        ax.plot(index, values, c='k', zorder=1, alpha=.25)
        if face_color is None:
            colormap = self.get_random_colormap()
            import matplotlib as mpl
            cmap = mpl.colormaps.get_cmap(colormap)
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=values.min(), vmax=values.max())
            face_color = cmap(norm(values))
        if label is None:
            ax.scatter(
                index, values, s=30, lw=.5, c=face_color, edgecolors='k',
                zorder=2
            )
        else:
            ax.scatter(
                index, values, s=30, lw=.5, c=face_color, edgecolors='k',
                zorder=2, label=label
            )

    @staticmethod
    def get_random_colormap():
        import random
        import matplotlib.pyplot as plt

        return random.choice(plt.colormaps())

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

    @staticmethod
    def plot_grouped_pie_chart(
        df, column_name, slice_label, slice_cutoff=None, verbose=False
    ):
        """
        Create a grouped pie chart to visualize the distribution of
        values in a DataFrame column.

        This function generates a pie chart representing the distribution
        of unique values within a specified column (`column_name`) of a
        pandas DataFrame (`df`). It filters out any missing values (NaN)
        before counting the occurrences of each unique value.

        The function allows grouping less frequent values into a single
        slice labeled according to the provided `slice_label`. This helps
        focus on the most prominent categories while still acknowledging
        the presence of less frequent ones. The grouping threshold
        (`slice_cutoff`) can be customized, or it defaults to 2% of the
        total count.

        Parameters:
            df (pandas.DataFrame):
                The DataFrame containing the column to be plotted.
            column_name (str):
                The name of the column to be visualized.
            slice_label (str):
                The label for the slice grouping less frequent values
                (defaults to 'Other').
            slice_cutoff (int or float, optional):
                The count cutoff for grouping values into one slice. If None,
                defaults to 2% of the total count.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            matplotlib.pyplot.Figure:
                The matplotlib figure object representing the generated pie
                chart.
        """

        # Filter for non-null values in the specified column
        mark_series = ~df[column_name].isnull()
        filtered_values = df[mark_series][column_name]

        if verbose:
            print(f'Total number of rows in DataFrame: {len(df)}')
            print(
                "Number of non-null values"
                f" in column '{column_name}':"  # noqa E231
                f" {len(filtered_values)}"
            )

        # Count the occurrences of each unique value
        value_counts = filtered_values.value_counts()

        # Determine the cutoff for grouping smaller slices
        if slice_cutoff is None:
            slice_cutoff = value_counts.sum()*0.02
        if verbose:
            print(
                'Slice cutoff for grouping less frequent values:'
                f' {slice_cutoff}'
            )

        # Group values below cutoff and create a new slice for them
        grouped_value_counts = value_counts[value_counts > slice_cutoff]

        # Sum the values below the cutoff
        other_slice = value_counts[value_counts <= slice_cutoff].sum()

        # Add the grouped slice to the counts
        if other_slice:
            grouped_value_counts[slice_label] = other_slice

        # Print the grouped value counts if verbose
        if verbose:
            print('Grouped value counts:\n', grouped_value_counts)

        # Set the figure size
        plt.figure(figsize=(8, 8))

        # Plot the pie chart
        plt.pie(
            grouped_value_counts, labels=grouped_value_counts.index,
            autopct='%1.1f%%', startangle=90
        )

        # Set the title of the chart
        plt.title(f'Distribution of {column_name}')

        # Return the matplotlib figure object
        return plt

    @staticmethod
    def plot_right_circles(
        tuples_list, draw_a_two_circle_venn_diagram, verbose=False
    ):
        """
        Plot a series of two-circle Venn diagrams in a grid layout.

        This function takes a list of tuples, where each tuple contains
        data for a two-circle Venn diagram. It creates a subplot for each
        Venn diagram and displays them in a grid layout.

        Parameters:
            tuples_list (list of tuples):
                A list of tuples, each containing the data for one Venn
                diagram.
            draw_a_two_circle_venn_diagram (function):
                A function that draws a two-circle Venn diagram given a tuple
                of data and an axes object.
            verbose (bool, optional):
                Whether to print debug or status messages. Defaults to False.

        Returns:
            matplotlib.figure.Figure
                The matplotlib figure object of the grid of Venn diagrams.
        """

        # Get the number of plots to create
        plot_count = len(tuples_list)

        # Determine optimal grid layout to have maximum of 3 columns per row
        ncols = min(3, plot_count)

        # Determine the optimal grid layout to have at least 1 row
        nrows = max(1, int(math.ceil(plot_count / ncols)))

        # Create the Matplotlib figure and subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4))

        # Print type information for debugging if verbose flag is set
        if verbose:
            print(f'axes type: {type(axes)}')

        # Loop through data tuples and corresponding subplots
        for i, right_circle_tuple in enumerate(tuples_list):

            # Calculate column index based on loop counter and columns count
            col = i % ncols

            # Print debugging information if verbose flag is set
            if verbose:
                print(f'Index: {i}, Column: {col}, Columns: {ncols}')

            # Is there more than one row?
            if nrows > 1:

                # Base the row index on loop counter and columns
                row = int(math.floor(i / ncols))

                # Access the specific subplot
                ax = axes[row, col]

            # If only one plot, axes is not an array
            elif plot_count == 1:
                ax = axes

            # If only one row, access by column index
            else:
                ax = axes[col]

            # Assert expected type for the subplot object
            import matplotlib
            assert isinstance(ax, matplotlib.axes.SubplotBase), (
                f'ax is of type {type(ax)}, it should be'
                ' matplotlib.axes.SubplotBase'
            )

            # Draw the Venn diagram on the current axes
            draw_a_two_circle_venn_diagram(right_circle_tuple, ax=ax)

        # Hide any unused subplots in the grid (in reverse order)
        for i in range(nrows * ncols, plot_count, -1):

            # Calculate column index based on loop counter and columns count
            col = (i - 1) % ncols

            # Print debugging information if verbose flag is set
            if verbose:
                print(f'Index: {i}, Column: {col}, Columns: {ncols}')

            # Are there are more than one rows?
            if nrows > 1:

                # Determine subplot based on number of rows and plot count
                row = (i - 1) // 3
                ax = axes[row, col]

            # If its a single, the plot occupies the entire figure
            elif plot_count == 1:
                ax = axes

            else:
                ax = axes[i]

            # Turn off axis visibility for unused subplots
            ax.axis('off')

        # Adjust spacing between subplots for better presentation
        plt.tight_layout()

        # Return the Matplotlib figure object containing the generated plots
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

        Example:
            import matplotlib.pyplot as plt
            import random

            # Define the sequence of user actions
            sequence = ["SESSION_START", "LOGIN", "VIEW_PRODUCT"]

            # Generate more shopping elements
            for _ in range(19):
                if sequence[-1] != 'ADD_TO_CART':
                    sequence.append(random.choice(['VIEW_PRODUCT', 'ADD_TO_CART']))
                else:
                    sequence.append('VIEW_PRODUCT')

            # Finish up the shopping
            sequence += ["LOGOUT", "SESSION_END"]

            # Define n-grams to highlight
            highlighted_ngrams = [["VIEW_PRODUCT", "ADD_TO_CART"]]

            # Define a custom color dictionary for the actions
            color_dict = {
                "SESSION_START": "green",
                "LOGIN": "blue",
                "VIEW_PRODUCT": "orange",
                "ADD_TO_CART": "purple",
                "LOGOUT": "red",
                "SESSION_END": "black"
            }

            # Plot the sequence
            fig, ax = nu.plot_sequence(
                sequence=sequence,
                highlighted_ngrams=highlighted_ngrams,
                color_dict=color_dict,
                suptitle="User Session Sequence",
                verbose=False
            )

            # Show the plot
            plt.show()
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
        # if np_sequence.dtype.str not in ['<U21', '<U11']:
        #     int_sequence = np_sequence

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
                    print(fitted_equation)

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
                    print(fitted_equation)

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
                    import re
                    file_path = osp.join(
                        self.saves_png_folder,
                        re.sub(
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

    def plot_semantic_distances_dendogram(
        self, collection_name="collection", documents_list=[], ids_list=[],
        plot_title='A Test Dendogram'
    ):

        # Import the ChromaDB library for semantic search and document storage
        import chromadb

        # Initialize the ChromaDB client
        chroma_client = chromadb.Client()

        # Retrieve existing collections from ChromaDB
        existing_collections = chroma_client.list_collections()

        # Check if the collection already exists; if not, create it
        if any([
            collection.name == collection_name
            for collection in existing_collections
        ]):
            collection = chroma_client.get_collection(collection_name)

        # Create a collection in ChromaDB and add documents
        else:
            collection = chroma_client.create_collection(name=collection_name)

        # Check if the document is already in the collection; if not, add it
        existing_ids = collection.get()['ids']
        map_obj = map(lambda id: id in existing_ids, ids_list)
        if all(map_obj):
            pass
        elif any(map_obj):
            for document, id in zip(documents_list, ids_list):
                if id not in existing_ids:
                    collection.add(
                        documents=[document],
                        ids=[id]
                    )
            existing_ids = collection.get()['ids']

        # Add all documents to the collection
        else:
            collection.add(
                documents=documents_list,
                ids=ids_list
            )
            existing_ids = collection.get()['ids']

        # Calculate semantic distances between documents in the collection
        existing_documents = collection.get()['documents']
        distance_matrix = []
        for i, document_name in enumerate(existing_ids):
            results = collection.query(
                query_texts=[existing_documents[i]],
                n_results=len(existing_ids)
            )
            distance_matrix.append(results['distances'][0])
        distance_matrix = np.array(distance_matrix)

        # Perform hierarchical clustering on the semantic distance matrix
        from scipy.cluster.hierarchy import dendrogram, linkage
        import matplotlib.pyplot as plt

        # Generate a dendrogram to visualize the clustering
        linked = linkage(distance_matrix, 'single')
        fig, ax = plt.subplots(figsize=(18, 7))
        dendrogram(linked, labels=existing_ids)
        ax.set_ylabel('Semantic Distance')
        fig.suptitle(plot_title, fontsize=24)
        plt.tight_layout()
        plt.show()

        return (collection, fig, ax)

    def show_subgraph(
        self, sub_graph, suptitle='Within-function Function Calls',
        nodes_list_list=None, node_color='b', verbose=False
    ):
        """
        Visualize a subgraph with a custom layout and optional node grouping.

        Parameters:
            sub_graph : networkx.Graph
                The input subgraph to be visualized.
            suptitle : str, optional
                The title of the graph visualization. Default is
                'Within-function Function Calls'.
            nodes_list_list : list of list of str, optional
                A list of lists, where each inner list contains node names to
                be grouped and colored separately. If None, all nodes are
                displayed with the same color.
            node_color : str or ndarray, optional
                The default color for all nodes if `nodes_list_list` is None.
                Default is 'b' (blue).
            verbose : bool, optional
                If True, prints additional debug information during execution.
                Default is False.

        Returns:
            None
                The function generates a visualization and does not return any
                value.

        Notes:
            - The graph layout is adjusted for readability using
              `nx.spring_layout`.
            - If `nodes_list_list` is provided, each group of nodes is colored
              differently.

        Examples:
            import networkx as nx
            sub_graph = nx.erdos_renyi_graph(10, 0.3)
            nu.show_subgraph(sub_graph)
            nodes_list_list = [['node1', 'node2'], ['node3', 'node4']]
            nu.show_subgraph(
                sub_graph, nodes_list_list=nodes_list_list, verbose=True
            )
        """

        # Vertically separate the labels for easier readability
        import networkx as nx
        layout_items = nx.spring_layout(sub_graph).items()
        left_lim, right_lim = (-1500, 1500)
        bottom_lim = left_lim * self.twitter_aspect_ratio
        top_lim = right_lim * self.twitter_aspect_ratio
        rows_list = [{
            'node_name': node_name, 'layout_x': pos_array[0],
            'layout_y': pos_array[1]
        } for node_name, pos_array in layout_items]
        df = DataFrame(rows_list).sort_values('layout_x')
        df['adjusted_x'] = [int(round(el)) for el in pd.cut(
            np.array([left_lim, right_lim]), len(sub_graph.nodes)+1,
            retbins=True
        )[1]][1:-1]
        df = df.sort_values('layout_y')
        df['adjusted_y'] = [int(round(el)) for el in pd.cut(
            np.array([bottom_lim, top_lim]), len(sub_graph.nodes)+1,
            retbins=True
        )[1]][1:-1]

        # Create the layout dictionary
        layout_dict = {}
        for row_index, row_series in df.iterrows():
            node_name = row_series.node_name
            layout_x = row_series.adjusted_x
            layout_y = row_series.adjusted_y
            layout_dict[node_name] = np.array([
                float(layout_x), float(layout_y)
            ])

        # Draw the graph using the layout
        fig, ax = plt.subplots(figsize=(18, 7), facecolor='white')

        # Make the nodes the node_color
        if nodes_list_list is None:
            nx.draw_networkx_nodes(
                G=sub_graph, pos=layout_dict, alpha=0.33,
                node_color=node_color.reshape(1, -1), node_size=150, ax=ax
            )
            nx.draw_networkx_edges(
                G=sub_graph, pos=layout_dict, alpha=0.25, ax=ax
            )
            nx.draw_networkx_labels(
                G=sub_graph, pos=layout_dict, font_size=10, ax=ax
            )

        # Color each nodes list differently
        else:
            if verbose:
                display(nodes_list_list)
            color_cycler = self.get_color_cycler(len(nodes_list_list))
            for nodes_list, fcd in zip(nodes_list_list, color_cycler()):
                if verbose:
                    display(fcd['color'])
                node_color = fcd['color'].reshape(1, -1)
                sub_subgraph = nx.subgraph(sub_graph, nodes_list)
                nx.draw_networkx_nodes(
                    G=sub_subgraph, pos=layout_dict, alpha=0.5,
                    node_color=node_color, node_size=300, ax=ax
                )
                nx.draw_networkx_edges(
                    G=sub_subgraph, pos=layout_dict, alpha=0.25, ax=ax
                )
                nx.draw_networkx_labels(
                    G=sub_subgraph, pos=layout_dict, font_size=9, ax=ax
                )

        plt.axis('off')
        plt.xticks([], [])
        plt.yticks([], [])
        fig.suptitle(suptitle, fontsize=24)
        plt.show()

        return (layout_dict, fig, ax)


# print('\\b(' + '|'.join(dir()) + ')\\b')
