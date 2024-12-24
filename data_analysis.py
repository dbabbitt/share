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
from bs4 import BeautifulSoup as bs
from numpy import nan
from os import (
    path as osp
)
from pandas import (
    DataFrame, Series, concat, get_dummies, notnull,
    to_datetime
)
from re import (
    Pattern, sub
)
import humanize
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import subprocess

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


class DataAnalysis(BaseConfig):
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

    # -------------------
    # Numeric Functions
    # -------------------

    # -------------------
    # String Functions
    # -------------------

    # -------------------
    # List Functions
    # -------------------

    # -------------------
    # File Functions
    # -------------------

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

    @staticmethod
    def clean_numerics(df, columns_list=None, verbose=False):
        import re
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

    # -------------------
    # 3D Point Functions
    # -------------------

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
        assert len(first_point) == len(second_point), (
            f'Mismatched dimensions: {len(first_point)}'
            f' != {len(second_point)}'
        )

        # Check if the points are in 3D
        if len(first_point) == 3:

            # Unpack the coordinates of the first and second points
            x1, y1, z1 = first_point
            x2, y2, z2 = second_point

            # Calculate the Euclidean distance for 3D points
            euclidean_distance = math.sqrt(
                (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
            )

        # Check if both points are 2D
        elif len(first_point) == 2:

            # Unpack the coordinates of the first and second points
            x1, z1 = first_point
            x2, z2 = second_point

            # Calculate the Euclidean distance for 2D points
            euclidean_distance = math.sqrt((x1 - x2)**2 + (z1 - z2)**2)

        # Return the calculated Euclidean distance
        return euclidean_distance

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

    # -------------------
    # Plotting Functions
    # -------------------

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

    def inspect_spread_points(self, spread_points):
        """
        Visualize the spread points in color space using a pie chart and a 3D
        scatter plot.

        This function creates two visualizations to represent the spread
        points:
        1. A pie chart showing the colors of the spread points, where the
           first point (assumed to be the fixed point) is highlighted with an
           exploded wedge.
        2. A 3D scatter plot showing the spatial distribution of the spread
           points in a unit RGB color cube, with the fixed point highlighted
           by a black edge.

        Assumptions:
            - The first element in `spread_points` is the fixed point.
            - The `spread_points` parameter is a 2D array-like object where
              each row represents a color in RGB format (e.g., `[R, G, B]`
              values scaled between 0 and 1).

        Parameters:
            spread_points (array-like):
                A collection of points in RGB color space. Each point is
                represented as an array-like structure containing three
                numerical values (red, green, blue).

        Visualizations:
            - Pie Chart:
                - Displays the colors of the spread points.
                - The fixed point (first element in `spread_points`) is
                  highlighted with an exploded wedge and annotated as "(fixed
                  point)".
            - 3D Scatter Plot:
                - Plots the spread points in a 3D RGB color cube.
                - Points are colored according to their RGB values.
                - The fixed point is highlighted with a black edge for better
                  visibility.
                - The corners of the unit cube (representing `[0, 0, 0]` for
                  black and `[1, 1, 1]` for white) are annotated as "Black
                  Corner" and "White Corner" respectively.

        Returns:
            None:
                The function displays the visualizations but does not return
                any value.

        Notes:
            - The function assumes that the nearest neighbors of the fixed
              point are determined and reordered using a helper method
              `self.get_nearest_neighbor`.
            - Ensure that `spread_points` is a NumPy array or can be
              converted to one for slicing and indexing operations.

        Example:
            import numpy as np
            spread_points = np.array([
                [0.2, 0.4, 0.6],  # Fixed point (assumed to be the first)
                [0.7, 0.2, 0.3],
                [0.1, 0.8, 0.4],
                [0.5, 0.5, 0.5]
            ])
            nu.inspect_spread_points(spread_points)
        """

        # Prepare colors for the pie chart and 3D scatter plot
        colors = [
            tuple(color) for color in spread_points
        ]  # Convert spread points to RGB tuples

        # Get locations list and color order
        fixed_point = colors[0]
        locations_list = colors[1:]
        color_order = [fixed_point]

        # Pop the nearest neighbor off the list and add it to the color order
        while locations_list:
            nearest_neighbor = self.get_nearest_neighbor(
                color_order[-1], locations_list
            )
            nearest_neighbor = locations_list.pop(
                locations_list.index(nearest_neighbor)
            )
            color_order.append(nearest_neighbor)
        num_points = len(color_order)

        # Create a figure with two subplots
        fig = plt.figure(figsize=(14, 6))

        # Left panel: Pie chart
        ax1 = fig.add_subplot(121)  # 1 row, 2 columns, 1st subplot
        ax1.pie(
            [1 for _ in range(num_points)],
            colors=color_order,
            explode=[0.1] + [0.0] * (num_points - 1),
            startangle=90,
        )

        # Add a curved arrow annotation pointing to the exploded wedge
        ax1.annotate(
            '(fixed point)',  # Text label
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=-0.2",
                facecolor='black',
            ),  # Arrow style
            fontsize=12,
            ha='center',
            xy=(-0.15, 1),  # Arrow tip location (near the exploded wedge)
            xytext=(-1, 1.0),  # Text location
        )

        # Add title and adjust aspect ratio
        ax1.set_title("Colors of Spread Points")
        ax1.set_aspect("equal")

        # Right panel: 3D scatter plot
        ax2 = fig.add_subplot(
            122, projection="3d"
        )  # 1 row, 2 columns, 2nd subplot
        ax2.scatter(
            spread_points[:, 0], spread_points[:, 1], spread_points[:, 2],
            c=colors, s=100, edgecolors=fixed_point, linewidth=3,
            label='Spread Points', alpha=1.0
        )

        # Highlight the fixed point with a black edge
        ax2.scatter(
            fixed_point[0], fixed_point[1], fixed_point[2],
            color=fixed_point, s=100, edgecolors='black', linewidth=3,
            label='Fixed Point', alpha=1.0
        )
        ax2.set_title("Spread Points in Unit Cube with Colored Edges")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()

        # Add annotations for the corners (for a unit cube)
        black_corner = [0, 0, 0]  # Black corner (origin)
        white_corner = [1, 1, 1]  # White corner (opposite corner)
        ax2.text(
            black_corner[0], black_corner[1], black_corner[2],
            'Black Corner',  # Text label
            color='black',  # Text color
            horizontalalignment='left',  # Text alignment
            bbox=dict(
                facecolor='white', edgecolor='none', alpha=0.3
            ),  # Add contrast background
        )
        ax2.text(
            white_corner[0], white_corner[1], white_corner[2],
            'White Corner',  # Text label
            color='white',  # Text color
            horizontalalignment='right',  # Text alignment
            bbox=dict(
                facecolor='black', edgecolor='none', alpha=0.3
            ),  # Add contrast background
        )

        # Display the combined plot
        plt.tight_layout()
        plt.show()

# print('\\b(' + '|'.join(dir()) + ')\\b')
