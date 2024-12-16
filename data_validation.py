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

from os import (
    listdir as listdir, makedirs as makedirs, path as osp
)
from pandas import (
    DataFrame
)
import matplotlib.pyplot as plt
import numpy as np
import os
import time


class DataValidation:
    def __init__(
        self, saves_folder_path=None, verbose=False
    ):

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

        # Create the assumed directories
        self.saves_pickle_folder = osp.join(self.saves_folder, 'pkl')
        makedirs(name=self.saves_pickle_folder, exist_ok=True)

        # Handy list of the different types of encodings
        self.encoding_types_list = ['utf-8', 'latin1', 'iso8859-1']
        self.encoding_type = self.encoding_types_list[0]

        # Various aspect ratios
        self.twitter_aspect_ratio = 16/9

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
            # and not in the excluded indices list
            if (
                current_idx in splitting_indices_list
                and current_idx not in excluded_indices_list
            ):

                # Add it to the current list
                current_list.append(current_idx)

            # Otherwise, if the current list is not empty, add it to the split
            # list and start a new current list
            else:
                if current_list:
                    split_list.append(current_list)
                current_list = []

        # If the current list is not empty, add it to the split list
        if current_list:
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

                    # Update the maximum similarity and corresponding item if
                    # a higher similarity is found
                    if this_similarity > max_similarity:
                        max_similarity = this_similarity
                        max_item = second_item

            # Create a row dictionary to store information for each similar
            # item pair
            row_dict = {'first_item': first_item, 'second_item': max_item}

            # Convert items to byte arrays and join them with '-' for string
            # representation
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
            sd_set = set(some_dict.keys()).symmetric_difference(set(
                df.similar_key
            ))
            typos_df = check_for_typos(
                list(set(df.similar_key).intersection(sd_set)),
                list(set(some_dict.keys()).intersection(sd_set)),
                verbose=False
            ).sort_values(
                ['max_similarity', 'left_item', 'right_item'],
                ascending=[False, True, True]
            )
            for i, r in typos_df.iterrows():
                print(
                    f"some_dict['{r.left_item}'] ="
                    f" some_dict.pop('{r.right_item}')"
                )
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

            # Iterate through items in the right list and find the most
            # similar item
            for right_item in right_list:
                this_similarity = self.compute_similarity(
                    left_item, right_item
                )
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

        # Create a data frame from the list of rows, rename columns if
        # necessary
        name_similarities_df = DataFrame(
            rows_list, columns=column_list
        ).rename(columns=rename_dict)

        # Print the time taken for the computation if verbose is True
        if verbose:
            t1 = time.time()
            print(t1-t0, time.ctime(t1))

        # Return the resulting data frame
        return name_similarities_df

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

        # Filter the file names to include only pickle files (.pkl or .pickle
        # extensions)
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

    # -------------------
    # Path Functions
    # -------------------

    # -------------------
    # Storage Functions
    # -------------------

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

        # Retrieve the coordinates for both points, defaulting to the origin
        # for the first point if not provided
        x1, x2, y1, y2, z1, z2 = self.get_coordinates(
            second_point, first_point=first_point
        )

        # Calculate the relative position by adding corresponding coordinates
        # and rounding to one decimal place
        relative_position = (
            round(x1 + x2, 1), round(y1 + y2, 1), round(z1 + z2, 1)
        )

        # Return the calculated relative position as a tuple
        return relative_position

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

        # Iterate over the randomly selected keys and add their corresponding
        # values to the sub-dictionary
        for key in random_keys:
            sub_dict[key] = super_dict[key]

        return sub_dict

    # -------------------
    # Plotting Functions
    # -------------------

    @staticmethod
    def get_color_cycler(n):
        """
        Generate a color cycler for plotting with a specified number of colors.

        This static method creates a color cycler object (`cycler.Cycler`)
        suitable for Matplotlib plotting. The color cycler provides a
        sequence of colors to be used for lines, markers, or other plot
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
            color_cycler = cycler('color', plt.cm.Accent(np.linspace(0, 1, n)))

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
