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
    makedirs as makedirs, path as osp
)
from re import (
    IGNORECASE
)
import matplotlib.pyplot as plt
import numpy as np
import re
import os


class BaseConfig:
    def __init__(self, data_folder_path=None, saves_folder_path=None):
        self.data_folder_path = data_folder_path
        self.saves_folder_path = saves_folder_path
        
        if data_folder_path and not os.path.exists(data_folder_path):
            os.makedirs(data_folder_path)
            
        if saves_folder_path and not os.path.exists(saves_folder_path):
            os.makedirs(saves_folder_path)

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

        # Handy list of the different types of encodings
        self.encoding_types_list = ['utf-8', 'latin1', 'iso8859-1']
        self.encoding_type = self.encoding_types_list[0]

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

        # Compile a regex pattern to match non-alphanumeric characters
        self.lower_ascii_regex = re.compile('[^a-z0-9]+')

        # Various aspect ratios
        self.facebook_aspect_ratio = 1.91
        self.twitter_aspect_ratio = 16/9

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

    # -------------------
    # Sub-sampling Functions
    # -------------------

    # -------------------
    # Plotting Functions
    # -------------------

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

# print('\\b(' + '|'.join(dir()) + ')\\b')
