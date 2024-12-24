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
from datetime import timedelta
from os import (
    listdir as listdir, makedirs as makedirs, path as osp,
    walk as walk
)
from pandas import (
    DataFrame, Series, concat
)
from re import (
    IGNORECASE
)
import humanize
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import subprocess
import sys

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


class Uncategorized(BaseConfig):
    def __init__(
        self, data_folder_path=None, saves_folder_path=None, verbose=False
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
        self.pip_command_str = f'{sys.executable} -m pip'

        # Create the assumed directories
        self.saves_text_folder = osp.join(self.saves_folder, 'txt')
        makedirs(name=self.saves_text_folder, exist_ok=True)
        self.saves_wav_folder = osp.join(self.saves_folder, 'wav')
        makedirs(name=self.saves_wav_folder, exist_ok=True)
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

        # Module lists
        self.object_evaluators = [
            fn for fn in dir(inspect) if fn.startswith('is')
        ]

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

    # -------------------
    # List Functions
    # -------------------

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

    # -------------------
    # Module Functions
    # -------------------

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

    # -------------------
    # Pandas Functions
    # -------------------

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

    # -------------------
    # Sub-sampling Functions
    # -------------------

    # -------------------
    # Plotting Functions
    # -------------------

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
