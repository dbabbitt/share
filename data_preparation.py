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
from os import (
    makedirs as makedirs, path as osp
)
from pandas import (
    read_html
)
from re import (
    split, sub
)
import numpy as np
import os
import sys
import urllib

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


class DataPreparation(BaseConfig):
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

    # -------------------
    # Numeric Functions
    # -------------------

    # -------------------
    # String Functions
    # -------------------

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

    def get_evaluations(self, obj):
        evaluations_list = []
        for evaluator in self.object_evaluators:
            try:
                evaluation = eval(f'inspect.{evaluator}(obj)')
                if evaluation:
                    evaluations_list.append(evaluator[2:])
            except Exception:
                continue

        return evaluations_list

    def get_dir_tree(
        self, module_name, function_calls=[], contains_str=None,
        not_contains_str=None, recurse_classes=True, recurse_modules=False,
        import_call=None, verbose=False
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

        Notes:
            This function dynamically imports the specified module and
            retrieves its attributes, filtering them based on the provided
            criteria. It can also recursively explore classes and modules if
            specified.
        """

        # Try to get the module object by first importing it
        if import_call is None:
            import_call = 'import ' + module_name.split('.')[0]
        if verbose:
            print(import_call)
        try:
            exec(import_call)  # Execute the import statement
        except ImportError:
            pass  # Ignore import errors and continue
        module_obj = eval(module_name)

        # Iterate over the attributes of the module, excluding standard and
        # built-in modules
        for library_name in sorted(set(dir(module_obj)).difference(
            set(self.standard_lib_modules)
        ).difference(
            set(sys.builtin_module_names)
        )):
            if library_name.startswith('__'):
                continue  # Skip special attributes

            # Construct the full attribute name
            function_call = f'{module_name}.{library_name}'

            # Evaluate the function or class
            try:
                function_obj = eval(function_call)
            except Exception:
                function_obj = None

            # Get evaluations of the object from the inspect library
            evaluations_list = self.get_evaluations(function_obj)
            if evaluations_list:
                function_calls.append(function_call)

            if verbose:
                print(function_call, evaluations_list)

            # Recursively explore classes if specified
            if recurse_classes and 'class' in evaluations_list:
                function_calls = self.get_dir_tree(
                    module_name=function_call, function_calls=function_calls,
                    verbose=verbose
                )
                continue

            # Recursively explore modules if specified
            if recurse_modules and 'module' in evaluations_list:
                function_calls = self.get_dir_tree(
                    module_name=function_call, function_calls=function_calls,
                    verbose=verbose
                )
                continue

        # Apply filtering criteria if provided
        if not bool(contains_str) and bool(not_contains_str):
            function_calls = [
                fn
                for fn in function_calls
                if not_contains_str not in fn.lower()
            ]
        elif bool(contains_str) and (not bool(not_contains_str)):
            function_calls = [
                fn
                for fn in function_calls
                if contains_str in fn.lower()
            ]
        elif bool(contains_str) and bool(not_contains_str):
            function_calls = [fn for fn in function_calls if (
                contains_str in fn.lower()
            ) and (not_contains_str not in fn.lower())]

        # Return a sorted list of unique function calls
        return sorted(set(function_calls))

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

        # Get the parent td tag object
        tag_obj = self.get_td_parent(tag_obj, verbose=verbose)
        if verbose:
            display(tag_obj)

        # Traverse the siblings of the table tag object backward until a
        # style column is found
        from bs4.element import NavigableString
        while isinstance(
            tag_obj, NavigableString
        ) or not tag_obj.has_attr('style'):
            tag_obj = tag_obj.previous_sibling
            if verbose:
                display(tag_obj)

        # Display the text content of the found style column if verbose is
        # True
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

        # If the download directory is not specified, use the downloads
        # subdirectory
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

        # If the page URL or filepath is not a URL, ensure it exists and open
        # it using open() and get the page HTML that way
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
            page_tables_list = nu.get_page_tables(tables_url)

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

            # If it's not a URL or a filepath, assume it's a string
            # representation of the tables
            from io import StringIO

            # Create a StringIO object from the string
            f = StringIO(tables_url_or_filepath)

            # Read the tables from the StringIO object using
            # pandas.read_html()
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

            # Recursively get the DataFrames for all the tables on the
            # Wikipedia page
            table_dfs_list = []
            for table_soup in table_soups_list:
                table_dfs_list += self.get_page_tables(
                    str(table_soup), verbose=False
                )

            # If verbose is True, print a sorted list of the tables by their
            # number of rows and columns
            if verbose:
                print(sorted([(i, df.shape) for i, df in enumerate(
                    table_dfs_list
                )], key=lambda x: x[1][0] * x[1][1], reverse=True))

        except Exception as e:

            # If there is an error, print the error message
            if verbose:
                print(str(e).strip())

            # Recursively get the DataFrames for the tables on the Wikipedia
            # page again, but with verbose=False
            table_dfs_list = self.get_page_tables(
                tables_url_or_filepath, verbose=False
            )

        # Return the list of DataFrames
        return table_dfs_list

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

# print('\\b(' + '|'.join(dir()) + ')\\b')
