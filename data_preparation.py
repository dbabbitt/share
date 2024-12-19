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
import importlib
import inspect
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

        # Evaluator list
        self.object_evaluators = [
            fn for fn in dir(inspect) if fn.startswith('is')
        ]
        
        # Get built-in module names
        self.built_in_modules = set(sys.builtin_module_names)

        # Get pure Python modules from the standard library
        self.std_lib_path = osp.dirname(os.__file__)
        self.std_lib_modules = set([
            module_info.name
            for module_info in pkgutil.iter_modules([self.std_lib_path])
        ])

        # Combine both lists and sort for easier reading
        self.standard_library_modules = sorted(self.built_in_modules | self.std_lib_modules)

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
            if library_name in self.built_in_modules:
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
                    f'function_call: {function_call},'
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
