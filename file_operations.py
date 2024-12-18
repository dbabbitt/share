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
    listdir as listdir, path as osp, remove as remove,
    walk as walk
)
from pandas import (
    DataFrame, read_csv, read_pickle,
)
import os
import sys
try:
    import dill as pickle
except Exception:
    try:
        import pickle5 as pickle
    except Exception:
        import pickle


class FileOperations(BaseConfig):
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

    # -------------------
    # List Functions
    # -------------------

    # -------------------
    # File Functions
    # -------------------

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

                # Search for function definitions using the regular expression
                match_obj = self.simple_defs_regex.search(line)

                # If function definition is found
                if match_obj:

                    # Extract the function name from the match and add it
                    scraping_util = match_obj.group(1)
                    utils_set.add(scraping_util)

        return utils_set

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

                                    # Extract the function name from the match
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

    def show_dupl_fn_defs_search_string(
        self, util_path=None, github_folder=None
    ):
        """
        Identify and report duplicate function definitions in Jupyter
        notebooks and suggest how to consolidate them.

        Parameters:
            util_path (str, optional):
                The path to the utility file where refactored functions will
                be added. Defaults to '../py/notebook_utils.py'.
            github_folder (str, optional):
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
            util_path = osp.join(os.pardir, 'py', 'notebook_utils.py')

        # Set the GitHub folder path if not provided
        if github_folder is None:
            github_folder = self.github_folder

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
                f' {github_folder} folder for this pattern:'  # noqa: E231
            )
            print('\\s+"def (' + '|'.join(duplicate_fns_list) + ')\\(')
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

            # Print the absolute path to the pickle file if verbose is enabled
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

        # If no pickle path provided, construct default path using object name
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

            # If reading the pickle file fails, fall back to the pickle module
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
                            csv_name=frame_name, folder_path=self.saves_folder
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

        # Iterate over dfs in the kwargs dictionary and save them to CSV files
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

# print('\\b(' + '|'.join(dir()) + ')\\b')
