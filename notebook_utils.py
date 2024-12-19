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

from data_analysis import DataAnalysis
from data_preparation import DataPreparation
from data_validation import DataValidation
from file_operations import FileOperations
from sequence_analysis import SequenceAnalysis
from uncategorized import Uncategorized
from os import (
    makedirs as makedirs, path as osp
)
import os


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

        # Instantiate the smaller classes
        self.data_analysis = DataAnalysis()
        self.data_preparation = DataPreparation(
            data_folder_path=data_folder_path,
            saves_folder_path=saves_folder_path
        )
        self.data_validation = DataValidation(
            saves_folder_path=saves_folder_path
        )
        self.file_operations = FileOperations(
            data_folder_path=data_folder_path,
            saves_folder_path=saves_folder_path
        )
        self.sequence_analysis = SequenceAnalysis(
            data_folder_path=data_folder_path,
            saves_folder_path=saves_folder_path
        )
        self.uncategorized = Uncategorized(
            data_folder_path=data_folder_path,
            saves_folder_path=saves_folder_path
        )

    def __getattr__(self, name):

        # Check if the method exists in one of the smaller classes
        for component in [
            self.data_analysis, self.data_preparation, self.data_validation,
            self.file_operations, self.sequence_analysis, self.uncategorized
        ]:
            if hasattr(component, name):
                return getattr(component, name)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __dir__(self):
        """
        Override the __dir__ method to include attributes and methods
        from the smaller classes.
        """

        # Start with the default attributes and methods of this class
        attributes = set(super().__dir__())

        # Add attributes and methods from the smaller classes
        for attr_name in self.__dict__:
            attr = getattr(self, attr_name)
            if hasattr(attr, '__dir__'):
                attributes.update(attr.__dir__())

        return sorted(attributes)

# print('\\b(' + '|'.join(dir()) + ')\\b')
