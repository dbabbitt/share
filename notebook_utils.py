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
import os


# Create classes lazily to avoid circular imports
class NotebookUtilities:
    def __init__(self, data_folder_path=None, saves_folder_path=None, verbose=False):
        # Load dependencies on demand
        from data_analysis import DataAnalysis
        from data_preparation import DataPreparation 
        from data_validation import DataValidation
        from file_operations import FileOperations
        from sequence_analysis import SequenceAnalysis
        from uncategorized import Uncategorized

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
        # Delegate attribute access to the appropriate component
        if hasattr(self.data_analysis, name):
            return getattr(self.data_analysis, name)
        elif hasattr(self.data_preparation, name):
            return getattr(self.data_preparation, name)
        elif hasattr(self.data_validation, name):
            return getattr(self.data_validation, name)
        elif hasattr(self.file_operations, name):
            return getattr(self.file_operations, name)
        elif hasattr(self.sequence_analysis, name):
            return getattr(self.sequence_analysis, name)
        elif hasattr(self.uncategorized, name):
            return getattr(self.uncategorized, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

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
