# Share: Notebook Utilities for Jupyter Notebooks

**Share** is a Python library designed to streamline workflows for machine learning engineers, data analysts, and data scientists working with Jupyter Notebooks. It provides a comprehensive set of utility functions for data preparation, validation, analysis, visualization, and file operations, all organized into modular classes for better maintainability and usability.

The core functionality is implemented in the `NotebookUtilities` class, which acts as a **facade** to delegate method calls to smaller, focused modules. This ensures backward compatibility while improving internal organization and scalability.

---

## 📂 Project Structure

The project is structured into modular components. Each component focuses on a specific aspect of notebook utilities. 

To discover a natural structure, I performed a market basket analysis of all the instantiated functions in the notebooks containing them. I then used this to make a a directed graph, the edges representing the maximum confidence in each direction between pairs of functions (so that each node has one "in" edge and one "out" edge). I applied three generations of the Girvan-Newman algorithm to this graph. This analysis helped detect communities, which I used to organize the modules. (You can see some of this work in the [Breaking Up notebook_utils](https://github.com/dbabbitt/notebooks/blob/master/visualizations/Breaking%20Up%20notebook_utils.ipynb) notebook.)

Below is an overview of the structure:

- **`notebook_utils.py`**: The original `NotebookUtilities` class, now a facade that delegates calls to the smaller, focused classes.
- **`base_config.py`**: Contains the `BaseConfig` class, which provides common attributes and functions inherited by other modules.

### Modules

#### 1. **Data Preparation**
   - **Module Name**: `data_preparation.py`
   - **Key Functions**:
     - `get_first_year_element`, `get_evaluations`, `get_library_names`, `get_dir_tree`, `get_filename_from_url`, `get_style_column`, `get_td_parent`, `download_file`, `get_page_soup`, `get_page_tables`, `get_wiki_tables`.
   - **Description**: Provides the `DataPreparation` class for functions related to data extraction and preparation workflows.

#### 2. **Data Validation**
   - **Module Name**: `data_validation.py`
   - **Key Functions**:
     - `compute_similarity`, `conjunctify_nouns`, `check_for_typos`, `list_dfs_in_folder`, `get_relative_position`, `get_random_subdictionary`, `plot_inauguration_age`.
   - **Description**: Contains the `DataValidation` class for functions related to validation and utility workflows.

#### 3. **Data Analysis and Visualization**
   - **Module Name**: `data_analysis.py`
   - **Key Functions**:
     - `open_path_in_notepad`, `get_wiki_infobox_data_frame`, `get_inf_nan_mask`, `get_column_descriptions`, `modalize_columns`, `get_regexed_columns`, `get_regexed_dataframe`, `one_hot_encode`, `get_flattened_dictionary`, `get_numeric_columns`, `get_euclidean_distance`, `get_nearest_neighbor`, `get_minority_combinations`, `plot_line_with_error_bars`, `plot_histogram`, `get_r_squared_value_latex`, `get_spearman_rho_value_latex`, `first_order_linear_scatterplot`.
   - **Description**: Provides the `DataAnalysis` class for functions related to data analysis and visualization.

#### 4. **File Operations**
   - **Module Name**: `file_operations.py`
   - **Key Functions**:
     - `get_utility_file_functions`, `get_notebook_functions_dictionary`, `get_notebook_functions_set`, `show_duplicated_util_fns_search_string`, `show_dupl_fn_defs_search_string`, `delete_ipynb_checkpoint_folders`, `get_random_py_file`, `attempt_to_pickle`, `csv_exists`, `load_csv`, `pickle_exists`, `load_object`, `load_data_frames`, `save_data_frames`, `store_objects`, `get_random_function`.
   - **Description**: Contains the `FileOperations` class for functions related to file cleanup and basic file operations.

#### 5. **Sequence Analysis**
   - **Module Name**: `sequence_analysis.py`
   - **Key Functions**:
     - `split_list_by_gap`, `count_ngrams`, `get_sequences_by_count`, `split_list_by_exclusion`, `convert_strings_to_integers`, `get_ndistinct_subsequences`, `get_turbulence`, `replace_consecutive_elements`, `count_swaps_to_perfect_order`, `get_color_cycled_list`, `plot_sequence`, `plot_sequences`.
   - **Description**: Contains the `SequenceAnalysis` class for analyzing, manipulating, and visualizing sequences, including numeric, string, and plotting utilities.

---

## 🚀 Features

- **Backward Compatibility**: The `NotebookUtilities` class retains its original interface, ensuring existing notebooks can continue using it without modification.
- **Modular Design**: Functionality is split into smaller, focused classes for better organization and maintainability.
- **Comprehensive Utility Functions**: Covers a wide range of tasks, including data preparation, validation, analysis, visualization, and file operations.
- **Scalable and Extensible**: The modular structure makes it easy to add new functionality or extend existing features.

---

## 📦 Installation

To use the `share` library, clone the repository and install the required dependencies:

1. Clone the repository and add it as a submodule to your notebooks:
   ```bash
   repo="path/to/your/notebooks"
   cd "$repo"
   cd ../
   git clone https://github.com/dbabbitt/share.git
   if [ -d "$repo/.git" ]; then
       cd "$repo"
       repo_name=$(basename "$repo")
       # Update the submodule if it already exists
       if [ -d "share" ]; then
           echo "Updating share submodule in repository: $repo_name..."
           git submodule update --remote --merge
       # Add it if the submodule doesn't exist
       else
           echo "Adding share submodule in repository: $repo_name..."
           git submodule add -f https://github.com/dbabbitt/share.git share
           git commit -m "Added share submodule"
       fi
   fi
   ```
2. Navigate to the project directory:
   ```bash
   cd share
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

To remove the `share` submodule from your repository:
   ```bash
   repo="path/to/your/repository"
   if [ -d "$repo/.git" ]; then
       cd "$repo"
       
       # Check if the submodule exists
       if [ -d "share" ]; then
           repo_name=$(basename "$repo")
           echo "Removing submodule in repository: $repo_name..."
           git config -f .gitmodules --remove-section submodule.share
           git rm --cached share
           
           # Fully remove the share folder
           rm -rf share
           
           git commit -m "Removed submodule share"
       fi
       
   fi
   ```

---

## 🛠️ Usage

Here’s how you can use the `NotebookUtilities` class in your Jupyter notebook:

1. Import the `NotebookUtilities` class:
   ```python
   import os.path as osp
   import os
   import sys
   
   # Assuming here you're running this in a Jupyter notebook cell in a subfolder
   shared_folder = osp.abspath(osp.join(os.pardir, 'share'))
   assert osp.exists(shared_folder), f"The share submodule is not at {shared_folder}"
   if shared_folder not in sys.path: sys.path.insert(1, shared_folder)
   
   from notebook_utils import NotebookUtilities
   ```
2. Initialize the class:
   ```python
   nu = NotebookUtilities(
       
       # This will create a data folder if there isn't one already
       data_folder_path=osp.abspath(osp.join(os.pardir, 'data')),
       
       # This will create a saves folder if there isn't one already
       saves_folder_path=osp.abspath(osp.join(os.pardir, 'saves'))
       
   )
   ```
3. Use the utility functions as needed:
   ```python   
   import random
   
   # Generate a random fixed point in the CIELAB color space
   ranges = [(0, 100), (-128, 127), (-128, 127)]
   fixed_point = tuple([random.uniform(a, b) for a, b in ranges])
   
   # Generate a random number of additional points to spread, between 1 and 15
   random_point_count = random.randint(1, 15)
   
   # Initialize a list to store trial results
   trials = []
   
   # Perform up to 5 trials to find the best spread of points
   from colormath.color_objects import LabColor
   while len(trials) < 5:
       
       # Attempt to spread the points evenly within a unit cube
       try:
           spread_points = nu.spread_points_in_cube(
            random_point_count, fixed_point, *ranges, verbose=False
           )
   
           # Ensure spread points have all unique XKCD names
           xkcd_set = set()
           for lab_color in spread_points:
               rgb_color = nu.lab_to_rgb(LabColor(*lab_color))
               nearest_neighbor = nu.get_nearest_neighbor(rgb_color, XKCD_COLORS)
               xkcd_set.add(NEAREST_NAME_DICT[nearest_neighbor])
           if len(xkcd_set) == len(spread_points):
               
               # Calculate the spread value, which measures how far the points are from the fixed point
               spread_value = nu.calculate_spread(spread_points[1:], fixed_point, verbose=False)
               
               # Store the result as a tuple of (spread_points, spread_value)
               trial_tuple = (spread_points, spread_value)
               trials.append(trial_tuple)
       
       # If an error occurs (e.g., a spread point too close to black or white), skip this trial
       except Exception:
           continue
   
   # Select the trial with points as far away from the fixed point as possible
   trial_tuple = max(trials, key=lambda x: x[1])
   
   # Extract the spread points from the best trial
   spread_points = []
   for lab_color in trial_tuple[0]:
       rgb_color = nu.lab_to_rgb(LabColor(*lab_color))
       spread_points.append(rgb_color)
   
   # Visualize the spread points
   nu.inspect_spread_points(spread_points, verbose=False)
   ```
   ```python
   # Count the occurrences of a sequence of elements (n-grams) in a list
   actions = ['jump', 'run', 'jump', 'run', 'jump']
   ngrams = ['jump', 'run']
   nu.count_ngrams(actions, ngrams)  # 2
   ```
   ```python
   # Convert a sequence of strings into a sequence of integers and a mapping dictionary
   sequence = ['apple', 'banana', 'apple', 'cherry']
   new_sequence, mapping = nu.convert_strings_to_integers(sequence)
   print(new_sequence)  # array([0, 1, 0, 2])
   print(mapping)  # {'apple': 0, 'banana': 1, 'cherry': 2}
   ```
   ```python
   # Take a list of strings with indentation and prefix them based on a
   # multi-level list style
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
   ```
   ```python
   # Get the absolute file path where a function object is stored
   import os.path as osp
   my_function = lambda: None
   file_path = nu.get_function_file_path(my_function)
   print(osp.abspath(file_path))
   ```
   ```python
   # Calculate the Euclidean distance between two colors in RGB space
   print(nu.color_distance_from('white', (255, 0, 0)))  # 360.62445840513925
   print(nu.color_distance_from(
       '#0000FF', (255, 0, 0)
   ))  # 360.62445840513925
   ```
   ```python
   # Generate a step-by-step description of how to perform a function by hand from its comments
   nu.describe_procedure(nu.describe_procedure)
   ```
   ```python
   # Check the closest names for typos by comparing items from two different lists
   commonly_misspelled_words = ["absence", "consensus", "definitely", "broccoli", "necessary"]
   common_misspellings = ["absense", "concensus", "definately", "brocolli", "neccessary"]
   typos_df = nu.check_for_typos(
      commonly_misspelled_words,
      common_misspellings,
      rename_dict={'left_item': 'commonly_misspelled', 'right_item': 'common_misspelling'}
   ).sort_values(
      ['max_similarity', 'commonly_misspelled', 'common_misspelling'],
      ascending=[False, True, True]
   )
   mask_series = (typos_df.max_similarity < 1.0)
   typos_df[mask_series]
   ```
   ```python
   # Open a file in Notepad
   nu.open_path_in_notepad(r'C:\this_example.txt')
   ```
   ```python
   # Retrieve <table>s from a given URL as a list of DataFrames
   tables_url = 'https://en.wikipedia.org/wiki/'
   tables_url += 'Provinces_of_Afghanistan'
   page_tables_list = nu.get_page_tables(tables_url, verbose=True)
   page_tables_list[2]
   ```
   ```python
   import numpy as np
   import pandas as pd
   
   # Create a new column in a DataFrame representing
   # the modal value of the D and E columns
   df = pd.DataFrame({
   'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3], 'C': ['a', 'b', 'c']
   })
   df['D'] = pd.Series([np.nan, 2, np.nan])
   df['E'] = pd.Series([1, np.nan, 3])
   df = nu.modalize_columns(df, ['D', 'E'], 'F')
   display(df)
   assert all(df['A'] == df['F'])
   ```
   ```python
   # Identify numeric columns in a DataFrame
   import pandas as pd
   df = pd.DataFrame({
       'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3], 'C': ['a', 'b', 'c']
   })
   nu.get_numeric_columns(df)  # ['A', 'B']
   ```
   ```python
   # Introspect a Python module to discover available functions and classes programmatically
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
   ```
   ```python
   # Create a standard sequence plot where each element corresponds to a position on the y-axis
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
   ```

For detailed examples and use cases, refer to the documentation (TODO: write docs).

---

## 🧑‍💻 Contributing

We welcome contributions to improve the `share` library! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request on GitHub.

---

## 🐛 Reporting Issues

If you encounter any bugs or have feature requests, please open an issue in the [Issues](https://github.com/dbabbitt/share/issues) section of the repository. Be sure to include:

- A clear description of the issue.
- Steps to reproduce the problem.
- Any relevant logs or screenshots.

---

## 🛡️ License

This project is licensed under the [MIT License](https://github.com/dbabbitt/share/blob/master/LICENSE). Feel free to use, modify, and distribute it as per the terms of the license.

---

## 🌟 Support

If you find this project helpful, please consider giving it a ⭐ on GitHub to show your support!
