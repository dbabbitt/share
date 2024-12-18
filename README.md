# Share: Notebook Utilities for Jupyter Notebooks

**Share** is a Python library designed to streamline workflows for machine learning engineers, data analysts, and data scientists working with Jupyter Notebooks. It provides a comprehensive set of utility functions for data preparation, validation, analysis, visualization, and file operations, all organized into modular classes for better maintainability and usability.

The core functionality is implemented in the `NotebookUtilities` class, which acts as a **facade** to delegate method calls to smaller, focused modules. This ensures backward compatibility while improving internal organization and scalability.

---

## 📂 Project Structure

The project is structured into modular components, each focusing on a specific aspect of notebook utilities. Below is an overview of the structure:

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
     - `compute_similarity`, `conjunctify_nouns`, `split_list_by_exclusion`, `check_for_typos`, `list_dfs_in_folder`, `get_relative_position`, `get_random_subdictionary`, `plot_inauguration_age`.
   - **Description**: Contains the `DataValidation` class for functions related to validation and utility workflows.

#### 3. **Data Analysis and Visualization**
   - **Module Name**: `data_analysis.py`
   - **Key Functions**:
     - `get_jitter_list`, `split_list_by_gap`, `count_swaps_to_perfect_order`, `open_path_in_notepad`, `get_wiki_infobox_data_frame`, `get_inf_nan_mask`, `get_column_descriptions`, `modalize_columns`, `get_regexed_columns`, `get_regexed_dataframe`, `one_hot_encode`, `get_flattened_dictionary`, `get_numeric_columns`, `get_euclidean_distance`, `get_nearest_neighbor`, `get_minority_combinations`, `plot_line_with_error_bars`, `plot_histogram`, `get_r_squared_value_latex`, `get_spearman_rho_value_latex`, `first_order_linear_scatterplot`.
   - **Description**: Provides the `DataAnalysis` class for functions related to data analysis and visualization.

#### 4. **File Operations**
   - **Module Name**: `file_operations.py`
   - **Key Functions**:
     - `get_utility_file_functions`, `get_notebook_functions_dictionary`, `get_notebook_functions_set`, `show_duplicated_util_fns_search_string`, `show_dupl_fn_defs_search_string`, `delete_ipynb_checkpoint_folders`, `get_random_py_file`, `attempt_to_pickle`, `csv_exists`, `load_csv`, `pickle_exists`, `load_object`, `load_data_frames`, `save_data_frames`, `store_objects`, `get_random_function`.
   - **Description**: Contains the `FileOperations` class for functions related to file cleanup and basic file operations.

---

## 🚀 Features

- **Backward Compatibility**: The `NotebookUtilities` class retains its original interface, ensuring existing notebooks can continue using it without modification.
- **Modular Design**: Functionality is split into smaller, focused classes for better organization and maintainability.
- **Comprehensive Utility Functions**: Covers a wide range of tasks, including data preparation, validation, analysis, visualization, and file operations.
- **Scalable and Extensible**: The modular structure makes it easy to add new functionality or extend existing features.

---

## 📦 Installation

To use the `Share` library, clone the repository and install the required dependencies:

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
3. Install dependencies (TODO: write a requirements.txt file):
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

Here’s how you can use the `NotebookUtilities` class in your Jupyter Notebook:

1. Import the `NotebookUtilities` class:
   ```python
   import os.path as osp
   import os
   import sys
   
   shared_folder = osp.abspath(osp.join(os.pardir, 'share'))
   assert osp.exists(shared_folder), "The share submodule needs to be added"
   if shared_folder not in sys.path: sys.path.insert(1, shared_folder)
   
   from notebook_utils import NotebookUtilities
   ```
2. Initialize the class:
   ```python
   nu = NotebookUtilities(
       data_folder_path=osp.abspath(osp.join(os.pardir, 'data')),
       saves_folder_path=osp.abspath(osp.join(os.pardir, 'saves'))
   )
   ```
3. Use the utility functions as needed:
   ```python
   # Count the occurrences of a sequence of elements (n-grams) in a list
   actions = ['jump', 'run', 'jump', 'run', 'jump']
   ngrams = ['jump', 'run']
   nu.count_ngrams(actions, ngrams)
   
   # Convert a sequence of strings into a sequence of integers and a mapping dictionary
   sequence = ['apple', 'banana', 'apple', 'cherry']
   new_sequence, mapping = nu.convert_strings_to_integers(sequence)
   display(new_sequence)  # array([0, 1, 0, 2])
   display(mapping)  # {'apple': 0, 'banana': 1, 'cherry': 2}
   
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
   
   # Get the absolute file path where a function object is stored
   import os.path as osp
   my_function = lambda: None
   file_path = nu.get_function_file_path(my_function)
   print(osp.abspath(file_path))
   
   # Calculate the Euclidean distance between two colors in RGB space
   nu.color_distance_from('white', (255, 0, 0))  # 360.62445840513925
   nu.color_distance_from(
       '#0000FF', (255, 0, 0)
   )  # 360.62445840513925
   
   # Visualize a subgraph with a custom node grouping
   import networkx as nx
   sub_graph = nx.erdos_renyi_graph(10, 0.3)
   show_subgraph(sub_graph)
   nodes_list_list = [['node1', 'node2'], ['node3', 'node4']]
   show_subgraph(
       sub_graph, nodes_list_list=nodes_list_list, verbose=True
   )
   
   # Generate a color cycler with a specified number of colors
   color_cycler = nu.get_color_cycler(len(possible_cause_list))
   for possible_cause, face_color_dict in zip(
       possible_cause_list, color_cycler()
   ):
       face_color = face_color_dict['color']
   
   # Check the closest names for typos by comparing items from two different lists
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
   
   # Open a file in Notepad
   nu.open_path_in_notepad(r'C:\this_example.txt')
   
   # Identify and report duplicate function definitions in Jupyter notebooks
   # and suggest how to consolidate them
   nu.show_dupl_fn_defs_search_string()
   
   # Retrieve <table>s from a given URL and return a list of DataFrames
   tables_url = 'https://en.wikipedia.org/wiki/'
   tables_url += 'Provinces_of_Afghanistan'
   page_tables_list = nu.get_page_tables(tables_url, verbose=True)
   
   # Return a mask indicating which elements of X_train and y_train are not inf or nan
   inf_nan_mask = nu.get_inf_nan_mask(X_train, y_train)
   X_train_filtered = X_train[inf_nan_mask]
   y_train_filtered = y_train[inf_nan_mask]
   
   # Identify numeric columns in a DataFrame
   import pandas as pd
   df = pd.DataFrame({
       'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3], 'C': ['a', 'b', 'c']
   })
   nu.get_numeric_columns(df)  # ['A', 'B']
   
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

This project is licensed under the [MIT License](https://github.com/dbabbitt/share/blob/main/LICENSE). Feel free to use, modify, and distribute it as per the terms of the license.

---

## 🌟 Support

If you find this project helpful, please consider giving it a ⭐ on GitHub to show your support!
