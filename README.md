# A Decision Support Model for Colorectal Cancer Screening

- [Project Description](#project-description)
  - [Paper Information](#paper-information)
  - [Abstract](#abstract)
  - [Image](#image)
  - [Project Files](#project-files)
- [Installation](#installation)
- [Usage](#usage)
  - [Notebook Use Cases](#notebook-use-cases)
- [License](#license)

## Project Description

This project contains the code for the paper you published, titled "A Decision Support Model for Colorectal Cancer Screening". The paper explores an approach for personalising optimal CRC screening strategies. This repository includes various Python scripts and Jupyter notebooks that implement the methodologies and experiments discussed in the paper.

### Paper Information

- **Title**: A Decision Support Model for Colorectal Cancer Screening
- **Authors**: Daniel Corrales, David Rios-Insua, Marino J. González
- **DOI**: [DOI link](https://doi.org/10.48550/arXiv.2502.21210)

### Abstract

We present a decision analysis-based approach to provide personalized colorectal cancer (CRC) screening suggestions. Based on an earlier CRC predictive model, we support decisions concerning whether and which screening method to consider and/or whether a colonoscopy should be administered. We include comfort, costs, complications, and information as decision criteria integrated within a multi-attribute utility model. Use cases concerning individual decision support, screening program assessment and design, and screening device benchmarking are presented.

![Project Image](outputs/id_24_01_page-0001%20(1).jpg)

### Project Files

Below is a detailed description of each file included in this repository:

- `calculo.py`: Contains functions and methods for performing various calculations used throughout the project.
- `CITATION.cff`: Citation file for the project, providing information on how to cite the project in academic works.
- `config.yaml`: Configuration file that holds various settings and parameters used in the project.
- `designing_new_strategy.py`: Script for designing and evaluating new strategies based on the project's methodology.
- `df_plot.py`: Contains functions for plotting dataframes, used for visualizing data.
- `elicit_lambda.py`: Script for eliciting lambda values, which are parameters used in the project's models.
- `elicitation.py`: Contains methods for the elicitation process, gathering expert knowledge and converting it into model parameters.
- `full_example.py`: A comprehensive example script demonstrating the usage of various components of the project.
- `functions.py`: Contains various utility functions that are used across different scripts in the project.
- `get_info_values.py`: Script to extract and compute information values from the data.
- `info_value_to_net.py`: Converts information values into a network representation for further analysis.
- `LICENSE`: License file for the project, detailing the terms under which the project can be used.
- `main.py`: Main script to run the project, integrating various components and executing the primary workflow.
- `network_functions.py`: Contains functions related to network analysis and manipulation.
- `plots.py`: Script for generating various plots and visualizations used in the project.
- `preprocessing.py`: Script for preprocessing data, including cleaning and transforming data for analysis.
- `pysmile_license.py`: License file for PySMILE, a library used in the project.
- `README.md`: This file, providing an overview and documentation for the project.
- `requirements.txt`: List of dependencies required for the project, which can be installed using pip.
- `save_info_values.py`: Script to save computed information values to files for later use.
- `sens_analysis_elicitation.py`: Script for performing sensitivity analysis on the elicitation process.
- `sens_analysis_param_U.py`: Script for performing sensitivity analysis on parameter U.
- `sens_analysis_PE_method.py`: Script for performing sensitivity analysis on the PE method.
- `simulations.py`: Script for running simulations based on the project's models and strategies.
- `update.py`: Script for updating the project, including refreshing data and recalculating values.
- `use_case_new_strategy.py`: Script demonstrating a use case with a new strategy, showcasing the project's application.
- `use_cases.ipynb`: Jupyter notebook containing various use cases and examples from the paper, demonstrating the project's methodology and results.

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

To run the main script, use:

```sh
python main.py
```

### Notebook Use Cases

The `use_cases.ipynb` notebook contains specific examples from the paper. It demonstrates various use cases and scenarios discussed in the publication. To open and run the notebook, use:

```sh
jupyter notebook use_cases.ipynb
```

This will open the Jupyter Notebook interface in your web browser, where you can interact with the examples provided.

## License

This project is licensed under the terms of the LICENSE file.

