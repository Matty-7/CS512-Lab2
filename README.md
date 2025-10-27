# Lab #2 - MapReduce

## Overview

MapReduce is a programming model and processing technique designed to process large datasets in a parallel and distributed manner. Originally developed by Google, this framework has inspired the development of other data processing systems like Apache Hadoop and Apache Spark. The model is based on two main operations:

1. **Map**: In this phase, a side-effect free mapper function is applied to all elements of the input dataset independently. The interface can be abstracted as `map(K1, V1) -> [(K2, V2)]`, where each input key-value pair is transformed into a list of intermediate key-value pairs.
2. **Reduce**: In this phase, the output from the map phase is aggregated by key, and a reducer function is applied to aggregate the values in each group. The interface can be abstracted as `reduce(K2, [V2]) -> [V3]`.

Each of these phases can be parallelized across multiple machines, which enables MapReduce to scale efficiently to handle large data volumes. In this lab, you’ll learn the MapReduce programming model by implementing a variety of applications.

## Environment Setup

Before starting the lab, ensure your development environment meets the following requirements:

- **Python 3.10 or higher** installed on your system
- Set up a Python virtual environment (recommended) to isolate project dependencies
- Install the project dependencies by running `pip install -e .` in the directory containing `setup.py`. This command installs the package in editable mode, allowing you to modify the source code without reinstalling.

After cloning the repository, create a separate branch for your development work. This will allow you to pull any updates or fixes we may release later and integrate them into your work smoothly.

## Assignment - MapReduce Programming Model

### Simple MapReduce Framework

The `PandasMapReduce` class provides a framework for implementing the MapReduce programming model using `pandas`, a powerful library in Python for manipulating and analyzing structured data. `pandas` provides a two-dimensional tabular data structure called a `DataFrame`, which is similar to a spreadsheet or SQL table.

- **Initialization**: The class is initialized with user-defined `mapper_func` and `reducer_func`.
- **Map Phase**: Apply the mapper function to each row of the input `DataFrame`.
- **Shuffle Phase**: Group intermediate results by key using the `groupby` method.
- **Reduce Phase**: Apply the reducer function to each group of values.
- **Execution**: Combine the map, shuffle, and reduce phases to produce the final output.

### Tasks

In this part, you will implement a series of algorithms using the `PandasMapReduce` framework. Below are the tasks, along with their input and expected output. Please refer to `prog_model_sol.py` for examples of input and output data. You are ONLY required to implement the mapper and reducer function for each task. You should only modify `prog_model_sol.py`.

1. **Word Count**: Count the number of occurrences of each word in a collection of text documents.
   - **Input**: A `DataFrame` where each row represents a document, containing the document ID and the text of the document.
   - **Output**: A `DataFrame` where each row represents a word and its count of occurrences across all documents.
  
2. **Inverted Index**: Create an index mapping each word to the list of document IDs in which it appears.
   - **Input**: A `DataFrame` where each row represents a document, containing the document ID and the text of the document.
   - **Output**: A `DataFrame` where each row represents a word and the list of document IDs where the word appears.

3. **N-Gram Model**: Generate a character-level n-gram model from text data. An n-gram is a contiguous sequence of n items from a given sample of text. For example, the 2-gram (bigram) model of the word "hello" is `["he", "el", "ll", "lo"]`.
    - **Input**: A `DataFrame` where each row represents a document, containing the document ID and the text of the document.
    - **Output**: A `DataFrame` where each row represents an (n-1)-gram and the probability distribution of the next character.

4. **Table Join**: Perform an inner join on two tables based on a common key.
    - **Input**: Two tables encoded into a single `DataFrame`.
    - **Output**: A `DataFrame` representing the inner join of the two tables.
  
5. **Matrix Multiplication**: Perform matrix multiplication on two matrices.
   - **Input**: Two matrices encoded into a single `DataFrame`.
   - **Output**: A `DataFrame` where each row represents a position in the resulting matrix and the computed value at that position.

### Testing

We have included a simple test case for each task in `prog_model_sol.py`, which also serves as a reference for the expected input and output data. You can add some logging statements to the `PandasMapReduce` class or print intermediate results to help debug your implementation.

## Submission Instructions

- Run `python submission.py` to create a `submission.zip` file that you can submit to the Gradescope auto-grader.

Good luck!
