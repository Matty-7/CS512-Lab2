from collections import Counter, defaultdict
from typing import Any

import fire
import numpy as np
import pandas as pd
from cs512_lab2.prog_model import PandasMapReduce

""" Word count """


def word_count_mapper(key: Any, value: Any) -> list[tuple[Any, Any]]:
    words = value.split()
    return [(word, 1) for word in words]


def word_count_reducer(key: Any, values: list[Any]) -> list[Any]:
    return [sum(values)]


def word_count_test():
    print("\nWord count test")

    # Define input data
    input_df = pd.DataFrame(
        {
            "key": [1, 2, 3],  # Document IDs
            "value": [
                "hello world",
                "hello there",
                "world of warcraft",
            ],
        }
    )

    print("\nInput data:")
    print(input_df)

    """
    Input data:
    key              value
    0    1        hello world
    1    2        hello there
    2    3  world of warcraft
    """

    map_reduce = PandasMapReduce(word_count_mapper, word_count_reducer)
    output_df = map_reduce.execute(input_df)
    output_df = output_df.astype({"value": "int64"})
    # Sort the output dataframe by the key column for easier comparison
    output_df = output_df.sort_values(by="key").reset_index(drop=True)

    print("\nOutput data:")
    print(output_df)

    print("\nExpected output:")
    expected_output = pd.DataFrame(
        {
            "key": [
                "hello",
                "world",
                "there",
                "of",
                "warcraft",
            ],
            "value": [2, 2, 1, 1, 1],  # Word counts
        }
    )
    # Sort the expected dataframe by the key column for easier comparison
    expected_output = expected_output.sort_values(by="key").reset_index(drop=True)
    print(expected_output)

    """
    Expected output:
            key  value
    0     hello      2
    1        of      1
    2     there      1
    3  warcraft      1
    4     world      2
    """

    # Compare the two dataframes
    if output_df.equals(expected_output):
        print("\nword_count_test passed")
    else:
        print("\nword_count_test failed")


""" Inverted index """


def inverted_index_mapper(key: Any, value: Any) -> list[tuple[Any, Any]]:
    words = value.split()
    return [(word, key) for word in words]


def inverted_index_reducer(key: Any, values: list[Any]) -> list[Any]:
    return [values]


def inverted_index_test():
    print("\nInverted index test")

    # Define input data
    input_df = pd.DataFrame(
        {
            "key": [1, 2, 3],  # Document IDs
            "value": [
                "hello world",
                "hello there",
                "world of warcraft",
            ],
        }
    )

    print("\nInput data:")
    print(input_df)

    """
    Input data:
    key              value
    0    1        hello world
    1    2        hello there
    2    3  world of warcraft
    """

    map_reduce = PandasMapReduce(inverted_index_mapper, inverted_index_reducer)
    output_df = map_reduce.execute(input_df)
    # Sort the output dataframe by the key column for easier comparison
    output_df = output_df.sort_values(by="key").reset_index(drop=True)

    print("\nOutput data:")
    print(output_df)

    print("\nExpected output:")
    expected_output = pd.DataFrame(
        {
            "key": [
                "hello",
                "world",
                "there",
                "of",
                "warcraft",
            ],
            "value": [[1, 2], [1, 3], [2], [3], [3]],  # Document IDs
        }
    )
    # Sort the expected dataframe by the key column for easier comparison
    expected_output = expected_output.sort_values(by="key").reset_index(drop=True)
    print(expected_output)

    # Compare the two dataframes
    if output_df.equals(expected_output):
        print("\ninverted_index_test passed")
    else:
        print("\ninverted_index_test failed")


""" Character-level n-gram model """

# Global constants used in automated testing
n_gram_n = 3  # n-gram length


def ngram_mapper(key: Any, value: Any) -> list[tuple[Any, Any]]:
    result = []
    for i in range(len(value) - n_gram_n + 1):
        n_gram = value[i : i + n_gram_n]
        prefix = n_gram[:-1]  # First n-1 characters
        next_char = n_gram[-1]  # Last character
        result.append((prefix, next_char))
    return result


def ngram_reducer(key: Any, values: list[Any]) -> list[Any]:
    # Count occurrences of each character
    char_counts = Counter(values)
    total_count = sum(char_counts.values())
    
    # Convert counts to probabilities
    probabilities = {char: count / total_count for char, count in char_counts.items()}
    
    return [probabilities]


def ngram_test():
    print("\nCharacter-level n-gram model test")

    # Define input data
    input_df = pd.DataFrame(
        {
            "key": [1, 2, 3],  # Document IDs
            "value": [
                "hello world",
                "hello there",
                "world of warcraft",
            ],
        }
    )

    print("\nInput data:")
    print(input_df)

    """
    Input data:
    key              value
    0    1        hello world
    1    2        hello there
    2    3  world of warcraft
    """

    map_reduce = PandasMapReduce(ngram_mapper, ngram_reducer)
    output_df = map_reduce.execute(input_df)
    # Sort the output dataframe by the key column for easier comparison
    output_df = output_df.sort_values(by="key").reset_index(drop=True)

    print("\nOutput data:")
    print(output_df)

    print("\nExpected output:")
    # Compute the frequency of each character following a prefix (first n-1 characters)
    n_gram_counter = defaultdict(Counter)
    for text in input_df["value"]:
        for n_gram in [text[i : i + n_gram_n] for i in range(len(text) - n_gram_n + 1)]:
            n_gram_counter[n_gram[:-1]].update([n_gram[-1]])

    # Convert the n-gram counter to a dataframe
    expected_output = pd.DataFrame(
        [
            {
                "key": prefix,
                "value": {
                    char: count / sum(counter.values())  # Convert count to probability
                    for char, count in counter.items()
                },
            }
            for prefix, counter in n_gram_counter.items()
        ]
    )

    # Sort the expected dataframe by the key column for easier comparison
    expected_output = expected_output.sort_values(by="key").reset_index(drop=True)
    print(expected_output)

    """
    Expected output:
    key                                              value
    0    o                                         {'f': 1.0}
    1    t                                         {'h': 1.0}
    2    w                               {'o': 0.5, 'a': 0.5}
    3   af                                         {'t': 1.0}
    4   ar                                         {'c': 1.0}
    5   cr                                         {'a': 1.0}
    6   d                                          {'o': 1.0}
    7   el                                         {'l': 1.0}
    8   er                                         {'e': 1.0}
    9   f                                          {'w': 1.0}
    10  he  {'l': 0.6666666666666666, 'r': 0.3333333333333...
    11  ld                                         {' ': 1.0}
    12  ll                                         {'o': 1.0}
    13  lo                                         {' ': 1.0}
    14  o                                {'w': 0.5, 't': 0.5}
    15  of                                         {' ': 1.0}
    16  or                                         {'l': 1.0}
    17  ra                                         {'f': 1.0}
    18  rc                                         {'r': 1.0}
    19  rl                                         {'d': 1.0}
    20  th                                         {'e': 1.0}
    21  wa                                         {'r': 1.0}
    22  wo                                         {'r': 1.0}
    """

    # Compare the two dataframes
    if output_df.equals(expected_output):
        print("\nngram_test passed")
    else:
        print("\nngram_test failed")


""" Table join """

# Global constants used in automated testing
# You may use these constants in your implementation of the worker functions
join_column = "join_col"  # The column to join on
table1_name = "left"  # The name of the left table
table2_name = "right"  # The name of the right table


def inner_join_mapper(key: Any, value: Any) -> list[tuple[Any, Any]]:
    join_key = value[join_column]
    return [(join_key, (key, value))]


def inner_join_reducer(key: Any, values: list[Any]) -> list[Any]:
    # Separate rows by table
    left_rows = []
    right_rows = []
    
    for table_name, row_data in values:
        if table_name == table1_name:
            left_rows.append(row_data)
        elif table_name == table2_name:
            right_rows.append(row_data)
    
    # Perform Cartesian product of left and right rows
    result = []
    for left_row in left_rows:
        for right_row in right_rows:
            # Merge the two rows
            joined_row = {**left_row, **right_row}
            result.append(joined_row)
    
    return result


def inner_join_test():
    print("\nTable join test")

    # Define input data
    orders_df = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104],
            "customer_id": [1, 2, 3, 1],
            "amount": [200, 150, 300, 250],
        }
    )
    customers_df = pd.DataFrame(
        {"customer_id": [1, 2, 3], "customer_name": ["Alice", "Bob", "Charlie"]}
    )
    orders_df = orders_df.rename(columns={"customer_id": join_column})
    customers_df = customers_df.rename(columns={"customer_id": join_column})

    input_df = pd.DataFrame(
        {
            "key": [table1_name] * len(orders_df) + [table2_name] * len(customers_df),
            "value": (
                orders_df.to_dict(orient="records")
                + customers_df.to_dict(orient="records")
            ),
        }
    )

    print("\nInput data:")
    print(input_df)

    """
    Orders Table (left)
        order_id  customer_id  amount
    0       101            1     200
    1       102            2     150
    2       103            3     300
    3       104            1     250

    Customers Table (right)
        customer_id customer_name
    0            1         Alice
    1            2           Bob
    2            3       Charlie

    Input data:
        key                                            value
    0   left  {'order_id': 101, 'join_col': 1, 'amount': 200}
    1   left  {'order_id': 102, 'join_col': 2, 'amount': 150}
    2   left  {'order_id': 103, 'join_col': 3, 'amount': 300}
    3   left  {'order_id': 104, 'join_col': 1, 'amount': 250}
    4  right        {'join_col': 1, 'customer_name': 'Alice'}
    5  right          {'join_col': 2, 'customer_name': 'Bob'}
    6  right      {'join_col': 3, 'customer_name': 'Charlie'}
    """

    map_reduce = PandasMapReduce(inner_join_mapper, inner_join_reducer)
    output_df = map_reduce.execute(input_df)
    # Sort the output dataframe by the key column for easier comparison
    output_df = output_df.sort_values(by="key").reset_index(drop=True)

    print("\nOutput data:")
    print(output_df)

    print("\nExpected output:")
    # Compute the ground truth result using pandas's merge function
    expected_df = orders_df.merge(customers_df, on=join_column, how="inner")
    expected_df = pd.DataFrame(
        [
            {"key": row[join_column], "value": row}
            for row in expected_df.to_dict(orient="records")
        ]
    )
    # Sort the expected dataframe by the key column for easier comparison
    expected_df = expected_df.sort_values(by="key").reset_index(drop=True)
    print(expected_df)

    """
    Expected output:
       key                                              value
    0    1  {'order_id': 101, 'join_col': 1, 'amount': 200...
    1    1  {'order_id': 104, 'join_col': 1, 'amount': 250...
    2    2  {'order_id': 102, 'join_col': 2, 'amount': 150...
    3    3  {'order_id': 103, 'join_col': 3, 'amount': 300...
    """

    # Compare the two dataframes
    if output_df.equals(expected_df):
        print("\ninner_join_test passed")
    else:
        print("\ninner_join_test failed")


""" Matrix multiplication """

# Global constants used in automated testing
# You may use these constants in your implementation of the worker functions
m, k, n = 3, 4, 5  # (m x k) and (k x n) matrices


def matmul_mapper(key: Any, value: Any) -> list[tuple[Any, Any]]:
    col_a, row_b = value  # k-th column of A (size m), k-th row of B (size n)
    
    result = []
    for i in range(len(col_a)):  # Iterate over rows of A (m rows)
        for j in range(len(row_b)):  # Iterate over columns of B (n columns)
            # Compute partial product for position (i, j)
            partial_product = col_a[i] * row_b[j]
            result.append(((i, j), partial_product))
    
    return result


def matmul_reducer(key: Any, values: list[Any]) -> list[Any]:
    return [sum(values)]


def matmul_test():
    print("\nMatrix multiplication test")

    # Define input data
    np.random.seed(42)
    mat_a = np.random.randint(1, 10, size=(m, k))
    mat_b = np.random.randint(1, 10, size=(k, n))

    # Key is the index of the reduce dimension
    # Value is (k-th column of `mat_a`, k-th row of `mat_b`)
    input_df = pd.DataFrame(
        {"key": range(k), "value": [(mat_a[:, k], mat_b[k, :]) for k in range(k)]}
    )

    print("\nInput data:")
    print(input_df)

    """
       key                               value
    0    0  ([7, 3, 8, 2], [1, 6, 9, 1, 3, 7])
    1    1  ([4, 7, 8, 8], [4, 9, 3, 5, 3, 7])
    2    2  ([8, 8, 3, 6], [5, 9, 7, 2, 4, 9])
    3    3  ([5, 5, 6, 2], [2, 9, 5, 2, 4, 7])
    4    4  ([7, 4, 5, 5], [8, 3, 1, 4, 2, 8])
    """

    map_reduce = PandasMapReduce(matmul_mapper, matmul_reducer)
    output_df = map_reduce.execute(input_df)
    # Convert the value column to int64 type for easier comparison
    output_df["value"] = output_df["value"].astype("int64")
    # Sort the output dataframe by the key column for easier comparison
    output_df = output_df.sort_values(by="key").reset_index(drop=True)

    print("\nOutput data:")
    print(output_df)

    print("\nExpected output:")
    # Compute the ground truth result using numpy's dot function
    expected_output = np.dot(mat_a, mat_b)
    expected_output = pd.DataFrame(
        {
            "key": [(i, j) for i in range(m) for j in range(n)],
            "value": expected_output.flatten(),
        }
    )
    # Sort the expected dataframe by the key column for easier comparison
    expected_output = expected_output.sort_values(by="key").reset_index(drop=True)
    print(expected_output)

    """
    Expected output:
        key  value
    0   (0, 0)    129
    1   (0, 1)    216
    2   (0, 2)    163
    ...
    20  (3, 2)     99
    21  (3, 3)     78
    22  (3, 4)     72
    23  (3, 5)    178
    """

    # compare the two dataframes
    if output_df.equals(expected_output):
        print("\nmatmul_test passed")
    else:
        print("\nmatmul_test failed")


def run_all_tests():
    word_count_test()
    inverted_index_test()
    ngram_test()
    inner_join_test()
    matmul_test()


if __name__ == "__main__":
    # Run individual tests: `python prog_model_sol.py word_count_test`
    # Run all tests: `python prog_model_sol.py run_all_tests | grep -E "passed|failed"`
    fire.Fire()
