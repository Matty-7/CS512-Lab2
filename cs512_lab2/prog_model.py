from typing import Any, Callable

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy


class PandasMapReduce:
    def __init__(
        self,
        mapper_func: Callable[[Any, Any], list[tuple[Any, Any]]],
        reducer_func: Callable[[Any, list[Any]], list[Any]],
    ):
        """
        Initializes the MapReduce object with user-defined mapper and reducer functions.

        Parameters:
        -----------
        mapper_func : Callable[[K1, V1], list[tuple[K2, V2]]]
            The mapper function that processes each key-value pair and returns
            a list of key-value pairs.

        reducer_func : Callable[[K2, list[V2]], list[V3]]
            The reducer function that processes each key and its list of values,
            returning the reduced result. Note that the reducer function should
            return a list of values, even if there is only one value to return.
        """
        self.mapper_func = mapper_func
        self.reducer_func = reducer_func

    def map(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the mapper function to each row of the input DataFrame.

        Parameters:
        -----------
        input_df : pd.DataFrame
            The input DataFrame to be processed by the mapper function. It contains two columns,
            'key' (K1) and 'value' (V1).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing intermediate results, with columns 'key' (K2) and 'value' (V2).
        """
        if set(input_df.columns) != {"key", "value"}:
            raise ValueError("Input DataFrame must have 'key' and 'value' columns")

        intermediate_df = (
            input_df.apply(
                lambda row: self.mapper_func(row["key"], row["value"]), axis=1
            )  # Map each row to a list of key-value pairs
            .explode()  # Convert list of key-value pairs to separate rows
            .dropna()  # Handle the case where ``mapper_func`` returns an empty list
        )
        return pd.DataFrame(intermediate_df.tolist(), columns=["key", "value"])

    def shuffle(self, intermediate_df: pd.DataFrame) -> DataFrameGroupBy:
        """
        Groups the intermediate results by key.

        Parameters:
        -----------
        intermediate_df : pd.DataFrame
            A DataFrame containing intermediate results from the map phase. It contains two columns,
            'key' (K2) and 'value' (V2).

        Returns:
        --------
        DataFrameGroupBy
            A grouped DataFrame, where values are grouped by key (K2), preparing them
            for the reduce phase.
        """
        return intermediate_df.groupby("key")

    def reduce(self, grouped_df: DataFrameGroupBy) -> pd.DataFrame:
        """
        Applies the reducer function to each group of values.

        Parameters:
        -----------
        grouped_df : DataFrameGroupBy
            A grouped DataFrame obtained from the shuffle phase.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the final reduced results, with columns 'key' (K2)
            and 'value' (V3).
        """
        reduced_df = (
            grouped_df.apply(
                lambda group: self.reducer_func(group.name, group["value"].tolist())
            )
            .explode()  # Convert list of values to separate rows
            .reset_index()
        )
        reduced_df.columns = ["key", "value"]
        return reduced_df

    def execute(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the complete MapReduce job, combining the map, shuffle, and reduce phases.

        Parameters:
        -----------
        input_df : pd.DataFrame
            The input DataFrame to be processed by the MapReduce job. It contains two columns,
            'key' (K1) and 'value' (V1).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the final results of the MapReduce job, with columns
            'key' (K2) and 'reduced_value' (V3).
        """
        intermediate_df = self.map(input_df)
        grouped_df = self.shuffle(intermediate_df)
        output_df = self.reduce(grouped_df)
        return output_df
