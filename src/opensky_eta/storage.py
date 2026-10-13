"""Reading and writing the hourly parquet files of the processing pipeline."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from opensky_eta.config import BATCH_SIZE


def list_parquet_files(directory, n_files=None):
    """Sorted parquet files of a directory (only the first `n_files` if set, for a dry run)."""
    files = sorted(Path(directory).glob("*.parquet"))
    return files if n_files is None else files[:n_files]


def file_time_range(path):
    """(min, max) of the `time` column, read from the parquet statistics."""
    metadata = pq.ParquetFile(path).metadata
    mins, maxs = [], []
    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        for j in range(row_group.num_columns):
            column = row_group.column(j)
            if column.path_in_schema == "time":
                mins.append(column.statistics.min)
                maxs.append(column.statistics.max)
    return min(mins), max(maxs)


def to_utc(timestamp):
    """Unix timestamp(s) -> UTC datetime(s)."""
    return pd.to_datetime(timestamp, unit="s", utc=True)


def read_flight(directory, flight, columns):
    """All state vectors of one flight (`flight` holds the FLIGHT_KEY values)."""
    condition = (
        (ds.field("icao24") == flight["icao24"])
        & (ds.field("firstSeen") == flight["firstSeen"])
        & (ds.field("lastSeen") == flight["lastSeen"])
    )
    table = ds.dataset(directory, format="parquet").to_table(
        columns=columns, filter=condition
    )
    return table.to_pandas().sort_values("time").reset_index(drop=True)


def write_filtered(source, destination, keep, batch_size=BATCH_SIZE):
    """Copy a parquet file, keeping only the rows where `keep` is True."""
    writer, offset = None, 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size):
        mask = pa.array(keep[offset:offset + batch.num_rows])
        offset += batch.num_rows
        table = pa.Table.from_batches([batch]).filter(mask)
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema)
        writer.write_table(table)
    writer.close()
