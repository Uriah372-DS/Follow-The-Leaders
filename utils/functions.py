import pyspark
import pyspark.sql.functions as F


def parse_row(row: pyspark.sql.Row):
    return row.asDict(True)