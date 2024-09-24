import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename", type=str,
                    help="CSV file of reference data", required= True)

args = parser.parse_args()
# Load the CSV containing the names and their respective photo filename
df = pd.read_csv(args.filename, header=None)

name_list = df[0].to_list()
img_list = df[1].to_list()
