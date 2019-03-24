import fishfinder 
import pandas as pd

# create FishFinder object
ff = fishfinder.FishFinder()

# generate datasets and display a sample of the generated data
ff.generate_datasets(save_to_csv=True)
df = ff.get_dataframe()
print(df.head())

# create path 
ff.generate_map()
# get straight line and write coordinates into a table
x_straight, y_straight = ff.get_straight_path()
df_grid = pd.DataFrame()
df_grid.insert(0, 'X', x_straight)
df_grid.insert(1, 'Y', y_straight)
print(df_grid)
