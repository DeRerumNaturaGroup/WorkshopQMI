import pandas as pd

descriptors = pd.read_csv('qm9_description.csv')

def print_record(item):
  print("Number of atoms: ", len(item[0][0]))
  print("Atomic numbers of atoms: ", item[0][0])
  df_pos = pd.DataFrame(item[0][1], columns = ['x', 'y','z', 'Partial Charge'])
  df_params = pd.DataFrame(item[1], index = list(descriptors.description) )

  display(df_pos)
  display(df_params)

  return