from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def scale_data(data,scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    return scaled_data,scaler


def preprocess(J_,A_,H_,is_split=False):

  '''
  J_: Jones coefficients
  A_: Alexander coefficients
  H_: Homfly coefficients
  is_split: False - If not split, True - If split
  '''
  jones = J_[~J_['knot_id'].str.contains('!')]
  if is_split:
    Homfly = H_.iloc[jones['knot_id']]
  jones = jones.drop(columns=['knot_id', 'representation','is_alternating','signature','minimum_exponent','maximum_exponent'])
  jones['knot'] = jones['number_of_crossings'].astype(str) + '_' + jones['table_number'].astype(str)
  jones = jones.drop(columns=['number_of_crossings','table_number'])
  col = jones.pop('knot')
  jones.insert(0, 'knot', col)
  jones = jones.drop(columns=["knot"])

  alexander = A_[A_["number_of_crossings"] < 16]
  alexander['knot'] = alexander['number_of_crossings'].astype(str) + '_' + alexander['table_number'].astype(str)
  col = alexander.pop('knot')
  alexander.insert(0, 'knot', col)
  alexander = alexander.drop(columns=["N/A_1","number_of_crossings","table_number","table_number","is_alternating","signature","minimum_exponent","maximum_exponent"])
  alexander = alexander.drop(columns=["knot"])

  return alexander, jones, Homfly



# Suppose your DataFrame is called df and has a column called 'signature'
# And you want to select 6135 random samples for each of the following values:
def sampling(df,n_,target_signatures = [0, 2, 4, 6, 8]):
  '''
  df: DataFrame
  n: Number of samples
  target_signatures: List of target signatures

  '''
  df_ = (
      df[df['signature'].isin(target_signatures)]
      .groupby('signature', group_keys=False)
      .apply(lambda x: x.sample(n=n_, random_state=42))
  )
  df_.drop(columns=['signature'], inplace=True)

  return df_


def drop_(X,index_):
  A = X.drop(index=index_).reset_index(drop=True)
  return A

def split_data(X,Y,signature_data,test_size = 0.2, random_state = 42, batch_size=32):
    X_train, X_test, y_train, y_test, sig_train, sig_test = train_test_split(
    X, Y, signature_data, test_size=0.2, random_state=42
    )
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader,test_loader,X_test,y_test,sig_train,sig_test