from typing import Optional, Union
import pandas as pd
import numpy as np
import scipy.stats

def remove_anomalies_z_score(df: pd.DataFrame, z_score_lim: float) -> pd.DataFrame:
	"""
	Function to remove anomalies in the data, according to the Z-score.
	"""
	ori_len = len(df)
	z_scores = scipy.stats.zscore(df)
	abs_z_scores = np.abs(z_scores)
	filtered_entries = (abs_z_scores < z_score_lim).all(axis=1) # Remove the anomalies that have a large Z-score
	df = df[filtered_entries]
	new_len = len(df)
	print('Removed ',ori_len-new_len, 'anomalies.')
	return df

def add_bias_term(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Add first column ("00") of 1 for the bias term.
	"""
	df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)
	return df
	
def feature_scaling(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Function to do standardisation for each columns.
	Calculating the Z-scores for the columns.
	"""
	for i in range(1, len(df.columns)):
		df[i-1] = scipy.stats.zscore(df[i-1])
	return df

def init_theta(df: pd.DataFrame) -> np.ndarray:
	"""
	Function to return the initiate features parameters, theta
	"""
	theta = np.array([0.1]*len(df.columns))
	return theta