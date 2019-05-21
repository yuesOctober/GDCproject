# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
from utils import logger
if __name__ == '__main__':


	data_dir ="/Users/yueshi/Downloads/GDCProject/data/"
	# Input directory and label file. The directory that holds the data. Modify this when use.
	# dirname = data_dir + "brain_miRNA"
	liver_miRNA = data_dir + "miRNA_matrix.csv"
	brain_miRNA = data_dir + "brain_miRNA_matrix.csv"
	df_liver_both = pd.read_csv(liver_miRNA)
	# print(df_liver_both)
	df_liver = df_liver_both.loc[df_liver_both['label'] == 1]
	df_brain_both = pd.read_csv(brain_miRNA)
	df_brain = df_brain_both.loc[df_brain_both['label'] == 1]
	df_brain.label[df_brain.label == 1] = 0
	#output file
	outputfile = data_dir + "merged_miRNA.csv"
	frames = [df_liver, df_brain]
	result = pd.concat(frames)

	
	result.to_csv(outputfile, index=False)
	#print (labeldf)