import pandas as pd
tsv_file='train.tsv'
csv_table=pd.read_table(tsv_file,sep='\t')
csv_table.to_csv('sst2.csv',index=False)