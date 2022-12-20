import joblib
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt


# open a file, where you stored the pickled data
data = joblib.load('experiments/econ_reg/Dec19_23-14_ECON_experiments/seed0/2022_12_19-23_33_stBERT/teacher_dump/att_scores_test_iter18.pkl')

df = pd.DataFrame(data)
df.columns = ['Okun', 'Taylor', 'AR(1)', 'AR(2)', 'AR(3)', 'AR(4)', 'NKPC', 'Student']

fred = Fred(api_key='30adf5295a539a48e57fe367896a60e9')
GDP = fred.get_series('GDPC1', units='pc1', frequency='q')
dev_cutoff = '2018-01-01'
df.index = GDP.index[(GDP.index >= dev_cutoff)]
df.index = pd.PeriodIndex(df.index, freq='Q')

# Displaying dataframe as an heatmap
# with diverging colourmap as RdYlBu
plt.imshow(df, cmap ="RdYlBu", aspect='auto')
  
# Displaying a color bar to understand
# which color represents which range of data
plt.colorbar()
  
# Assigning labels of x-axis 
# according to dataframe
plt.xticks(range(len(df.columns)), df.columns)
  
# Assigning labels of y-axis 
# according to dataframe
plt.yticks(range(len(df.index)), df.index)

# Displaying the figure
plt.show()