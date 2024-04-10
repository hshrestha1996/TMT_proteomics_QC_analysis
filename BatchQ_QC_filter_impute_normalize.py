#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from itertools import cycle
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import to_rgba

def z_scores_for_proteins(data):
    # Calculate the mean and standard deviation for each protein (row)
    z_scores=data.sub(np.mean(data, axis=1), axis="index").div(np.std(data, axis=1), axis="index")

    return z_scores

def plot_batch_boxplot(proteins_log,title="Boxplot Before Any Correction", ylabel="Log2 Abundance", xlabel= "Samples across batches"):
    # Extract batch information from column names
    batches = proteins_log.columns.str.split("_").str.get(0)
    unique_batches = batches.unique()
    n = int(len(batches) / len(unique_batches))

    # Create a list to store color names for each batch
    colors = cycle(plt.cm.tab10.colors)
    batch_color_names = [plt.cm.tab10(i) for i in range(len(unique_batches))]
    batch_color_names = [value for value in batch_color_names for _ in range(n)]

    # Before any correction
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    proteomics_data = proteins_log  # 100 samples and 5 proteins
    ax = axs

    # Plot the boxplot using batch colors
    boxplot = ax.boxplot(proteomics_data, vert=True, patch_artist=True)

    # Set colors for each batch
    for box, color_name in zip(boxplot['boxes'], batch_color_names):
        box.set_facecolor(color_name)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(proteomics_data.min().min(), proteomics_data.max().max())

    # Adjust layout
    plt.tight_layout()
    plt.savefig(title, dpi=300)

def mean_center_sub(data, mean=True, center_zero=True):
    ndata = data.copy()
    
    # Compute the mean or median of each column in the data
    center = np.mean(data, axis=0) if mean else np.median(data, axis=0)
    
    # Transform the center vector into a matrix
    center_matrix = np.tile(center, (data.shape[0], 1))
    
    if center_zero:
        # Center the data at zero
        ndata -= center_matrix
    else:
        # Center the data at the maximum mean or median
        new_average = np.nanmax(center)
        ndata = ndata - center_matrix + new_average
    
    return ndata

# Redirect stdout to a log file
sys.stdout = open('log_file.txt', 'w')

# Command-line arguments
excel_batchQname = sys.argv[1]
first_row = int(sys.argv[2])
NaN_threshold = float(sys.argv[3])
IR_channel = sys.argv[4]

# Load libraries and functions
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from itertools import cycle
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import to_rgba

# Load Excel data
df = pd.read_excel(excel_batchQname, skiprows=first_row)
df = df.set_index("ProteinAccession")

# Print basic info
info = df.loc[:, df.columns.str.contains("sig")].columns.str.split("_").str.get(0)
uniq_batches = info.unique()
n = int(len(info) / len(uniq_batches))
print(df.shape[0], 'total proteins are identified in batchQ.')
print("TMT:", n, "was used.")
print(len(uniq_batches), 'unique batches are present.')

# Proteins quantified across batches
data = df
total_peptides_columns = data.filter(like="TotalPeptides")
total_peptides_summary = total_peptides_columns.describe()
data['batchMissed'] = total_peptides_columns.apply(lambda x: (x == 0).sum(), axis=1)

# Visualization
plt.figure(figsize=(8, 5))
tmp = data['batchMissed'].value_counts(normalize=True) * 100
tmp = tmp.sort_index()
bp = tmp.plot(kind='bar', ylim=(0, 100), ylabel="% proteins (total n = {})".format(len(data)),
              xlabel="# batches in which a protein is not quantified", color='grey', edgecolor='black')
for i, v in enumerate(tmp):
    plt.text(i, v + 2, f"{round(v, 2)}%", ha='center', va='bottom', fontname='Arial')
plt.title("Proteins quantified across batches", fontname='Arial')
plt.savefig('Proteins_quantified_across_batches.png', dpi=300)
plt.show()

# Impute user-defined threshold
print('You select to keep data with', (100 - NaN_threshold * 100), "%. NaN values. The NaN values will be imputed using KNN method.")
df_filtered = df.loc[:, df.columns.str.contains("sig")]
df_filtered.dropna(thresh=(df_filtered.shape[1] * NaN_threshold), inplace=True)
print(df_filtered.shape[0], 'proteins will be in the final matrix.')
imputer = KNNImputer(n_neighbors=10, weights='distance')
imputed_data = imputer.fit_transform(df_filtered)
df_imputed = pd.DataFrame(imputed_data, index=df_filtered.index, columns=df_filtered.columns)
df_imputed.to_csv("proteindf_raw_withNaN_impute.csv")

flattened_data_original = df_filtered.applymap(np.log2).values.flatten()
flattened_data_imputed = df_imputed.applymap(np.log2).values.flatten()

plt.figure(figsize=(8, 6))
plt.hist(flattened_data_imputed, bins=50, color='orange', alpha=0.7, label='Imputed Data', edgecolor='black')
plt.hist(flattened_data_original, bins=50, color='skyblue', alpha=0.7, label='Original Data', edgecolor='black')
plt.title('Histogram of Original and Imputed Data')
plt.xlabel('Abundance Values (log2)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig("Before_and_after_impute_histogram.png", dpi=300)

proteins_log = df_imputed.applymap(np.log2)

plot_batch_boxplot(proteins_log, title="Boxplot Before Batch Correction", ylabel="log2", xlabel="Samples across batches")

IR = proteins_log.loc[:, proteins_log.columns.str.contains(IR_channel)]
Average_IR = IR.mean(axis=1)
IF_factor = -(IR.sub(Average_IR, axis=0))

batches = proteins_log.columns.str.split("_").str.get(0)
unique_batches = batches.unique()
median_int = pd.DataFrame()
for batch in unique_batches:
    batch = batch + "_"
    for_median = proteins_log.loc[:, proteins_log.columns.str.contains(batch)]
    median = for_median.median(axis=1)
    median_int[batch] = median
Average_medInt = median_int.mean(axis=1)
medInt_factor = -(median_int.sub(Average_medInt, axis=0))
medInt_factor.columns = IF_factor.columns
Factor = (IF_factor + medInt_factor) / 2

print(' Protein matrix will now be corrected with IR + median factor and then they will be mean centered.')

corrected_dataframes = []
for batch in unique_batches:
    batch = batch + "_"
    for_correction = proteins_log.loc[:, proteins_log.columns.str.contains(batch)]
    batch_factor = Factor.loc[:, Factor.columns.str.contains(batch)]
    corrected = for_correction.add(batch_factor.values, axis=1)
    corrected_dataframes.append(corrected)

corrected_proteins = pd.concat(corrected_dataframes, axis=1)

plot_batch_boxplot(corrected_proteins, title="Boxplot After Batch Correction", ylabel="log2", xlabel="Samples across batches")

mydata_log_centered = mean_center_sub(corrected_proteins, mean=True, center_zero=False)

plot_batch_boxplot(mydata_log_centered, title="Boxplot After Batch Correction & Mean Centering",
                   ylabel="Log2 Abundance", xlabel="Samples across batches")

original_corrected = 2 ** mydata_log_centered
original_corrected.to_csv("raw_intenisty_batch_corrected_and_centered.csv")

mydata_log_centered_z = z_scores_for_proteins(mydata_log_centered)

fig, axs = plt.subplots(1, 2, figsize=(16, 7))
ax = axs[0]
pca = PCA()
principal_components = pca.fit_transform(mydata_log_centered_z)
batches = mydata_log_centered.columns.str.split("_").str.get(0)
unique_batches = batches.unique()
colors = cycle(plt.cm.tab10.colors)
batch_colors = {batch: next(colors) for batch in unique_batches}
for batch in unique_batches:
    indices = [i for i, b in enumerate(batches) if b == batch]
    ax.scatter(principal_components[indices, 0], principal_components[indices, 1], label=batch,
               color=batch_colors[batch])
    cov_matrix = np.cov(principal_components[indices, 0], principal_components[indices, 1])
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = np.sqrt(eigenvalues[0]) * 2 * 1.96
    height = np.sqrt(eigenvalues[1]) * 2 * 1.96
    ellipse = Ellipse(xy=(np.mean(principal_components[indices, 0]), np.mean(principal_components[indices, 1])),
                      width=width, height=height, angle=angle, fill=False, color=batch_colors[batch],
                      linestyle='dashed')
    ax.add_patch(ellipse)

ax.set_title('PCA Plot with 95% confidence interval ellipse')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.grid()
ax = axs[1]
pca = PCA()
principal_components = pca.fit_transform(mydata_log_centered_z)
batches = mydata_log_centered.columns.str.split("_").str.get(0)
unique_batches = batches.unique()
colors = cycle(plt.cm.tab10.colors)
batch_colors = {batch: next(colors) for batch in unique_batches}
for batch in unique_batches:
    indices = [i for i, b in enumerate(batches) if b == batch]
    ax.scatter(principal_components[indices, 0], principal_components[indices, 1], label=batch,
               color=batch_colors[batch])

ax.set_title('PCA Plot')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend(title='Batches', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid()
fig.savefig("PCA_plot_after_batch_correction_and_mean_centering_zscore", dpi=300)

# Close the log file
sys.stdout.close()

