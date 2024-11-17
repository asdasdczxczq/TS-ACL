import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given data for each method and dataset with accuracy and standard deviation
data = {
    "Dataset": ["UCI-HAR", "UWave", "DSA", "GRABMyo", "WISDM"],
    "Naive": [(32.00, 2.90), (25.36, 0.59), (17.04, 0.82), (19.44, 0.38), (14.89, 1.39)],
    "Offline": [(94.94, 2.17), (96.61, 1.05), (99.65, 0.46), (93.63, 1.22), (85.31, 1.30)],
    "LwF": [(35.96, 11.33), (44.67, 11.64), (24.82, 8.02), (19.22, 0.49), (14.89, 2.55)],
    "MAS": [(52.34, 15.92), (53.80, 10.36), (31.82, 6.36), (19.04, 1.45), (10.74, 2.79)],
    "DT2W": [(53.23, 16.33), (64.44, 5.92), (19.56, 3.82), (21.34, 4.75), (17.29, 6.76)],
    "GR": [(65.46, 13.78), (70.28, 8.21), (79.75, 18.11), (47.03, 6.82), (41.69, 7.54)],
    "ER": [(65.46, 13.78), (70.28, 8.21), (79.75, 18.11), (47.03, 6.82), (41.69, 7.54)],
    "DER": [(74.41, 7.57), (70.88, 7.71), (59.19, 17.56), (31.38, 4.65), (28.99, 9.19)],
    "Herding": [(69.58, 19.66), (78.47, 2.87), (82.42, 10.04), (47.14, 7.43), (42.42, 4.99)],
    "ASER": [(92.36, 2.78), (82.74, 2.01), (97.26, 1.59), (56.50, 4.24), (48.36, 16.79)],
    "CLOPS": [(72.87, 11.91), (71.04, 1.66), (74.10, 15.43), (43.75, 4.03), (32.95, 5.89)],
    "FastICARL": [(79.69, 7.77), (67.77, 8.62), (67.28, 15.06), (40.55, 4.97), (32.72, 6.23)],
    "TS-ACL": [(88.41, 1.52), (91.89, 1.72), (98.33, 1.34), (57.06, 3.80), (85.35, 2.81)]
}

# Initialize dictionaries to store the mean accuracy and mean standard deviation for each method
mean_accuracy = {}
mean_stddev = {}

# Compute mean accuracy and mean standard deviation for each method
for method in list(data.keys())[1:]:
    accuracies = [x[0] for x in data[method]]
    stddevs = [x[1] for x in data[method]]
    
    mean_accuracy[method] = np.mean(accuracies)
    mean_stddev[method] = np.mean(stddevs)

# Create a DataFrame for displaying
results = pd.DataFrame({
    'Method': list(mean_accuracy.keys()),
    'Mean Accuracy': list(mean_accuracy.values()),
    'Mean Std Dev': list(mean_stddev.values())
})

# Remove 'Naive' and 'Offline' from the data for visualization
filtered_results = results[~results['Method'].isin(['Naive', 'Offline'])]
# Define custom colors for each method
custom_colors = {
    'LwF': '#54B345',
    'MAS': '#32B897',
    'DT2W': '#05B9E2',
    'ER': '#BB9727',
    'DER': '#C76DA2',
    'Herding': '#A1A9D0',
    'ASER': '#F0988C',
    'CLOPS': '#96CCCB',
    'FastICARL': '#F6CAE5',
    'TS-ACL': '#FFC107'
}

# Plotting the results with custom colors
plt.figure(figsize=(10, 6))

# Plot the mean accuracy with specific colors
plt.bar(filtered_results['Method'], filtered_results['Mean Accuracy'], 
        yerr=filtered_results['Mean Std Dev'], 
        capsize=10, 
        alpha=0.7, 
        color=[custom_colors[method] for method in filtered_results['Method']], 
        label='Mean Accuracy')

plt.xlabel('Method')
plt.ylabel('Mean Accuracy (%)')
plt.title('Mean Accuracy and Standard Deviation of Methods (Excluding Naive and Offline)')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Save the plot as an image file
plt.savefig('method_accuracy_plot_colored.png', dpi=300)

# Show the plot
plt.show()