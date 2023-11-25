import re
import matplotlib.pyplot as plt

# File paths for the uploaded logs
base_add = "/export/livia/home/vision/Abahri/projects/Hard_Patches_Point_Cloud/HPM/logs_PMAE_PM2AE_PMAE-SA3D_PM2AE-SA3D/"
#file_paths = [base_add + "PM2AE.log", base_add + "PM2AE-SA3DF.log", base_add + "PMAE.log", base_add + "PMAE-SA3DF.log"]

file_paths = [base_add + "PM2AE.log", base_add + "PM2AE-SA3DF.log"]
#file_paths = [base_add + "PMAE.log", base_add + "PMAE-SA3DF.log"]     
#file_paths = [base_add + "PMAE.log", base_add + "PMAE-SA3DP.log", base_add + "PMAE-SA3DF.log"] 

# Function to extract accuracy values from a given file
def extract_accuracies_v2(file_path):
    accuracies = []
    with open(file_path, 'r') as file:
        for line in file:
            # Searching for the accuracy patterns in each line
            match = re.search(r'Linear Accuracy : (\d+\.\d+)|"val_acc": (\d+\.\d+)|acc = (\d+\.\d+)|"val_svm_acc": (\d+\.\d+)', line)
            if match:
                # Extract the matched accuracy value
                matched_groups = match.groups()
                accuracy = next((float(m) for m in matched_groups if m is not None), None)
                if accuracy:
                    accuracies.append(accuracy)
    return accuracies

# Extract accuracies from each log file
accuracies_all_files = [extract_accuracies_v2(file_path) for file_path in file_paths]

# Plotting the accuracies
plt.figure(figsize=(10, 6))
#colors = ['b', 'g', 'r', 'c']  # Different colors for each file
#labels = ['PM2AE', 'PM2AE-SA3DF', 'PMAE', 'PMAE-SA3DF']  # Labels for each file

#colors = ['b', 'g']  # Different colors for each file
#labels = ['PM2AE', 'PM2AE-SA3DF']  # Labels for each file

colors = ['r', 'b']  # Different colors for each file
#labels = ['PointMAE', 'PointMAE+SA3D', 'PointMAE+SA3DF']  # Labels for each file
labels = ['PointM2AE', 'PointM2AE+SA3DF']  # Labels for each file

for accuracies, color, label in zip(accuracies_all_files, colors, labels):
    epochs = range(1, len(accuracies[:40]) + 1)
    plt.plot(epochs, accuracies[:40], color=color, label=label, linewidth=3)

plt.xlabel('Epochs', fontsize=16)
plt.ylabel('SVM Accuracy', fontsize=16)
plt.title('Accuracy vs Epochs for Different Models', fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig(base_add + "accuracy_vs_epochs_plot_PM2AE_PM2AESA3DF_3.png")