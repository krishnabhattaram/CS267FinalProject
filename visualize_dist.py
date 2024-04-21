import matplotlib.pyplot as plt
from collections import Counter

# Load text from a file
input_filename = '/global/homes/n/nerscii/RBMCuda/build/slurm-24600666.out'
with open(input_filename, 'r') as file:
    # Read lines and filter out lines containing "Simulation Time"
    lines = [line.strip() for line in file if "Simulation Time" not in line]

# Join the filtered lines back into a single string
text = "\n".join(lines)

# Extract numbers from each line
numbers = [line.split()[2] for line in text.strip().split("\n")]

# Count occurrences of each number pattern
histogram = Counter(numbers)

# Calculate total count for normalization
total_count = sum(histogram.values())

# Print the histogram as percentages
print("Number Pattern Histogram (Percentages):")
for number, count in histogram.items():
    percentage = (count / total_count) * 100
    print(f"{number}: {percentage:.4f}%")

# Plot the histogram
plt.bar(histogram.keys(), histogram.values())
plt.xlabel('Number Pattern')
plt.ylabel('Frequency')
plt.title('Number Pattern Histogram')

# Save the plot based on the input filename
output_filename = input_filename.replace('.out', '_histogram.jpg')
plt.savefig(output_filename)
