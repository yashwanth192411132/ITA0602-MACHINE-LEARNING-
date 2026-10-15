
# FIND-S Algorithm

import pandas as pd

# Training dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Same'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print("Training Dataset:")
print(df)

# Initialize the most specific hypothesis
hypothesis = None

# Apply FIND-S
for _, row in df.iterrows():
    if row['EnjoySport'] == 'Yes':
        attributes = list(row[:-1])

        if hypothesis is None:
            hypothesis = attributes
        else:
            for i in range(len(hypothesis)):
                if hypothesis[i] != attributes[i]:
                    hypothesis[i] = '?'

    print("Current Hypothesis:", hypothesis)

print("\nFinal Most Specific Hypothesis:")
print(hypothesis)
