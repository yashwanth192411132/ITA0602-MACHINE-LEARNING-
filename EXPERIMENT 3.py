
import pandas as pd
import math
from collections import Counter

# Play Tennis dataset
data = {
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rainy",
        "Rainy", "Rainy", "Overcast", "Sunny",
        "Sunny", "Rainy", "Sunny", "Overcast",
        "Overcast", "Rainy"
    ],
    "Temperature": [
        "Hot", "Hot", "Hot", "Mild", "Cool", "Cool",
        "Cool", "Mild", "Cool", "Mild", "Mild", "Mild",
        "Hot", "Mild"
    ],
    "Humidity": [
        "High", "High", "High", "High", "Normal", "Normal",
        "Normal", "High", "Normal", "Normal", "Normal", "High",
        "Normal", "High"
    ],
    "Wind": [
        "Weak", "Strong", "Weak", "Weak", "Weak", "Strong",
        "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
        "Weak", "Strong"
    ],
    "PlayTennis": [
        "No", "No", "Yes", "Yes", "Yes", "No", "Yes",
        "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"
    ]
}

df = pd.DataFrame(data)
target = "PlayTennis"

# Calculate entropy
def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    result = 0

    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)

    return result

# Calculate information gain
def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    weighted_entropy = 0

    for value in df[attribute].unique():
        subset = df[df[attribute] == value]
        weighted_entropy += (
            len(subset) / len(df)
        ) * entropy(subset[target])

    return total_entropy - weighted_entropy

# Build ID3 decision tree
def id3(df, target, attributes):
    labels = df[target]

    # If all examples have the same class
    if labels.nunique() == 1:
        return labels.iloc[0]

    # If no attributes remain
    if not attributes:
        return labels.mode()[0]

    # Select attribute with maximum information gain
    best = max(
        attributes,
        key=lambda a: information_gain(df, a, target)
    )

    tree = {best: {}}
    remaining = [a for a in attributes if a != best]

    for value in sorted(df[best].unique()):
        subset = df[df[best] == value]

        if subset.empty:
            tree[best][value] = labels.mode()[0]
        else:
            tree[best][value] = id3(
                subset, target, remaining
            )

    return tree

# Display decision tree
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "-> " + str(tree))
        return

    attribute = next(iter(tree))

    for value, subtree in tree[attribute].items():
        print(f"{indent}{attribute} = {value}")
        print_tree(subtree, indent + "  ")

# Classify a new sample
def predict(tree, sample, default="Yes"):
    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))
    value = sample.get(attribute)

    if value not in tree[attribute]:
        return default

    return predict(tree[attribute][value], sample, default)

attributes = [col for col in df.columns if col != target]

print("Information Gain:")
for attribute in attributes:
    print(
        f"{attribute}: "
        f"{information_gain(df, attribute, target):.3f}"
    )

tree = id3(df, target, attributes)

print("\nDecision Tree:")
print_tree(tree)

# New sample to classify
sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

print("\nNew Sample:", sample)
print("Predicted Class:", predict(tree, sample))
