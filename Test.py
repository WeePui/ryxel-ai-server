import pickle

# Load the saved rules
with open("d:/ryxel-server-ts/src/Python_Server/association_rules.pkl", "rb") as f:
    loaded_rules = pickle.load(f)

# Test basket
test_basket = set(["637d100014b63b001c7e9d01", "637d100014b63b001c7e9d04"])

# Filter rules where antecedents match the test basket
relevant_rules = loaded_rules[
    loaded_rules["antecedents"].apply(lambda x: x.issubset(test_basket))
]

relevant_rules = relevant_rules.sort_values(by="confidence", ascending=False)

# Limit to 10 recommendations
top_10_recommendations = relevant_rules.head(10)
print("Top 10 Recommended rules:")
print(top_10_recommendations[["antecedents", "consequents", "confidence"]])
