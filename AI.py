import pickle
import pandas as pd

# Đọc file dữ liệu đơn hàng
df_orders = pd.read_csv("D:/fake_carts.csv")

# Xem dữ liệu
print(df_orders.head())

from mlxtend.frequent_patterns import apriori, association_rules

# Chuyển đổi dữ liệu thành dạng one-hot encoding
basket = df_orders.groupby(['cartId', 'productId'])['quantity'].sum().unstack().fillna(0)

# Chuyển đổi giá trị > 0 thành 1 (one-hot)
basket = basket > 0  # Converts values > 0 to True and <= 0 to False
basket = basket.astype(bool)

# Áp dụng thuật toán Apriori
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

# Sinh luật kết hợp
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

print(rules.head(10))

with open("D:/ryxel-server-ts/src/Python_Server/cart_rules.pkl", "wb") as file:
    pickle.dump(rules, file)

print("Model computed and saved to 'rules.pkl' successfully.")