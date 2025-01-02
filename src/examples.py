import pandas as pd

train_df = pd.read_csv("outputs/ml-1m/MF/train_outputs_1.csv")
test_df = pd.read_csv("outputs/ml-1m/MF/test_outputs_1.csv")

# Compute average train rating per user
user_avg_rating = train_df.groupby("user_id")["rating"].mean().rename("user_avg_rating")

# Compute average train rating per item
item_avg_rating = train_df.groupby("item_id")["rating"].mean().rename("item_avg_rating")

# Merge the average ratings with the test dataframe
test_df = test_df.merge(user_avg_rating, on="user_id", how="left")
test_df = test_df.merge(item_avg_rating, on="item_id", how="left")

filtered_df = test_df[
    (test_df["user_avg_rating"] > 4.25) & (test_df["item_avg_rating"] > 4.25) & (test_df["rating"] <= 1)
]

movies_info = pd.read_csv("data/ml-1m/movies.dat", sep="::", engine="python", header=None, encoding="latin-1")
users_info = pd.read_csv("data/ml-1m/users.dat", sep="::", engine="python", header=None, encoding="latin-1")

filtered_df = filtered_df.merge(movies_info, left_on="item_id", right_on=0)
filtered_df = filtered_df.merge(users_info, left_on="user_id", right_on=0)

print(filtered_df)

######################
# CTRPv2
######################

train_df = pd.read_csv("outputs/ctrpv2/MF/train_outputs_1.csv")
test_df = pd.read_csv("outputs/ctrpv2/MF/test_outputs_1.csv")

# Compute average train rating per user
user_avg_rating = train_df.groupby("user_id")["rating"].mean().rename("user_avg_rating")
item_avg_rating = train_df.groupby("item_id")["rating"].mean().rename("item_avg_rating")

# Merge the average ratings with the test dataframe
test_df = test_df.merge(user_avg_rating, on="user_id", how="left")
test_df = test_df.merge(item_avg_rating, on="item_id", how="left")

filtered_df = test_df[
    (test_df["user_avg_rating"] < 13) & (test_df["item_avg_rating"] < 13) & (test_df["rating"] > 18.5)
]

drug_info = pd.read_csv("data/ctrpv2/drug_id_mapping.csv")
cell_line_info = pd.read_csv("data/ctrpv2/user_id_mapping.csv")

filtered_df = filtered_df.merge(drug_info, left_on="item_id", right_on="drug_id")
filtered_df = filtered_df.merge(cell_line_info, left_on="user_id", right_on="new_user_id")

print(filtered_df)


######################
# DOT
######################

train_df = pd.read_csv("outputs/dot_2023/MF/train_outputs_1.csv")
test_df = pd.read_csv("outputs/dot_2023/MF/test_outputs_1.csv")

# Compute average train rating per country (user or item)
avg_ratings = {}
for country_id in train_df["user_id"].unique():
    country_ratings = train_df[train_df["user_id"] == country_id]["rating"]
    avg_ratings[country_id] = (len(country_ratings), country_ratings.sum())

for country_id in train_df["item_id"].unique():
    country_ratings = train_df[train_df["item_id"] == country_id]["rating"]
    country_avg = avg_ratings.get(country_id, (0, 0))
    avg_ratings[country_id] = (country_avg[0] + len(country_ratings), country_avg[1] + country_ratings.sum())

avg_ratings = {k: v[1] / v[0] for k, v in avg_ratings.items()}

# Merge the average ratings with the test dataframe
test_df["user_avg_rating"] = test_df["user_id"].map(avg_ratings)
test_df["item_avg_rating"] = test_df["item_id"].map(avg_ratings)

test_df["avg_diff"] = abs(test_df["user_avg_rating"] - test_df["item_avg_rating"])

test_df["abs_error"] = abs(test_df["rating"] - test_df["pred"])
filtered_df = test_df[(test_df["avg_diff"] > 3) & (test_df["abs_error"] > 1.5)]

country_info = pd.read_csv("data/dot_2023/country_id_mapping.csv")

filtered_df = filtered_df.merge(country_info, left_on="user_id", right_on="new_id")
filtered_df = filtered_df.merge(country_info, left_on="item_id", right_on="new_id")

print(filtered_df)
