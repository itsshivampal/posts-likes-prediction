import pandas as pd
import numpy as np
import matplotlib as plt

training_data = pd.read_csv("data/train.csv")

# Drop rows which contain any missing value
training_data.dropna(axis = 0, how = "any", inplace = True)

# Data Normalization
views = training_data['#views'].to_numpy()
comments = training_data['#comments'].to_numpy()
likes = training_data['#likes'].to_numpy()

max_views, min_views = max(views), min(views)
max_comments, min_comments = max(comments), min(comments)
max_likes, min_likes = max(likes), min(likes)

def normalise_views(num):
    return float(num - min_views)/float(max_views - min_views)

def normalise_comments(num):
    return float(num - min_comments)/float(max_comments - min_comments)

def normalise_likes(num):
    return float(num - min_likes)/float(max_likes - min_likes)

category_list = list(training_data.category.unique())
country_list = list(training_data.country.unique())

category_dict = {}

index = 0
for category in category_list:
    category_dict[category] = index
    index += 1

def get_category_val(cat):
    one_hot = np.zeros((len(category_list)), dtype = int)
    index = category_dict[cat]
    one_hot[index] = 1
    return one_hot

def get_country_enc(country):
    one_hot = np.zeros((len(country_list)), dtype = int)
    one_hot[int(country)] = 1
    return one_hot

print("Step 1 done")

for i in range(5000):# range(training_data.shape[0]):
    training_data.at[i, "norm_views"] = normalise_views(training_data[["#views"]].iloc[i].values[0])
    training_data.at[i, "norm_comments"] = normalise_comments(training_data[["#comments"]].iloc[i].values[0])
    training_data.at[i, "norm_likes"] = normalise_likes(training_data[["#likes"]].iloc[i].values[0])

    category_val = get_category_val(training_data[["category"]].iloc[i].values[0])
    training_data.at[i, "cat1"] = category_val[0]
    training_data.at[i, "cat2"] = category_val[1]
    training_data.at[i, "cat3"] = category_val[2]
    training_data.at[i, "cat4"] = category_val[3]
    training_data.at[i, "cat5"] = category_val[4]
    training_data.at[i, "cat6"] = category_val[5]
    training_data.at[i, "cat7"] = category_val[6]
    training_data.at[i, "cat8"] = category_val[7]
    training_data.at[i, "cat9"] = category_val[8]

    country_val = get_country_enc(training_data[["country"]].iloc[i].values[0])
    training_data.at[i, "count1"] = country_val[0]
    training_data.at[i, "count2"] = country_val[1]
    training_data.at[i, "count3"] = country_val[2]
    training_data.at[i, "count4"] = country_val[3]
    training_data.at[i, "count5"] = country_val[4]
    training_data.at[i, "count6"] = country_val[5]
    training_data.at[i, "count7"] = country_val[6]
    training_data.at[i, "count8"] = country_val[7]
    training_data.at[i, "count9"] = country_val[8]
    training_data.at[i, "count10"] = country_val[9]
    training_data.at[i, "count11"] = country_val[10]
    training_data.at[i, "count12"] = country_val[11]
    training_data.at[i, "count13"] = country_val[12]
    training_data.at[i, "count14"] = country_val[13]
    training_data.at[i, "count15"] = country_val[14]
	
	if i%5000 == 0:
		print(i)

print("Step 2 done")

# Feature Extraction
user_list = list(training_data.user_id.unique())

category_avg_data = {}
for category in category_list:
    df = training_data[training_data['category'] == category]
    views = list(df['norm_views'].to_numpy())
    comments = list(df['norm_comments'].to_numpy())
    avg_views = sum(views)/len(views)
    avg_comments = sum(comments)/len(comments)
    category_avg_data[category] = [avg_views, avg_comments]

country_avg_data = {}
for country in country_list:
    df = training_data[training_data['country'] == country]
    views = list(df['norm_views'].to_numpy())
    comments = list(df['norm_comments'].to_numpy())
    avg_views = sum(views)/len(views)
    avg_comments = sum(comments)/len(comments)
    country_avg_data[country] = [avg_views, avg_comments]

user_avg_data = {}
for user in user_list:
    df = training_data[training_data['user_id'] == user]
    views = list(df['norm_views'].to_numpy())
    comments = list(df['norm_comments'].to_numpy())
    avg_views = sum(views)/len(views)
    avg_comments = sum(comments)/len(comments)
    user_avg_data[user] = [avg_views, avg_comments]

print("Step 3 start")


for i in range(5000):# range(training_data.shape[0]):
    training_data.at[i, 'user_avg_views'] = user_avg_data[training_data[["user_id"]].iloc[i].values[0]][0]
    training_data.at[i, 'user_avg_comments'] = user_avg_data[training_data[["user_id"]].iloc[i].values[0]][1]

    training_data.at[i, 'country_avg_views'] = country_avg_data[training_data[["country"]].iloc[i].values[0]][0]
    training_data.at[i, 'country_avg_comments'] = country_avg_data[training_data[["country"]].iloc[i].values[0]][1]

    training_data.at[i, 'category_avg_views'] = category_avg_data[training_data[["category"]].iloc[i].values[0]][0]
    training_data.at[i, 'category_avg_comments'] = category_avg_data[training_data[["category"]].iloc[i].values[0]][1]

	if i%5000 == 0:
		print(i)


# Save Final Data
training_data.to_csv("data/processed_data.csv")

print("All steps done")
