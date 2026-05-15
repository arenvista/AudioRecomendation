import math
import csv # uses first row of CSV as keys

# calculates euclidean distance between two songs in 3D PCA space
def euclidean_distance(song1, song2):
    return math.sqrt(
        (float(song1["PC1"]) - float(song2["PC1"]))**2 +
        (float(song1["PC2"]) - float(song2["PC2"]))**2 + 
        (float(song1["PC3"]) - float(song2["PC3"]))**2)

# target_song is the song we want to find similar songs to
# dataset is a list of songs, where each song has attributes "Filename", "PC1", "PC2", and "PC3"
# recs is the number of similar songs to return
def find_similar(target_song, dataset, recs=5):
    distances = []

    for song in dataset:
        if song["Filename"] == target_song["Filename"]:
            continue
        
        # euclidean distance in 3D space
        distance = euclidean_distance(target_song, song)
        
        distances.append({"Filename": song["Filename"], "Distance": distance})

    distances.sort(key=lambda x: x["Distance"])

    return [song["Filename"] for song in distances[:recs]]

# determines distance between songs in a user's cluster
def intra_cluster_distance(user_dataset):
    # groups songs by user
    users = {}
    for song in user_dataset:
        if int(song["User_ID"]) not in users:
            users[int(song["User_ID"])] = []
        users[int(song["User_ID"])].append(song)
    
    
    intra_distances = {}
    for user in users:
        songs = users[user]
        distance = 0

        # finds distance between every combination of songs for each user
        for i in range(len(songs)):
            for j in range(i+1, len(songs)):
                distance += euclidean_distance(songs[i], songs[j])
        # divides by number of combinations to get average distance
        intra_distances[user] = distance / (len(songs) * (len(songs) - 1) / 2)

    
    for user in intra_distances:
        print(f"User {user} | Intra-cluster distance: {intra_distances[user]}")

    return intra_distances

# determines distance between clusters for each pair of users
def inter_cluster_distance(user_dataset):
    # groups songs by user
    users = {}
    for song in user_dataset:
        if int(song["User_ID"]) not in users:
            users[int(song["User_ID"])] = []
        users[int(song["User_ID"])].append(song)

    inter_distances = {}
    for user1 in users:
        for user2 in users:
            if user1 >= user2:
                continue
            distance = 0

            # finds distance between every combination of songs for each pair of users
            for song1 in users[user1]:
                for song2 in users[user2]:
                    distance += euclidean_distance(song1, song2)
            # divides by number of combinations to get average distance
            inter_distances[(user1, user2)] = distance / (len(users[user1]) * len(users[user2]))

    for user_pair in inter_distances:
        print(f"User {user_pair[0]} & User {user_pair[1]} | Inter-cluster distance: {inter_distances[user_pair]}")
    
    return inter_distances

# loads csv
def load_dataset(PCA_dataset):
    dataset = []
    with open(PCA_dataset, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
    return dataset

# PCA_dataset is a csv that contains PCA data for all songs in dataset
# each row has "Filename", "Full_Path", "PC1", "PC2", and "PC3"
PCA_dataset = load_dataset("path_here/PCA_dataset.csv")

# PCA_user is a csv that contains PCA data for songs by user
# each row has "User_ID", "Filename", "Full_Path", "PC1", "PC2", and "PC3"
PCA_user = load_dataset("path_here/PCA_user.csv")

## RECOMMENDATIONS ##
users = {}
num_recs = 5
for song in PCA_user:
    if int(song["User_ID"]) not in users:
        users[int(song["User_ID"])] = set()
    users[int(song["User_ID"])].update(find_similar(song, PCA_dataset, recs=num_recs))

for user in users:
    print(f"User {user} recommendations:")
    for i, recs in enumerate(users[user]):
        print(f"  {i+1}. {recs}")

## INTRA-CLUSTER DISTANCE ##
intra_cluster_distance(PCA_user)

## INTER-CLUSTER DISTANCE ##
inter_cluster_distance(PCA_user)
