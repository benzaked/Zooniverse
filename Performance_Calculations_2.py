import numpy as np
import pandas as pd
from more_itertools import unique_everseen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Recommender
from CFItemItem import CFItemItem
from CFUserUser import CFUserUser
from PopularityBased import PopularityBased
from Recommender import is_online_project_recommended
from SVD import SVD

projects_data = pd.read_csv('projects_data.csv', index_col=0)
projects_data = projects_data.fillna('')
records = pd.read_pickle('historical_records.pkl')
data_items_train = pd.read_pickle('data_items_train.pkl')


def calculate_similarity_by_content():
    tf_idf = TfidfVectorizer()
    total_similarity = pd.DataFrame(0, index=range(len(projects_data)), columns=range(len(projects_data)))
    for feature in projects_data.columns:
        feature_similarity = tf_idf.fit_transform(projects_data[feature])
        feature_similarity = pd.DataFrame(cosine_similarity(feature_similarity))
        total_similarity = total_similarity + feature_similarity / 4  # check the division!!!
    total_similarity.index = [str(i) for i in projects_data.index]
    total_similarity.columns = [str(i) for i in projects_data.index]
    return total_similarity


data_matrix = calculate_similarity_by_content()


def calc_special_precision(relevant_projects, known_user_likes_test):
    known_user_likes_test = [int(x) for x in known_user_likes_test]
    precision = np.intersect1d(relevant_projects, known_user_likes_test).size / len(relevant_projects)
    rejected_recs = list(set(relevant_projects) - set(known_user_likes_test))
    similarity_sums = np.sum([calc_max_sim(rp, known_user_likes_test) for rp in rejected_recs])
    return precision + similarity_sums / len(relevant_projects)


def calc_max_sim(rejected_project, chosen_projects):
    sim = []
    for cp in chosen_projects:
        if str(cp) in data_matrix.index and str(rejected_project) in data_matrix.index:
            sim.append(data_matrix.loc[str(cp)][str(rejected_project)])
        else:
            sim.append(0)
    return np.max(sim)


def precision_recall(user_index, known_user_likes_test, relevant_projects, algorithm, k_values):
    precision_recall_by_k = []
    for k in k_values:
        if relevant_projects:
            relevant_projects_by_k = relevant_projects[:k]
            if not is_online_project_recommended(relevant_projects_by_k):
                relevant_projects_by_k[-1] = algorithm.get_highest_online_project()
            # print ("recommendations: ", relevant_projects)
            if len(relevant_projects_by_k) < k:  # for debugging
                print("problem with user: ", user_index)
            # calculate recall and precision - this is the same value since the sets are the same size
            precision = np.intersect1d(relevant_projects_by_k, known_user_likes_test).size / len(relevant_projects_by_k)
            special_percision = calc_special_precision(relevant_projects_by_k, known_user_likes_test)
            recall = np.intersect1d(relevant_projects_by_k, known_user_likes_test).size / len(known_user_likes_test)
            # hit_rate = 1 if len(np.intersect1d(relevant_projects_by_k, known_user_likes_test)) > 0 else 0
            precision_recall_by_k.append([precision, recall, special_percision])
        else:
            precision_recall_by_k.append([-1, -1, -1])
    return precision_recall_by_k


def get_precision_and_recall_by_time_split(user, k, algorithm, ip_address):
    user_data = records[records.profile == user]
    projects_list = list(user_data['project'].values)
    projects_list = [str(int(x)) for x in projects_list if x is not None and x == x]
    projects_list = list(unique_everseen(projects_list))
    projects_list = [int(x) for x in projects_list]
    if len(projects_list) >= Recommender.HISTORY_THRES:
        splitter_index = max(1, int(0.9 * len(projects_list)))
        # split to train and test by timeline!!
        known_user_likes_train = projects_list[:splitter_index]
        known_user_likes_test = projects_list[splitter_index:]
        user_index_place = Recommender.data[Recommender.data['user'] == user].index
        user_index = user_index_place[0] if len(user_index_place) > 0 else -1

        relevant_projects = algorithm.get_recommendations(user_index, known_user_likes_train, k, ip_address)
        relevant_projects = Recommender.make_sure_k_recommendations(relevant_projects, user_index, k, ip_address)
        print(user, relevant_projects, known_user_likes_test)
        return relevant_projects, known_user_likes_test
    return None, None


def precision_recall_at_k(k_values, test_users, algorithm):
    results = []
    ip_addresses = ['64.233.160.0', '74.125.224.72', '132.72.235.23', '13.68.172.47']
    i = 0
    for user in test_users:
        print(i)
        relevant_projects, known_user_likes_test = get_precision_and_recall_by_time_split(user, 10, algorithm, None)
        precision_recall_by_k = precision_recall(user, known_user_likes_test, relevant_projects, algorithm, k_values)
        results.append(precision_recall_by_k)  # ip_addresses[i%4]))
        i += 1
    for k in range(len(k_values)):
        print("algorithm:" + str(algorithm) + " k " + str(k))
        print(str([i[k][0] for i in results if i[k][0] >= 0]) + "\n")
        print(str([i[k][1] for i in results if i[k][1] >= 0]) + "\n")
        precisions = np.mean([i[k][0] for i in results if i[k][0] >= 0])
        recalls = np.mean([i[k][1] for i in results if i[k][1] >= 0])
        special_precisions = np.mean([i[k][2] for i in results if i[k][2] >= 0])
        print("algorithm:", algorithm, "k", k)
        print(precisions, recalls, special_precisions)


if __name__ == '__main__':
    print(Recommender.data_items.shape)
    print(data_items_train.shape)
    # for ip change get_recommendation param
    # precision_recall_at_k([1, 5, 7, 10], Recommender.data['user'].values, CFUserUser(data_items_train))
    precision_recall_at_k([1, 3, 5, 7, 10], Recommender.data['user'].values, CFItemItem(data_items_train))
    precision_recall_at_k([1, 3, 5, 7, 10], Recommender.data['user'].values, PopularityBased(data_items_train))
    precision_recall_at_k([1, 3, 5, 7, 10], Recommender.data['user'].values, SVD(data_items_train))
