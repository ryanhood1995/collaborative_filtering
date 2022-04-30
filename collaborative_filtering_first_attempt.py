import math

def convert_txt_file_to_representation(file_path):
    """ This method takes either the training text file or the testing text file and converts it to a
    list of list.  Each sub-list represents a different example, and the elements of that sub-list correspond
    to the values of that example. """
    file = open(file_path, "r")
    rep = []
    for line in file:
        string = file.readline()
        example = string.split(',')
        example[0] = int(example[0])
        example[1] = int(example[1])
        example[2] = float(example[2].rstrip())
        rep.append(example)
    return rep


def mean_vote(train_rep, user_id):
    """ This method finds the average score given by the user corresponding to user_id in the training
    data. """
    num_ratings = 0
    sum_of_scores = 0
    for index in range(0, len(train_rep)):
        if train_rep[index][1] == user_id:
            num_ratings = num_ratings + 1
            sum_of_scores = sum_of_scores + train_rep[index][2]
    return (sum_of_scores / num_ratings)



def get_ratings_lists(train_rep, active_user_id, other_user_id):
    """ This method takes the training representation along with 2 user ids.  It finds the shared ratings for the movies.
    And then it returns 2 list representing the ratings among those shared movies for the two users."""
    # First we get the indices in the train_rep that correspond to the active user.
    active_user_movie_ids = []
    for index in range(0, len(train_rep)):
        if train_rep[index][1] == active_user_id:
            active_user_movie_ids.append(train_rep[index][0])
    # And we do the same for the other user.
    other_user_movie_ids = []
    for index in range(0, len(train_rep)):
        if train_rep[index][1] == other_user_id:
            other_user_movie_ids.append(train_rep[index][0])
    # Now we compare the two previous list and keep the ids that are shared.
    common_movie_ids = []
    for movie_id in active_user_movie_ids:
        if movie_id in other_user_movie_ids:
            common_movie_ids.append(movie_id)

    # Now we get the two ratings lists. First for active user.
    active_user_ratings = []
    other_user_ratings = []
    for index in range(0, len(train_rep)):
        if train_rep[index][0] in common_movie_ids and train_rep[index][1] == active_user_id:
            active_user_ratings.append(train_rep[index][2])
        if train_rep[index][0] in common_movie_ids and train_rep[index][1] == other_user_id:
            other_user_ratings.append(train_rep[index][2])
    return active_user_ratings, other_user_ratings




def correlation(train_rep, active_user_id, other_user_id, active_mean_vote, other_mean_vote):
    """ This method calculates the correlation between the "active" user (who we wish to make a prediction),
    and another user."""
    # First we get two list.  The first is a list of the ratings by the active user (on the j items connecting the active
    # user to the other user).  The second is a list of the ratings by the other user (on the same j items).
    active_list, other_list = get_ratings_lists(train_rep, active_user_id, other_user_id)

    # Once we have those lists, calculating the correlation is easy.
    numerator_sum = 0
    denominator_sum_1 = 0
    denominator_sum_2 = 0
    for index in range(0, len(active_list)):
        numerator_term = (active_list[index] - active_mean_vote)*(other_list[index] - other_mean_vote)
        denominator_term_1 = (active_list[index] - active_mean_vote)**2
        denominator_term_2 = (other_list[index] - other_mean_vote)**2
        numerator_sum = numerator_sum + numerator_term
        denominator_sum_1 = denominator_sum_1 + denominator_term_1
        denominator_sum_2 = denominator_sum_2 + denominator_term_2
    complete_denominator = math.sqrt(denominator_sum_1 * denominator_sum_2)
    return (numerator_sum / complete_denominator)



def kappa(weight_vector):
    sum = 0
    for index in range(0, len(weight_vector)):
        sum = sum + weight_vector[index]
    return (1/sum)

def vij(train_rep, other_user_id, movie_id):
    """ This method finds the rating by user i on movie j.  If user i did not rate movie j, then -5 is returned. """
    for index in range(0, len(train_rep)):
        if train_rep[index][0] == movie_id and train_rep[index][1] == other_user_id:
            return train_rep[index][2]
    return -5



def driver(train_file_path, test_file_path):
    """ This is the driving method of the program.  For each row in the testing directory, a predicted score will be
    returned.  This prediction list and the correct list are returned."""
    # First we get the representations of the train and test set.
    train_rep = convert_txt_file_to_representation(train_file_path)
    test_rep = convert_txt_file_to_representation(test_file_path)

    # Now we go through our test_rep and get a prediction for each row.
    prediction_list = []
    correct_list = []
    for row in test_rep:
        correct_list.append(row[2])
        movie_id = row[0]
        active_user_id = row[1]
        prediction = get_prediction(train_rep, movie_id, active_user_id)
        print("NEW PREDICTION!")
        print(prediction)
        prediction_list.append(prediction)

    return prediction_list, correct_list

def get_prediction(train_rep, movie_id, active_user_id):
    """ This method gets the prediction for the active user for the movie_id.  The training representation is needed as well
    so we also pass that."""
    active_mean_vote = mean_vote(train_rep, active_user_id)
    weight_dict = get_weight_dict(train_rep, active_user_id, active_mean_vote)
    kappa = kappa(weight_dict)
    sum = 0
    for index in range(0, len(weight_dict)):
        other_user_id = list(weight_dict)[index]
        weight_value = list(weight_dict.values())[index]
        other_mean_vote = mean_vote(train_rep, other_user_id)
        vij = vij(train_rep, other_user_id, movie_id)
        sum = sum + weight_value*(vij - other_mean_vote)
    sum = sum * kappa
    sum = sum + active_mean_vote
    return sum

def get_weight_dict(train_rep, active_user_id, active_mean_vote):
    """ This method finds a dictionary where the keys are the other_user_ids and the values are the weights between
    the current active user and the other user."""
    # Let's first initialize our final weight dict.
    weight_dict = {}

    # Let's first get a list of the unique other_user_ids.
    other_ids_list = []
    for index in range(0, len(train_rep)):
        if train_rep[index][1] != active_user_id:
            other_ids_list.append(train_rep[index][1])
    # Now remove duplicates from our list.
    other_ids_list = list(set(other_ids_list))

    # Now for each other_id, we can get the correlation and then add it to thr weight dict.
    print("LENGTH OF OTHER IDS LIST: ")
    print(len(other_ids_list))
    for other_user_id in other_ids_list:
        other_mean_vote = mean_vote(train_rep, other_user_id)
        corr = correlation(train_rep, active_user_id, other_user_id, active_mean_vote, other_mean_vote)
        print("CORRELATION FOUND: ")
        print(corr)
        weight_dict.update({other_user_id: corr})

    # Now our weight dict is finished, so we return it.
    return weight_dict


def mean_absolute_error(prediction_list, correct_list):
    """ This method finds the mean absolute error of all of the predictions in prediction_list assuming the
    corresponding correct values are stored in correct_list. """
    sum = 0
    for index in range(0, len(prediction_list)):
        abs_error = abs(prediction_list[index] - correct_list[index])
        sum = sum + abs_error
    return (sum/len(prediction_list))

def root_mean_squared_error(prediction_list, correct_list):
    """ This method finds the root mean squared error of all of the predictions in prediction_list assuming the
    corresponding correct values are stored in correct_list. """
    sum = 0
    for index in range(0, len(prediction_list)):
        squared_error = (prediction_list[index] - correct_list[index])**2
        sum = sum + squared_error
    mean_squared_error = (sum/len(prediction_list))
    return math.sqrt(mean_squared_error)

train_path = "C:\\Users\\ryanc\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw3\\TrainingRatings.txt"
test_path = "C:\\Users\\ryanc\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw3\\TestingRatings.txt"

prediction_list, correct_list = driver(train_path, test_path)
print(prediction_list)
