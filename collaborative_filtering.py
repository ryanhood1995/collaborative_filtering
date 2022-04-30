# ======================================================================================================================================
# Author: Ryan Hood
#
# Description: This file is part of homework 3.  The goal is to take data which was part of the Neflix recommender system challenge and
# implement a collaborative filtering algorithm based around the idea of correlation coefficient.
# ======================================================================================================================================

import sys
import math
import numpy as np

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

def remove_nan(prediction_list, correct_list):
    """ Sometimes my implementation gives a NaN response for the predicted rating.  These very rare cases need to be removed from both lists
    (although the NaN will only show up in the prediction list). """
    # Only the prediction list might have nan.
    pred_array = np.array(prediction_list)
    correct_array = np.array(correct_list)
    index_of_nan = []
    for array_index in range(0, len(pred_array)):
        if math.isnan(pred_array[array_index]):
            index_of_nan.append(array_index)
    for nan_index in index_of_nan:
        prediction_list.pop(nan_index)
        correct_list.pop(nan_index)
    return prediction_list, correct_list

def get_num_lines(file):
    for i, l in enumerate(file):
        pass
    return i+1


def convert_txt_file_to_representation(file_path):
    """ This method takes either the training text file or the testing text file and converts it to a numpy array.
    Each sub-list represents a different example, and the elements of that sub-list correspond
    to the values of that example. """
    file = open(file_path, "r")
    # Count lines.
    count = get_num_lines(file)

    # Close file.
    file.close()

    # Create array of zeros.
    rep = np.zeros((count, 3))

    # Reopen file.
    file = open(file_path, "r")

    for index in range(0, count):
        string = file.readline()
        example = string.split(',')
        example[0] = int(example[0])
        example[1] = int(example[1])
        example[2] = float(example[2].rstrip())
        rep[index][0] = example[0]
        rep[index][1] = example[1]
        rep[index][2] = example[2]
    return rep



def create_user_dict(rep):
    """ This method takes a representation and figures a dict with the keys being the user ids and the values
    being an index which starts at 0 and counts up by 1.  This will be helpful to be able to convert a user_id
    into the approapriate index for our sparse matrix. """
    user_id_array = rep[:,1]
    user_id_array = np.unique(user_id_array)
    user_id_list = list(user_id_array)
    dict = {k: v for v,k in enumerate(user_id_list)}
    return dict

def create_movie_dict(rep):
    """ This method does the same as the above, but for movie ids. """
    movie_id_array = rep[:,0]
    movie_id_array = np.unique(movie_id_array)
    movie_id_list = list(movie_id_array)
    dict = {k: v for v,k in enumerate(movie_id_list)}
    return dict


def create_sparse_matrix(movie_dict, user_dict, rep):
    """ This method creates a sparse matrix of ratings.  The row index will be the user ids, the column index will be the movie ids,
    and the values will be the rating given by the ith person for the jth movie. """
    # We first initialize a 2D array (very large) of the correct size.
    sm = np.zeros((len(user_dict), len(movie_dict)))
    # We go through rep line by line.
    for sub_array in rep:
        movie_id = sub_array[0]
        user_id = sub_array[1]
        rating = sub_array[2]
        # Now we look up the correct indices.
        movie_index = movie_dict[movie_id]
        user_index = user_dict[user_id]

        # Now we can insert the rating in the correct spot.
        sm[user_index][movie_index] = rating

    return sm


def get_average_rating_all_users(sm, user_dict):
    """This method returns a dict where the keys are the user ids
    and the values are the average ratings for that user."""
    new_dict = {}
    for entry in user_dict:
        new_key = entry
        index = user_dict[entry]
        # Now we have to get the average rating for the entry.
        # Get the row of sm.
        array = sm[index,:]
        # We need the sum of these elements and the number of non-zero elements.
        non_zero = np.count_nonzero(array)
        sum = np.sum(array)
        av = sum / non_zero

        # Now we store this sum in the new_dict with key new_key
        new_dict[new_key] = av
    return new_dict



def calculate_correlation_matrix(sm, av_dict, user_dict, movie_dict):
    """ I thought I might implement this, but then changed my mind.  The result would be an incredibly large
    and sparse matrix where the [i,j] entry would be the correlation between the ith and jth users. But since
    we have a lot of users, the size of this would be roughly 700 M, so I decided against it.  But it might be something
    I try later. """
    return



def calculate_correlation(active_user_id, other_user_id, sm, user_dict, movie_dict, av_dict):
    """ This method is rather simple despite its size.  It calculates the correlation between two users."""
    # First let's get the indices in our sparse matrix.
    active_index = user_dict[active_user_id]
    other_index = user_dict[other_user_id]

    # Now let's get the average movie scores for both.
    active_av = av_dict[active_user_id]
    other_av = av_dict[other_user_id]

    # Now let's get the corresponding rows of the sm.
    active_array = sm[active_index, :]
    other_array = sm[other_index, :]

    # Initializations.
    num_sum = 0
    denom_sum_1 = 0
    denom_sum_2 = 0

    # For every movie.... we loop.
    for movie_index in range(0, len(active_array)):
        vaj = active_array[movie_index]
        vij = other_array[movie_index]
        if (vaj != 0 and vij != 0):
            # Then both users have rated the movie.
            num_term = (vaj - active_av)*(vij - other_av)
            denom_term_1 = (vaj - active_av)**2
            denom_term_2 = (vij - other_av)**2

            # Now update.
            num_sum += num_term
            denom_sum_1 += denom_term_1
            denom_sum_2 += denom_term_2

    # Now return value.
    if ((denom_sum_1 == 0) or (denom_sum_2 == 0)):
        return 0.0
    return (num_sum/math.sqrt(denom_sum_1*denom_sum_2))


def calculate_sum_term(active_user_id, movie_id, sm, user_dict, movie_dict, av_dict):
    """ This method calculates the sum term which is a very large part of any given prediction.
    The sum term along with the sum of all of the weights is returned.  The sum of weights will be
    used as a normalizing factor. """
    # We want to loop through all other users.
    # So we loop through the user dict
    sum_of_weights = 0
    sum_term = 0
    for other_user_id in user_dict:
        if (other_user_id != active_user_id):
            # We calculate the correlation between active user and user_id.
            other_user_index = user_dict[other_user_id]
            movie_index = movie_dict[movie_id]
            vij = sm[other_user_index, movie_index]
            if vij == 0.0:
                continue

            weight = calculate_correlation(active_user_id, other_user_id, sm, user_dict, movie_dict, av_dict)
            sum_of_weights += weight

            # Now we get vij
            other_user_av = av_dict[other_user_id]

            # Now add to sum term.
            #print("vij: ", vij)
            #print("diff: ", (vij - other_user_av))
            #print("weight: ", weight)
            #print("other_user_av: ", other_user_av)
            sum_term += (weight)*(vij - other_user_av)
            #print("NEW SUM TERM: ", sum_term)

    return sum_of_weights, sum_term

def get_prediction(active_user_id, movie_id, sm, user_dict, movie_dict, av_dict):
    """ This method get a single prediction for a given user and a given movie. """
    # First we get the average score by active user.
    av_score = av_dict[active_user_id]

    # Now we get the prediction by calling a method and doing some simple arithmetic.
    sum_of_weights, sum_term = calculate_sum_term(active_user_id, movie_id, sm, user_dict, movie_dict, av_dict)
    if sum_of_weights == 0:
        prediction = 2.5
    else:
        prediction = av_score + (sum_term / sum_of_weights)
    return prediction

def driver(train_path, test_path):
    """ This method drives the whole program by getting rating predictions for every case and then returning
    the prediction list and correct list. """
    # First we get everything required.
    train_rep = convert_txt_file_to_representation(train_path)
    test_rep = convert_txt_file_to_representation(test_path)

    movie_dict = create_movie_dict(train_rep)
    user_dict = create_user_dict(train_rep)

    sm = create_sparse_matrix(movie_dict, user_dict, train_rep)

    av_dict = get_average_rating_all_users(sm, user_dict)

    # Do stuff.
    prediction_list = []
    correct_list = []
    count = 0
    max_iterations = 1000
    for sub_array in test_rep:
        movie_id = sub_array[0]
        user_id = sub_array[1]
        rating = sub_array[2]
        count += 1

        # Below I set a hard limit on the number of iterations before ending.
        if count == max_iterations:
            break

        print("We are: ", count, " / ", max_iterations, " of the way done.")
        # We get the estimated rating.
        prediction = get_prediction(user_id, movie_id, sm, user_dict, movie_dict, av_dict)
        # We insert the predicted rating into prediction list.
        prediction_list.append(prediction)
        print("New Prediction: ", prediction)
        print("Actual Rating: ", rating)

        # We also insert the actual rating into correct_list.
        correct_list.append(rating)
    return prediction_list, correct_list


if __name__ == '__main__':
    # Set default arguments.
    train_path = "C:\\Users\\ryanc\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw3\\TrainingRatings.txt"
    test_path = "C:\\Users\\ryanc\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw3\\TestingRatings.txt"


    # Set new arguments if the correct number of them were given.
    if len(sys.argv) == 3:
        print("You provided the correct number of parameters.  Congrats!")
        train_path = sys.argv[1]
        test_path = sys.argv[2]
    else:
        print("You did not provide the correct number of parameters.  Using default selections.")


    # Now we perform the process.
    prediction_list, correct_list = driver(train_path, test_path)
    prediction_list, correct_list = remove_nan(prediction_list, correct_list)
    mae = mean_absolute_error(prediction_list, correct_list)
    rmse = root_mean_squared_error(prediction_list, correct_list)
    print("MAE: ", mae)
    print("RMSE: ", rmse)
