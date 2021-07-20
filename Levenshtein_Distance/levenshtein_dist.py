# ---------------------------------------------------------------------------
# @Title: Levenstein Distance (Minimum Edit Distance)
# @Author: Nico Solitana
# @Course: Natural Language Processing
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd

# Create base matrix


def init_matrix(firstWord, secondWord):
    matrix = []
    for i in range(len(firstWord)):
        a = []
        x = 0
        for j in range(len(secondWord)):
            if(i == 0):
                x = j
            elif(j == 0):
                x = i
            else:
                x = 0
            a.append(x)
        matrix.append(a)
    return matrix

# Prints the matrix


def print_matrix(matrix, firstWord, secondWord):
    for i in range(len(firstWord)):
        for j in range(len(secondWord)):
            print(matrix[i][j], end=" ")
        print()


# Computes the minimum edit distance
def min_edit_distance(matrix, firstWord, secondWord):
    for i in range(1, len(firstWord)):
        for j in range(1, len(secondWord)):
            a = []
            a.append(matrix[i-1][j]+1)
            a.append(matrix[i][j-1]+1)
            if(firstWord[i] == secondWord[j]):
                a.append(matrix[i-1][j-1])
            else:
                a.append(matrix[i-1][j-1]+2)
            matrix[i][j] = min(a)
    return matrix


# Performs the levenstein distance computation
def levenstein_distance(firstWord, secondWord):
    matrix = init_matrix(firstWord, secondWord)
    min_edit_distance(matrix, firstWord, secondWord)
    return matrix


# Create a dataframe from the matrix
def create_ld_df(matrix, firstWord, secondWord):
    numpy_data = np.array(matrix)
    df = pd.DataFrame(data=numpy_data, index=list(
        firstWord), columns=list(secondWord))
    return df


# Reverses the array and joins to form a string of alignment
def CombineStr(fWord, sWord):
    fWord.reverse()
    sWord.reverse()
    fWordStr = "".join(fWord)
    sWordStr = "".join(sWord)
    print(fWordStr)
    print(sWordStr)


# Checks Diagonal Values
def DiagCheck(i, j, fWord, sWord, matrix, firstWord, secondWord):
    nextMin = min([matrix[i][j-1], matrix[i-1][j-1], matrix[i-1][j]])
    if (firstWord[i] == secondWord[j]) or ((firstWord[i] != secondWord[j]) and (nextMin+2 == matrix[i][j])):
        fWord.append(firstWord[i])
        sWord.append(secondWord[j])
        flag = True
    else:
        if (nextMin == matrix[i-1][j]):
            fWord.append(firstWord[i])
            sWord.append("_")
        else:
            fWord.append("_")
            sWord.append(secondWord[j])
        flag = False
    return flag, fWord, sWord


# BackTracking
def BackTracking(matrix, firstWord, secondWord):
    fWord = []
    sWord = []
    i = len(firstWord)-1
    j = len(secondWord)-1
    flag = False
    prev = 0
    while(True):
        if (i == len(firstWord)-1 and j == len(secondWord)-1) or (matrix[i][j] == min([matrix[i][j+1], matrix[i][j], matrix[i+1][j]])) and (flag == False):
            flag, fWord, sWord = DiagCheck(
                i, j, fWord, sWord, matrix, firstWord, secondWord)
        elif (matrix[i+1][j] == min([matrix[i][j+1], matrix[i][j], matrix[i+1][j]])) and (flag == False):
            if (firstWord[i+1] == secondWord[j]):
                fWord.append(firstWord[i+1])
                sWord.append(secondWord[j])
                flag = True
            else:
                fWord.append("_")
                sWord.append(secondWord[j])
                flag = False
            i += 1
        elif (matrix[i][j+1] == min([matrix[i][j+1], matrix[i][j], matrix[i+1][j]])) and (flag == False):
            if (firstWord[i] == secondWord[j+1]):
                fWord.append(firstWord[i])
                sWord.append(secondWord[j+1])
                flag = True
            else:
                fWord.append(firstWord[i])
                sWord.append("_")
                flag = False
            j += 1
        else:
            flag, fWord, sWord = DiagCheck(
                i, j, fWord, sWord, matrix, firstWord, secondWord)
            flag = False
        prev = matrix[i][j]
        i -= 1
        j -= 1
        if(i < 0):
            i = 0
        if(j < 0):
            j = 0
        if(i == 0) and (j == 0):
            break
    CombineStr(fWord, sWord)


# Prints the computed minimum edit distance
def GetMinimumDistance(matrix, firstWord, secondWord):
    print('Minimum edit distance: ' +
          str(matrix[len(firstWord)-1][len(secondWord)-1]))
