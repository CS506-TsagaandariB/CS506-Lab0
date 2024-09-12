## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 0, 0])
    vector2 = np.array([0, 1, 0])
    
    result = cosine_similarity(vector1, vector2)
    
    expected_result = 0.0 # Cosine of 90 degrees (orthogonal vectors)
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    target_vector = np.array([1, 2, 3])
    vectors = [
        np.array([4, 5, 6]),
        np.array([1, 0, 0]),
        np.array([7, 8, 9])
    ]
    
    result = nearest_neighbor(target_vector, vectors)
    
    expected_index = 2 # The vector [7, 8, 9] is closest to [1, 2, 3] based on cosine similarity
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
