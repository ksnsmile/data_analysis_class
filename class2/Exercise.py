# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Exercise
# =============================================================================

with open('Exercise.txt', 'r') as f:
    count=0
    words = f.readlines()
    for word in words:
        if len(word.strip('\n')) <= 5:
            count += 1
    print(count)
