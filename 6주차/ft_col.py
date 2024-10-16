# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Data-driven Manufacturing
# Professor : Ju Yeon Lee
# Contents : Data Pre-processing
# =============================================================================


# quantile() : a function to find percentile, parameter (0~1)
# Q1 = variable.quantile(1/4)
# Q3 = variable.quantile(3/4)
# IQR = Q3 - Q1
# Q2 = variable.quantile(2/4), variable.quantile(1/2)
# Q4 = variable.quantile(1)

def outlier_test(x):
    Q1=x.quantile(1/4) # Q1 : 1/4
    Q3=x.quantile(3/4) # Q3 : 3/4
    IQR = Q3-Q1
    LL= Q1-(1.5*IQR) # Lower limit
    UU= Q3+(1.5*IQR) # Upper limit
    outlier = (x < LL)|(x > UU)
    return outlier
