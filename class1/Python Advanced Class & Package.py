# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Python Advanced, Class & Package
# =============================================================================


### class

result=0

def adder(num):
    global result
    result += num
    return result

print(adder(3))
print(adder(4))

result1=0
result2=0

def adder1(num):
    global result1
    result1 += num
    return result1

def adder2(num):
    global result2
    result2 += num
    return result2

print(adder1(3))
print(adder1(4))
print(adder2(5))
print(adder2(10))

class Calculator:
    def __init__(self):
        self.result=0
        
    def adder(self, num):
        self.result += num
        return self.result
    
cal1=Calculator()
cal2=Calculator()

print(cal1.adder(3))
print(cal1.adder(4))
print(cal2.adder(3))
print(cal2.adder(7))

class Calculator:
    def __init__(self):
        self.result=0
        
    def adder(self, num):
        self.result += num
        return self.result
    def sub(self, num):
        self.result -= num
        return self.result

cal3=Calculator()

print(cal3.sub(10))
print(cal3.sub(20))

class FourCal:
    pass
    
a=FourCal()
a.setdata(4,2)
a.add()
a.sub()
a.mul()
a.div()

class FourCal:
    pass

a= FourCal()
print(type(a))

class FourCal:
    def setdata(self, first, second):
        self.first = first
        self.second = second

a= FourCal()
a.setdata(4,2)
print(a.first)
print(a.second)

a= FourCal()
b= FourCal()

a.setdata(4,2)
b.setdata(3,7)

print(a.first)
print(b.first)

print(id(a.first))
print(id(b.first))

class FourCal:
    def setdata(self, first, second):
        self.first = first
        self.second = second
        
    def add(self):
        result = self.first + self.second
        return result

a= FourCal()
a.setdata(4,2)
print(a.add())

class FourCal:
    def setdata(self, first, second):
        self.first = first
        self.second = second
        
    def add(self):
        result = self.first + self.second
        return result
    
    def sub(self):
        result = self.first - self.second
        return result

    def mul(self):
        result = self.first * self.second
        return result

    def div(self):
        result = self.first / self.second
        return result

a= FourCal()
a.setdata(4,2)
print(a.add())
print(a.sub())
print(a.mul())
print(a.div())

b= FourCal()
b.setdata(15,3)
print(b.add())
print(b.sub())
print(b.mul())
print(b.div())


c=FourCal()
c.add()

class FourCal:
    def __init__(self, first, second):
        self.first = first
        self.second = second
        
    def add(self):
        result = self.first + self.second
        return result
    
    def sub(self):
        result = self.first - self.second
        return result

    def mul(self):
        result = self.first * self.second
        return result

    def div(self):
        result = self.first / self.second
        return result
    
a=FourCal()

a=FourCal(4,2)
print(a.add())

class MoreFourCal(FourCal):
    pass

a= MoreFourCal(4,2)
print(a.add())

class MoreFourCal(FourCal):
   
    def pow(self):
        result = self.first ** self.second
        return result  

a= MoreFourCal(4,2)
print(a.pow())

class SafeFourCal(FourCal):
    
    def div(self):
        if self.second == 0:
            return 0
        else:
            return self.first / self.second

a=SafeFourCal(4,0)
print(a.div())

b=FourCal(4,0)
print(b.div())

class FourCal:
    
    first = 10
    second = 5
# =============================================================================
#     def __init__(self, first, second):
#         self.first = first
#         self.second = second
# =============================================================================
        
    def add(self):
        result = self.first + self.second
        return result
    
    def sub(self):
        result = self.first - self.second
        return result

    def mul(self):
        result = self.first * self.second
        return result

    def div(self):
        result = self.first / self.second
        return result

a=FourCal()
b=FourCal()
print(a.first)
print(b.first)


### module & package

import mod1 

print(mod1.add(3,4))
print(mod1.sub(4,2))


from mod2 import *

print(add(3,4))
print(sub(4,2))


from mod3 import *

print(PI)

a=math()
print(a.solv(2))

print(add(PI, 4.4))


import sys

sys.path
print(sys.path)

sys.path.append("C:/myModules")
print(sys.path)
