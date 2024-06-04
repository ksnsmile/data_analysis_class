# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Python Basics, Functions (1)
# =============================================================================


### functions

def add(a, b):
    result = a+b
    return result

a=3
b=4
c=add(a, b)
print(c)

d=add(3, 4)
print(d)

print(add(3,4))


def say():
    return 'Hi'

print(say())


def sum(a, b):
    print("%d, %d의 합은 %d입니다. " % (a, b, a+b))

sum(3,4)


exList = [1, 2, 3]
exList.append(4)
exList.pop()


def say():
    print('Hi')

say()


def sum_many(*args):
    sum=0
    for i in args:
        sum = sum + i
    return sum

result = sum_many(1,2,3)
print(result)


def sum_many(choice, *args):
    if choice == "add":
        sum=0
        for i in args:
            sum = sum+i
    elif choice == "mul":
        sum=1
        for i in args:
            sum = sum*i
    return sum

result = sum_many('mul', 1,2,3,4,5) 
print(result)

result = sum_many('add', 1,2,3,4,5)
print(result)


def print_kwargs(**kwargs):
    print(kwargs)

print_kwargs(a=1)


def print_kwargs(**kwargs):
    print(kwargs)
    for k in kwargs.keys():
        if (k== 'name'):
            print("당신의 이름은 :" + kwargs.get(k))

print_kwargs(name="Juyeon Lee", age=10, phone="111-111-1111")


def sum_and_mul (a, b):
    return a+b, a*b

result = sum_and_mul(3,4)
print(result)

result1, result2 = sum_and_mul(3,4)
print(result1, result2)


def say_myself(name, old, man=True):
    print("나의 이름은 %s 입니다." % name)
    print("나이는 %d살입니다. " % old)
    if man:
        print("남자입니다.")
    else:
        print("여자입니다.")

say_myself("박응용", 27)
say_myself("박응용", 27, man=True)
say_myself(27, "박응용", man=True)
 

def say_myself(name, man=True, old):
    print("나의 이름은 %s 입니다." % name)
    print("나이는 %d살입니다. " % old)
    if man:
        print("남자입니다. ")
    else:
        print("여자입니다. ")


### gloval variable & local variable      

a=1
def vartest(a):
    a=a+1

vartest(a)
print(a)


a=1
def vartest(a):
    a=a+1
    return a

a = vartest(a)
print(a)    


a=1
def vartest():
    global a
    a=a+1
    
vartest()
print(a)


