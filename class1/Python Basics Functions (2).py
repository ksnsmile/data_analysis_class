# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# Course : Introduction to Data Analytics
# Professor : Ju Yeon Lee
# Contents : Python Basics, Functions (2)
# =============================================================================


### inputs & outputs

a=input()


number = input("숫자를 입력하세요: ")
print(type(number))


number1 = int(input("첫번째 숫자를 입력하세요: "))
number2 = int(input("두번째 숫자를 입력하세요: "))
print(number1+number2)


number1, number2 = map(int, input("숫자 2개를 입력하세요: ").split())
print(number1+number2)


print("life" "is" "too short") 
print("life"+"is"+"too short")


print("life", "is", "too short")


for i in range(10):
    print(i, end='')
    

### file read & write
       
f = open("새파일.txt", 'w')
f.close()


f=open("C:/NewFile-Test/새파일.txt", 'w')
f.close()


f = open("새파일.txt", 'w')
for i in range(1,11):
    data= "%d번째 줄입니다.\n" %i
    f.write(data)
f.close()


for i in range(1,11):
    data= "%d번째 줄입니다.\n" %i
    print(data)
    
    
f=open("새파일.txt", 'r')
line=f.readline()
print(line)
f.close()


f=open("새파일.txt", 'r')
while True:
    line=f.readline()
    if not line: break
    print(line)
f.close()


f=open("새파일.txt", 'r')
lines=f.readlines()
for line in lines:
    print(line)
f.close()


f=open("새파일.txt", 'r')
lines=f.readlines()
for line in lines:
    print(line.strip("\n"))
f.close()


f=open("새파일.txt", 'r')
data=f.read()
print(data)
f.close()


f=open("새파일.txt", 'a')
for i in range(11, 21):
    data = "%d번째 줄입니다.\n" %i
    f.write(data)
f.close()


with open("foo.txt", 'w') as f:
    f.write("Life is too short, you need python")

