def sor_low_to_high(ls: list):
    output = [0]*len(ls)

    for i in ls:
        count = 0
        du = 0
        for j in ls:
            if i>j:
                count+=1
            if i == j:
                du +=1
            else:
                count = count

        if du == 0:
            output[count]=i
        if du >0:
            for k in range(0,du,1):
                output[count+k] = i

    print(output)
    return output
def sor_high_to_low(ls: list):
    output = [0]*len(ls)

    for i in ls:
        count = -len(ls)
        du = 0
        for j in ls:
            if i<j:
                count+=1
            if i == j:
                du +=1
            else:
                count = count

        if du == 0:
            output[count]=i
        if du >0:
            for k in range(0,du,1):
                output[count+k] = i

    print(output)
    return output
def fibonacci(num):
    if num == 1:
        return 0
    elif num == 2:
        return 1
    return fibonacci(num-2) + fibonacci(num - 1)
import matplotlib.pyplot as p

x = []
y = []
print(fibonacci(10))
for a in range(0,3,1):
    xi = a
    yi = fibonacci(a+1)
    print(f"{fibonacci(a+1)} AND {a+1}")
    x.append(xi)
    y.append(yi)
#p.plot(x,y)
#p.show()

def main():
    class Person:

        def __init__(self, name, age):
            self.name = name

            self.age = age


    class Student(Person):
        def __init__(self, name, age, id):
            super().__init__(name, age)
            self.id = id

    obj1 = Student("John", 23, 1234)  # object instance from the Student class

    dict_att = obj1.__dict__
    print(dict_att)
    return dict_att


if tracker == "You Win!":
    win += 1
elif tracker == "you lose!":
    lose += 1
elif tracker == "Draw!":
    draw += 1
