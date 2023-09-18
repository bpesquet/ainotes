# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Python

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %%
import platform

print(f"Python version: {platform.python_version()}")

# Relax some linting rules for code examples
# pylint: disable=invalid-name,redefined-outer-name,consider-using-f-string,duplicate-value,unnecessary-lambda-assignment,protected-access,too-few-public-methods,wrong-import-position,unused-import

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Python for AI and Data Science


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Python in a nutshell
#
# [Python](https://www.python.org) is a multi-purpose programming language created in 1989 by [Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum) and developed under a open source license.
#
# It has the following characteristics:
#
# - multi-paradigms (procedural, fonctional, object-oriented);
# - dynamic types;
# - automatic memory management;
# - and much more!

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The Python syntax
#
# For more examples, see the [cheatsheet]() below.


# %%
def hello(name):
    """Say hello to someone"""

    print(f"Hello, {name}")


friends = ["Lou", "David", "Iggy"]

for friend in friends:
    hello(friend)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### What is Data Science?
#
# - Expression born in 1997 in the statistician community.
# - "A Data Scientist is a statistician that lives in San Francisco."
# - Main objective: extract insight from data.
# - 2012 : "Sexiest job of the 21st century" (Harvard Business Review).
# - There is a [controversy](https://en.wikipedia.org/wiki/Data_science#Relationship_to_statistics) on the expression's real usefulness.

# %% [markdown] slideshow={"slide_type": "slide"}
# [![Data Science Venn diagram by Conway](images/DataScience_VD_conway.png)](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram)

# %% [markdown] slideshow={"slide_type": "slide"}
# [![Data Science Venn diagram by Kolassa](images/DataScience_VD_Kolassa.png)](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### A prominent language
#
# For the following reasons, Python has become the language of choice in these fields:
#
# - language qualities (ease of use, simplicity, versatility);
# - involvement of the scientific and academical communities;
# - rich ecosystem of dedicated open source libraries.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Gems from the ecosystem
#
# Essential tools from the vast Python ecosytem for AI and Data Science include:
#
# - [Anaconda](https://www.anaconda.com/distribution/), a scientific distribution including Python and many (1500+) specialized packages. It is the easiest way to setup a work environment for AI and Data Science with Python.
# - The [Jupyter Notebook](https://jupyter.org/), an open-source web application for creating and managing documents (_.ipynb_ files) that may contain live code, equations, visualizations and text. This format has become the *de facto* standard for sharing research results in numerical fields.
# - [Google Colaboratory](https://colab.research.google.com), a cloud environment for executing Jupyter notebooks with access to specialized processors (GPU or TPU).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## (Yet another) Python cheatsheet
#
# Inspired by [A Whirlwind Tour of Python](https://jakevdp.github.io/WhirlwindTourOfPython/) and [another Python Cheatsheet](https://www.pythoncheatsheet.org/).


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Basics

# %% slideshow={"slide_type": "-"}
# Print statement
print("Hello World!")

# Optional separator
print(1, 2, 3)
print(1, 2, 3, sep="--")

# Variables (dynamically typed)
mood = "happy"  # or 'happy'

print("I'm", mood)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### String formatting

# %% slideshow={"slide_type": "-"}
name = "Garance"
age = 16

# Original language syntax
message = "My name is %s and I'm %s years old." % (
    name,
    age,
)
print(message)

# Python 2.6+
message = "My name is {} and I'm {} years old.".format(name, age)
print(message)

# f-string (Python 3.6+)
# https://realpython.com/python-f-strings/
# https://cito.github.io/blog/f-strings/
message = f"My name is {name} and I'm {age} years old."
print(message)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Numbers and arithmetic

# %%
# Type: int
a = 0

# Type: float
b = 3.14

# Variable swapping
a, b = b, a
print(a, b)

# Float and integer divisions
print(13 / 2)
print(13 // 2)

# Exponential operator
print(3**2)
print(2**3)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Flow control

# %% [markdown]
# #### The if/elif/else statement

# %%
name = "Bob"
age = 30
if name == "Alice":
    print("Hi, Alice.")
elif age < 12:
    print("You are not Alice, kiddo.")
else:
    print("You are neither Alice nor a little kid.")

# %% [markdown] slideshow={"slide_type": "slide"}
# #### The while loop

# %%
num = 1

while num <= 10:
    print(num)
    num += 1

# %% [markdown] slideshow={"slide_type": "slide"}
# #### The for/else loop
#
# The optional `else`statement is only useful when a `break` condition can occur in the loop.

# %%
for i in [1, 2, 3, 4, 5]:
    if i == 3:
        print(i)
        break
else:
    print("No item of the list is equal to 3")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Data structures

# %% [markdown]
# #### Lists

# %% slideshow={"slide_type": "-"}
countries = ["France", "Belgium", "India"]

print(len(countries))
print(countries[0])
print(countries[-1])

# Add element at end of list
countries.append("Ecuador")

print(countries)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### List indexing and slicing

# %%
print(countries[1:3])
print(countries[0:-1])
print(countries[:2])
print(countries[1:])
print(countries[:])
print(countries[::-1])

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Tuples
#
# Contrary to lists, tuples are *immutable* (read-only).

# %% slideshow={"slide_type": "-"}
eggs = ("hello", 42, 0.5)

print(eggs[0])
print(eggs[1:3])

# TypeError: a tuple is immutable
# eggs[0] = "bonjour"

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Dictionaries

# %%
numbers = {"one": 1, "two": 2, "three": 3}

numbers["ninety"] = 90
print(numbers)

for key, value in numbers.items():
    print(f"{key} => {value}")

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Sets
#
# A set is an unordered collection of unique items.

# %%
# Duplicate values are automatically removed
s = {1, 2, 3, 2, 3, 4}
print(s)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Union, intersection and difference of sets

# %%
primes = {2, 3, 5, 7}
odds = {1, 3, 5, 7, 9}

print(primes | odds)
print(primes & odds)
print(primes - odds)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Functions

# %% [markdown]
# #### Function definition


# %%
def square(x):
    """Returns the square of x"""

    return x**2


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Function call

# %%
# Print function docstring
help(square)

print(square(0))
print(square(3))


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Default parameter values


# %%
def fibonacci(n, a=0, b=1):
    """Returns a list of the n first Fibonacci numbers"""

    l = []
    while len(l) < n:
        a, b = b, a + b
        l.append(a)
    return l


print(fibonacci(7))


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Flexible function arguments


# %%
def catch_all_args(*args, **kwargs):
    """Demonstrates the use of *args and **kwargs"""

    print(f"args = {args}")
    print(f"kwargs = {kwargs}")


catch_all_args(1, 2, 3, a=10, b="hello")

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Lambda (anonymous) functions

# %%
add = lambda x, y: x + y

print(add(1, 2))

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Iterators

# %% [markdown] slideshow={"slide_type": "-"}
# #### A unified interface for iterating

# %%
for element in [1, 2, 3]:
    print(element)
for element in (4, 5, 6):
    print(element)
for key in {"one": 1, "two": 2}:
    print(key)
for char in "ABC":
    print(char)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Under the hood
#
# - An **iterable** is a object that has an `__iter__` method which returns an **iterator** to provide iteration support.
# - An **iterator** is an object with a `__next__` method which returns the next iteration element.
# - A **sequence** is an iterable which supports access by integer position. Lists, tuples, strings and range objects are examples of sequences.
# - A **mapping** is an iterable which supports access via keys. Dictionaries are examples of mappings.
# - Iterators are used implicitly by many looping constructs.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### The range() function
#
# It doesn't return a list, but a `range` object (which exposes an iterator).

# %% slideshow={"slide_type": "-"}
for i in range(10):
    if i % 2 == 0:
        print(f"{i} is even")
    else:
        print(f"{i} is odd")

# %% slideshow={"slide_type": "subslide"}
for i in range(0, 10, 2):
    print(i)

# %% slideshow={"slide_type": "-"}
for i in range(5, -1, -1):
    print(i)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### The enumerate() function

# %%
supplies = ["pens", "staplers", "flame-throwers", "binders"]

for i, supply in enumerate(supplies):
    print(f"Index {i} in supplies is: {supply}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Comprehensions
#
# #### Principle
#
# - Provide a concise way to create sequences.
# - General syntax: `[expr for var in iterable]`.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### List comprehensions

# %% slideshow={"slide_type": "-"}
# Using explicit code
squared_numbers = []
stop = 10

for n in range(stop):
    squared_numbers.append(n**2)

print(squared_numbers)

# Using a list comprehension
print([n**2 for n in range(stop)])

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Set and dictionary comprehensions

# %%
# Create an uppercase set
s = {"abc", "def"}
print({e.upper() for e in s})

# Obtains modulos of 4 (eliminating duplicates)
print({a % 4 for a in range(1000)})

# Switch keys and values
d = {"name": "Prosper", "age": 12}
print({v: k for k, v in d.items()})

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Generators
#
# #### Principle
#
# - A **generator** defines a recipe for producing values.
# - A generator does not actually compute the values until they are needed.
# - It exposes an iterator interface. As such, it is a basic form of iterable.
# - It can only be iterated once.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Generators expressions
#
# They use parentheses, not square brackets like list comprehensions.

# %%
g1 = (n**2 for n in range(stop))

print(list(g1))
print(list(g1))


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Generator functions
#
# - A function that, rather than using `return` to return a value once, uses `yield` to yield a (potentially infinite) sequence of values.
# - Useful when the generator algorithm gets complicated.


# %%
def gen():
    """Generates squared numbers"""

    for n in range(stop):
        yield n**2


g2 = gen()
print(list(g2))
print(list(g2))


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Object-oriented programming

# %% [markdown]
# #### Class definition


# %%
class Account:
    """Represents a bank account"""

    def __init__(self, initial_balance):
        self.balance = initial_balance

    def credit(self, amount):
        """Credits money to the account"""

        self.balance += amount


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Class instanciation

# %%
new_account = Account(100)
new_account.credit(-40)
print(new_account.balance)


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Instance properties


# %%
class Vehicle:
    """Represents a vehicle"""

    def __init__(self, number_of_wheels, type_of_tank):
        # The leading underscore designates internal ("private") attributes
        self._number_of_wheels = number_of_wheels
        self._type_of_tank = type_of_tank

    @property
    def number_of_wheels(self):
        """Number of wheels"""

        return self._number_of_wheels

    @number_of_wheels.setter
    def number_of_wheels(self, number):
        self._number_of_wheels = number


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Using instance properties

# %%
my_strange_vehicle = Vehicle(4, "electric")
my_strange_vehicle.number_of_wheels = 2
print(my_strange_vehicle.number_of_wheels)
# Works, but frowned upon (accessing a private attribute)
# We should use a property instead
print(my_strange_vehicle._type_of_tank)


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Class attributes


# %%
class Employee:
    """Represents an employee"""

    empCount = 0

    def __init__(self, name, salary):
        self._name = name
        self._salary = salary
        Employee.empCount += 1

    @staticmethod
    def count():
        """Count the number of employees"""

        return f"Total employees: {Employee.empCount}"

    def __str__(self):
        return f"Name: {self._name}, salary: {self._salary}"


e1 = Employee("Ben", "30")
print(e1)
print(Employee.count())


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Inheritance


# %%
class Animal:
    """Represents an animal"""

    def __init__(self, species):
        self.species = species


class Dog(Animal):
    """Represents a specific animal: a dog"""

    def __init__(self, name):
        Animal.__init__(self, "Mammal")
        self.name = name


doggo = Dog("Fang")
print(doggo.name)
print(doggo.species)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Modules and packages

# %% slideshow={"slide_type": "slide"}
# Importing all module content into a namespace
import math

print(math.cos(math.pi))  # -1.0

# Importing specific module content into local namespace
from math import cos, pi

print(cos(pi))  # -1.0

# Aliasing an import
import numpy as np

print(np.cos(np.pi))  # -1.0


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Python good practices

# %% slideshow={"slide_type": "slide"}
import this
