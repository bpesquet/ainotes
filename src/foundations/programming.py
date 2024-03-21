# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Programming

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Learning objectives
#
# - Review the essential concepts of programming: variables, flow control, data structures, etc.
# - Demonstrate their implementation in the [Python](https://www.python.org/) programming language.
# - Discover what is a [Jupyter notebook](https://docs.jupyter.org/en/latest/).
# - Learn about some good practices for Python-based software development.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %% slideshow={"slide_type": "skip"} tags=["hide-output"]
# !pip install papermill

# %% slideshow={"slide_type": "slide"}
# Relax some linting rules not needed here
# pylint: disable=invalid-name,redefined-outer-name,consider-using-f-string,duplicate-value,unnecessary-lambda-assignment,protected-access,too-few-public-methods,wrong-import-position,unused-import,consider-swap-variables,consider-using-enumerate,too-many-lines

import platform
import os

print(f"Python version: {platform.python_version()}")


# %% [markdown] slideshow={"slide_type": "slide"}
# ## The Python language


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
# > For many more examples, see the cheatsheet below.


# %%
def hello(name):
    """Say hello to someone"""

    print(f"Hello, {name}")


friends = ["Lou", "David", "Iggy"]

for friend in friends:
    hello(friend)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### A prominent language
#
# Python has become the language of choice for artificial intelligence and [Data Science](https://en.wikipedia.org/wiki/Data_science), for the following reasons:
#
# - language qualities (ease of use, simplicity, versatility);
# - involvement of the scientific and academical communities;
# - rich ecosystem of dedicated open source libraries.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The Jupyter Notebook format
#
# The [Jupyter Notebook](https://jupyter.org/) is an open-source web application for creating and managing documents (called *notebooks*) that may contain executable code, equations, visualizations and text.
#
# A notebook file has an _.ipynb_ extension. It contains blocks of text called *cells*, written in either code or [Markdown](https://www.markdownguide.org/). Notebooks have become a standard for experimenting and sharing results in many scientific fields.
#
# [![IPython](_images/jupyterpreview.jpg)](https://jupyter.org/)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Running Python code
#
# Python code can be run either:
#
# - **locally**, after installing a Python environment. [Anaconda](https://www.anaconda.com/download), a scientific distribution including many (1500+) specialized packages, is the easiest way to setup a work environment for AI with Python.
# - **in the cloud**, using an online service for executing raw Python code or Jupyter notebooks. For example, [Google Colaboratory](https://colab.research.google.com) offers free access to specialized processors.

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
# https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions

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
# https://docs.python.org/3/tutorial/classes.html


class Account:
    """Represents a bank account"""

    def __init__(self, initial_balance):
        self.balance = initial_balance

    def credit(self, amount):
        """Credits money to the account"""

        self.balance += amount

    def __str__(self):
        return f"Account balance: {self.balance}"


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Class instanciation

# %%
new_account = Account(100)
print(new_account)

new_account.credit(-40)
print(new_account.balance)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Instance properties


# %%
# https://docs.python.org/3/library/functions.html#property


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

    empCount = 0  # Class-level attribute, shared by all instances

    def __init__(self, name, salary):
        self._name = name
        self._salary = salary
        Employee.empCount += 1

    @staticmethod
    def count():
        """Count the number of employees"""

        return f"Total employees: {Employee.empCount}"


e1 = Employee("Ben", "30")
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
# #### Dataclasses
#
# Simplified syntax for attribute-centric classes.

# %%
# https://realpython.com/python-data-classes/

from dataclasses import dataclass


@dataclass
class Student:
    """Represents a student"""

    name: str
    id: int


student = Student("Jack", 123456)
print(student)

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

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Writing pythonic code

# %%
import this

# %% [markdown] slideshow={"slide_type": "slide"}
# #### What does "Pythonic" mean?
#
# - Python code is considered _pythonic_ if it:
#   - conforms to the Python philosophy;
#   - takes advantage of the language's specific features.
# - Pythonic code is nothing more than **idiomatic Python code** that strives to be clean, concise and readable.
#
# > Most linting tools (see below) enforce pythonicity.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Example: swapping two variables

# %%
a = 3
b = 2

# Non-pythonic
tmp = a
a = b
b = tmp
print(a, b)

# Pythonic
a, b = b, a
print(a, b)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Example: iterating on a list

# %%
my_list = ["a", "b", "c"]


# Non-pythonic
i = 0
while i < len(my_list):
    print(my_list[i])
    i += 1

# Still non-pythonic
for i in range(len(my_list)):
    print(my_list[i])

# Pythonic
for item in my_list:
    print(item)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Example: indexed traversal

# %%
my_list = ["a", "b", "c"]

# Non-pythonic
for i in range(len(my_list)):
    print(i, "->", my_list[i])

# Pythonic
for i, item in enumerate(my_list):
    print(i, "->", item)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Example: searching in a list

# %%
fruits = ["apples", "oranges", "bananas", "grapes"]
fruit = "cherries"

# Non-pythonic
found = False
size = len(fruits)
for i in range(0, size):
    if fruits[i] == fruit:
        found = True
print(found)

# Pythonic
found = fruit in fruits
print(found)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Example: generating a list

# %%
numbers = [1, 2, 3, 4, 5, 6]

# Non-pythonic
doubles = []
for i in range(len(numbers)):
    if numbers[i] % 2 == 0:
        doubles.append(numbers[i] * 2)
    else:
        doubles.append(numbers[i])
print(doubles)

# Pythonic
doubles = [x * 2 if x % 2 == 0 else x for x in numbers]
print(doubles)

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Code style
#
# - [PEP8](https://www.python.org/dev/peps/pep-0008/) is the official style guide for Python:
#   - use 4 spaces for indentation;
#   - define a maximum value for line length (around 80 characters);
#   - organize imports at beginning of file;
#   - surround binary operators with a single space on each side;
#   - ...
# - Code style should be enforced upon creation by a tool like [Black](https://github.com/psf/black).

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Beyond PEP8
#
# Focusing on style and PEP8-compliance might make you miss more fundamental code flaws.

# %%
from IPython.display import YouTubeVideo

YouTubeVideo("wf-BqAjZb8M")


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Docstrings
#
# A [docstring](https://www.python.org/dev/peps/pep-0257/) is a string literal that occurs as the first statement in a module, function, class, or method definition to document it.
#
# All modules, classes, public methods and exported functions should include a docstring.


# %%
def create_complex(real=0.0, imag=0.0):
    """Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """

    # ... (do something useful with parameters)
    _ = real, imag


# %% [markdown] slideshow={"slide_type": "slide"}
# #### Code linting
#
# - _Linting_ is the process of checking code for syntactical and stylistic problems before execution.
# - It is useful to catch errors and improve code quality in dynamically typed, interpreted languages, where there is no compiler.
# - Several linters exist in the Python ecosystem. A popular choice is [Pylint](https://pylint.readthedocs.io).

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Type annotations
#
# - Added in Python 3.5, [type annotations](https://www.python.org/dev/peps/pep-0484/) allow to add type hints to code entities like variables or functions, bringing a statically typed flavour to the language.
# - [mypy](http://mypy-lang.org/) can automatically check the code for annotation correctness.


# %%
def greeting(name: str) -> str:
    """Greet someone"""

    return "Hello " + name


greeting("Alice")  # OK

# greeting(3)  # mypy error: incompatible type "int"; expected "str"

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Unit tests
#
# Unit tests automate the testing of individual code elements like functions or methods, thus decreasing the risk of bugs and regressions.
#
# They can be implemented in Python using tools like [unittest](https://docs.python.org/3/library/unittest.html) or [pytest](https://docs.pytest.org).


# %%
def inc(x):
    """Increment a value"""

    return x + 1


assert inc(3) == 4

# assert inc(3) == 5  # AssertionError


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Packaging and dependency management

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Managing dependencies in Python
#
# - Most Python apps depend on third-party libraries and frameworks (NumPy, Flask, Requests...).
# - These tools may also have external dependencies, and so on.
# - **Dependency management** is necessary to prevent version conflicts and incompatibilities. it involves two things:
#   - a way for the app to declare its dependencies;
#   - a tool to resolve these dependencies and install compatible versions.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Semantic versioning
#
# - Software versioning convention used in many ecosystems.
# - A version number comes as a suite of three digits `X.Y.Z`.
#   - X = major version (potentially including breaking changes).
#   - Y = minor version (only non-breaking changes).
#   - Z = patch.
# - Digits are incremented as new versions are shipped.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### pip and requirements.txt
#
# A `requirements.txt` file is the most basic way of declaring dependencies in Python.
#
# ```text
# certifi>=2020.11.0
# chardet==4.0.0
# click>=6.5.0, <7.1
# download==0.3.5
# Flask>=1.1.0
# ```
#
# The [pip](https://pypi.org/project/pip/) package installer can read this file and act accordingly, downloading dependencies from [PyPI](https://pypi.org/).
#
# ```bash
# pip install -r requirements.txt
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Virtual environments
#
# - A **virtual environment** is an isolated Python environment where a project's dependencies are installed.
# - Using them prevents the risk of mixing dependencies required by different projects on the same machine.
# - Several tools exist to manage virtual environments in Python, for example [virtualenv](https://virtualenv.pypa.io) and [conda](https://docs.conda.io).

# %% [markdown] slideshow={"slide_type": "slide"}
# #### conda and environment.yml
#
# Installed as part of the [Anaconda](https://www.anaconda.com/) distribution, the [conda](https://docs.conda.io) package manager reads an `environment.yml` file to install the dependencies associated to a specific virtual environment.
#
# ```yaml
# name: example-env
#
# channels:
#   - conda-forge
#   - defaults
#
# dependencies:
#   - python=3.7
#   - matplotlib
#   - numpy
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Poetry
#
# [Poetry](https://python-poetry.org) is a recent packaging and dependency management tool for Python. It downloads packages from [PyPI](https://pypi.org/) by default.
#
# ```bash
# # Create a new poetry-compliant project in the my-project folder
# poetry new my-project
#
# # Initialize an already existing project for Poetry
# poetry init
#
# # Install defined dependencies
# poetry install
#
# # Add a package to project dependencies and install it
# poetry add <package name>
#
# # Update dependencies to sync them with configuration file
# poetry update
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Poetry and virtual environments
#
# By default, Poetry creates a virtual environment for the configured project in a user-specific folder. A standard practice is to store it in the project's folder.
#
# ```bash
# # Tell Poetry to store the environment in the local project folder
# poetry config virtualenvs.in-project true
#
# # Activate the environment
# poetry shell
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# #### The pyproject.toml file
#
# Poetry configuration file, soon-to-be standard for Python projects.
#
# ```toml
# [tool.poetry]
# name = "poetry example"
# version = "0.1.0"
# description = ""
#
# [tool.poetry.dependencies]
# python = ">=3.7.1,<3.10"
# jupyter = "^1.0.0"
# matplotlib = "^3.3.2"
# sklearn = "^0.0"
# pandas = "^1.1.3"
# ipython = "^7.0.0"
#
# [tool.poetry.dev-dependencies]
# pytest = "^6.1.1"
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Caret requirements
#
# Offers a way to precisely define dependency versions.
#
# | Requirement | Versions allowed |
# | :---------: | :--------------: |
# |   ^1.2.3    |  >=1.2.3 <2.0.0  |
# |    ^1.2     |  >=1.2.0 <2.0.0  |
# |   ~1.2.3    |  >=1.2.3 <1.3.0  |
# |    ~1.2     |  >=1.2.0 <1.3.0  |
# |    1.2.3    |    1.2.3 only    |

# %% [markdown] slideshow={"slide_type": "slide"}
# #### The poetry.lock file
#
# - The first time Poetry install dependencies, it creates a `poetry.lock` file that contains the exact versions of all installed packages.
# - Subsequent installs will use these exact versions to ensure consistency.
# - Removing this file and running another Poetry install will fetch the latest matching versions.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Working with notebooks

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Advantages of Jupyter notebooks
#
# - Standard format for mixing text, images and (executable) code.
# - Open source and platform-independant.
# - Useful for experimenting and prototyping.
# - Growing ecosystem of [extensions](https://tljh.jupyter.org/en/latest/howto/admin/enable-extensions.html) for various purposes and cloud hosting solutions ([Colaboratory](https://colab.research.google.com/), [AI notebooks](https://www.ovhcloud.com/en/public-cloud/ai-notebook/)...).
# - Integration with tools like [Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Drawbacks of Jupyter notebooks
#
# - Arbitrary execution order of cells can cause confusing errors.
# - Notebooks don't encourage good programming habits like modularization, linting and tests.
# - Being JSON-based, their versioning is more difficult than for plain text files.
# - Dependency management is also difficult, thus hindering reproducibility.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Collaborating with notebooks
#
# A common solution for sharing notebooks between a team is to use [Jupytext](https://jupytext.readthedocs.io). This tool can associate an `.ipynb` file with a Python file to facilitate collaboration and version control.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Code organization
#
# Monolithic notebooks can grow over time and become hard to understand and maintain.
#
# Just like in a traditional software project, it is possible to split them into separate parts, thus following the [separation of concerns](https://en.wikipedia.org/wiki/Separation_of_concerns) design principle.
#
# Code can be splitted into several sub-notebooks and/or external Python files. The latter facilitates unit testing and version control.

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Notebook workflow
#
# Tools like [papermill](https://papermill.readthedocs.io) can orchestrate the execution of several notebooks in a row. External parameters can be passed to notebooks, and the runtime flow can depend on the execution results of each notebook.

# %% tags=["hide-output"] slideshow={"slide_type": "slide"}
import papermill as pm

# Doesn't work on Google Colaboratory. Workaround here:
# https://colab.research.google.com/github/rjdoubleu/Colab-Papermill-Patch/blob/master/Colab-Papermill-Driver.ipynb
notebook_dir = "./_papermill"
result = pm.execute_notebook(
    os.path.join(notebook_dir, "simple_input.ipynb"),
    os.path.join(notebook_dir, "simple_output.ipynb"),
    parameters={"msg": "Hello"},
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Additional resources
#
# [The Little Book of Python Anti-Patterns](https://docs.quantifiedcode.com/python-anti-patterns/index.html)

# %%
