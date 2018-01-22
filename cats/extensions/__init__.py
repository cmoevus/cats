"""Extension submodule.

Classes in CATS are extensible. To add a method to a CATS class, one simply need to write that method as a function and bind it to the desired class via the `_extension` dictionary. The `_extension` dictionary is written within the extension file with the following syntax:
    _extension = {'class0_name': {
                    'method0_name': function0,
                    'method1_name': function1,
                    ...
                 },
                 'class1_name': {
                    'method0_name': function1,
                    ...
                 }
                ...
    }
The function has to be written like a method and will be appended automatically to the class. A function bound to 'class0_name' with name 'method0_name' will be accessed from the object as follow:
    c = class0_name()
    c.method0_name()

A typical extension file will look like this:
    def useless_func(self, arg):
        return arg

    _extension = {
        'class0': {
            'method0': useless_func
        }
    }
"""
from __future__ import absolute_import, division, print_function
import importlib
import os
from glob import glob


def accepts_extensions(self):
    """Return True if the object accepts extensions. Doesn't exist if not..."""
    return True


def append(clas):
    """Append extensions to the given class."""
    if not hasattr(clas, 'accepts_extensions'):
        setattr(clas, 'accepts_extensions', accepts_extensions)
    if clas.__name__ in extensions:
        for method, func in extensions[clas.__name__].items():
            setattr(clas, method, func)
            getattr(clas, method).is_extension = True
    return clas


# Load and sort extensions
extensions = dict()
for extension_file in glob(os.path.dirname(__file__) + '/*.py'):
    name = os.path.splitext(os.path.basename(extension_file))[0]
    if name != '__init__':
        extension = importlib.import_module('cats.extensions.' + name)
        for clas, methods in extension._extension.items():
            if clas not in extensions:
                extensions[clas] = dict()
            extensions[clas].update(methods)
