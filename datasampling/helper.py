__author__ = "Georgi Tancev"
__copyright__ = "Copyright (C) 2019 Georgi Tancev"

TINY = 1e-20
import matplotlib
matplotlib.use('Agg')
import re
import numpy as np
import copy
import os, errno
import scipy.linalg as la
from scipy.stats import special_ortho_group as sog
import logging
import urllib.request
import functools


def argget(dt, key, default=None, keep=False, ifset=None):
    """ Evaluates function arguments.

    It takes a dictionary dt and a key "key" to evaluate, if this key is contained in the dictionary. If yes, return
    its value. If not, return default. By default, the key and value pair are deleted from the dictionary, except if
    keep is set to True. ifset can be used to override the value, and it is returned instead (except if None).
    :param dt: dictionary to be searched
    :param key: location in dictionary
    :param default: default value if key not found in dictionary
    :param keep: bool, indicating if key shall remain in dictionary
    :param ifset: value override.
    :return: chosen value for key if available. Else default or ifset override.
    """
    if key in dt and dt[key] is not None:
        if keep:
            val = dt[key]
        else:
            val = dt.pop(key)
        if ifset is not None:
            return ifset
        else:
            return val
    else:
        return default

def compile_arguments(cls, kw, transitive=False, override_static=False, keep_entries=True):
    """Extracts valid keywords for cls from given keywords and returns the resulting two dicts.

    :param cls: instance or class having property or attribute "_defaults", which is a dict of default parameters.
    :param transitive: determines if parent classes should also be consulted
    :param kw: the keyword dictionary to separate into valid arguments and rest
    :return: tuple of dicts, one with extracted/copied relevant keywords and one with the rest
    """
    if keep_entries:
        kw = copy.copy(kw) # using copy from library copy (shallow copy)
    if hasattr(cls, 'compile_arguments') and not override_static:
        new_kw, kw = cls.compile_arguments(kw, keep_entries=keep_entries)
    else:
        new_kw = {}
    if transitive:
        for b in cls.__bases__:
            if hasattr(b, '_defaults'):
                temp_kw, kw = compile_arguments(b, kw, transitive=True, keep_entries=keep_entries)
                new_kw.update(temp_kw)
    # compile defaults array from complex _defaults dict:
    defaults = {k: v['value'] if isinstance(v, dict) else v for k, v in cls._defaults.items() if
                not isinstance(v, dict) or 'value' in v}
    required = [k for k, v in cls._defaults.items() if isinstance(v, dict) and not 'value' in v]
    new_kw.update({k: argget(kw, k, v) for k, v in defaults.items()})
    new_kw.update({k: argget(kw, k) for k in required})
    return new_kw, kw

def collect_parameters(cls, kw_args={}):
    """
    helper function to collect all parameters defined in cls' _defaults needed for both compile arguments and define arguments
    :param cls:
    :param kw_args:
    :return:
    """
    args = copy.copy(kw_args)
    for b in cls.__bases__:
        if hasattr(b, '_defaults'):
            args = collect_parameters(b, args)
    params = {k: v for k, v in cls._defaults.items() if isinstance(v, dict) and 'help' in v}
    args.update(params)
    return args

def define_arguments(cls, parser):
    """ Requires cls to have field _defaults! Parses all fields defined in _defaults and creates a argparse compatible
    structure from them. These are then appended to the parser structure.

    :param cls: cls which contains (at least an empty) _defaults dict.
    :param parser: parser to add parameters to
    :return: parser with added parameters
    """
    if hasattr(cls, 'collect_parameters'):
        args = cls.collect_parameters()
    else:
        args = collect_parameters(cls)
    for k, v in args.items():
        key = k
        kw = {'help': v['help']}

        if 'type' in v:
            kw['type'] = v['type']
        if 'nargs' in v:
            kw['nargs'] = v['nargs']
        elif 'value' in v and isinstance(v['value'], (list, dict)):
            kw['nargs'] = '+'

        propname = copy.copy(key)
        if 'value' in v:
            if isinstance(v['value'], bool):
                if v['value'] == False:
                    kw['action'] = 'store_true'
                else:
                    kw['dest'] = key
                    kw['action'] = 'store_false'
                    if 'invert_meaning' in v and not 'name' in v:
                        propname = v['invert_meaning'] + key
                    else:
                        propname = "no_" + key
            else:
                kw['default'] = v['value']
        else:
            # we do not provide a default, this value is required!
            kw['required'] = True
        if 'name' in v:  # overrides invert_meaning!
            propname = v['name']
            kw['dest'] = key
        props = ["--" + propname]
        if 'short' in v:
            props = ['-' + v['short']] + props
        if 'alt' in v:
            props += ['--' + x for x in v['alt']]
        parser.add_argument(*props, **kw)
    return parser

def generate_defaults_info(cls):
    """
    Adds description for keyword dict to docstring of class cls

    Parameters
    ----------
    cls: Class to extract _defaults field from to generate additional docstring info from.
    """
    if hasattr(cls, "_defaults"):

        desc = """kw : dict containing the following options.\n"""

        # try to find Parameters heading:
        for k, v in cls._defaults.items():
            if isinstance(v, dict):
                if 'help' in v.keys():
                    if 'value' in v.keys():
                        val = str(v['value']).replace('<', '').replace('>', '')
                        desc += "    - **{}** [default: {}] {}\n".format(k, val, v['help'])
                    else:
                        desc += "    - **{}** {}\n".format(k, v['help'])
                else:
                    if 'value' in v.keys():
                        val = str(v['value']).replace('<', '').replace('>', '')
                        desc += "    - **{}** [default: {}]\n".format(k, val)
                    else:
                        desc += "    - **{}**\n".format(k)
            else:
                v = str(v).replace('<', '').replace('>', '')
                desc += "    - **{}** [default: {}]\n".format(k, v)

        if cls.__doc__ is None:
            desc = "\n".join(["        " + d for d in desc.split('\n') if len(d)]) + "\n"
            if cls.__init__.__doc__ is None:
                cls.__init__.__doc__ = desc
            else:
                if len(re.findall("Parameters\s*\n\s*[-]+\s*\n", cls.__init__.__doc__)):
                    cls.__init__.__doc__ = re.sub('(Parameters\s*\n\s*[-]+\s*\n)', '\g<1>' + desc,
                                                  cls.__init__.__doc__, 1)
                elif len(re.findall('([ ]+)(:param.*\n)', cls.__init__.__doc__)):
                    cls.__init__.__doc__ = re.sub('([ ]*)(:param.*\n)',
                                                  '\g<1>\g<2>' + desc.replace('kw :', ':param kw:', 1),
                                                  cls.__init__.__doc__, 1)
                else:
                    cls.__init__.__doc__ = cls.__init__.__doc__.rstrip() + """

        Parameters
        ----------
""" + desc

        else:
            desc = "\n".join(["    " + d for d in desc.split('\n') if len(d)]) + "\n"
            if len(re.findall("Parameters\s*\n\s*[-]+\s*\n", cls.__doc__)):
                cls.__doc__ = re.sub('(Parameters\s*\n\s*[-]+\s*\n)', '\g<1>' + desc, cls.__doc__, 1)
            elif len(re.findall('([ ]+)(:param.*\n)', cls.__doc__)):
                cls.__doc__ = re.sub('([ ]*)(:param.*\n)', '\g<1>\g<2>' + desc.replace('kw :', ':param kw:', 1),
                                     cls.__doc__, 1)
            else:
                cls.__doc__ = cls.__doc__.rstrip() + """

    Parameters
    ----------
""" + desc

def counter_generator(maxim):
    """ Generates indices over multidimensional ranges.

    :param maxim: Number of iterations per dimension
    :return: Generator yielding next iteration
    """
    maxim = np.asarray(maxim)
    count = np.zeros(maxim.shape)
    yield copy.deepcopy(count)
    try:
        while True:
            arr = (maxim - count) > 1
            lind = len(arr) - 1 - arr.tolist()[::-1].index(True)
            count[lind] += 1
            count[lind + 1:] = 0
            yield copy.deepcopy(count)
    except ValueError:
        pass