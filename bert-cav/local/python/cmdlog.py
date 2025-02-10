# Make CMDs directory with the current command.

#! /usr/bin/python

import sys
import os

from path import makeDir

def makeCmd (description = None):
    commandPath = 'CMDs'
    step = os.path.basename (sys.argv [0])
    separator = 30 * '-' + '\n'

    makeDir (commandPath, False)
    f = open ('%s/%s.cmds.txt' % (commandPath, step), 'a')

    f.write (separator)
    if description != None:
        f.write ('# ' + description + '\n')
    f.write (' '.join (sys.argv) + '\n')
    f.write (separator)

def makeCmdPath (cmdpath, description = None):
    commandPath = os.path.join('CMDs', cmdpath)
    step = os.path.basename (sys.argv [0])
    separator = 30 * '-' + '\n'

    makeDir (commandPath, False)
    f = open ('%s/%s.cmds.txt' % (commandPath, step), 'a')

    f.write (separator)
    if description != None:
        f.write ('# ' + description + '\n')
    f.write (' '.join (sys.argv) + '\n')
    f.write (separator)

def makeMainCmd (prog, description, *args):
    commandPath = 'CMDs'
    #step = os.path.basename (prog)
    separator = 30 * '-' + '\n'

    makeDir (commandPath, False)
    f = open ('%s/%s.cmds.txt' % (commandPath, prog), 'a')

    f.write (separator)
    if description != None:
        f.write ('# ' + description + '\n')
    f.write (' '.join (args) + '\n')
    f.write (separator)

