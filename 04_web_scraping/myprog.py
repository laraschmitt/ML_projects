import sys
import argparse

parser = argparse.ArgumentParser(description='my first program')

# positional arguments
parser.add_argument("message", help="display the string you use here")

# Defaults
# for many options, you may want to define a default value
# this requires to give a shortcut for the command
parser.add_argument("-s", "--square", help="display the square of a given number", type= int, default=0)


# in mqny programs you find a verbosity option that generates more detailed output
# add one as well, but make it optional
#parser.add_argument("-v", "--verbosity", help="increase output verbosity", action="store_true")
# the store_true stores the verbosity as a boolean.



#sometimes you may want to restrict the values one could use
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], help="increase output verbosity", default=0)


args = parser.parse_args()
print(args.message)
print(args.square**2)

if args.verbosity:
	print("verbosity turned on")
