import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--foo", help="Foo help")
parser.add_argument("-b", "--bar", help="Bar help")
args = parser.parse_args()

print("#"*50)
print("Our flags:")
print(f"Foo: { args.foo}")
print(f"bar: { args.bar}")
print()

print("Our environment variables:")
print("BAZ: ", os.environ.get('BAZ'))
print("QUUX: ", os.environ.get('QUUX'))
print("Done")
print("#"*50)
