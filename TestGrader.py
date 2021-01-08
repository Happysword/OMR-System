from difflib import SequenceMatcher , ndiff
import argparse
import os

args_parser = argparse.ArgumentParser()
args_parser.add_argument('accurate_path', help="Accurate test file")
args_parser.add_argument('to_check_path', help="To be checked file")
args = args_parser.parse_args()

accurate = open(args.accurate_path, 'r').read()
to_check = open(args.to_check_path, 'r').read()

seq = SequenceMatcher(a=accurate, b=to_check)
diff = ndiff(a=open(args.accurate_path, 'r').readlines(), b=open(args.to_check_path, 'r').readlines())

print("Testing:", os.path.basename(args.accurate_path))
print(''.join(diff))
print(f"Accuracy {seq.ratio() * 100}%")
