from difflib import SequenceMatcher

truth = open('PublicTestCases-version1.1/test-set-scanned/gt/05.txt', 'r').read()

output = open('Output/05.txt', 'r').read()

seq = SequenceMatcher(a=truth,b=output)

print(str(seq.ratio()*100) + "%")