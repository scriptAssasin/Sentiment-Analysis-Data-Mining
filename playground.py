import pickle

# open a file, where you stored the pickled data
file = open('eclass_all_with_sentiment_v2.pkl', 'rb')

# # dump information to that file
data = pickle.load(file)

print(data)
# # close the file
# file.close()

# print('Showing the pickled data:')

# cnt = 0
# for item in data:
#     print('The data ', cnt, ' is : ', item)
#     cnt += 1
