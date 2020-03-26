# dataset name 
dataset = 'yelp_dataset'

# paths
# main_path = '/home/share/guoyangyang/recommendation/NCF-Data/'
main_path = 'data/'

# train_rating = main_path + '{}.train.rating'.format(dataset)
# test_rating = main_path + '{}.test.rating'.format(dataset)
# test_negative = main_path + '{}.test.negative'.format(dataset)
train_data = main_path + 'train_data.pickle'
test_data = main_path + 'test_data.pickle'
test_negative = main_path + 'test_with_neg_sample.pickle'
user_data = main_path + 'users-complete.pickle'
item_data = main_path + 'businesses-complete.pickle'

model_path = './models/'
# BPR_model_path = model_path + 'NeuMF.pth'
BPR_model_path = model_path + 'checkpoint.pth'
