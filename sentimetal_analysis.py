
# coding: utf-8

# # sentimental analysis 1o1

# In[13]:


import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
import pandas as pd
import tensorflow as tf


# In[2]:


lemmatizer = WordNetLemmatizer()
hm_lines = 100000



# In[3]:


lexicon = []
with open('pos.txt','r') as f:
	contents = f.readlines()
    
	for l in contents[:hm_lines]:
		all_words = word_tokenize(l)
		lexicon += list(all_words)

print(len(contents))
print(len(lexicon))


# In[4]:


lexicon = []
with open('pos.txt','r') as f:
	contents = f.readlines()
	for l in contents[:hm_lines]:
		all_words = word_tokenize(l)
		lexicon += list(all_words)
            
with open('neg.txt','r') as f:
	contents = f.readlines()
	for l in contents[:hm_lines]:
		all_words = word_tokenize(l)
		lexicon += list(all_words)


# In[5]:


len(lexicon)


# In[6]:


lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
w_counts = Counter(lexicon)
l2 = []
for w in w_counts:
	#print(w_counts[w])
	if 1000 > w_counts[w] > 9:
		l2.append(w)
print(l2)



# In[7]:


def create_lexicon(pos,neg):

	lexicon = []
	with open(pos,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)
            
	with open(neg,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:
		#print(w_counts[w])
		if 1000 > w_counts[w] > 50:
			l2.append(w)
	print((l2))
	return l2


# In[8]:


featureset = []

with open('pos.txt','r') as f:
    contents = f.readlines()
    
    for l in contents[:10]:
        current_words = word_tokenize(l.lower())
        print(current_words)
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))
        print(len(features))
        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                features[index_value] += 1
                print(features)
            else:
                print("s")
                
        features = list(features)
        featureset.append([features,[1,0]])
print((featureset[1][2]))
       
        
			


# In[9]:



def sample_handling(sample,lexicon,classification):

	featureset = []

	with open('sample','r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			features = list(features)
			featureset.append([features,classification])

	return featureset


# In[10]:











def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling('pos.txt',lexicon,[1,0])
	features += sample_handling('neg.txt',lexicon,[0,1])
    
	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y




# In[11]:


train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')


# In[ ]:


from create_sentiment_featuresets import create_feature_sets_and_labels
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','/path/neg.txt')

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Nothing changes
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
	    
		for epoch in range(2):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
				epoch_loss += c
				i+=batch_size
				
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

	    
train_neural_network(x)


# In[ ]:


if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	# if you want to pickle this data:
	with open('/path/to/sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)

