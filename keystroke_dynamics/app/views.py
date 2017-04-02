from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files import File
from django.conf import settings
import json
import numpy as np
import pickle
import pandas as pd
import os
import random
import tensorflow as tf
BASE = os.path.dirname(os.path.abspath(__file__))

tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)

data = list()

with open(os.path.join(BASE, 'Saumitra1' + '.csv')) as f:
    skipped_first_line = False
    for line in f.readlines():
        if not skipped_first_line:
            skipped_first_line = True
            continue
        else:
            line = line.strip().split(',')
            d = [int(float(x)) for x in line[1:]]
            data.append(d)

######################### Neural Network #######################

z_dim = 5
batch_size = 10
n_epoch = 100

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 24])

D_W1 = tf.Variable(xavier_init([24, 6]))
D_b1 = tf.Variable(tf.zeros(shape=[6]))

D_W2 = tf.Variable(xavier_init([6, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

Z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, 6]))
G_b1 = tf.Variable(tf.zeros(shape=[6]))

G_W2 = tf.Variable(xavier_init([6, 24]))
G_b2 = tf.Variable(tf.zeros(shape=[24]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

n_batch = len(data) // batch_size
batched_data = np.array_split(data, n_batch)

# Start session
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

# Epoch-training
for epoch in range(n_epoch):
    err_G = []
    err_D = []

    # Batch training
    for b_idx in range(n_batch):
        x_btch = batched_data[b_idx]
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x_btch, Z: sample_Z(batch_size, z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, z_dim)})

        err_D.append(D_loss_curr)
        err_G.append(G_loss_curr)

    print("Epoch %d G:%f  D:%f" % (epoch, np.mean(err_G), np.mean(err_D)))


def predict(data):
    d_res = sess.run(D_real, feed_dict={X: np.reshape(data, [1, -1])})
    result = d_res[0][0]

    if result < 0.5:
        return False
    else:
        return True


# Create your views here.

def index(req):
    return HttpResponse('hello')


def login(req):
	return render(req, 'login.html', {"username": "Saumitra"})


def cont_auth(req):
	return render(req, 'auth.html', {"username": "Saumitra"})


def send_details(req):
	username = req.POST['username']
	password = req.POST['password']
	jsonR = req.POST['json']
	wasEntered = not (jsonR == 'nope')
	# Authenticate here
	authenticated = True
	# Authenticate end

	if authenticated and not jsonR == 'nope':
		processedJson = preprocess(jsonR)
		verify_dynamics = check_keyboard_dynamics(username, processedJson)
		if not verify_dynamics:
			return redirect(req, reverse('otp_login'))
		else:
			with open(os.path.join(BASE,  username + '.csv'), 'a+') as f:
				dFile = File(f)
				dFile.write(processedJson)

	return JsonResponse({"authenticated": authenticated, "wasEntered": wasEntered})


def send_login_details(req):
    username = req.POST['username']
    password = req.POST['password']
    jsonR = req.POST['json']
    # Authenticate here
    authenticated = True
    # Authenticate wasEntere
    if authenticated:
        processedJson = preprocessLogin(jsonR)
        print('PROCESSED', processedJson)
        verify = predict(processedJson)
        print('VERIFICATION', verify)

        if not verify:
            authenticated = False

    return JsonResponse({"authenticated": authenticated})


def predict_login(username, processed):
	print('login', processed)
	return True


def check_keyboard_dynamics(username, processed):
	# processed.csv or processed.nparray
	print(processed)
	username = username.strip()
	try:
		with open(os.path.join(BASE, username + '.csv'), 'r') as f:
			ff = File(f)
			processed_data = ff.read()

	except IOError:
		#file doesnt exist
		pass

	return True


def formatForLogin(dataS):
    finS = []
    ind = 0
    for r in dataS:
        curr = 0
        finS.append({})
        for i in r:
            keyVal = i['key'] + '-' +str(curr) + '-'
            finS[ind][keyVal+'kftime'] = i['kftime']
            finS[ind][keyVal+'ftime'] = i['ftime']
            finS[ind][keyVal+'time'] = i['time']
            curr+=1
            
        finS[ind]['totaltime'] = sum([x['time'] for x in r])

        ind += 1
        
    return finS


def formatData(dataS):
    finS = []
    for r in dataS:
        for i in r:
            finS.append(i)

    return finS


def preprocessLogin(dataS):
	finS = formatForLogin(json.loads('[' + dataS[:-1] + ']'))
	dfS = pd.DataFrame(finS)
	dfS.drop('p-0-ftime', axis=1, inplace=True)
	dfS.fillna(dfS.mean(), inplace=True)
	print(dfS.head())
	return dfS.as_matrix().ravel()


def preprocess(dataS):
	finS = formatData(json.loads('[' + dataS[:-1] + ']'))
	dfS = pd.DataFrame(finS)
	dfS.fillna(dfS.mean(), inplace=True)
	return dfS.to_csv(index=False, header=None)
