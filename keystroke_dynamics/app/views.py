from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files import File
from django.conf import settings
import json
import numpy as np
import pickle
import pandas as pd
import os
BASE = os.path.dirname(os.path.abspath(__file__))
# Create your views here.

def index(req):
    return HttpResponse('hello')


def login(req):
	return render(req, 'login.html', {})


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


def check_keyboard_dynamics(username, processed):
	# processed.csv or processed.nparray
	# print processed
	username = username.strip()
	try:
		with open(os.path.join(BASE, username + '.csv'), 'r') as f:
			ff = File(f)
			processed_data = ff.read()

	except IOError:
		#file doesnt exist
		pass

	return True


def formatData(dataS):
    finS = []
    for r in dataS:
        for i in r:
            finS.append(i)
                    
    return finS


def preprocess(dataS):
	finS = formatData(json.loads('[' + dataS[:-1] + ']'))
	dfS = pd.DataFrame(finS)
	dfS.fillna(dfS.mean(), inplace=True)
	return dfS.to_csv(index=False)