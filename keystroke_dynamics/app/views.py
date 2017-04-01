from django.shortcuts import render
from django.shortcuts import render_to_response

# Create your views here.

def login(request):
	return render(request,'login.html')


def index(request):
	return render(request,'index.html')

def send_otp(request):
	return render(request,'send_otp.html')

def verify_otp(request):
	return render(request,'verify_otp.html')
