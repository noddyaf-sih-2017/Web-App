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

def table(request):
	return render(request,'tables_dynamic.html')
<<<<<<< HEAD

def contacts(request):
	return render(request,'contacts.html')

def projects(request):
	return render(request,'projects.html')
=======
>>>>>>> f697bc50d30a344694dc4da9b99c101f9862e15d
