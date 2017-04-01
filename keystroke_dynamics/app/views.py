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

=======
>>>>>>> e4b469a5126b82617e016763d82dc1396781502a

def contacts(request):
	return render(request,'contacts.html')

def projects(request):
	return render(request,'projects.html')

<<<<<<< HEAD

def project_detail(request):
	return render(request,'project_detail.html')	

def profile(request):
	return render(request,'profile.html')
=======
>>>>>>> e4b469a5126b82617e016763d82dc1396781502a
