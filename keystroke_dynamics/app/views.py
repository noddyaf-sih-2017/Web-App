from django.shortcuts import render
from django.shortcuts import render_to_response

# Create your views here.

def login(request):
	return render(request,'login.html')


def index(request):
	return render(request,'index.html')

def send_otp(request):
	return render(request,'send_otp.html')

<<<<<<< HEAD
def verify_otp(request):
	return render(request,'verify_otp.html')
=======
def table(request):
	return render(request,'tables_dynamic.html')



>>>>>>> d5aaf4de050640cfdca93c0b2e6c4bca46b74fbe
