from django.shortcuts import render
import urllib
import http.client
import json
from app.models import OTP
from django.shortcuts import render_to_response

def send_otp(request):
	conn = http.client.HTTPConnection("2factor.in")
	OTP.objects.create(number = '9769953291')
	payload = "{}"
	url = "/API/V1/643743b1-1698-11e7-9462-00163ef91450/SMS/"+number+"/AUTOGEN/ABCDEF"
	conn.request("GET", url, payload)
	res = conn.getresponse()
	data = res.read()
	data = str(data,'utf-8')
	data = json.loads(data)
	objects = OTP.objects.filter(number__icontains = '9769953291')
	objects[0].session_id = data["Details"]
	print(data)
	return render(request,'send_otp.html')

def submit_otp(request):	
	if request.method=='POST':
		if form.is_valid():
			cd = otp_form.cleaned_data
			objects = OTP.objects.all()
			objects[0].otp = cd['otp']
			return render(request,'verify_otp.html',{'form':otp_form})

	else:
		form = OTPForm()
	return render(request,'submit_otp.html',{'form':otp_form})


def verify_otp(request):
	objects = OTP.objects.all()
	conn = http.client.HTTPConnection("2factor.in")
	payload = "{}"
	conn.request("GET", "/API/V1/643743b1-1698-11e7-9462-00163ef91450/SMS/VERIFY/"+objects[0].session_id+"/"+objects[0].otp, payload)
	res = conn.getresponse()
	data = res.read()	
	print(data.decode("utf-8"))

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

def contacts(request):
	return render(request,'contacts.html')

def projects(request):
	return render(request,'projects.html')

def project_detail(request):
	return render(request,'project_detail.html')	

def profile(request):
	return render(request,'profile.html')