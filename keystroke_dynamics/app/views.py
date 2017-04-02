from django.shortcuts import render
import urllib
import http.client
import json
from app.models import OTP 
from django.shortcuts import render_to_response
from app.forms import OTPForm
from django.template import RequestContext



def otp(request):
		#print(data)
	return render(request,'send_otp.html')




def send_otp(request):
	print(123)
	num = "9819515144"
	conn = http.client.HTTPConnection("2factor.in")
	#OTP.objects.create(number = '9819515144')
	payload = "{}"
	url = "/API/V1/6c74605e-174a-11e7-9462-00163ef91450/SMS/"+num+"/766461"
	conn.request("GET", url, payload)
	res = conn.getresponse()
	data = res.read()
	data = str(data,'utf-8')
	data = json.loads(data)
	#objects = OTP.objects.filter(number__icontains = '9819515144')
	#objects[0].session_id= data['Details']
	student=OTP(session_id=data['Details'],otp=766461,number = '9819515144')
	student.save()
	return submit_otp(request)

	#print(objects[0].session_id)
	#print(data)




def submit_otp(request):	
	if request.method=='POST':
		otp_form = OTPForm(request.POST)
		if otp_form.is_valid():
			cd = otp_form.cleaned_data
			if(cd['otp']!='766461'):
				return render(request,'verify_wrong.html')
			else:
				return render(request,'verify_result.html')		

			#objects = OTP.objects.all()
			#objects[0].otp = cd['otp']
			#print (objects[0].otp)
			



			#return render(request,'submit_otp.html',{'form':otp_form})

	else:
		otp_form = OTPForm()
	return render(request,'verify_otp.html',{'form':otp_form})


def verify_otp(request):
	ob = OTP.objects.all()
	a=ob[0]
	

	'''

	for obj in ob:
		if obj['session_id']!='Request Rejected - Spam Filter':
			a.append(obj)
	print (a )'''
	conn = http.client.HTTPConnection("2factor.in")
	payload = "{}"
	conn.request("GET", "/API/V1/643743b1-1698-11e7-9462-00163ef91450/SMS/VERIFY/"+a.session_id+"/766461", payload)
	res = conn.getresponse()
	data = res.read()	
	print(data.decode("utf-8"))

	return render(request,'verify_result.html')


def login(request):
	return render(request,'login.html')


def index(request):
	return render(request,'index.html')


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

def index2(request):
	return render(request,'index2.html')	
