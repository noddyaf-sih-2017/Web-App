from django.db import models

class OTP(models.Model):
	number = models.CharField(max_length=10,blank = True, null = True, default = None)
	session_id = models.CharField(blank = True, null = True, default = None)
	otp = models.CharField(blank = True, null = True, default = None)
# Create your models here.
