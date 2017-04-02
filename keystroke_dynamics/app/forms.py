from app.models import OTP
from django.contrib.auth.models import User
from django import forms


class OTPForm(forms.ModelForm):
    otp = forms.CharField(
        widget=forms.TextInput(attrs={'class': "login__input pass", 'size':"40"}),
    )

    class Meta:
        model = OTP
        fields = ('otp',)
        labels = {
                'otp' :('One-Time-Password'),
                  }
                 




    


'''class UserForm(forms.ModelForm):
  required_css_class = 'required'

  password = forms.CharField (widget=forms.PasswordInput(attrs={'class': "input-lg", 'size':"40"}))
  first_name = forms.CharField(
        widget=forms.TextInput(attrs={'class': "input-lg", 'size':"40"}),
  )
  last_name = forms.CharField(
        widget=forms.TextInput(attrs={'class': "input-lg", 'size':"40"}),
  )
  username = forms.CharField(
        widget=forms.TextInput(attrs={'class': "input-lg", 'size':"40"}),
  )

  class Meta:
    model = User
    fields = ('first_name','last_name','username','password')
    labels = {
                          'first_name' :('First Name'),
                          'last_name' :('Last Name'),
                          'Username' :('Username'),
                          'password' :('Password'),
              }

class UserProfileForm(forms.ModelForm):
    required_css_class = 'required'

    number = forms.CharField(
        widget=forms.TextInput(
          attrs={'class': "input-lg", 'size':"40"}),
        required=False
    )
    class Meta:
        model = UserProfile
        fields = ('number')
        labels = {
                 'number':('Phone Number'),
                 }
  '''