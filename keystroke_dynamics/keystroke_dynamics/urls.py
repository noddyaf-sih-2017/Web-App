"""keystroke_dynamics URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from app import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^index/', views.index),
    url(r'^register/', views.register),
    url(r'^train/', views.train),
    url(r'^send_otp/', views.send_otp),
    url(r'^verify_otp/', views.verify_otp),
    url(r'^table/', views.table),
    url(r'^contacts/', views.contacts),
    url(r'^projects/', views.projects),
    url(r'^project_detail/', views.project_detail),
    url(r'^profile/', views.profile),
    url(r'^submit/', views.submit_otp),
    url(r'^otp/', views.otp),
    url(r'^index2/', views.index2),
    url(r'^login/$', views.login, name='login'),
    url(r'cont_auth/$', views.cont_auth, name='cont_auth'),
    url(r'^send_details/$', views.send_details, name='send_details'),
    url(r'^send_login_details/$', views.send_login_details, name='send_login_details'),
]
