from django.conf.urls import url
from . import views
urlpatterns = [
	url(r'^login/$', views.login, name='login'),
	url(r'cont_auth/$', views.cont_auth, name='cont_auth'),
	url(r'^send_details/$', views.send_details, name='send_details'),
    url(r'$', views.index, name='index'),
]
