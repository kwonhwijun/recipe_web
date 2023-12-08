from django.urls import path

from . import views

app_name = 'polls'

urlpatterns = [
    path('', views.index, name='index'),
    path('<int:question_id>/', views.detail, name='detail'),
    path('test/', views.test, name='test'),
    path('selecttest/', views.selecttest, name='selecttest'),

]