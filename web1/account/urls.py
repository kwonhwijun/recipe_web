from django.urls import path

from . import views

app_name = 'account'

urlpatterns = [
    path('login/', views.login, name='login'),
    path('signup_ajax/', views.signup_ajax, name='signup_ajax'),
    path('login_ajax/', views.login_ajax, name='login_ajax'),
    path('logout/', views.logout, name='logout'),
    path('delete/', views.delete, name='delete'),
    path('profile/',views.profile, name='profile'),
    path('edit_profile/', views.edit_profile, name='edit_profile'),
]