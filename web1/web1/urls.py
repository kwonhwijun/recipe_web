"""web1 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
import polls.views
import account.views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('polls/', include('polls.urls')),
    path('', polls.views.home, name='home'),
    path('home/', polls.views.home, name='home'),
    path('input/', polls.views.input, name='input'),
    path('result/', polls.views.input_result, name = 'input_result'),
    path('input_ajax/', polls.views.input_ajax, name='input_ajax'),
    path('login/', account.views.login, name='login'),
    path('signup_ajax/', account.views.signup_ajax, name='signup_ajax'),
    path('login_ajax/', account.views.login_ajax, name='login_ajax'),
    path('logout/', account.views.logout, name='logout'),
    path('delete/', account.views.delete, name='delete'),
    path('profile/',account.views.profile, name='profile'),
    path('edit_profile/', account.views.edit_profile, name='edit_profile'),
    path('auto_complete/', polls.views.auto_complete, name='auto_complete'),
    path('select_pill/', polls.views.select_pill, name='select_pill'),
    path('recipe/', polls.views.input_recipe, name = 'input_recipe'),
    path('recipe_ajax/', polls.views.recipe_ajax, name='recipe_ajax'),
    path('load_ajax/', polls.views.load_ajax, name='load_ajax')
]
