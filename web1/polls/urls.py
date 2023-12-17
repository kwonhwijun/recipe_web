from django.urls import path

from . import views

app_name = 'polls'

urlpatterns = [
    path('', views.home, name='home'),
    path('home/', views.home, name='home'),
    path('input/', views.input, name='input'),
    path('result/', views.input_result, name = 'input_result'),
    path('input_ajax/', views.input_ajax, name='input_ajax'),
    path('auto_complete_pill/', views.auto_complete_pill, name='auto_complete_pill'),
    path('auto_complete_disease/', views.auto_complete_disease, name='auto_complete_disease'),
    path('recipe/', views.input_result, name = 'input_result'),
    # path('recipe_ajax/', views.recipe_ajax, name='recipe_ajax'),
    path('load_ajax/', views.load_ajax, name='load_ajax'),
    path('testpage/', views.testpage, name='testpage'),
    path('output_test/', views.output_test, name='output_test'),
    path('delete_info/', views.delete_info, name='delete_info'),
    path('detail/', views.detail, name='detail'),
    path('detail_ajax/', views.detail_ajax, name='detail_ajax'),
]