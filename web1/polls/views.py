from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import Http404
from .models import Question
import oracledb as od

def index(request):
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    context = {'latest_question_list': latest_question_list}
    return render(request, 'polls/index.html', context)

def detail(request, question_id):
    try:
        question = Question.objects.get(pk=question_id)
    except Question.DoesNotExist:
        raise Http404("Question does not exist")
    return render(request, 'polls/detail.html', {'question': question})

def selecttest(request):
    try:
        od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
        conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
        cursor = conn.cursor()
        
        query = 'select 대표식품명, "에너지(kcal)", "수분(g)" from nutrient_table'
        cursor.execute(query)
        datas = cursor.fetchall()
        
        conn.close()
        
        tests = []
        for data in datas:
            row = {'nutri': data[0],
                   'energy': data[1],
                   'water': data[2]}
            tests.append(row)
    except:
        conn.rollback()
        print("Failed selecting")
    
    return render(request, 'polls/test.html', {'tests': tests})