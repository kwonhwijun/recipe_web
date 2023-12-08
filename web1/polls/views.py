from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.template import loader
from django.http import Http404
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login
from django.http import JsonResponse
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


@csrf_exempt
def test(request):
    return render(request, 'polls/test.html')

@csrf_exempt
def selecttest(request):
    try:
        if request.method == 'POST':
            input_value = request.POST.get('input_value')

            od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
            conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
            cursor = conn.cursor()

            # 검색 조건
            query = f'select nutrient, "에너지(Kcal)", "탄수화물", "단백질", "지방" from nutrient_data_table where nutrient like \'%{input_value}%\''
            cursor.execute(query)
            datas = cursor.fetchall()

            conn.close()

            tests = []
            for data in datas:
                row = {
                    'nutri': data[0],
                    'energy': data[1],
                    'carbo': data[2],
                    'protein': data[3],
                    'fat': data[4]
                }
                tests.append(row)

            return JsonResponse({'result': f'{input_value}', 'tests': tests})

    except Exception as e:
        conn.rollback()
        print(f"조회 중 오류 발생: {str(e)}")

    return JsonResponse({'error': 'error'})


def home(request):
    return render(request, 'polls/home.html')

def input(request):
    return render(request, 'polls/input.html')

def generic(request):
    return render(request, 'polls/generic.html')