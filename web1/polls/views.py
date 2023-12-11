from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.http import JsonResponse
import oracledb as od

def connection(query):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
    conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data




def index(request):
    return render(request, 'polls/index.html')


@csrf_exempt
def test(request):
    return render(request, 'polls/test.html')

@csrf_exempt
def selecttest(request):
    # try:
    if request.method == 'POST':
        input_value = request.POST.get('input_value')


        # 검색 조건
        query = f'select nutrient, "에너지(Kcal)", "탄수화물", "단백질", "지방" from nutrient_data_table where nutrient like \'%{input_value}%\''
        datas = connection(query)

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

    # except Exception as e:
    #     conn.rollback()
    #     print(f"조회 중 오류 발생: {str(e)}")

    return JsonResponse({'error': 'error'})


def home(request):
    return render(request, 'polls/home.html')

def input(request):
    return render(request, 'polls/input.html')


# ajax test
@csrf_exempt
def input_ajax(request):
    if request.method == 'POST':
        disease = request.POST.get('disease', '')
        pill = request.POST.get('pill', '')
        response_data = {
            'disease': disease,
            'pill': pill,
        }
    return JsonResponse(response_data)

# form태그 test
def input_result(request):
    disease = request.POST.get('disease','')
    pill = request.POST.get('pill','')
    return render(request, 'polls/input_result.html', {'disease': disease, 'pill': pill})



# 자동완성 ajax
@csrf_exempt
def auto_complete(request):
    if request.method == 'POST':
        input_value = request.POST.get('input_value')
        
        od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
        conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
        query = (
            "SELECT dname FROM med_test "
            "WHERE UPPER(dname) LIKE UPPER(:input_value)||'%' "
            "ORDER BY CASE WHEN UPPER(dname) LIKE UPPER(:input_value)||'%' THEN 1 ELSE 2 END, dname"
                )
        with conn.cursor() as cursor:
            cursor.execute(query, input_value=f'%{input_value}%')
            datas = [row[0] for row in cursor.fetchmany(30)] # 30개만 출력

        return JsonResponse(datas, safe=False)

# 자동완성 후 클릭한 option ajax
@csrf_exempt
def select_pill(request):
    if request.method == 'POST':
        selected_option = request.POST.get('selected_option', '')
        result = {'result': selected_option}
        return JsonResponse(result)