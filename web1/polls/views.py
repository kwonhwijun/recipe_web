from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.http import JsonResponse
import oracledb as od

# db connection
def connection():
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
    conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
    return conn

# /index
def index(request):
    return render(request, 'polls/index.html')

# /polls/test
@csrf_exempt
def test(request):
    return render(request, 'polls/test.html')

# /polls/test-ajax
@csrf_exempt
def selecttest(request):
    # try:
    if request.method == 'POST':
        input_value = request.POST.get('input_value')


        # 검색 조건
        query = f'select nutrient, "에너지(Kcal)", "탄수화물", "단백질", "지방" from nutrient_data_table where nutrient like \'%{input_value}%\''
        exe = connection().cursor()
        exe.execute(query)
        datas = exe.fetchall()

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

# /home
def home(request):
    return render(request, 'polls/home.html')

# /input
def input(request):
    return render(request, 'polls/input.html')


# /input - ajax test
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

# /input - form태그 test
def input_result(request):
    disease = request.POST.get('disease','')
    pill = request.POST.get('pill','')
    return render(request, 'polls/input_result.html', {'disease': disease, 'pill': pill})

# /input - form태그 >> /recipe 페이지
def input_recipe(request):
    # 레시피 input페이지에서 recipe페이지로 보내기
    recipe = request.POST.get('recipe','')
    
    # recipe페이지에서 버튼 생성할 value
    query = f'select classification from tag_table group by CLASSIFICATION'

    exe = connection().cursor()
    exe.execute(query)
    datas = exe.fetchall()

    tests = []
    for data in datas:
        row = {'recipe': data}
        tests.append(row)
    
    return render(request, 'polls/recipe.html', {'tests': tests, 'recipe': recipe})

# /input - 자동완성 ajax
@csrf_exempt
def auto_complete(request):
    if request.method == 'POST':
        input_value = request.POST.get('input_value')
        conn = connection()
        query = (
            "SELECT dname FROM med_test "
            "WHERE UPPER(dname) LIKE UPPER(:input_value)||'%' "
                )
        with conn.cursor() as cursor:
            cursor.execute(query, input_value=f'%{input_value}%')
            datas = [row[0] for row in cursor.fetchmany(30)] # 30개만 출력

        return JsonResponse(datas, safe=False)

# /input - 자동완성 후 클릭한 option ajax
@csrf_exempt
def select_pill(request):
    if request.method == 'POST':
        selected_option = request.POST.get('selected_option', '')
        result = {'result': selected_option}
        return JsonResponse(result)
    
# 레시피와 버튼value를 가져와 쿼리문 실행
@csrf_exempt
def recipe_ajax(request):
    if request.method == 'POST':
        recvalue = request.POST.get('recvalue', '').strip() #공백제거
        btnvalue = request.POST.get('btnvalue', '').strip()
        query = f'''
                  SELECT recipe_title, recipe_url
                  FROM tag_table
                  WHERE recipe_title LIKE \'%{recvalue}%\' AND CLASSIFICATION = \'{btnvalue}\'
                  '''
        exe = connection().cursor()
        exe.execute(query)
        datas = exe.fetchall()
        
        result = []
        columns = [desc[0] for desc in exe.description]

        for data in datas:
            row = {columns[i]: data[i] for i in range(len(columns))}
            result.append(row)

        return JsonResponse({'results': result})

    