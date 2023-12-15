from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.http import JsonResponse
import oracledb as od
import requests
from bs4 import BeautifulSoup
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from . import recipe
from . import db
from . import svd

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity



# db connection(select문)
def connection(query):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
    conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
    exe = conn.cursor()
    exe.execute(query)
    result = exe.fetchall()
    exe.close()
    return result 

# db connection(insert, delete, update문)
def connection_idu(query):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
    conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
    exe = conn.cursor()
    exe.execute(query)   
    conn.commit()
    exe.close()   

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
    if request.method == 'POST':
        input_value = request.POST.get('input_value')
        
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
    return JsonResponse({'error': 'error'})

# /home
def home(request):
    return render(request, 'polls/home.html')

# /input
def input(request):
    return render(request, 'polls/input.html')


# /input - 유저 id, 질병명, 약이름 db에 insert
@csrf_exempt
def input_ajax(request):
    if request.method == 'POST':
        disease = request.POST.get('disease', '')
        pill = request.POST.get('pill', '')
        user_id = request.POST.get('user_id', '')
        
        query = f'insert into user_input values (\'{user_id}\', \'{disease}\', \'{pill}\')'  
              
        connection_idu(query)
        
        response_data = {
            'disease': disease,
            'pill': pill,
        }
    return JsonResponse(response_data)


# /input - 페이지 로딩되었을때 입력했던 질병, 약정보 출력하는 ajax
@csrf_exempt
def load_ajax(request):
    user_id = request.POST.get('user_id', '')
    query = f'''
            SELECT disease, pill
            FROM user_input
            WHERE userid = \'{user_id}\'
            ''' 
    datas = connection(query)
    result = []
    for data in datas:
        row = {
            'disease': data[0],
            'pill': data[1]
        }
        result.append(row)

    return JsonResponse({'results': result})


# /input - x버튼 누르면 해당 질병명, 약이름 행 찾아서 db에서 삭제
@csrf_exempt
def delete_info(request):
    user_id = request.POST.get('user_id', '')
    disease = request.POST.get('disease', '')
    pill = request.POST.get('pill','')

    if disease == 'null':
        query = f'''
                delete from user_input
                where userid =  \'{user_id}\' and disease is null and pill = \'{pill}\'
                '''
    elif pill == 'null':
        query = f'''
                delete from user_input
                where userid =  \'{user_id}\' and disease=\'{disease}\' and pill is null
                '''
    else:
        query = f'''
                delete from user_input
                where userid =  \'{user_id}\' and disease=\'{disease}\' and pill = \'{pill}\'
                '''
    connection_idu(query)

    response_data = {
            'user_id' : user_id,
            'disease': disease,
            'pill': pill,
        }
    return JsonResponse(response_data)


# /input - form으로 질병명, 약이름 보내기
def input_result(request):
    disease = request.POST.get('disease','')
    pill = request.POST.get('pill','')
    return render(request, 'polls/input_result.html', {'disease': disease, 'pill': pill})

# /input - form태그 >> /recipe 페이지
def input_recipe(request):
    # 레시피 input페이지에서 recipe페이지로 보내기
    recipe = request.POST.get('recipe','')
    
    # recipe페이지에서 버튼 생성할 value
    query = f'select recipe_category_type from final_recipe group by RECIPE_CATEGORY_TYPE'
    datas = connection(query)

    tests = []
    for data in datas:
        row = {'recipe': data}
        tests.append(row)
    
    return render(request, 'polls/recipe.html', {'tests': tests, 'recipe': recipe})

# /input - 약이름 자동완성 ajax
@csrf_exempt
def auto_complete_pill(request):
    if request.method == 'POST':        
        input_value = request.POST.get('input_value')
        
        od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
        conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
        
        query = (
            "SELECT dname FROM med_test "
            "WHERE UPPER(dname) LIKE UPPER(:input_value)||'%' "
                )
        with conn.cursor() as cursor:
            cursor.execute(query, input_value=f'%{input_value}%')
            datas = [row[0] for row in cursor.fetchmany(30)] # 30개만 출력

        return JsonResponse(datas, safe=False)
    
    
# /input - 질병명 자동완성 ajax
@csrf_exempt
def auto_complete_disease(request):
    if request.method == 'POST':        
        input_value = request.POST.get('input_value')
        
        od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
        conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
        
        query = (
            "SELECT 질병명 FROM disease_table "
            "WHERE UPPER(질병명) LIKE UPPER(:input_value)||'%' "
                )
        with conn.cursor() as cursor:
            cursor.execute(query, input_value=f'%{input_value}%')
            datas = [row[0] for row in cursor.fetchmany(30)] # 30개만 출력

        return JsonResponse(datas, safe=False)
    
# /recipe - 레시피와 버튼 value(분류)를 가져와 쿼리문 실행
@csrf_exempt
def recipe_ajax(request):
    if request.method == 'POST':
        recvalue = request.POST.get('recvalue', '').strip() #공백제거
        btnvalue = request.POST.get('btnvalue', '').strip()

        page = int(request.POST.get('page', 1))

        # 페이지당 아이템 수
        items_per_page = 10

        # 쿼리 조건에 페이징을 적용
        query = f'''
            SELECT recipe_title, recipe_url
            FROM (
                SELECT recipe_title, recipe_url, ROWNUM AS rnum
                FROM final_recipe
                WHERE recipe_title LIKE :recvalue AND RECIPE_CATEGORY_TYPE = :btnvalue
            )
            WHERE rnum > :start_row AND rnum <= :end_row
        '''

        
        
        od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
        conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
        exe = conn.cursor()
        exe.execute(query, {'recvalue': f'%{recvalue}%', 'btnvalue': btnvalue, 'start_row': (page - 1) * items_per_page, 'end_row': page * items_per_page})
        datas = exe.fetchall()
        
        result = []
        columns = [desc[0] for desc in exe.description]

        for data in datas:
            row = {columns[i]: data[i] for i in range(len(columns))}
            result.append(row)

        # 이미지 URL을 저장할 리스트
        image_urls = []

        for recipe in result:
            # 각 레시피의 URL에서 이미지를 가져옴
            response = requests.get(recipe.get('RECIPE_URL', ''))
            soup = BeautifulSoup(response.content, 'html.parser')
            img_tag = soup.find('img', id='main_thumbs')

            if img_tag:
                image_url = img_tag['src']
                image_urls.append({'recipe': recipe.get('RECIPE_TITLE', ''), 'image_url': image_url})
                
        conn.close()
        return JsonResponse({'image_urls': image_urls})

    return JsonResponse({'error': 'Invalid request'})

# test페이지
def testpage(request):
    return render(request, 'polls/testpage.html')

# /testpage > /output - 인덱스 입력, 식재료 제거 전 후 5개씩 출력
def output_test(request):
    matrix = pd.read_csv(r'polls/data/food_matrix_2001.csv')
    matrix = matrix.loc[matrix.recipe_title.notna()].copy() 
    matrix.index = range(len(matrix)) 
    
    recipe_index = int(request.POST.get('recipe_index',''))
    remove_ingre = request.POST.get('remove_ingre','')

    def draw_recipe_recommend(matrix, rec_num, ingd_name):
        rec_title, ingd_list, rec_vec, ingd_vec = svd.matrix_decomposition(matrix)

        ingd_idx = ingd_list.index(ingd_name) # 삼겹살 -> 삼겹살의 index
        target_inge = ingd_vec[ingd_idx] # 삼결살의 100차원 벡터


        myfood = rec_vec[rec_num] # 내가 원하는 음식의 벡터                          
        myfood_new  = myfood - 1* target_inge

        sim_before = cosine_similarity([myfood], rec_vec)[0] # 식재료 제거하기 전
        sim_after = cosine_similarity([myfood_new], rec_vec)[0] #식재료 제거한 후
        # 차원 맞춰주기
        sim_idx =  np.argsort(sim_before)[::-1][:5] # 기존 레시피와 유사한 레시피
        recommend_idx = np.argsort(sim_after)[::-1][:5] # 식재료 정보를 반영한 레시피

        recommend_list_before =  [rec_title[i] for i in sim_idx]
        recommend_list_after = [rec_title[i] for i in recommend_idx]
        print(f"선택 음식 : {rec_title[rec_num]} - 먹으면 안돼는 식재료 : {ingd_name}")
        return recommend_list_before, recommend_list_after 
        
    output = draw_recipe_recommend(matrix, recipe_index, remove_ingre)
    return render(request, 'polls/output.html', {'output': output})



# # 유사도 5개출력 테스트
# def output_test(request):
     
#     recipe_title = int(request.POST.get('recipe_title',''))
#     matrix = pd.read_csv(r'polls/data/test_matrix.csv')
#     title, rec_vec, ingre_vec = svd.matrix_decomposition(matrix.iloc[:, 1:])
#     print(recipe_title)

#     def draw_TSNE(rec_vec, n = 0):
#         # recipe_vec 간의 유사도 구하기
#         sim_recipe = cosine_similarity(rec_vec , rec_vec)
#         # TSNE로 차원 축소하기
#         tsne = TSNE(n_components= 2)
#         reduced_vec = tsne.fit_transform(rec_vec)
#         def find_5idx(title, similarity, row_num = 0):
#             similarity_pd = pd.DataFrame(similarity, columns=title)
#             sim_list = similarity_pd.loc[row_num].sort_values(ascending= False)[1:6]
#             idx = []
#             for sim_title in list(sim_list.index) :
#                 idx.extend(list(title.index[title == sim_title]))
#             return idx
#         output = []
#         for i in find_5idx(title, sim_recipe, n):
#             output.append(title[i])
#         return output
#     output = draw_TSNE(rec_vec, recipe_title)
    
#     return render(request, 'polls/output.html', {'output': output})





