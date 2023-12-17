from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.http import JsonResponse
import oracledb as od
import requests
from bs4 import BeautifulSoup
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import os

from . import recipe
from . import db
from . import svd

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pickle

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
        items_per_page = 12

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

# /detail > 메뉴가 떴을 때 그 카드를 누르면 넘어가는 페이지 (레시피 상세설명: 식재료, 조리순서, 영양소 등 출력)
def detail(request):
    return render(request, 'polls/detail.html')

# div눌렀을때 레시피명을 가져와서 recipe_step을 /detail로 보내기
@csrf_exempt
def detail_ajax(request):
    if request.method == 'POST':
       recipetitle = request.POST.get('recipetitle', '')  
       query = f"select recipe_step from final_recipe where recipe_title = \'{recipetitle}\'"
       datas = connection(query)
       
       response_data = {'recipetitle': recipetitle, 'recipe_steps': datas}
    return JsonResponse(response_data)
    

#-------------------------------------------------------------------------------------------------------#
# 레시피 추천
def output_test(request):    
    pill_name = request.POST.get('pill','')
    disease_name = request.POST.get('disease','')
    
    # 받아온값 리스트 형식으로 변환
    pill_name = [pill_name]
    disease_name = [disease_name]
    
    with open(r'polls/data/recipe/recipe.pickle', 'rb') as file:
        recipe_dict = pickle.load(file)

    with open(r'polls/data/recipe/ingre.pickle', 'rb') as file:
        ingre_dict = pickle.load(file)

    with open(r'polls/data/recipe/category.pickle', 'rb') as file:
        category_dict = pickle.load(file)   

    rec_title = list(recipe_dict.keys()) # 키 리스트
    rec_vec = [recipe_dict[rec_title[i]] for i in range(len(recipe_dict))]
    
    # 2. 상호작용 딕셔너리 불러오기 
    hos = pd.read_csv(r'polls/data/med/disease_interaction.csv')
    dis_dict= {hos['질병명']: {'권장영양소' : hos['권장영양소'], '주의영양소' : hos['주의영양소'], '권장식재료' : hos['권장식재료'], '주의식재료' : hos['주의식재료']} for _, hos in hos.iterrows()}

    hos = pd.read_csv(r'polls/data/med/med_interaction.csv')
    med_dict= {hos['dname']: {'권장영양소' : hos['권장영양소'], '주의영양소' : hos['주의영양소'], '권장식재료' : hos['권장식재료'], '주의식재료' : hos['주의식재료']} for _, hos in hos.iterrows()}
    
    
    def my_interaction(med_list, dis_list):
        interaction = {'권장식재료' : [], '주의식재료' : [], '권장영양소' : [], '주의영양소' : []}
        # 약물 정보 반영
        for med in med_list :
            for inter in interaction.keys():
                if not pd.isna(med_dict[med][inter]):
                    interaction[inter].extend([value.strip() for value in med_dict[med][inter].split(',')])
        # 질병 정보 반영
        for dis in dis_list :
            for inter in interaction.keys() :
                if not pd.isna(dis_dict[dis][inter]):
                    interaction[inter].extend([value.strip() for value in dis_dict[dis][inter].split(',')])
        return interaction
    
    def recommend(my_med_list, my_dis_list, my_food):
        my_inter_dict = my_interaction(my_med_list, my_dis_list) # 입력한 질병, 약물을 상호작용 식재료, 영양소 불러오기
        my_food_vec = category_dict[my_food] # 내가 선택한 카테고리의 벡터
        my_food_vec_recommend = my_food_vec

        # 권장 : + 0.01 , 주의 -0.01 배로 해줌
        for food in my_inter_dict['권장식재료']:
            if food in ingre_dict :
                my_food_vec_recommend += 0.01* ingre_dict[food]
        for food in my_inter_dict['주의식재료']:
            if food in ingre_dict :
                my_food_vec_recommend -=  0.01 * ingre_dict[food]
        sim = cosine_similarity([my_food_vec_recommend], rec_vec)[0]
        recommend_idx = np.argsort(sim)[::-1][:100]

        return [rec_title[i] for i in recommend_idx]
    # 수정해야될 부분
    output = recommend(pill_name, disease_name, '찌개')    
    return render(request, 'polls/output.html', {'output': output})
