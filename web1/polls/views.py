from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import oracledb as od
import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
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
    
    query = f'select recipe_category_type from final_recipe group by RECIPE_CATEGORY_TYPE'
    datas = connection(query)

    tests = []
    for data in datas:
        row = {'recipe': data}
        tests.append(row)
    
    return render(request, 'polls/recipe.html', {'tests': tests, 'disease': disease, 'pill': pill})

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

# /detail > 메뉴가 떴을 때 그 카드를 누르면 넘어가는 페이지 (레시피 상세설명: 식재료, 조리순서, 영양소 등 출력)
def detail(request):
    return render(request, 'polls/detail.html')

# div눌렀을때 레시피명을 가져와서 recipe_step을 /detail로 보내기
@csrf_exempt
def detail_ajax(request):
    if request.method == 'POST':
       recipetitle = request.POST.get('recipetitle', '') 
       query1 = f"select recipe_step from final_recipe where recipe_title = \'{recipetitle}\'"
       query2 = f"select recipe_ingredients from final_recipe where recipe_title = \'{recipetitle}\'"
       data1 = connection(query1)
       data2 = connection(query2)
    response_data = {'recipetitle': recipetitle, 'recipe_step': data1, 'recipe_ingre': data2}
    return JsonResponse(response_data)

# 상호작용 출력
@csrf_exempt
def interaction_ajax(request):
    inter_disease = request.POST.get('disease','')
    inter_pill = request.POST.get('pill', '')
    query1 = f"select 권장식재료, 주의식재료 from disease_table where 질병명 = \'{inter_disease}\'"
    query2 = f"select MERGED_CONTENTS from pill_table where DNAME = \'{inter_pill}\'"
    disease = connection(query1)
    pill = connection(query2)
    
    good_ingre = []
    bad_ingre = []

    for row in disease:
        good_ingre.append(row[0])  # Assuming '권장식재료' is the first column
        bad_ingre.append(row[1])   # Assuming '주의식재료' is the second column
    
    response_data = {
        'result1': {
            'good_ingre' : good_ingre,
            'bad_ingre' : bad_ingre
        },
        'result2': pill
    }
    return JsonResponse(response_data)
    
    

#-------------------------------------------------------------------------------------------------------#
# 레시피 추천
# /recipe - 레시피와 버튼 value(분류)를 가져와 쿼리문 실행
@csrf_exempt
def output_test(request):    
    pill_name = request.POST.get('pillvalue','')
    disease_name = request.POST.get('diseasevalue','')
    btn_value = request.POST.get('btnvalue', '').strip()
    
    # 받아온값 리스트 형식으로 변환
    pill_name = [pill_name]
    disease_name = [disease_name]
    print(pill_name, disease_name, btn_value)
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

    hos = pd.read_csv(r'polls/data/med/med_interaction_merged.csv')
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
        try : 
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
        except Exception as e:
            return []
    
    # 수정해야될 부분
    output = recommend(pill_name, disease_name, btn_value)
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
    conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
    
    per_page = 10  # 한 페이지에 표시할 레시피 수

    # 페이지 번호 가져오기
    page = int(request.POST.get('page', 1))

    # 페이징 처리를 위해 시작 인덱스와 종료 인덱스 계산
    start_index = (page - 1) * per_page
    end_index = start_index + per_page

    # 페이지에 해당하는 데이터 가져오기
    output_page = output[start_index:end_index]
    result = []
    
    for recipe in output_page:
        query = "SELECT recipe_title, recipe_url FROM final_recipe WHERE recipe_title = :recipe"
        exe = conn.cursor()
        exe.execute(query, {'recipe': recipe})
        data = exe.fetchone()
        exe.close()

        if data:
            result.append({
                'RECIPE_TITLE': data[0],
                'RECIPE_URL': data[1]
            })
    
    image_urls = []
    print(result)
    for recipe in result:
        # 각 레시피의 URL에서 이미지를 가져옴
        response = requests.get(recipe.get('RECIPE_URL', ''))
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tag = soup.find('img', id='main_thumbs')

        if img_tag:
            image_url = img_tag['src']
            image_urls.append({'recipe': recipe.get('RECIPE_TITLE', ''), 'image_url': image_url})

    conn.close()
    return JsonResponse({'image_urls': image_urls, 'page': page, 'has_next': len(output_page) == per_page})
