from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import oracledb as od
import requests
from bs4 import BeautifulSoup
import json
from konlpy.tag import Okt
import re
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity as cosine_similarity1

from tqdm import tqdm
import numpy as np


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
        if not disease and not pill:
            return
        if not disease.replace(" ", "") and not pill.replace(" ", ""):
            return
        
        if not disease or not disease.replace(" ", ""):
            disease = ' '
        if not pill or not pill.replace(" ", ""):
            pill = ' '
            
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
    pill = request.POST.getlist('pill')
        
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
        # div를 누르면 name로 저장된 recipetitle값을 받아옴
        recipetitle = request.POST.get('recipetitle', '') 
        
        # recipe_step과 ingrdients를 받기위해 조회
        query1 = f"select recipe_step from final_recipe where recipe_title = \'{recipetitle}\'"
        query2 = f"select recipe_ingredients from final_recipe where recipe_title = \'{recipetitle}\'"
        
        # db select함수 실행
        data1 = connection(query1)
        data2 = connection(query2)
        
    # 저장된 값을 ajax 데이터 형식으로 변경    
    response_data = {'recipetitle': recipetitle, 'recipe_step': data1, 'recipe_ingre': data2}
    return JsonResponse(response_data)

# /recipe - 상호작용 출력
@csrf_exempt
def interaction_ajax(request):
    # /recipe에서 입력되어 있는 질병, 약 정보를 받아옴
    inter_disease = request.POST.get('disease','')
    pill_json = request.POST.get('pillvalue','')
    inter_pill = json.loads(pill_json) # JSON.stringify로 보냈기 때문에 json문자열 -> python 문자열로 변환
    
    # 입력받은 질병명으로 db에서 조회
    query1 = f"select 권장식재료, 주의식재료 from disease_table where 질병명 = \'{inter_disease}\'"
    disease = connection(query1)
    
    # 다제약물로 받을 수 있기 때문에 for문으로 여러번 조회 후 append로 저장
    result_list = []
    for pill in inter_pill:
        query2 = f"select MERGED_CONTENTS from pill_table where DNAME = \'{pill}\'"
        pill_result = connection(query2)
        result_list.append(pill_result)
        
    good_ingre = []
    bad_ingre = []
     
    # 권장식재료, 주의식재료 2 컬럼으로 받아서 첫번째 컬럼은 good_ingre 리스트에, 두번째컬럼은 bad_ingre
    for row in disease:
        good_ingre.append(row[0])
        bad_ingre.append(row[1])
    
    # ajax 데이터 형식으로 변환
    response_data = {
        'result1': {
            'good_ingre' : good_ingre,
            'bad_ingre' : bad_ingre
        },
        'result2': result_list
    }
    return JsonResponse(response_data)
    
    

#-------------------------------------------------------------------------------------------------------#
# 레시피 추천
# /recipe - 레시피와 버튼 value(분류)를 가져와 쿼리문 실행
@csrf_exempt
def output_test(request):    
    pill_json = request.POST.get('pillvalue','')
    # pill_name = request.POST.getlist('pillvalue[]', '')
    disease_name = request.POST.get('diseasevalue','')
    btn_value = request.POST.get('btnvalue', '').strip()
    pill_name = json.loads(pill_json)
  
  
    # 받아온값 리스트 형식으로 변환
    disease_name = [disease_name]
    
    with open(r'polls/data/recipe/recipe.pickle', 'rb') as file:
        recipe_dict = pickle.load(file)

    with open(r'polls/data/recipe/ingre.pickle', 'rb') as file:
        ingre_dict = pickle.load(file)
        
    # with open(r'polls/data/recipe/nutri_vec_150.pickle', 'rb') as file:
    #     nutri_dict = pickle.load(file)

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
    
    per_page = 9  # 한 페이지에 표시할 레시피 수

    # 페이지 번호 가져오기
    page = int(request.POST.get('page', 1))

    # 페이징 처리를 위해 시작 인덱스와 종료 인덱스 계산
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    
    results=[]
    for recipe in output:
        query = "SELECT recipe_title FROM final_recipe WHERE recipe_title = :recipe AND recipe_category_type = :btn_value"
        exe = conn.cursor()
        exe.execute(query, {'recipe': recipe, 'btn_value': btn_value})
        data = exe.fetchone()
        exe.close()
        if data:
            results.append(data[0])

    output_page = results[start_index:end_index]
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
    print(result)
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
    return JsonResponse({'image_urls': image_urls, 'page': page, 'has_next': len(output_page) == per_page})

# /rnn 식재료 바꿨을때 레시피 출력
        # 식재료 RNN 모델 정의
        
class RNNNoun(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNNoun, self).__init__()
        self.hidden_size = hidden_size
        self.dense = nn.Linear(input_size, hidden_size)  # Dense layer
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.dense(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out1 = out[:, -1, :]
        out2 = self.fc(out1)
        out3 = self.softmax(out2)
        return out1, out2, out3

@csrf_exempt
def rnn_ajax(request):
    if request.method == 'POST':
        recipe_step = request.POST.get('recipe_step','')
        before_ingre = request.POST.get('before', '')
        after_ingre = request.POST.get('after', '')
        
        new_input_data = recipe_step.replace(before_ingre, after_ingre)
        
        with open(r'polls/data/rnn/ingre.pickle', 'rb') as file:
            test = pickle.load(file)
        with open(r'polls/data/rnn/unique_stems.pkl', 'rb') as file:
            unique_stems = pickle.load(file)
        with open(r'polls/data/rnn/recipe_tensor_list.pkl', 'rb') as file:
            recipe_tensor_list = pickle.load(file)
        with open(r'polls/data/rnn/recipe_step_dict.pkl', 'rb') as f:
            df1 = pickle.load(f)
        
        # Okt 형태소 분석기 초기화
        okt = Okt()

        # 동사와 명사를 추출하여 어간으로 변환하는 함수
        def extract_verb_noun(text, food_vec):
            vn = []
            # 특수문자 및 숫자 제거
            text = re.sub('[^가-힣\s]', '', text)
            # 형태소 분석
            morphs = okt.pos(text)
            # 명사와 동사 추출

            for word, pos in morphs:
                if pos.startswith('N') and word in test:  # 명사일 경우
                    vn.append(word)
                elif pos.startswith('V'):  # 동사일 경우
                    # 동사 그대로 저장
                    verb_stem = okt.pos(word, stem=True)[0][0]
                    if verb_stem in unique_stems:
                        vn.append(verb_stem)
            return vn

        # 데이터에 대해 함수 적용하여 리스트에 추가
        verb_noun = extract_verb_noun(new_input_data, test) # <- test = 레시피벡터, recipe_step = 바뀐 레시피 
        # 명사, 동사만 남겨진 레시피 5개의 토큰으로 옮겨가면서 묶기
        slice_num = 5
        split_lists = [verb_noun[i:i+slice_num] for i in range(len(verb_noun)-slice_num+1)]
        
        # 단어 사전의 크기와 임베딩 차원을 정의합니다.
        vocab_size = len(unique_stems)  # 단어 사전의 크기
        embedding_dim = 100  # 임베딩 차원

        # 임베딩 레이어 초기화
        embedding = nn.Embedding(vocab_size, embedding_dim)
        dim = 100
        zero_array = np.zeros(dim)

        recipe_vec = []
        for i, sliced_recipe in enumerate(tqdm(split_lists)):
            sliced_vec = []
            for token in sliced_recipe:
                if token in unique_stems:
                    index = unique_stems.index(token)
                    word_index = torch.LongTensor([index])
                    word_embed = embedding(word_index)
                    word_embed = word_embed.squeeze(0)
                    sliced_vec.append(word_embed.detach().numpy())
                    
                elif token in test:
                    food_token = test[token]
                    sliced_vec.append(food_token)
                    
                elif token == '':
                    sliced_vec.append(zero_array)

            recipe_vec.append(sliced_vec)
            
        checkpoint = torch.load(r'polls/data/rnn/model_checkpoint1.pth')
        # 불러온 모델의 상태와 옵티마이저 상태 적용 
        model_noun = RNNNoun(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['output_size'])
        model_noun = model_noun.float()

        optimizer_noun = torch.optim.Adam(model_noun.parameters(), lr=0.001)

        # 불러온 모델의 상태와 옵티마이저 상태 적용
        model_noun.load_state_dict(checkpoint['model_state_dict'])
        optimizer_noun.load_state_dict(checkpoint['optimizer_state_dict'])
        

                    
        # 하이퍼파라미터 설정
        # input_size_noun = checkpoint['input_size'] # 입력 크기
        # hidden_size_noun =  checkpoint['hidden_size']  # 은닉 상태 크기
        # output_size_noun = checkpoint['output_size']  # 출력 크기

        sliced_recipe_tensor = []
        for sliced_recipe in recipe_vec:
            sliced_recipe = torch.tensor(sliced_recipe).unsqueeze(0)
            sliced_recipe = sliced_recipe.float()
            sliced_recipe = sliced_recipe.squeeze(-1)
            out1, out2, out3 = model_noun(sliced_recipe)
            sliced_recipe_tensor.append(out1)

            
        mean_tensor = torch.mean(torch.stack(sliced_recipe_tensor), dim=0)
        # 손실 함수와 옵티마이저 설정
        # criterion = nn.CrossEntropyLoss()
        reference_tensor = mean_tensor  # 기준이 되는 텐서
        
        # 각 텐서들 간의 코사인 유사도 계산
        similarities = []
        for i, tensor in enumerate(recipe_tensor_list):
            sim = cosine_similarity1(reference_tensor, tensor, dim = 1)
            similarities.append((i, sim.item()))
        # 두 번째 값에 따라 튜플 정렬
        sorted_data = sorted(similarities, key=lambda x: x[1], reverse=True)

        # 결과 출력
        result_item = []
        for item in sorted_data:
            result_item.append(item[0])
        
        keys_list = list(df1.keys())
        values_list = list(df1.values())

        recipe_dict = [{'title': keys_list[i], 'recipe': values_list[i]} for i in result_item]
        print(recipe_dict)
        return JsonResponse(recipe_dict, safe= False)
      

# /detail -> /rnn form태그로 변경 전, 후 식재료 입력정보 보내기
def rnn_page(request):
    bf_ingre = request.POST.get('before', '')
    af_ingre = request.POST.get('after', '')
    return render(request, 'polls/rnn.html', {'bf_ingre': bf_ingre, 'af_ingre': af_ingre})