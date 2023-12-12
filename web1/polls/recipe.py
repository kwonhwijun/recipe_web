import oracledb as od
import pandas as pd
import numpy as np
from . import recipe
from tqdm import tqdm
import ast
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import datetime
import warnings

#0. 데이터 불러오기
def load_recipe(n =1000):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12") # db connection
    conn = od.connect(user = config.DB_CONFIG['user'], password = config.DB_CONFIG['password'],  dsn = config.DB_CONFIG['dsn'])
    exe = conn.cursor()
    exe.execute(f'select * from (select * from recipe_table order by row_cnt asc) where row_cnt <= {n}')
    row = exe.fetchall() # row 불러오기
    column_name = exe.description # column 불러오기
    columns=[]
    for i in column_name:
        columns.append(i[0])
    result = pd.DataFrame(row, columns=columns) # row, column을 pandas DataFrame으로 나타내기
    result.rename(mapper=str.lower, axis='columns', inplace=True)
    conn.close()
    return result

def load_recipe_tiny(n=100):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12") # DB 연결
    conn = od.connect(user=config.DB_CONFIG['user'], password=config.DB_CONFIG['password'], dsn=config.DB_CONFIG['dsn'])
    exe = conn.cursor()
    exe.execute(f'SELECT * FROM (SELECT * FROM recipe_table ORDER BY row_cnt ASC) WHERE row_cnt <= {n}')
    result = pd.DataFrame(exe.fetchall(), columns=[col[0].lower() for col in exe.description])  # row와 column 이름을 가져와 DataFrame 생성
    conn.close() #실험 # 수정
    return result

# query문 직접 작성해서 select 할때 사용
def select_table(query):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12") # db connection
    conn = od.connect(user = config.DB_CONFIG['user'], password = config.DB_CONFIG['password'],  dsn = config.DB_CONFIG['dsn'])
    exe = conn.cursor()
    exe.execute(query)
    row = exe.fetchall() # row 불러오기
    column_name = exe.description # column 불러오기
    columns=[]
    for i in column_name:
        columns.append(i[0])
    result = pd.DataFrame(row, columns=columns) # row, column을 pandas DataFrame으로 나타내기
    result.rename(mapper=str.lower, axis='columns', inplace=True)
    conn.close()
    return result

def recipe_preprocessing(raw):
    data = raw.loc[raw['recipe_ingredients'].notnull()].copy()  # None 값 제거
    def clean_ingredients(ingredients):
        if ingredients is not None:
            ingredients = ingredients.replace('\\ufeff', '').replace('\\u200b', '')
        return ingredients
    
    # recipe_ingredinents가 비어있지 않은 행만 남기기
    def not_empty_ingredients(row):
        return row['recipe_ingredients'].strip() != '{}' 

    data["recipe_ingredients"] = data["recipe_ingredients"].apply(clean_ingredients)
    data = data[data.apply(not_empty_ingredients, axis=1)]
    result = data[['recipe_title', 'recipe_ingredients']].copy()

    title_idx = result[result['recipe_title'].isnull()].index # title이 null값인 행 인덱스 찾기
    del_idx = result[result['recipe_ingredients'].str.startswith('소시지')].index #소시지~ 로 시작해서 오류 일으키는 행 인덱스 찾기
    result.drop(del_idx, inplace=True) # 오류 일으키는 행 제거
    result.drop(title_idx, inplace=True) # title null값인 행 제거
    result = result.drop_duplicates() # 중복 제거

    return result

#1. 식재료 단위 별로 쪼개기
def split_ingredient(data):
    num_ingredients = 74

    list = [[f'ingredient{i}', f'quantity{i}', f'unit{i}'] for i in range(1, num_ingredients + 1)]
    column_names = []
    for i in list :
        column_names.extend(i)

    empty_columns = pd.DataFrame(columns=column_names)
    data = pd.concat([data, empty_columns], axis=1)

    non_matching_items = {} # 패턴과 일치하지 않는 데이터를 저장할 딕셔너리

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]): #tqdm으로 진행상황 확인
        if row['recipe_ingredients']:
            ingredients_dict = ast.literal_eval(row["recipe_ingredients"]) #딕셔너리 형태로 저장된 recipe_ingredients 불러오기
            ingredient_count = 1

            for items in ingredients_dict.values():
                for item in items:
                    match = re.match(r'([가-힣a-zA-Z]+(\([가-힣a-zA-Z]+\))?|\d+[가-힣a-zA-Z]*|\([가-힣a-zA-Z]+\)[가-힣a-zA-Z]+)([\d.+/~-]*)([가-힣a-zA-Z]+|약간|조금)?', item)

                    if match:
                        ingredient, _, quantity, unit = match.groups()

                        data.at[idx, f'ingredient{ingredient_count}'] = ingredient
                        data.at[idx, f'quantity{ingredient_count}'] = quantity
                        data.at[idx, f'unit{ingredient_count}'] = unit

                        ingredient_count += 1
                    else:
                        non_matching_items[idx] = item

    data = data.drop([k for k, v in non_matching_items.items() if v != ''])

    #i가 75 이상인 경우 제거하는 조건문
    data = data.copy()

    columns_to_drop = []
    for i in range(data.shape[1]):
        if i >= 75:
            column_prefixes = [f'ingredient{i}', f'quantity{i}', f'unit{i}']
            columns_to_drop.extend(column_prefixes)

    # 실제로 데이터프레임에 존재하는 열만 삭제
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(existing_columns_to_drop, axis=1, inplace=True)

    

    return data

# 2. 식재료 종류 전처리 (돌리면 코랩 기준 약 9분 30초 정도 걸림)
def process_ingredient(dataframe):
    dataframe = dataframe.copy()
    def process_pattern(dataframe, pattern, replacement):
        for i in tqdm(range(1, 75)):
            col_name = f'ingredient{i}'
            unit_col_name = f'unit{i}'

            dataframe[unit_col_name] = np.where(dataframe[col_name].notna() & dataframe[col_name].str.contains(pattern, regex=True), replacement, dataframe[unit_col_name])

            dataframe[col_name] = dataframe[col_name].str.replace(pattern, '', regex=True)

        dataframe = dataframe.drop_duplicates()

        return dataframe

    # '약간', '적당량', '조금', '톡톡', '적당히' 패턴 처리
    dataframe = process_pattern(dataframe, r'약간', '약간')
    dataframe = process_pattern(dataframe, r'적당량', '적당량')
    dataframe = process_pattern(dataframe, r'적당히', '적당량')
    dataframe = process_pattern(dataframe, r'적당양', '적당량')
    dataframe = process_pattern(dataframe, r'조금.*', '조금')
    dataframe = process_pattern(dataframe, r'톡톡(톡)?', '톡톡')

    # 괄호 제거
    for i in tqdm(range(1, 75)):
        col_name = f'ingredient{i}'
        dataframe[col_name] = dataframe[col_name].str.replace(r'\([^)]*\)', '', regex=True)
        dataframe = dataframe.drop_duplicates() # 중복 제거

    return dataframe


# 식재료 2개 만 있는 레시피는 의미 없다. => EDA 
# 한번만 반영된 식재료는 제거 한다 (레시피도 같이) 
#   EX) 콩고전통요리 만들기: 콩고옥수수 : 1 => 제거.

# => 50번이상 등장한 식재료 목록 -> 각 행에 대해서 조회? 






# 4. Matrix 변환
def recipe_food_matrix(data):
    data.index = range(len(data)) # index 초기화

    def parse_quantity(quantity):
        if '~' in quantity:
            numbers = re.findall(r'\d+\.?\d*', quantity)  # 숫자들을 찾음
            numbers = [float(num) for num in numbers]  # 문자열을 실수로 변환
            return float(sum(numbers) / len(numbers))  # 평균 계산
        try:
            return float(quantity)  # 일반적인 경우, 숫자로 변환
        except ValueError:
            return float(1) # 비어있는 경우 1로 변환 
         
        
    # 단위를 g으로 : convert_unit_to_number('조금') = 10
    def convert_unit_to_number(unit):
        file_path = r"data\change.txt"
        unit_conversion = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            unit_conversion = {line.split()[0]: float(line.split()[1]) for line in file if line.split()[1].isdigit()}
        return unit_conversion.get(unit, 1)
    # all_ingredients: 모든 식재료 리스트

    ingredient_columns = data.filter(like='ingredient')
    if 'recipe_ingredients' in ingredient_columns.columns:
        data = data.drop(columns=['recipe_ingredients'])

    all_ingredients = set()
    if data.shape[1] > 200 :
        for i in range(1, 75):  
            all_ingredients.update(data[f'ingredient{i}'].dropna().unique())
    
    if data.shape[1] < 100 :
        for i in range(1, 26):  
            all_ingredients.update(data[f'ingredient{i}'].dropna().unique())

    # 레시피 식재료 Matrix 만들기 
    col_name = ['recipe_title'].append(list(all_ingredients))
    recipe_ingredients_df = pd.DataFrame(columns=col_name) # 

    # 레시피 하나씩 붙이기 
    recipe_rows = []
    for idx, row in tqdm(data.iterrows(), total = data.shape[0]) : # tqdm으로 진행상황 확인
        recipe_data = {ingredient: 0.0 for ingredient in all_ingredients}  # 모든 식재료를 None으로 초기화
        for i in range(1, 26):  
            ingredient = row[f'ingredient{i}']
            quantity = row[f'quantity{i}']
            unit = row[f'unit{i}']
            if pd.notna(ingredient) and pd.notna(quantity):
                quantity_float = parse_quantity(quantity)
                if quantity_float is not None:
                    unit_number = convert_unit_to_number(unit) if pd.notna(unit) else 1
                    recipe_data[ingredient] = quantity_float * unit_number
        recipe_rows.append(recipe_data)

    # 새로운 데이터프레임 생성 (모든 식재료를 열로 가짐)
    recipe_ingredients_df = pd.concat([pd.DataFrame([row]) for row in recipe_rows], ignore_index=True)
    recipe_ingredients_df = recipe_ingredients_df.astype('float64')
    recipe_ingredients_df['recipe_title'] = data['recipe_title']

    # RECIPE_TITLE 컬럼을 젤 앞으로
    recipe_ingredients_df = recipe_ingredients_df[['recipe_title'] + [col for col in recipe_ingredients_df.columns if col != 'recipe_title']]

    return recipe_ingredients_df

#---------------------------------------------------------------------------------------------------#
# 재료 쪼갠 후 레시피별 영양소 나오는 테이블

def recipe_nutri(new_recipe1, nutri_df):
    file_path = r"data\change.txt"
    unit_conversion = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        unit_conversion = {line.split()[0]: line.split()[1] for line in file if line.split()[1].isdigit()}


#-------------------- 여기서 부터 --------------------#
# # txt 파일 경로
# file_path = r"C:\Users\admin\OneDrive\바탕 화면\change2.txt"

# # 빈 리스트 초기화
# data = []

# # 텍스트 파일 읽기
# with open(file_path, 'r', encoding='utf-8') as file:
#     lines = file.readlines()

# # 각 줄에 대해 처리
# for line in lines:
#     # 공백을 기준으로 열과 값 분리
#     parts = line.split()    
#     # 딕셔너리로 저장
#     row_data = {'ingredients': parts[0], 'unit': parts[1], 'value': parts[2]}    
#     # 리스트에 추가
#     data.append(row_data)

# df11  = pd.DataFrame(data)
# # dict으로 저장해서 속도 향상
# df11_dict = df11.set_index(['ingredients', 'unit']).to_dict()['value']

# for index, row in new_recipe1.iterrows():
#     for i in range(1, 25):
#         ingredient_col = f"ingredient{i}"
#         quantity_col = f"quantity{i}"
#         unit_col = f"unit{i}"

#         ingredient_value = row[ingredient_col] # ingredient{i} 행 데이터
#         unit_value = row[unit_col] # unit_col{i} 행 데이터

#         # df11_dict에서 일치하는 값을 찾아서 new_recipe1에 채우기
#         if (ingredient_value, unit_value) in df11_dict:
#             new_recipe1.at[index, unit_col] = df11_dict[(ingredient_value, unit_value)]
    #-------------------- 이부분까지 --------------------#


def recipe_nutri(new_recipe1, nutri_df):
    warnings.filterwarnings('ignore', category= UserWarning)
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

    #-------------------- 여기서 부터 --------------------#
    # txt 파일 경로 (딕셔너리 수정시 수정 필요함)
    file_path = r"data\change.txt"
    unit_conversion = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split()
            unit = line[0]
            value = line[1] if line[1].isdigit() else None
            unit_conversion[unit] = value

    # DataFrame 생성
    df11 = pd.DataFrame(list(unit_conversion.items()), columns=['unit', 'value'])

    # DataFrame을 딕셔너리로 변환
    df11_dict = df11.set_index('unit')['value'].to_dict()

    # 딕셔너리의 값을 숫자로 변환하여 새로운 딕셔너리 생성
    df_dict = {key: int(value) if value is not None else None for key, value in df11_dict.items()}
    df_dict

    # unit{i} 컬럼에 딕셔너리로 지정한 key : value값으로 치환
    for i in range(1, 15):
        column_name = f'unit{i}'
        if column_name in new_recipe1.columns:
            new_recipe1[column_name] = new_recipe1[column_name].apply(lambda x: df_dict.get(x) if pd.notna(x) and not re.match(r'\d+[^\d]*$', str(x)) else x)
    
    #-------------------- 이부분까지 --------------------#
    
    # new_recipe1에 recipe_title, ingredient{i}, quantity{i}, unit{i}만 저장
    new_recipe1 = new_recipe1[['recipe_title'] + [f'{name}{i}' for i in range(1, 14) for name in ['ingredient', 'quantity', 'unit']]]
 
    #계산을 위해 quantity의 타입변경 str => float
    for i in range(1, 14):
        try:
            new_recipe1.loc[:, f'quantity{i}'] = pd.to_numeric(new_recipe1[f'quantity{i}'], errors='coerce').astype('float16')
        except ValueError:
            new_recipe1.loc[:, f'quantity{i}'] = 0
    
    #mulit{i} 컬럼 생성 후 quantity * unit 값 대입
    for i in range(1,14):
        new_recipe1.loc[:, f'multi{i}'] = None
    for i in range(1, 14):
        new_recipe1.loc[:, f'multi{i}'] = new_recipe1.loc[:, f'quantity{i}'] * new_recipe1.loc[:, f'unit{i}']

        
    # quantity, unit 컬럼 전부 삭제
    for i in range(1,14):
        new_recipe1 = new_recipe1.drop(f'quantity{i}',axis = 1)
        new_recipe1 = new_recipe1.drop(f'unit{i}',axis = 1)
    
    # new_recipe1의 컬럼 재배열 (recipe_title ingredient1 multi1 ... 식으로)
    new_columns = [
        'recipe_title', 
        'ingredient1', 'multi1', 
        'ingredient2', 'multi2', 
        'ingredient3', 'multi3', 
        'ingredient4', 'multi4', 
        'ingredient5', 'multi5', 
        'ingredient6', 'multi6', 
        'ingredient7', 'multi7', 
        'ingredient8', 'multi8', 
        'ingredient9', 'multi9', 
        'ingredient10', 'multi10', 
        'ingredient11', 'multi11', 
        'ingredient12', 'multi12', 
        'ingredient13', 'multi13'
    ]
    new_recipe1 = new_recipe1[new_columns]
    
    # 단위 제거
    nutri_df =  nutri_df.apply(lambda x: x.str.extract(r'([\d.]+)', expand=False) if x.name not in ['nutrient'] else x)
    nutri_df 
    
    # 영양소 테이블의 컬럼명 변경
    nutri_df.rename(columns={'nutrient':'ingredient'}, inplace=True)
    
    # 타입변경 후 concat
    nutri_col_1 = nutri_df.iloc[:, :1]
    nutri_col_2 = nutri_df.iloc[:, 1:].astype('float64')
    nutri_df = pd.concat([nutri_col_1 , nutri_col_2], axis=1)
    
    # 영양소 테이블에서 컬럼명 추출 후 list에 담음
    nutrient_list = nutri_df.columns[1:].tolist()    
    
    for i in range(1, 14):  # ingredient1부터 ingredient13까지 처리
        ingredient_col = f'ingredient{i}'
        multi_col = f'multi{i}'
        
        # 필요한 컬럼만 추출하여 병합
        merged_df = pd.merge(new_recipe1[['recipe_title', ingredient_col, multi_col]],
                            nutri_df,
                            left_on=ingredient_col,
                            right_on='ingredient',
                            how='left')
        
        # 각 값에 대해 계산
        for index, row in merged_df.iterrows():
            if pd.notna(row['ingredient']) or str(row['ingredient']) in str(row[ingredient_col]):
                multiplier = row[multi_col] / 100 # row[index]로 변경가능
                for nutrient in nutrient_list:
                    new_recipe1.at[index, f'{nutrient}{i}'] = row[nutrient] * multiplier
            else:
                for nutrient in nutrient_list:
                    new_recipe1.at[index, f'{nutrient}{i}'] = None  # 또는 0 또는 다른 값으로 설정할 수 있음

    # NaN값 제거
    new_recipe1 = new_recipe1.dropna(subset=['recipe_title'])
    
    # ingredient, multi 컬럼 전부 삭제
    for i in range(1,14):
        new_recipe1 = new_recipe1.drop(f'ingredient{i}',axis = 1)
        new_recipe1 = new_recipe1.drop(f'multi{i}',axis = 1)
        
    # 총합 영양소 컬럼 생성
    nutrient_list1 = ['총합_' + nutrient for nutrient in nutrient_list]
    new_recipe1[nutrient_list1] = 0
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    
    # 각 컬럼당 sum값을 방금 만든 총합 ~ 컬럼에 각각 적용
    for nutrient in nutrient_list:
        new_recipe1[f'총합_{nutrient}'] = new_recipe1[[f'{nutrient}{i}' for i in range(1, 14)]].sum(axis=1)
    
    # 남길 컬럼만 선택
    columns_to_keep = ['recipe_title'] + [f'총합_{nutrient}' for nutrient in nutrient_list]    
    new_recipe1 = new_recipe1.loc[:, columns_to_keep]
    
    # 소수점 3자리까지만 표시
    new_recipe1 = round(new_recipe1, 3)
    
    return new_recipe1

# def split_ingredient 까지 진행한 df로 사용해야함. recipe_food_matrix 진행 x
# 예시 recipe_nutri(저장한 df명, 영영소 테이블 df명)
#---------------------------------------------------------------------------------------------------#

# 한번에 매트릭스까지 처리하는 함수
def load_split(n = 1000):
    raw = load_recipe(n)
    print("load completed")
    raw_processed = recipe_preprocessing(raw)
    print("Preprocessing completed")
    recipe = split_ingredient(raw_processed)
    print("Ingredient split completed")
    return recipe

# 한번에 레시피X식재료 매트릭스를 출력하는 함수
def load_matrix(n = 1000):
    raw = load_recipe(n)
    print("load completed")
    raw_processed = recipe_preprocessing(raw)
    print("Preprocessing completed")
    recipe = split_ingredient(raw_processed)
    print("Ingredient split completed")
    result = recipe_food_matrix(recipe)
    print("Matrix creation completed")

    now = datetime.datetime.now()
    format = "%b %d %H:%M"
    filename = now.strftime(format)
    result.to_csv("matrix/" + filename + ".csv")
    print("recipe X food matrix is saved with the name" + "matrix/"+filename)
    return result 

def not_matching(n=100):
    raw = load_recipe(n)
    raw_processed = recipe_preprocessing(raw)
    data = raw_processed.copy()
    for i in range(1, 21):
        data.loc[:, f'ingredient{i}'] = None
        data.loc[:, f'quantity{i}'] = None
        data.loc[:, f'unit{i}'] = None
   
    non_matching_items = {} # 패턴과 일치하지 않는 데이터를 저장할 딕셔너리

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]): #tqdm으로 진행상황 확인
        if row['recipe_ingredients']:
            ingredients_dict = ast.literal_eval(row["recipe_ingredients"]) #딕셔너리 형태로 저장된 recipe_ingredients 불러오기
            ingredient_count = 1
            for category, items in ingredients_dict.items(): #category : 재료, 양념재료, items: 사과1개, 돼지고기600g
                if items:  # 아이템이 존재하는 경우
                    for item in items:
                        match = re.match(r'([가-힣a-zA-Z]+(\([가-힣a-zA-Z]+\))?|\d+[가-힣a-zA-Z]*|\([가-힣a-zA-Z]+\)[가-힣a-zA-Z]+)([\d.+/~-]*)([가-힣a-zA-Z]+|약간|조금)?', item) # 정규식
                        if match:
                            pass
                        else:
                            # 패턴과 일치하지 않는 경우 딕셔너리에 추가
                            non_matching_items[idx] = item
        else:
            pass
    return non_matching_items




def recipe_preprocessing_tiny(raw):
    result = (
        raw.dropna(subset=['recipe_ingredients', 'recipe_title']) # NA 제거
        .assign(recipe_ingredients=lambda x: x['recipe_ingredients'].str.replace(r'\\ufeff|\\u200b', '', regex=True))
        .loc[lambda x: x['recipe_ingredients'].str.strip() != '{}'] # 빈 값(띄여쓰기) 제거
        .loc[lambda x: ~x['recipe_ingredients'].str.startswith('소시지')] # 소시지에서 문제 발생
        .loc[:, ['recipe_title', 'recipe_ingredients']]
        .drop_duplicates()
    )
    return result

def split_ingredient_tiny(data):
    num_ingredients = 74

    # 식재료 이름, 양, 단위 칼럼 생성
    # 모든 재료에 대한 열을 한 번에 생성
    ingredient_columns = [f'ingredient{i}' for i in range(1, num_ingredients + 1)]
    quantity_columns = [f'quantity{i}' for i in range(1, num_ingredients + 1)]
    unit_columns = [f'unit{i}' for i in range(1, num_ingredients + 1)]

    # 새로운 DataFrame을 생성하여 모든 열을 한 번에 추가합니다.
    new_columns = ingredient_columns + quantity_columns + unit_columns
    data = pd.concat([data, pd.DataFrame(columns=new_columns)], axis=1)


    non_matching_items = {} # 패턴과 일치하지 않는 데이터를 저장할 딕셔너리

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]): #tqdm으로 진행상황 확인
        if row['recipe_ingredients']:
            ingredients_dict = ast.literal_eval(row["recipe_ingredients"]) #딕셔너리 형태로 저장된 recipe_ingredients 불러오기
            ingredient_count = 1
            
            for items in ingredients_dict.values():
                if ingredient_count <= 75 : # 개별 레시피의 식재료 75개까지만
                    for item in items:
                        match = re.match(r'([가-힣a-zA-Z]+(\([가-힣a-zA-Z]+\))?|\d+[가-힣a-zA-Z]*|\([가-힣a-zA-Z]+\)[가-힣a-zA-Z]+)([\d.+/~-]*)([가-힣a-zA-Z]+|약간|조금)?', item)
                        if match:
                            ingredient, _, quantity, unit = match.groups()

                            data.at[idx, f'ingredient{ingredient_count}'] = ingredient
                            data.at[idx, f'quantity{ingredient_count}'] = quantity
                            data.at[idx, f'unit{ingredient_count}'] = unit

                            ingredient_count += 1
                        else:
                            non_matching_items[idx] = item
                else : pass

    data = data.drop([k for k, v in non_matching_items.items() if v != ''])

    return data


def load_matrix_tiny(n = 1000):
    raw = load_recipe(n)
    print("load completed")
    raw_processed = recipe_preprocessing_tiny(raw)
    print("Preprocessing completed")
    recipe = split_ingredient_tiny(raw_processed)
    print("Ingredient split completed")
    recipe2 = process_ingredient(recipe)
    result = recipe_food_matrix(recipe2)
    print("Matrix creation completed")

    now = datetime.datetime.now()
    format = "%b %d %H:%M"
    filename = now.strftime(format)
    result.to_csv("matrix/" + filename + ".csv")
    print("recipe X food matrix is saved with the name" + "matrix/"+filename)
    return result 




