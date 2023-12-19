import oracledb as od
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
import ast
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import datetime
import warnings

# 0. 데이터 불러오기
def load_recipe(n=100):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12") # DB 연결
    conn = od.connect(user=config.DB_CONFIG['user'], password=config.DB_CONFIG['password'], dsn=config.DB_CONFIG['dsn'])
    exe = conn.cursor()
    exe.execute(f'SELECT * FROM (SELECT * FROM recipe_table ORDER BY row_cnt ASC) WHERE row_cnt <= {n}')
    result = pd.DataFrame(exe.fetchall(), columns=[col[0].lower() for col in exe.description])  # row와 column 이름을 가져와 DataFrame 생성
    conn.close()
    return result

def recipe_preprocessing(raw):
    data = raw.loc[raw['recipe_ingredients'].notnull()].copy()  # None 값 제거
    def clean_ingredients(ingredients):
        if ingredients is not None:
            ingredients = ingredients.replace('\\ufeff', '').replace('\\u200b', '') # 데이터 불러올 때 오류 나는 부분 제거
        return ingredients
    
    # recipe_ingredinents가 비어있지 않은 행만 남기기
    def not_empty_ingredients(row):
        return row['recipe_ingredients'].strip() != '{}' 

    data["recipe_ingredients"] = data["recipe_ingredients"].apply(clean_ingredients)
    data = data[data.apply(not_empty_ingredients, axis=1)]
    result = data[['recipe_title', 'recipe_ingredients']].copy()

    title_idx = result[result['recipe_title'].isnull()].index # title이 null값인 행 인덱스 찾기
    del_idx = result[result['recipe_ingredients'].str.startswith('소시지')].index # 소시지~ 로 시작해서 오류 일으키는 행 인덱스 찾기
    result.drop(del_idx, inplace=True) # 오류 일으키는 행 제거
    result.drop(title_idx, inplace=True) # title null값인 행 제거
    result = result.drop_duplicates() # 중복 제거

    return result

# 1. 식재료 단위 별로 쪼개기
def split_ingredient(data):
    num_ingredients = 74

    list = [[f'ingredient{i}', f'quantity{i}', f'unit{i}'] for i in range(1, num_ingredients + 1)]
    column_names = []
    for i in list :
        column_names.extend(i)

    empty_columns = pd.DataFrame(columns=column_names)
    data = pd.concat([data, empty_columns], axis=1)

    non_matching_items = {} # 패턴과 일치하지 않는 데이터를 저장할 딕셔너리

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]): # tqdm으로 진행상황 확인
        if row['recipe_ingredients']:
            ingredients_dict = ast.literal_eval(row["recipe_ingredients"]) # 딕셔너리 형태로 저장된 recipe_ingredients 불러오기
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

    # i가 75 이상인 경우 제거하는 조건문 (식재료 종류: 최대 75개)
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

# 2. 식재료 종류 전처리: 식재료명 자체에 '양파조금', '후추약간' 식으로 들어간 경우가 있음. 이 표현들을 식재료/단위로 쪼개서 처리
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



# 3-1. 식재료 양 처리
def parse_quantity(dataframe):
    for i in range(1, 25):
        column_name = f'quantity{i}'
        if column_name in dataframe.columns:
            dataframe[column_name] = dataframe[column_name].apply(lambda x: parse_single_quantity(x))
    return dataframe

def parse_single_quantity(quantity):
    if isinstance(quantity, float):
        result = quantity  # 이미 실수인 경우 그대로 반환
        return float(1) if pd.isna(result) else result  # NaN 값이면 1로 변환

    if '~' in quantity:
        numbers = re.findall(r'\d+\.?\d*', quantity)  # 숫자들을 찾음
        numbers = [float(num) for num in numbers]  # 문자열을 실수로 변환
        if len(numbers) == 0:
            return float(2)  # 0으로 나누기를 방지하기 위해 2로 반환
        return float(sum(numbers) / len(numbers))  # 평균 계산
    try:
        result =  float(quantity)  # 일반적인 경우, 숫자로 변환
        return float(1) if pd.isna(result) else result  # NaN 값이면 1로 변환
    except ValueError:
        return float(1) # 비어있는 경우 1로 변환 

# 3-2. 식재료 단위 처리 - 단위를 g으로 : parse_unit('조금') = 10
def parse_unit(dataframe):
    for i in range(1, 25):
        column_name = f'unit{i}'
        if column_name in dataframe.columns:
            dataframe[column_name] = dataframe[column_name].apply(lambda x: parse_single_unit(x))
    return dataframe

def parse_single_unit(unit):
    file_path = r"data\change.txt"
    unit_conversion = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        unit_conversion = {line.split()[0]: float(line.split()[1]) for line in file if line.split()[1].isdigit()}
    return unit_conversion.get(unit, 1)

# 4. Matrix 변환
def recipe_food_matrix(data):
    data.index = range(len(data)) # index 초기

    ingredient_columns = data.filter(like='ingredient')
    if 'recipe_ingredients' in ingredient_columns.columns:
        data = data.drop(columns=['recipe_ingredients'])

    all_ingredients = set()
    ingre_len = int((data.shape[1]-1)/3)
    for i in range(1, ingre_len):  
        all_ingredients.update(data[f'ingredient{i}'].dropna().unique())

    # 레시피 식재료 Matrix 만들기 
    col_name = ['recipe_title'].append(list(all_ingredients))
    recipe_ingredients_df = pd.DataFrame(columns=col_name)

    # 레시피 하나씩 붙이기 
    recipe_rows = []
    for idx, row in tqdm(data.iterrows(), total = data.shape[0]) : # tqdm으로 진행상황 확인
        recipe_data = {ingredient: 0.0 for ingredient in all_ingredients}  # 모든 식재료를 None으로 초기화
        for i in  range(1, ingre_len):  
            ingredient = row[f'ingredient{i}']
            quantity = row[f'quantity{i}']
            unit = row[f'unit{i}']
            if pd.notna(ingredient) and pd.notna(quantity):
                quantity_float = parse_quantity(quantity)
                if quantity_float is not None:
                    unit_number = parse_unit(unit) if pd.notna(unit) else 1
                    recipe_data[ingredient] = quantity_float * unit_number
        recipe_rows.append(recipe_data)

    # 새로운 데이터프레임 생성 (모든 식재료를 열로 가짐)
    recipe_ingredients_df = pd.concat([pd.DataFrame([row]) for row in recipe_rows], ignore_index=True)
    recipe_ingredients_df = recipe_ingredients_df.astype('float64')
    recipe_ingredients_df['recipe_title'] = data['recipe_title']

    # RECIPE_TITLE 컬럼을 제일 앞으로
    recipe_ingredients_df = recipe_ingredients_df[['recipe_title'] + [col for col in recipe_ingredients_df.columns if col != 'recipe_title']]
    recipe_ingredients_df.to_csv(f'matrix/food_matrix_{len(data)}.csv')
    return recipe_ingredients_df

# 5. 재료 쪼갠 후 레시피별 영양소 나오는 테이블
def recipe_nutri(new_recipe1, nutri_df):
    warnings.filterwarnings('ignore', category= UserWarning)
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
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
