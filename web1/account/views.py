from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib import auth
from django.contrib.auth import authenticate, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse


def login(request):
    return render(request, 'login.html')


def signup_ajax(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm = request.POST.get('confirm')
        email = request.POST.get('email')
        
        # 유효성검사
        validation_failed = False 

        # 중복 사용자명 검사
        if User.objects.filter(username=username).exists():
            response_data = {'success': False, 'message': '이미 존재하는 사용자명입니다.'}
        elif password != confirm:
            response_data = {'success': False, 'message': '비밀번호와 비밀번호 확인이 일치하지 않습니다.'}
        elif validation_failed:
            response_data = {'success': False, 'message': '유효성 검사 실패.'}
        else:
            user = User.objects.create_user(username=username, password=password, email=email)
            auth.login(request, user)
            response_data = {'success': True, 'message': '가입 성공.'}

        return JsonResponse(response_data)

    return render(request, 'login.html')

def login_ajax(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return JsonResponse({'success': True, 'message': '로그인 성공'})
        else:
            return JsonResponse({'success': False, 'message': '아이디와 비밀번호가 일치하지 않습니다.'})

    return render(request, 'login.html')

# 로그아웃
def logout(request):
    if request.method == 'POST':
        auth.logout(request)
        return redirect('/home/')
    return render(request, 'login.html')

# 내정보
@login_required
def profile(request):
    # request.user를 사용하여 현재 로그인한 사용자의 정보에 접근할 수 있습니다.
    user = request.user

    context = {'user': user}
    return render(request, 'myinfo.html', context)

# 내정보 수정
@login_required
def edit_profile(request):
    if request.method == 'POST':
        user = request.user
        new_username = request.POST.get('new_username')
        new_email = request.POST.get('new_email')

        # 중복확인        
        if User.objects.filter(username=new_username).exclude(email=user.email).exists():
            messages.error(request, '이미 사용 중인 사용자 이름입니다.')
        elif User.objects.filter(email=new_email).exclude(username=user.username).exists():
            messages.error(request, '이미 사용 중인 이메일 주소입니다.')
        else:
            # 업데이트
            user.username = new_username
            user.email = new_email
            user.save()
            messages.success(request, '프로필이 업데이트되었습니다.')
            return redirect('/profile/')

    return render(request, 'edit_myinfo.html')



# 회원탈퇴
def delete(request):
    if request.method == 'POST':
        password = request.POST.get('password')  # 모달에서 입력한 비밀번호
        user = request.user

        # 비밀번호 확인
        if not authenticate(username=user.username, password=password):
            return JsonResponse({'error': '비밀번호가 일치하지 않습니다.'}, status=400)

        # 비밀번호가 일치하면 사용자 삭제 및 로그아웃
        user.delete()
        logout(request)

        return JsonResponse({'message': '회원 탈퇴가 완료되었습니다.'}, status=200)

    return render(request, 'myinfo.html')