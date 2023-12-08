from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib import auth

# 회원가입
def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm')

        # 하나라도 안적은게 있을때
        if not username or not password or not confirm_password:
            messages.error(request, '모든 필드를 입력해주세요.')
            return render(request, 'signup.html')
        
        # 이미 가입중인 user명
        if User.objects.filter(username=username).exists():
            messages.error(request, '이미 존재하는 사용자 이름입니다.')
            return render(request, 'signup.html')

        # 비밀번호, 비밀번호 확인 일치하는지
        if password == confirm_password:
            user = User.objects.create_user(username=username, password=password)
            auth.login(request, user)
            return redirect('/home/')
        else:
            messages.error(request, '비밀번호가 일치하지 않습니다.')
            return render(request, 'signup.html')

    return render(request, 'signup.html')

# 로그인
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('/home/')
        else:
            try:
                user = User.objects.get(username=username)
                messages.error(request, '아이디와 비밀번호가 일치하지 않습니다.')
            except User.DoesNotExist:
                messages.error(request, '존재하지 않는 아이디입니다.')
            return render(request, 'login.html')
    else:
        return render(request, 'login.html')

# 로그아웃
def logout(request):
    if request.method == 'POST':
        auth.logout(request)
        return redirect('/home/')
    return render(request, 'login.html')