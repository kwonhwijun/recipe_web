from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib import auth
from django.contrib.auth import authenticate, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

# 회원가입
def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')
        confirm_password = request.POST.get('confirm')

        # 하나라도 안적은게 있을때
        if not username or not password or not confirm_password:
            messages.error(request, '모든 필드를 입력해주세요.')
            return render(request, 'signup.html')
        
        # 이미 가입중인 user명
        if User.objects.filter(username=username).exists():
            messages.error(request, '이미 존재하는 사용자 이름입니다.')
            return render(request, 'signup.html')

        # 비밀번호, 비밀번호 확인 일치하면 home으로 redirect
        if password == confirm_password:
            user = User.objects.create_user(username=username, password=password, email = email)
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

# 내정보
@login_required
def profile(request):
    # request.user를 사용하여 현재 로그인한 사용자의 정보에 접근할 수 있습니다.
    user = request.user

    # 이제 템플릿에 사용자 객체를 전달하여 렌더링할 수 있습니다.
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