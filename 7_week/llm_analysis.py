import ollama 

article = """ 
인공지능 기술은 최근 의료 산업 전반에 큰 변화를 가져오고 있다. 특히 영상 진단 분야에서 AI는 종양과 같은 질환을 더욱 빠르고 정확하게 발견하는 데 활용되며, 일부 사례에서는 기존 전문 의료진보다 높은 성능을 보이기도 한다. 예를 들어 삼성서울병원에서는 AI를 활용한 폐암 조기 진단 시스템을 도입해 진단의 정밀도를 크게 개선한 바 있다.

또한 AI는 신약 개발 과정에서도 중요한 역할을 하고 있다. 방대한 데이터를 분석하여 후보 물질을 빠르게 선별함으로써, 기존에 오랜 시간이 소요되던 개발 기간을 획기적으로 줄일 가능성이 제시되고 있다.

그러나 이러한 기술 발전과 함께 윤리적 문제와 개인정보 보호에 대한 우려 역시 커지고 있다. 따라서 의료 AI의 안전하고 책임 있는 활용을 위해 관련 제도와 법적 기준을 정비하는 것이 중요한 과제로 떠오르고 있다.
""" 
# 키워드 추출 
print("=== 키워드 추출 ===") 
response = ollama.chat( 
    model="gemma3:4b", 
    messages=[ 
        {"role": "system", "content": "주어진 텍스트에서 핵심 키워드 5개를 추출하세요. 키워드만 쉼표로 구분하여 나열하세요."}, 
        {"role": "user", "content": article} 
    ] 
) 
print(response["message"]["content"]) 

# 요약 
print("\n=== 3줄 요약 ===") 
response = ollama.chat( 
    model="gemma3:4b", 
    messages=[ 
        {"role": "system", "content": "주어진 텍스트를 정확히 3줄로 요약하세요. 각 줄은 한 문장으로 작성하세요."}, 
        {"role": "user", "content": article} 
    ] 
) 
print(response["message"]["content"])