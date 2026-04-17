import ollama 
import json 
reviews = [ 
"배송 빠르고 제품 품질도 좋아요! 재구매 의사 있습니다.", 
"불량품이 왔는데 교환도 안 해주네요. 최악입니다.", 
"가격은 저렴한데 품질은 보통이에요." 
]

results = [] 
for review in reviews: 
    response = ollama.chat( 
        model="gemma3:4b", 
        messages=[ 
            { 
"role": "system", 
"content": """당신은 리뷰 분석 전문가입니다. 
주어진 리뷰를 분석하여 반드시 아래 JSON 형식으로만 응답하세요. 
다른 텍스트는 포함하지 마세요. 
2026-04-15
 {"sentiment": "긍정/부정/중립", "confidence": 0.0~1.0, "keywords": ["키워드1", "키워
드2"]} 
※ confidence = 확신도 (0.0이면 자신 없음, 1.0이면 매우 확실)""" 
            }, 
            { 
"role": "user", 
"content": review 
            } 
        ] 
    ) 
    raw = response["message"]["content"] 
    # JSON 파싱 시도
    try: 
        # ── 1단계: LLM 응답에서 순수 JSON 문자열만 추출 ── 
        # # LLM이 ```json ... ``` 코드블록으로 감싸서 응답하는 경우가 많음 
        # # 예: '```json\n{"sentiment": "긍정"}\n```' 에서 중간 JSON만 꺼냄 
        clean = raw.strip() 
        if "```json" in clean: 
# "```json" 기준으로 잘라서 JSON 부분만 추출 
            clean = clean.split("```json")[1].split("```")[0].strip() 
        elif "```" in clean: 
# "```" 만 있는 경우에도 동일하게 처리 
            clean = clean.split("```")[1].split("```")[0].strip() 
# 만약 ```가 없으면 clean은 원본 그대로 (바로 JSON인 경우) 
# ── 2단계: JSON 문자열 → 파이썬 딕셔너리로 변환 ── 
        data = json.loads(clean)  # {"sentiment": "긍정", ...} → dict 
        results.append(data) 
        print(f"✅ 리뷰: {review[:25]}...") 
        print(f"   감성: {data['sentiment']}, 확신도: {data['confidence']}") 
        print(f"   키워드: {data['keywords']}") 
    except json.JSONDecodeError: 
        print(f"⚠ JSON 파싱 실패: {raw[:100]}") 
    print("-" * 50) 
# 결과를 DataFrame으로 변환 가능! 
print(f"\n총 {len(results)}개 리뷰 분석 완료")
