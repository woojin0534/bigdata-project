"""
TMDB API를 사용해 Netflix · Disney+ · Amazon Prime Video 실제 데이터를 수집합니다.

사전 준비:
    1. https://www.themoviedb.org/ 에서 무료 계정 생성
    2. 설정 > API > 'API 키 (v3 auth)' 복사
    3. 아래 API_KEY 에 붙여넣기
    4. pip install requests pandas
    5. python collect_real_data.py
"""

import os
import time

import pandas as pd
import requests

# ── 설정 ──────────────────────────────────────────────────────────────────────
API_KEY = "a9773d590a0b0703237ae1a4e4efc820"
BASE = "https://api.themoviedb.org/3"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

PLATFORMS = {
    "Netflix": 8,
    "Amazon Prime Video": 9,
    "Disney+": 337,
}

COUNTRY_MAP = {
    "US": "United States", "KR": "South Korea",  "JP": "Japan",
    "GB": "United Kingdom","IN": "India",         "FR": "France",
    "DE": "Germany",       "IT": "Italy",         "ES": "Spain",
    "CA": "Canada",        "AU": "Australia",     "BR": "Brazil",
    "MX": "Mexico",        "CN": "China",         "NG": "Nigeria",
    "AR": "Argentina",     "TR": "Turkey",        "TH": "Thailand",
    "PH": "Philippines",   "ID": "Indonesia",     "SE": "Sweden",
    "IL": "Israel",        "DK": "Denmark",       "EG": "Egypt",
}

MAX_PAGES = 30          # 플랫폼·타입 조합당 최대 페이지 (20개/페이지 → 최대 600편)
DELAY     = 0.27        # 요청 간격 (TMDB 제한: 40회/10초)


# ── 장르 코드 → 이름 매핑 ──────────────────────────────────────────────────────
def fetch_genres() -> dict:
    genres = {}
    for media in ("movie", "tv"):
        r = requests.get(
            f"{BASE}/genre/{media}/list",
            params={"api_key": API_KEY, "language": "en-US"},
            timeout=10,
        ).json()
        for g in r.get("genres", []):
            genres[g["id"]] = g["name"]
    return genres


# ── 제목 1건 상세 정보 (런타임·제작국가) ──────────────────────────────────────
def fetch_details(media: str, tmdb_id: int) -> tuple:
    r = requests.get(
        f"{BASE}/{media}/{tmdb_id}",
        params={"api_key": API_KEY},
        timeout=10,
    ).json()

    if media == "movie":
        runtime = r.get("runtime")
    else:
        ep = r.get("episode_run_time", [])
        runtime = ep[0] if ep else None

    countries = [c["iso_3166_1"] for c in r.get("production_countries", [])]
    country_code = countries[0] if countries else None
    country = COUNTRY_MAP.get(country_code, country_code)
    return runtime, country


# ── Discover 엔드포인트로 목록 수집 ───────────────────────────────────────────
def discover(media: str, provider_id: int, provider_name: str, genres: dict) -> list:
    endpoint = "movie" if media == "movie" else "tv"
    rows = []

    for page in range(1, MAX_PAGES + 1):
        try:
            r = requests.get(
                f"{BASE}/discover/{endpoint}",
                params={
                    "api_key": API_KEY,
                    "with_watch_providers": provider_id,
                    "watch_region": "US",
                    "sort_by": "vote_count.desc",
                    "vote_count.gte": 30,
                    "page": page,
                },
                timeout=10,
            ).json()
        except Exception as e:
            print(f"  [경고] page {page} 오류: {e}")
            break

        items = r.get("results", [])
        if not items:
            break

        for item in items:
            title = item.get("title") or item.get("name", "")
            date  = item.get("release_date") or item.get("first_air_date", "")
            year  = int(date[:4]) if date and len(date) >= 4 and date[:4].isdigit() else None
            gids  = item.get("genre_ids", [])
            genre = genres.get(gids[0], "Drama") if gids else "Drama"

            rows.append({
                "_id":         item["id"],
                "_media":      media,
                "title":       title,
                "type":        "Movie" if media == "movie" else "TV Show",
                "platform":    provider_name,
                "primary_genre": genre,
                "release_year": year,
                "imdb_rating": round(item.get("vote_average", 0.0), 1),
            })

        time.sleep(DELAY)
        print(f"  [{provider_name} / {'영화' if media=='movie' else 'TV'}] "
              f"p{page}/{MAX_PAGES}  누적 {len(rows)}편")

    return rows


# ── 상세 정보 보완 (런타임 · 제작국가) ────────────────────────────────────────
def enrich(rows: list) -> list:
    total = len(rows)
    for i, row in enumerate(rows):
        try:
            runtime, country = fetch_details(row["_media"], row["_id"])
            row["duration_minutes"] = runtime
            row["country"] = country
        except Exception:
            row["duration_minutes"] = None
            row["country"] = None
        time.sleep(DELAY)
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  상세 보완: {i+1}/{total}")
    return rows


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    if API_KEY == "YOUR_TMDB_API_KEY_HERE":
        print("❌  API_KEY를 입력해주세요.")
        print("    https://www.themoviedb.org/ 에서 무료 발급 가능합니다.")
        return

    print("1) 장르 코드 불러오는 중...")
    genres = fetch_genres()
    print(f"   장르 {len(genres)}개 로드 완료\n")

    all_rows = []
    for name, pid in PLATFORMS.items():
        print(f"=== {name} ===")
        for media in ("movie", "tv"):
            rows = discover(media, pid, name, genres)
            all_rows.extend(rows)
        print()

    print(f"2) 기본 수집 완료: {len(all_rows)}편")
    print("3) 런타임 · 제작국가 보완 중 (시간이 걸립니다)...\n")
    all_rows = enrich(all_rows)

    df = pd.DataFrame(all_rows).drop(columns=["_id", "_media"])

    # 중복 제거 (같은 제목이 여러 플랫폼에 있을 수 있으므로 platform 포함)
    df = df.drop_duplicates(subset=["title", "platform", "type"])

    # 연도 범위 정리
    df = df[df["release_year"].between(1990, 2025)]

    # imdb_rating 0점 → NaN 처리 (평점 없는 항목)
    df.loc[df["imdb_rating"] == 0.0, "imdb_rating"] = None

    print(f"\n4) 최종 데이터셋: {len(df)}편")
    print(df["platform"].value_counts().to_string())

    # ── streaming_catalog.csv 저장
    out_main = os.path.join(SAVE_DIR, "streaming_catalog.csv")
    df.to_csv(out_main, index=False)
    print(f"\n✅  저장: {out_main}")

    # ── yearly_release_trends.csv 자동 생성
    yearly = (
        df.groupby(["release_year", "type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    yearly.columns.name = None
    yearly = yearly.rename(columns={"Movie": "movies", "TV Show": "tv_shows"})
    for col in ("movies", "tv_shows"):
        if col not in yearly.columns:
            yearly[col] = 0
    out_yearly = os.path.join(SAVE_DIR, "yearly_release_trends.csv")
    yearly.to_csv(out_yearly, index=False)
    print(f"✅  저장: {out_yearly}")

    print("\n수집 완료! ott_storyboard.py 를 다시 실행하세요.")


if __name__ == "__main__":
    main()
