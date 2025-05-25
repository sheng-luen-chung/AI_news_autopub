import re
import arxiv
import json
import os
from gtts import gTTS
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

NEWS_PATH = "docs/news.jsonl"
PROCESSED_IDS_PATH = "processed_ids.txt"
AUDIO_DIR = "docs/audios"
QUERY = ["AI", "Foundation Model", "Diffusion Model"]

# === 設定 Gemini API 金鑰 ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# === 讀取與儲存已處理的 ID ===
def load_processed_ids(path=PROCESSED_IDS_PATH):
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f.readlines())

def save_processed_ids(ids, path=PROCESSED_IDS_PATH):
    with open(path, "w") as f:
        f.write("\n".join(sorted(ids)))

# === 抓取 arXiv 論文摘要 ===
def fetch_ai_papers(query, processed_ids, max_results=50):
    client = arxiv.Client()
    search = arxiv.Search(
        query=f'"{query}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    new_ids = set()

    for result in client.results(search):
        arxiv_id = result.get_short_id()
        if arxiv_id in processed_ids:
            continue

        papers.append({
            "query": query,
            "id": arxiv_id,
            "url": result.entry_id,
            "title": result.title,
            "summary": result.summary,
            "authors": [author.name for author in result.authors],
            "published_date": result.published.strftime("%Y-%m-%d"),
        })

        processed_ids.add(arxiv_id)
        new_ids.add(arxiv_id)
        break  # 每個類別只抓一篇

    return papers, new_ids

# === Gemini 生成中文摘要與 pitch ===
def summarize_to_chinese(title, summary):
    prompt = (
        f"請將以下arXiv論文標題與摘要翻譯成繁體中文，並完成以下三個任務：\n"
        f"1. 濃縮為適合收聽的簡明摘要。\n"
        f"2. 設想三個生活化應用場景。\n"
        f"3. 用創投視角說明潛在商業價值。\n\n"
        f"標題：{title}\n"
        f"摘要：{summary}\n"
        f"請回傳 JSON 格式：\n"
        f'{{"title_zh": "...", "summary_zh": "...", "applications": ["...", "...", "..."], "pitch": "..."}}'
    )

    response = model.generate_content(prompt)
    text = response.text.strip()

    # 移除程式區塊標記 ```json 或 ``` 等
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()

    # 嘗試逐行解析，只取第一段合法 JSON
    for part in text.splitlines():
        try:
            return json.loads(part)
        except json.JSONDecodeError:
            continue

    # fallback：嘗試整段載入
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("⚠️ 無法解析 Gemini 回傳的 JSON，原始回應如下：\n")
        print(text)
        raise e

# === 將摘要轉成語音檔 ===
def save_audio(text, filename):
    tts = gTTS(text, lang='zh-tw')
    tts.save(filename)

# === 主程式 ===
def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    all_processed_ids = load_processed_ids()
    new_processed_ids = set()
    papers = []

    for query in QUERY:
        print(f"正在抓取 {query} 相關文章...")
        result, new_ids = fetch_ai_papers(query, all_processed_ids)
        if result:
            papers.extend(result)
            new_processed_ids.update(new_ids)
            print(f"已抓取: {result[0]['title']}")
    print(f"總共抓取到 {len(papers)} 篇文章")

    if len(papers) == 0:
        print("⚠️ 沒有抓到任何新文章")
        return

    for i, paper in enumerate(papers):
        print(f"正在處理第 {i+1} 篇 {paper['title']}")
        result = summarize_to_chinese(paper['title'], paper['summary'])

        audio_text = (
            f"{result['title_zh']}\n\n"
            f"{result['summary_zh']}\n\n"
            f"這項技術有三個生活化的應用場景：\n"
            f"第一，{result['applications'][0]}\n"
            f"第二，{result['applications'][1]}\n"
            f"第三，{result['applications'][2]}\n\n"
            f"向創投推銷的說法：\n{result['pitch']}"
        )

        audio_filename = f"{paper['id']}.mp3"
        audio_path = f"{AUDIO_DIR}/{audio_filename}"
        save_audio(audio_text, audio_path)

        paper.update({
            "title_zh": result["title_zh"],
            "summary_zh": result["summary_zh"],
            "applications": result["applications"],
            "pitch": result["pitch"],
            "audio": f"audios/{audio_filename}",
            "timestamp": datetime.now().isoformat()
        })

        with open(NEWS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    save_processed_ids(all_processed_ids.union(new_processed_ids))
    print("✅ 更新完成：news.jsonl 和 MP3 音檔已產生")

if __name__ == "__main__":
    main()
