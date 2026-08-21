# blog-seo-ai

本地端 CLI 工具：把部落格文章的 `.md` 丟進去，用 LLM 一次搞定事後最煩人的三件事——
挑 SEO 關鍵字、寫 meta description、看懂圖片後給出有意義的檔名與 alt 文字。

支援同時呼叫多個模型，結果並列輸出成 JSON，再用互動式的 `review` 逐項挑選；
分析與挑選階段**絕不修改任何原始檔案**，真正改檔是 `apply`。

## 目前進度


| 階段  | 指令        | 狀態                                   |
| --- | --------- | ------------------------------------ |
| 分析  | `analyze` | 可用（預設 Hugging Face 免費模型；也可指定 Claude） |
| 挑選  | `review`  | 可用                                   |
| 套用  | `apply`   | 可用                                   |


Provider 方面，Anthropic（Claude）與 Hugging Face（文字 Qwen3-4B、圖片 GLM-4.6V-Flash）已完成；
OpenAI 與 Gemini 已在註冊表預留位置，尚未實作。

## 安裝

需要 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
```

金鑰放在專案根目錄的 `.env`：

```
CLAUDE_API_KEY=sk-ant-...
# 或使用 SDK 預設名稱
# ANTHROPIC_API_KEY=sk-ant-...

# Hugging Face 免費額度：https://huggingface.co/settings/tokens
# 權限勾選「Make calls to Inference Providers」
HF_TOKEN=hf_...
```

`.env` 已被 `.gitignore` 排除，不會進版控。

## 使用

```bash
# 分析（預設 Hugging Face 免費模型）
uv run blog-seo-ai analyze path/to/post.md

# 挑選（純本地，不花錢）
uv run blog-seo-ai review output/post-analysis.json

# 套用（會改 Markdown 與圖片檔名）
uv run blog-seo-ai apply output/post-selection.json
```



### analyze

```bash
# 基本用法：關鍵字/摘要用 Qwen3-4B，圖片用 GLM-4.6V-Flash
uv run blog-seo-ai analyze path/to/post.md

# 只覆寫文字模型（圖片仍用 GLM-4.6V-Flash）
uv run blog-seo-ai analyze post.md --models hf:Qwen/Qwen3-8B

# 改走 Claude（會計費）
uv run blog-seo-ai analyze post.md --models claude

# 只要其中幾樣：關鍵字、摘要、圖片可自由組合
uv run blog-seo-ai analyze post.md --fields keywords
uv run blog-seo-ai analyze post.md --fields keywords,summary

# 控制花費：只分析前 3 張圖
uv run blog-seo-ai analyze post.md --max-images 3

# 文章是中文但 alt 想用英文
uv run blog-seo-ai analyze post.md --alt-lang en

# 列出可用的模型別名
uv run blog-seo-ai providers
```

分析結果預設寫到 `output/<檔名>-analysis.json`，用 `--out` 可自訂路徑。


| 選項                     | 預設                        | 說明                     |
| ---------------------- | ------------------------- | ---------------------- |
| `--models` / `-m`      | `hf`                      | 逗號分隔的模型別名，支援 `別名:模型id` |
| `--out` / `-o`         | `output/`                 | 輸出 JSON 的路徑或目錄         |
| `--fields` / `-f`      | `keywords,summary,images` | 要產生哪些項目，逗號分隔           |
| `--max-images`         | 不限                        | 最多分析幾張圖，用來控制花費         |
| `--alt-lang`           | `auto`                    | alt 文字語言，`auto` 跟隨文章語言 |
| `--alt-max`            | `30`                      | alt 字元上限               |
| `--summaries`          | `3`                       | 每個模型產生幾個不同角度的摘要版本（1–5） |
| `--summary-min`        | `60`                      | 摘要字元下限                 |
| `--summary-max`        | `150`                     | 摘要字元上限                 |
| `--concurrency` / `-c` | `4`                       | 平行呼叫上限                 |
| `--timeout`            | `90`                      | 單次請求逾時秒數               |
| `--twd-rate`           | `31.8`                    | 費用換算成新台幣的匯率            |




### review

```bash
# 互動式挑選，結果寫到 output/post-selection.json
uv run blog-seo-ai review output/post-analysis.json

# 非互動：整份採用 claude 的結果，摘要取第 2 則
uv run blog-seo-ai review output/post-analysis.json --pick claude --summary-index 2 --yes
```

`review` 只讀寫本地 JSON，不呼叫任何模型，所以不會產生費用，挑到滿意為止都可以重跑。
每一項都可以選某個模型的建議、自己手動輸入，或是不採用。
`--pick` 在互動模式下只是把該模型設成預設選項，加上 `--yes` 才會完全不詢問。

圖片的**檔名與 alt 是分開問的**，因為兩者的品質彼此無關：檔名取得漂亮的模型，
alt 可能寫得又臭又長。你可以拿 A 模型的檔名配 B 模型的 alt，也可以保持原檔名
只補 alt、或只改名不動 alt。兩項都選「不改」的圖片不會出現在選擇檔裡。


| 選項                | 預設        | 說明                    |
| ----------------- | --------- | --------------------- |
| `--out` / `-o`    | `output/` | 輸出選擇檔的路徑或目錄           |
| `--pick`          | 第一個成功的模型  | 偏好的模型別名               |
| `--summary-index` | `1`       | 搭配 `--yes` 時採用第幾則摘要   |
| `--yes` / `-y`    | 否         | 不詢問，直接採用 `--pick` 的結果 |
| `--force`         | 否         | 文章在分析後被改過也繼續          |




### apply

```bash
# 先預覽，不寫入
uv run blog-seo-ai apply output/post-selection.json --dry-run

# 確認後寫入；加 --yes 可跳過詢問
uv run blog-seo-ai apply output/post-selection.json --yes
```

這是唯一會修改原始檔案的指令。它只讀選擇檔與當下的 Markdown，不呼叫模型，
也**不會改 front matter**（關鍵字與摘要留在選擇檔裡，要自己貼）。

會做兩件事：

1. 依字元位置改寫圖片語法的路徑與 alt（Markdown 與 HTML `<img>` 都支援），程式碼區塊裡的範例不會被改到。
2. 把圖片檔改名，目錄不變。兩張圖對調檔名、或只改大小寫，都會先改成暫存名再改過去。

套用前會重新計算正文雜湊。文章在 analyze 之後被改過，預設會拒絕，避免改到錯誤的位置；`--force` 會改用當下重新解析的位置繼續。目標檔名已經被別的檔占用時也會中止，不會覆蓋。


| 選項             | 預設  | 說明                |
| -------------- | --- | ----------------- |
| `--dry-run`    | 否   | 只顯示將要做的變更         |
| `--yes` / `-y` | 否   | 不詢問，直接寫入          |
| `--force`      | 否   | 文章被改過、或選擇檔沒有雜湊也繼續 |
| `--backup`     | 否   | 寫入前把原文複製成 `.bak`  |




## 控制 analyze 的產出項目

`--fields` 決定這次要花錢做哪些事，可用 `keywords`、`summary`、`images` 三種：


| 想要的結果         | 指令                          |
| ------------- | --------------------------- |
| 只要關鍵字         | `--fields keywords`         |
| 只要摘要          | `--fields summary`          |
| 關鍵字 + 摘要，不碰圖片 | `--fields keywords,summary` |
| 只分析圖片         | `--fields images`           |


沒被要求的項目不會送進 prompt，也不會出現在給模型的 JSON schema 裡，所以不只是
輸出時過濾掉而已，是真的不會產生那些 output token。JSON 中未要求的項目會是空陣列
（不是 `null`），並在 `metadata.requested_fields` 記錄本次要求了什麼。

只做圖片分析時，`keyword_summary` 會是空物件；不做圖片時，圖片會標記
`skipped_reason` 為「本次 --fields 未包含 images」。

## 花費估算

用量表格最後一欄是新台幣估算金額，並在表格下方顯示合計：

```
用量
模型    模型 id          呼叫  失敗  輸入 tokens  輸出 tokens  花費 NT$
claude  claude-sonnet-5     3     0        6,542          407    0.5455
整體耗時 12.4 秒　合計 NT$ 0.5455（US$ 0.0172，匯率 31.8）
```

單價寫在各 provider 的 `pricing` 類別屬性（Claude 的在 `llm/anthropic.py`），
官方調價時要手動更新；不在價目表中的模型會顯示「—」而不是猜一個數字。
Hugging Face 走帳號免費額度，費用欄就是「—」。
匯率預設 31.8，要精確計算就用 `--twd-rate` 帶入當日匯率。

## Hugging Face 免費模型

Claude 按 token 計費；若只是想先把 SEO 流程跑通，可以用 Hub 的 Inference Providers。
註冊帳號就能拿免費額度（有速率上限，不是無限）。

預設拆成兩顆較小的免費模型，避免熱門視覺模型在 Featherless 額滿：

- 關鍵字 / 摘要：**Qwen/Qwen3-4B-Instruct-2507**（Nscale）
- 圖片檔名 / alt：**zai-org/GLM-4.6V-Flash**（Novita）

金鑰到 [Hugging Face tokens](https://huggingface.co/settings/tokens) 建立，權限勾
「Make calls to the Inference Providers」，寫進 `.env` 的 `HF_TOKEN`。

到 [Inference Providers](https://huggingface.co/settings/inference-providers)
確認 **Nscale** 與 **Novita** 為開啟，該列不要填 `hf_` token（空白才走 Hugging Face 免費額度）。

額度用完、或該模型暫時額滿時，只會在該模型的格子填 `error`，其他模型
（例如同時指定 `claude`）不受影響。

## 摘要候選

每個模型預設會產生三則切入角度不同的摘要，而不是三句換句話說：

1. 點出讀者遇到的問題與本文的解法
2. 強調具體的技術、工具與關鍵數字
3. 說明讀完之後能達成什麼結果

字數預設要求落在 60 到 150 個字元之間，含中文字、標點與空白，以 Python `len()`
計算。終端機表格會列出每則的實際字數，超出範圍的會標成黃色；JSON 裡則有
`summary_lengths` 與 `metadata.summary_max_chars` 可以對照。

要調整就用 `--summaries`、`--summary-min`、`--summary-max`：

```bash
# 只要一則、限制在 80 到 120 字元
uv run blog-seo-ai analyze post.md --summaries 1 --summary-min 80 --summary-max 120
```

字數是靠 prompt 約束的，模型不見得每次都精準命中，所以工具會如實回報實際字數
讓你自己判斷，而不會擅自截斷句子。

## alt 文字

alt 預設上限 30 個字元，同樣以 Python `len()` 計算。螢幕閱讀器是逐字唸出 alt 的，
過長的描述聽起來很痛苦；圖片細節應該寫在正文，alt 只要一句話交代這張圖在講什麼。

終端機表格會列出實際字數，超出上限的標成黃色；JSON 裡則有 `alt_length` 與
`metadata.alt_max_chars` 可以對照。工具不會擅自截斷句子。

```bash
# 這張圖的資訊量比較大，放寬到 60 字
uv run blog-seo-ai analyze post.md --alt-max 60
```

> Claude 這邊刻意把 thinking effort 設成 `low`（見 `llm/anthropic.py` 的
> `_THINKING_EFFORT`）。實測開啟預設的深度思考會多花三千多個 output token、
> 單次呼叫從 7 秒拉長到 30 秒，但關鍵字與摘要品質沒有差異。



## 圖片路徑的處理範圍

假設圖片與 `.md` 放在同一個目錄或其下的子資料夾（Obsidian、Hexo 常見的擺法），
路徑以 Markdown 檔案所在位置為基準推算。以下情況會被列出但標記略過：

- `http(s)://` 開頭的外部連結
- `/images/x.png` 這類站台絕對路徑
- 本地找不到的檔案
- 副檔名不是圖片的連結

程式碼區塊（圍籬與行內）裡的圖片語法會被忽略，教學文章裡的範例不會被誤改。

## 挑選時的兩道檢查

**文章有沒有被改過。** 圖片引用的字元位置是相對於 analyze 當下的正文，文章一改就會
失效，套用時可能改到錯誤的位置。`analyze` 會把正文（不含 front matter）的 sha256 記在
`article_info.content_hash`，`review` 開始前重新計算並比對，不一致就警告並要求確認。
用 `--force` 可以略過這道檢查。1.3 之前產生的 JSON 沒有這個欄位，仍然可以挑選，只是
會提醒無從確認。

**檔名有沒有撞在一起。** 模型很常把兩張不同的圖都命名成 `result-chart`，另外目標檔名
也可能本來就存在。這兩種情況都會在寫出選擇檔之前列出來，互動模式下可以當場改名或
略過該張圖；留到 `apply` 才發現就只能中止。互相對調檔名（A 改成 B、B 改成 A）不算衝突。

## 輸出格式

analysis JSON 頂層包含 `article_info`、`keyword_summary`、`images`、`metadata` 四個區塊，
關鍵字與每張圖的建議都依模型分組並列，方便逐項比較後混搭挑選。
完整範例見 `[docs/schema-example.json](docs/schema-example.json)`。

個別模型或個別圖片失敗時，只會在該格填入 `error` 欄位，不會影響其他呼叫的結果。

`review` 產生的 selection JSON 則是收斂後的單一決定，結構定義在
`schemas/selection.py`：最終的 `keywords`、`summary`，以及每張圖獨立的
`new_filename` 與 `alt`（任一項都可以是 `null`，表示這一項維持原文）。
它同時記下來源 analysis 路徑、文章路徑與 `content_hash`，讓 `apply`
只讀這一份檔案就能完成套用並自行驗證。

## 開發

```bash
uv run pytest
```

專案慣例寫在 `.cursor/rules/`：provider 介面規範、JSON 結構規範、Python 風格。
新增 provider 時請先讀 `llm-provider-pattern.mdc`。

## 專案結構

```
src/blogseo/
├── cli.py              # typer 進入點
├── config.py           # 金鑰與常數的唯一入口
├── errors.py           # 自訂例外
├── llm/                # provider 抽象層與各家實作
│   ├── base.py         # BaseProvider 介面
│   ├── registry.py     # 別名 → provider 對應
│   ├── prompts.py      # 各家共用的 prompt
│   ├── anthropic.py    # Claude 實作
│   └── huggingface.py  # Hugging Face（文字 Qwen3-4B、圖片 GLM-4.6V-Flash）
├── markdown/
│   ├── parser.py       # front matter、圖片擷取、位置記錄
│   └── updater.py      # 改寫圖片語法，front matter 原樣接回
├── image/
│   ├── analyzer.py     # 讀圖、縮圖、格式轉換
│   └── renamer.py      # 檔名正規化與兩階段改名
├── schemas/
│   ├── result.py       # analysis JSON 的 Pydantic 定義
│   └── selection.py    # selection JSON 的 Pydantic 定義
└── seo/
    ├── analyzer.py     # 多模型平行調度與錯誤隔離
    ├── selector.py     # 候選列舉、衝突偵測（純函式，不碰網路）
    └── applier.py      # 讀選擇檔、組計畫、寫回檔案
```

