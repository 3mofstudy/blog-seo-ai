# blog-seo-ai

本地端 CLI 工具：把部落格文章的 `.md` 丟進去，用 LLM 一次搞定事後最煩人的三件事——
挑 SEO 關鍵字、寫 meta description、看懂圖片後給出有意義的檔名與 alt 文字。

支援同時呼叫多個模型，結果並列輸出成 JSON 讓你挑選；分析階段**絕不修改任何原始檔案**。

## 目前進度

| 階段 | 指令 | 狀態 |
| --- | --- | --- |
| 分析 | `analyze` | 可用（Anthropic / Claude） |
| 挑選 | `review` | 未實作 |
| 套用 | `apply` | 未實作 |

Provider 方面，Anthropic 已完成；OpenAI 與 Gemini 已在註冊表預留位置，尚未實作。

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
```

`.env` 已被 `.gitignore` 排除，不會進版控。

## 使用

```bash
# 基本用法：分析一篇文章，結果寫到 output/
uv run blog-seo-ai analyze path/to/post.md

# 指定模型；別名後面加冒號可以指定確切的模型 id
uv run blog-seo-ai analyze post.md --models claude:claude-opus-5

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

### 主要選項

| 選項 | 預設 | 說明 |
| --- | --- | --- |
| `--models` / `-m` | `claude` | 逗號分隔的模型別名，支援 `別名:模型id` |
| `--out` / `-o` | `output/` | 輸出 JSON 的路徑或目錄 |
| `--fields` / `-f` | `keywords,summary,images` | 要產生哪些項目，逗號分隔 |
| `--max-images` | 不限 | 最多分析幾張圖，用來控制花費 |
| `--alt-lang` | `auto` | alt 文字語言，`auto` 跟隨文章語言 |
| `--summaries` | `3` | 每個模型產生幾個不同角度的摘要版本（1–5） |
| `--summary-min` | `60` | 摘要字元下限 |
| `--summary-max` | `150` | 摘要字元上限 |
| `--concurrency` / `-c` | `4` | 平行呼叫上限 |
| `--timeout` | `90` | 單次請求逾時秒數 |
| `--twd-rate` | `31.8` | 費用換算成新台幣的匯率 |

## 挑選產出項目

`--fields` 決定這次要花錢做哪些事，可用 `keywords`、`summary`、`images` 三種：

| 想要的結果 | 指令 |
| --- | --- |
| 只要關鍵字 | `--fields keywords` |
| 只要摘要 | `--fields summary` |
| 關鍵字 + 摘要，不碰圖片 | `--fields keywords,summary` |
| 只分析圖片 | `--fields images` |

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
匯率預設 31.8，要精確計算就用 `--twd-rate` 帶入當日匯率。

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

## 輸出格式

JSON 頂層包含 `article_info`、`keyword_summary`、`images`、`metadata` 四個區塊，
關鍵字與每張圖的建議都依模型分組並列，方便逐項比較後混搭挑選。
完整範例見 [`docs/schema-example.json`](docs/schema-example.json)。

個別模型或個別圖片失敗時，只會在該格填入 `error` 欄位，不會影響其他呼叫的結果。

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
│   └── anthropic.py    # Claude 實作
├── markdown/parser.py  # front matter、圖片擷取、位置記錄
├── image/
│   ├── analyzer.py     # 讀圖、縮圖、格式轉換
│   └── renamer.py      # 檔名正規化
├── schemas/result.py   # 輸出 JSON 的 Pydantic 定義
└── seo/analyzer.py     # 多模型平行調度與錯誤隔離
```
