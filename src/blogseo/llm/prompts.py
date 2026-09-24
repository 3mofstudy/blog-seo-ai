"""共用 prompt。

所有 provider 共用同一組 prompt，這樣不同模型的輸出才有可比性——比較結果的
差異應該來自模型能力，而不是來自 prompt 寫法不同。
"""

from __future__ import annotations

from typing import Final

from blogseo.config import (
    ALT_MAX_CHARS,
    KEYWORD_COUNT,
    KEYWORD_MAX_CHARS,
    SUMMARY_MAX_CHARS,
    SUMMARY_MIN_CHARS,
    SUMMARY_VARIANT_COUNT,
)
from blogseo.zh import wants_traditional_chinese

#: 送進模型的正文字元上限，避免超長文章造成不必要的花費。
MAX_CONTENT_CHARS: Final[int] = 20_000

#: 各摘要版本的切入角度。對齊方格子沙龍列表常見寫法，目的是讓版本之間真的不同，
#: 而不是同一句話換幾個詞。超出清單長度的部分交由模型自行延伸。
_SUMMARY_ANGLES: Final[tuple[str, ...]] = (
    "系列或文章定位：點出這篇在解決什麼問題，接著列出本文實際會介紹的重點",
    "讀者痛點開場：用「如果你…卻…」點出卡關之處，再接到本文的工具或做法會帶你做什麼",
    "主題加適用對象：先說主題是什麼、適用於誰，再交代文章涵蓋的原理、實作與會對照的概念",
    "反直覺問題開場：用「為什麼…？」抓住注意力，再說明本文探討的核心概念與會介紹的具體方法",
    "承接前文：僅當文章確實接續前作時，用「在上一篇文章中…」回顧上一篇再引出本文要往哪走；否則改寫成對本文主題的鋪陳與轉折",
)

#: 方格子沙龍列表摘要的寫法示範。只給語氣與資訊密度，模型不得抄寫這些題材。
_SUMMARY_STYLE_EXAMPLES: Final[tuple[str, ...]] = (
    "本文為 MLflow 實戰系列的完結篇，著重於解決企業級 MLflow 部署所面臨的安全性風險與協作權限問題。透過手把手教學，介紹如何啟用 MLflow 內建的 Auth 權限管理系統，Docker 容器化部署",
    "如果你習慣 R 語言 ggplot2 的優雅繪圖，卻在轉往 Python 時對 Matplotlib 感到苦手，那麼 plotnine 將是你不能錯過的救星！這篇文章將帶你快速認識 plotnine，完美復刻 ggplot2 語法的 Python",
    "本文深入解析資料包絡分析 (DEA)，適用於評估銀行分行、醫院、學校或企業部門等決策單位。文章涵蓋 DEA 的歷史起源、數學原理，並提供 R 語言的實際操作範例，及如何理解效率值、CCR 與 BCC 模型差異",
    "為什麼機器學習分類問題中，99% 的準確率可能毫無價值？本文將深入探討類別不平衡（Class Imbalance）和準確率悖論（Accuracy Paradox），並介紹敏感度、精確率、F1-score 等更適合的評估指標",
    "在上一篇文章中，我們一起漫步於機率分布的世界，認識了像常態分布、二項分布、柏松分布這些基礎卻無比重要的「地標」。它們是統計學的基石，描述了數據世界中最常見的幾種規律。然而，機率的宇宙浩瀚無垠",
)

_LANGUAGE_NAMES: Final[dict[str, str]] = {
    "zh": "繁體中文",
    "zh-hant": "繁體中文",
    "zh-tw": "繁體中文",
    "en": "English",
    "ja": "日本語",
}


def language_name(code: str) -> str:
    """把語言代碼轉成寫進 prompt 的語言名稱。

    Args:
        code: 語言代碼，例如 ``zh``。

    Returns:
        語言名稱；未知代碼原樣回傳。
    """
    normalized = code.strip().lower().replace("_", "-")
    return _LANGUAGE_NAMES.get(normalized, code)


def _summary_angle_lines(count: int) -> str:
    """列出各摘要版本要採用的切入角度。

    Args:
        count: 摘要版本數。

    Returns:
        條列文字；只要一個版本時回傳空字串。
    """
    if count <= 1:
        return ""
    lines = []
    for index in range(count):
        if index < len(_SUMMARY_ANGLES):
            lines.append(f"   - 第 {index + 1} 則：{_SUMMARY_ANGLES[index]}")
        else:
            lines.append(f"   - 第 {index + 1} 則：再換一個上面沒用過的切入角度")
    return "\n".join(lines)


def _summary_style_rule() -> str:
    """方格子列表摘要的語氣說明與示範。"""
    examples = "\n".join(f"   - {example}" for example in _SUMMARY_STYLE_EXAMPLES)
    return (
        "寫法要像方格子沙龍的列表摘要：資訊密度高、具體，"
        "不要寫成搜尋引擎 meta description 或空洞的行銷文案。"
        "可依文章性質選用類似開頭，但後面必須立刻接上具體內容。"
        "以下是語氣與資訊密度的示範，只模仿寫法，禁止抄寫這些題材。\n"
        "示範有的在列表字數上限處被截斷，你產出的摘要必須是完整句子。\n"
        + examples
    )


def text_system_prompt(
    language: str,
    *,
    include_keywords: bool = True,
    include_summaries: bool = True,
    keyword_count: int = KEYWORD_COUNT,
    keyword_max_chars: int = KEYWORD_MAX_CHARS,
    summary_count: int = SUMMARY_VARIANT_COUNT,
    summary_min_chars: int = SUMMARY_MIN_CHARS,
    summary_max_chars: int = SUMMARY_MAX_CHARS,
) -> str:
    """建立文章分析的 system prompt。

    只要其中一項時，另一項的規則不會出現在 prompt 裡——既省下輸入 token，
    也避免模型分心去做沒被要求的事。

    Args:
        language: 文章語言代碼。
        include_keywords: 是否要產生關鍵字。
        include_summaries: 是否要產生摘要。
        keyword_count: 要產生幾個關鍵字。
        keyword_max_chars: 每個關鍵字的字元上限。
        summary_count: 要產生幾個摘要版本。
        summary_min_chars: 每則摘要的字元下限。
        summary_max_chars: 每則摘要的字元上限。

    Returns:
        system prompt 文字。

    Raises:
        ValueError: 兩項都沒要求。
    """
    if not include_keywords and not include_summaries:
        raise ValueError("關鍵字與摘要至少要產生一項")

    name = language_name(language)
    plural = f"{summary_count} 則不同的" if summary_count > 1 else "一則"

    goals: list[str] = []
    if include_keywords:
        goals.append(f"{keyword_count} 個關鍵字")
    if include_summaries:
        goals.append(f"{plural}文章摘要")

    rules: list[str] = [f"輸出一律使用{name}，與文章語言一致。"]

    if include_keywords:
        rules += [
            (
                "關鍵字要是讀者真的會拿去搜尋的詞，涵蓋主題、技術名詞與應用情境，"
                "避免過於籠統的字（例如「技術」「教學」）。"
            ),
            (
                "每個關鍵字必須是連貫的一個詞，中間不能有空白。"
                "例如寫「plotnine教學」不要寫「plotnine 教學」，"
                "寫「Python繪圖」不要寫「Python 繪圖」。"
            ),
            (
                f"每個關鍵字不得超過 {keyword_max_chars} 個字元。"
                "計算方式是逐字元計算，中文字、英文字母與數字各算一個字元。"
                "這是硬性要求，寫完請自行確認長度再輸出；超過上限的詞會被捨棄。"
            ),
            "關鍵字依重要性由高到低排序，彼此不重複、不互為子集。",
        ]

    if include_summaries:
        rules += [
            (
                "這些摘要會刊登在方格子沙龍（https://vocus.cc/salon/lucy-r）的文章列表，"
                "讓讀者決定要不要點進去。每一則都必須說清楚這篇文章主要介紹什麼："
                "主題、要解決的問題或適用對象，以及文中實際會講到的工具、方法、模型或步驟。"
                "禁止只寫氣氛、口號或「不容錯過」卻沒有資訊。"
            ),
            (
                f"每則摘要的長度必須落在 {summary_min_chars} 到 {summary_max_chars} 個字元之間。"
                "計算方式是逐字元計算，中文字、英文字母、數字、標點符號與空白各算一個字元。"
                "這是硬性要求，寫完請自行確認長度再輸出；少於下限通常講不完重點，"
                "超過上限會在列表被截斷。必須在上限內把句子寫完，不要寫到一半。"
            ),
            "每則摘要都是單一段落、語句完整通順，不要換行，也不要用條列。",
            _summary_style_rule(),
            (
                "專有名詞、工具名、模型名保留文章用字，不要改成「相關技術」「這個方法」這種空詞。"
                "文章不是系列就不要寫「完結篇」或「在上一篇文章中」。"
                "「本文為」「本文深入解析」「這篇文章將帶你」「為什麼…？本文將深入探討」"
                "這類開頭可以使用，但後面必須立刻接上具體內容，"
                "禁止整則只有「本文將介紹某某主題。」這種空話。"
            ),
        ]
        if summary_count > 1:
            rules.append(
                f"{summary_count} 則摘要之間必須有明顯不同的切入角度，"
                "不能只是同一句話換幾個詞。請依下列分工撰寫：\n"
                + _summary_angle_lines(summary_count)
            )
        else:
            rules.append(
                "請依文章性質，從上述寫法中選最貼切的一種來寫，"
                "不要硬套不存在的系列或前文。"
            )

    numbered = "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))
    role = (
        "你是方格子技術沙龍的編輯，文章都將刊登於 https://vocus.cc/salon/lucy-r。"
        if include_summaries
        else "你是資深的技術部落格 SEO 編輯。"
    )
    return f"{role}你的任務是閱讀一篇文章，產出{'與'.join(goals)}。\n\n規則：\n{numbered}"


def text_user_prompt(title: str | None, content: str) -> str:
    """建立文章分析的 user prompt。

    Args:
        title: 文章標題，可為 ``None``。
        content: 文章正文。

    Returns:
        user prompt 文字；超長時會截斷並標註。
    """
    body = content.strip()
    if len(body) > MAX_CONTENT_CHARS:
        body = body[:MAX_CONTENT_CHARS] + "\n\n（後續內容因長度限制已省略）"

    header = f"文章標題：{title}\n\n" if title else ""
    return f"{header}以下是文章內容：\n\n{body}"


def image_system_prompt(alt_language: str, *, alt_max_chars: int = ALT_MAX_CHARS) -> str:
    """建立圖片分析的 system prompt。

    Args:
        alt_language: alt 文字要使用的語言代碼。
        alt_max_chars: alt 文字的字元上限。

    Returns:
        system prompt 文字。
    """
    name = language_name(alt_language)
    alt_lang_rule = f"alt 文字使用{name}"
    if wants_traditional_chinese(alt_language):
        alt_lang_rule += "（台灣正體），禁止使用簡體字"
    return (
        "你是負責圖片 SEO 的編輯。你會看到一張部落格文章中的圖片，"
        "以及它在文章裡的上下文。請產出這張圖的檔名與 alt 文字。\n\n"
        "規則：\n"
        "1. 檔名一律使用英文小寫，單字之間用連字號，3 到 6 個單字，不含副檔名。\n"
        "2. 檔名要描述圖片的實際內容，不要用 image、photo、screenshot-1 這種無意義的字。"
        "如果圖片是某個工具的操作畫面，就寫出工具名稱與該畫面在做什麼。\n"
        f"3. {alt_lang_rule}，長度不得超過 {alt_max_chars} 個字元。"
        "計算方式是逐字元計算，中文字、英文字母、數字、標點符號與空白各算一個字元。"
        "這是硬性要求，寫完請自行數過再輸出。\n"
        "4. alt 只寫這張圖最重要的那一件事，一句話講完就好，不要條列細節、"
        "不要把畫面上看到的數值全部抄進去；螢幕閱讀器會逐字唸出來，越長越難聽懂。\n"
        "5. alt 要讓看不到圖的人知道這張圖在說什麼，"
        "開頭不要出現「圖片」「示意圖」「一張」「image of」這類贅詞。\n"
        "6. 只根據你實際看到的內容描述，不要臆測圖片沒有呈現的東西。\n"
        "7. 輸出 JSON 物件，欄位名稱必須正好是 suggested_filename 與 alt，"
        "不要改成 filename、file_name、caption。"
    )


def image_user_prompt(context: str, original_filename: str) -> str:
    """建立圖片分析的 user prompt。

    Args:
        context: 圖片在文章中的上下文。
        original_filename: 原始檔名，供模型參考但不應照抄。

    Returns:
        user prompt 文字。
    """
    parts = [f"原始檔名：{original_filename}（僅供參考，通常沒有意義，不要照抄）"]
    if context.strip():
        parts.append(context.strip())
    parts.append("請分析上方這張圖片。")
    return "\n\n".join(parts)
