"""analyze 階段輸出的 JSON 結構定義。

頂層結構依 ``.cursor/rules/json-schema.mdc`` 規範，包含
``article_info`` / ``keyword_summary`` / ``images`` / ``metadata`` 四個區塊，
另外加上 ``schema_version`` 以便日後結構演進時能辨識舊檔。

修改任何欄位時，務必同步更新 ``docs/schema-example.json``。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Final

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from blogseo.config import KEYWORD_COUNT
from blogseo.image.renamer import slugify

#: 輸出 JSON 的結構版本。欄位有不相容變動時才進位。
#: 1.1 起 ``summary`` 改為 ``summaries`` 陣列，一次提供多個角度的版本供挑選。
#: 1.2 起可用 ``--fields`` 只產生部分內容，未要求的欄位為空陣列而非 null，
#: 並在 metadata 記錄 ``requested_fields`` 與費用估算。
#: 1.3 起 ``article_info`` 帶 ``content_hash``，供 review 與 apply 確認
#: occurrences 的字元位置在分析之後仍然有效。
#: 1.4 起 metadata 記錄 ``alt_max_chars``，alt 長度不再是寫死的常數。
SCHEMA_VERSION: Final[str] = "1.4"


class AnalysisField(StrEnum):
    """可以單獨開關的分析項目。"""

    KEYWORDS = "keywords"
    SUMMARY = "summary"
    IMAGES = "images"


#: 未指定 ``--fields`` 時要產生的項目。
DEFAULT_FIELDS: Final[frozenset[AnalysisField]] = frozenset(AnalysisField)


def _clean_keyword_list(value: list[str]) -> list[str]:
    """去除空白與重複關鍵字，並截斷到規定數量。

    Args:
        value: 模型回傳的關鍵字。

    Returns:
        清理後的關鍵字，最多 :data:`~blogseo.config.KEYWORD_COUNT` 個。
    """
    seen: dict[str, None] = {}
    for item in value:
        cleaned = item.strip().strip("#\"'")
        if cleaned and cleaned.casefold() not in {k.casefold() for k in seen}:
            seen[cleaned] = None
    return list(seen)[:KEYWORD_COUNT]


def _clean_summary_list(value: list[str]) -> list[str]:
    """壓平換行、去除空白，並剔除重複或空白的版本。

    這裡不限制數量，因為版本數是可設定的；裁切到指定數量由 provider 處理。

    Args:
        value: 模型回傳的摘要。

    Returns:
        清理後的摘要。
    """
    cleaned: list[str] = []
    for item in value:
        flattened = " ".join(item.split())
        if flattened and flattened not in cleaned:
            cleaned.append(flattened)
    return cleaned


KeywordList = Annotated[list[str], AfterValidator(_clean_keyword_list)]
SummaryList = Annotated[list[str], AfterValidator(_clean_summary_list)]

_KEYWORDS_DESCRIPTION: Final[str] = (
    f"{KEYWORD_COUNT} 個 SEO 關鍵字，使用文章本身的語言，"
    "依重要性排序，彼此不重複，不要加井字號或引號"
)
_SUMMARIES_DESCRIPTION: Final[str] = (
    "多個不同切入角度的文章摘要，作為 meta description 使用。"
    "數量與字數範圍依指示辦理，每則都是單一段落不換行，使用文章本身的語言"
)


class KeywordsOnlySchema(BaseModel):
    """只要關鍵字時送給模型的 structured output schema。"""

    keywords: KeywordList = Field(..., description=_KEYWORDS_DESCRIPTION)


class SummariesOnlySchema(BaseModel):
    """只要摘要時送給模型的 structured output schema。"""

    summaries: SummaryList = Field(..., description=_SUMMARIES_DESCRIPTION)


class KeywordSummarySchema(BaseModel):
    """關鍵字與摘要都要時送給模型的 structured output schema。"""

    keywords: KeywordList = Field(..., description=_KEYWORDS_DESCRIPTION)
    summaries: SummaryList = Field(..., description=_SUMMARIES_DESCRIPTION)


def text_output_schema(fields: frozenset[AnalysisField]) -> type[BaseModel]:
    """依要產生的項目挑選送給模型的 schema。

    只要關鍵字時就不能把摘要欄位放進 schema，否則模型仍會照生，
    等於白付那些 output token。

    Args:
        fields: 這次要產生的項目。

    Returns:
        對應的 Pydantic schema 類別。

    Raises:
        ValueError: 兩個文字項目都沒要求，呼叫端不應該走到這裡。
    """
    want_keywords = AnalysisField.KEYWORDS in fields
    want_summaries = AnalysisField.SUMMARY in fields
    if want_keywords and want_summaries:
        return KeywordSummarySchema
    if want_keywords:
        return KeywordsOnlySchema
    if want_summaries:
        return SummariesOnlySchema
    raise ValueError("keywords 與 summary 至少要有一項才會呼叫文章分析")


class KeywordSummaryResult(BaseModel):
    """單一模型對整篇文章的 SEO 分析結果。

    對外的統一型別。未被要求的項目是空陣列，不是 ``None``，這樣下游不必到處
    做 null 檢查。實際送給模型的 schema 由 :func:`text_output_schema` 決定。
    """

    keywords: KeywordList = Field(
        default_factory=list, description=_KEYWORDS_DESCRIPTION
    )
    summaries: SummaryList = Field(
        default_factory=list, description=_SUMMARIES_DESCRIPTION
    )

    @classmethod
    def from_payload(cls, payload: BaseModel) -> KeywordSummaryResult:
        """把部分欄位的 schema 結果轉成統一型別。

        Args:
            payload: :func:`text_output_schema` 產生的 schema 實例。

        Returns:
            缺少的項目以空陣列補齊的結果。
        """
        return cls.model_validate(payload.model_dump())


class ImageAnalysisResult(BaseModel):
    """單一模型對單張圖片的分析結果。"""

    model_config = ConfigDict(populate_by_name=True)

    suggested_filename: str = Field(
        ...,
        validation_alias=AliasChoices(
            "suggested_filename",
            "filename",
            "file_name",
            "new_filename",
        ),
        description=(
            "SEO 友善的英文檔名主體，全小寫、單字之間用連字號、不含副檔名，"
            "3 到 6 個單字，描述圖片實際內容而非泛稱"
        ),
    )
    alt: str = Field(
        ...,
        validation_alias=AliasChoices(
            "alt",
            "alt_text",
            "altText",
            "caption",
        ),
        description=(
            "圖片的 alt 文字，長度必須符合指示中的字元上限，"
            "只寫最重要的那件事，開頭不要出現「圖片」「示意圖」「image of」這類贅詞"
        ),
    )

    @field_validator("suggested_filename")
    @classmethod
    def _normalize_filename(cls, value: str) -> str:
        """把模型給的檔名正規化成安全的 slug。"""
        return slugify(value)

    @field_validator("alt")
    @classmethod
    def _clean_alt(cls, value: str) -> str:
        """壓平換行並去除前後空白。"""
        return " ".join(value.split())


class ModelKeywordSummary(BaseModel):
    """某個模型的文章分析結果，含失敗資訊。"""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(..., description="實際呼叫的模型 id")
    result: KeywordSummaryResult | None = Field(
        default=None, description="分析成功時的結果，失敗為 null"
    )
    summary_lengths: list[int] = Field(
        default_factory=list,
        description=(
            "各摘要版本的字元數，與 result.summaries 同順序。"
            "字元含中文字、標點與空白，可直接對照 metadata.summary_max_chars"
        ),
    )
    error: str | None = Field(default=None, description="失敗原因，成功為 null")
    elapsed_ms: int = Field(default=0, description="本次呼叫耗時（毫秒）")


class ModelImageAnalysis(BaseModel):
    """某個模型對某張圖片的分析結果，含失敗資訊。"""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(..., description="實際呼叫的模型 id")
    result: ImageAnalysisResult | None = Field(
        default=None, description="分析成功時的結果，失敗為 null"
    )
    final_filename: str | None = Field(
        default=None,
        description="建議檔名接上原始副檔名後的完整檔名，apply 階段實際會用的值",
    )
    alt_length: int = Field(
        default=0,
        description="alt 的字元數，可直接對照 metadata.alt_max_chars",
    )
    error: str | None = Field(default=None, description="失敗原因，成功為 null")
    elapsed_ms: int = Field(default=0, description="本次呼叫耗時（毫秒）")


class ImageOccurrence(BaseModel):
    """圖片在 Markdown 正文中的一次引用位置。

    ``start`` / ``end`` 是相對於「去掉 front matter 之後的正文」的字元位置，
    apply 階段靠它做精準替換，避免全域字串取代誤傷同名文字。
    """

    line: int = Field(..., description="於正文中的行號，從 1 起算")
    start: int = Field(..., description="正文中的起始字元位置")
    end: int = Field(..., description="正文中的結束字元位置（不含）")
    syntax: str = Field(..., description="引用語法：markdown 或 html")
    raw: str = Field(..., description="原始的完整圖片語法字串")
    original_alt: str = Field(default="", description="原本的 alt 文字")


class ImageEntry(BaseModel):
    """一張實體圖片的完整資訊，以及各模型對它的建議。"""

    source: str = Field(..., description="Markdown 中寫的原始路徑字串")
    resolved_path: str | None = Field(
        default=None, description="推算出的本地絕對路徑，無法推算為 null"
    )
    exists: bool = Field(default=False, description="本地檔案是否存在")
    extension: str = Field(default="", description="原始副檔名，含點")
    size_bytes: int | None = Field(default=None, description="原始檔案大小")
    width: int | None = Field(default=None, description="原始圖片寬度（px）")
    height: int | None = Field(default=None, description="原始圖片高度（px）")
    skipped_reason: str | None = Field(
        default=None, description="未送進模型分析的原因，有分析則為 null"
    )
    occurrences: list[ImageOccurrence] = Field(
        default_factory=list, description="這張圖在正文中出現的所有位置"
    )
    models: dict[str, ModelImageAnalysis] = Field(
        default_factory=dict, description="依模型別名分組的分析結果"
    )


class ArticleInfo(BaseModel):
    """被分析文章的基本資訊。"""

    path: str = Field(..., description="Markdown 檔案的絕對路徑")
    title: str | None = Field(default=None, description="文章標題")
    language: str = Field(..., description="偵測到的主要語言，zh 或 en")
    word_count: int = Field(..., description="正文字數（中文計字、英文計詞）")
    image_count: int = Field(..., description="正文中引用到的相異圖片數量")
    content_hash: str = Field(
        ...,
        description=(
            "正文（不含 front matter）的 sha256 十六進位字串。"
            "分析之後文章若被編輯過，occurrences 的字元位置就會失效，"
            "review 與 apply 靠這個值察覺"
        ),
    )
    frontmatter_keys: list[str] = Field(
        default_factory=list, description="現有 front matter 的欄位名稱"
    )


class ModelUsage(BaseModel):
    """單一模型在本次執行中的資源用量。"""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(..., description="實際呼叫的模型 id")
    calls: int = Field(default=0, description="總呼叫次數")
    failed_calls: int = Field(default=0, description="失敗的呼叫次數")
    input_tokens: int = Field(default=0, description="輸入 token 總數")
    output_tokens: int = Field(default=0, description="輸出 token 總數")
    elapsed_ms: int = Field(default=0, description="累計耗時（毫秒，含平行重疊）")
    estimated_cost_usd: float | None = Field(
        default=None,
        description="以內建價目表估算的費用（美元）；模型不在價目表中時為 null",
    )
    estimated_cost_twd: float | None = Field(
        default=None,
        description="換算成新台幣的估算費用；匯率見 metadata.usd_to_twd_rate",
    )


class Metadata(BaseModel):
    """本次執行的整體資訊。"""

    generated_at: str = Field(..., description="產生時間，ISO 8601 格式")
    tool_version: str = Field(..., description="blogseo 版本")
    schema_version: str = Field(default=SCHEMA_VERSION, description="JSON 結構版本")
    total_elapsed_ms: int = Field(..., description="整體實際耗時（毫秒）")
    requested_fields: list[str] = Field(
        ...,
        description=(
            "本次要求產生的項目，keywords / summary / images 的子集。"
            "未列出的項目在結果中為空，不代表分析失敗"
        ),
    )
    alt_language: str = Field(..., description="產生 alt 文字使用的語言")
    summary_min_chars: int = Field(..., description="本次要求的摘要字元下限")
    summary_max_chars: int = Field(..., description="本次要求的摘要字元上限")
    summary_count: int = Field(..., description="本次要求的摘要版本數")
    alt_max_chars: int = Field(..., description="本次要求的 alt 字元上限")
    usd_to_twd_rate: float = Field(..., description="費用換算使用的美元兌新台幣匯率")
    total_cost_usd: float | None = Field(
        default=None, description="所有模型的估算費用合計（美元）"
    )
    total_cost_twd: float | None = Field(
        default=None, description="所有模型的估算費用合計（新台幣）"
    )
    models: dict[str, ModelUsage] = Field(
        default_factory=dict, description="依模型別名分組的用量統計"
    )


class AnalysisResult(BaseModel):
    """analyze 階段落地成 JSON 的頂層結構。"""

    schema_version: str = Field(default=SCHEMA_VERSION)
    article_info: ArticleInfo
    keyword_summary: dict[str, ModelKeywordSummary] = Field(
        default_factory=dict, description="依模型別名分組的文章分析結果"
    )
    images: dict[str, ImageEntry] = Field(
        default_factory=dict, description="依圖片路徑分組，再依模型分組"
    )
    metadata: Metadata

    def to_json(self) -> str:
        """序列化成適合寫檔的 JSON 字串。

        Returns:
            以兩空格縮排、保留非 ASCII 字元的 JSON 字串，結尾帶換行。
        """
        return self.model_dump_json(indent=2) + "\n"

    def summary_rows(self) -> list[dict[str, Any]]:
        """整理成適合終端機表格顯示的資料。

        Returns:
            每個模型一列，含關鍵字、各摘要版本與失敗計數。
        """
        rows: list[dict[str, Any]] = []
        for alias, entry in self.keyword_summary.items():
            failed = sum(
                1
                for image in self.images.values()
                if image.models.get(alias) is not None
                and image.models[alias].error is not None
            )
            rows.append(
                {
                    "alias": alias,
                    "model_id": entry.model_id,
                    "keywords": entry.result.keywords if entry.result else [],
                    "summaries": entry.result.summaries if entry.result else [],
                    "summary_lengths": entry.summary_lengths,
                    "error": entry.error,
                    "failed_images": failed,
                }
            )
        return rows
