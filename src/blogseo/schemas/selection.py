"""review 階段輸出的選擇檔結構定義。

analyze 產生的 JSON 是「多模型並列」，review 把它收斂成「單一決定」後
落地成這份 selection JSON，apply 只需要讀這一份就能完成套用。

檔案裡刻意帶上 ``article_path`` 與 ``content_hash``：apply 執行時會重新計算
文章正文的雜湊並比對，一旦文章在 analyze 之後被編輯過就會發現，避免拿失效的
字元位置去替換內容。
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

#: selection JSON 的結構版本，與 analysis JSON 的版本各自獨立演進。
#: 1.1 起檔名與 alt 是兩個獨立的決定，可以只改其中一項。
SELECTION_SCHEMA_VERSION: Final[str] = "1.1"

#: 來源欄位用這個值表示該項是使用者自己打的，不是任何模型的建議。
MANUAL_SOURCE: Final[str] = "manual"


class SelectionImage(BaseModel):
    """單張圖片的最終決定。

    檔名與 alt 分開挑選，因為兩者好壞無關：某個模型的檔名取得漂亮，alt 卻可能
    又臭又長。想保留原檔名只補 alt，或只改名不動 alt，都是常見的需求。

    ``new_filename`` 與 ``alt`` 用 ``null`` 表示「這一項不動」。alt 的空字串是
    有意義的值（裝飾性圖片就該給空 alt），因此不能用空字串代替 ``null``。
    """

    model_config = ConfigDict(protected_namespaces=())

    source: str = Field(..., description="Markdown 中寫的原始路徑字串")
    resolved_path: str = Field(..., description="圖片目前的本地絕對路徑")
    new_filename: str | None = Field(
        default=None, description="要改成的檔名，含副檔名；null 表示保持原檔名"
    )
    filename_model: str | None = Field(
        default=None,
        description=(
            f"檔名來自哪個模型別名；使用者自行輸入為 {MANUAL_SOURCE}，"
            "不改名為 null"
        ),
    )
    alt: str | None = Field(
        default=None, description="要寫入的 alt 文字；null 表示不動原本的 alt"
    )
    alt_model: str | None = Field(
        default=None,
        description=(
            f"alt 來自哪個模型別名；使用者自行輸入為 {MANUAL_SOURCE}，"
            "不改 alt 為 null"
        ),
    )

    @property
    def is_empty(self) -> bool:
        """檔名與 alt 都不動時為 ``True``，這種條目不需要寫進選擇檔。"""
        return self.new_filename is None and self.alt is None


class Selection(BaseModel):
    """review 階段落地成 JSON 的頂層結構。"""

    schema_version: str = Field(default=SELECTION_SCHEMA_VERSION)
    generated_at: str = Field(..., description="產生時間，ISO 8601 格式")
    tool_version: str = Field(..., description="blogseo 版本")
    source_analysis: str = Field(..., description="來源 analysis JSON 的絕對路徑")
    article_path: str = Field(..., description="Markdown 檔案的絕對路徑")
    content_hash: str = Field(
        ...,
        description="analyze 當下正文的 sha256，apply 套用前必須重新比對",
    )
    keywords: list[str] = Field(
        default_factory=list, description="最終採用的關鍵字；未挑選為空陣列"
    )
    summary: str = Field(default="", description="最終採用的摘要；未挑選為空字串")
    images: dict[str, SelectionImage] = Field(
        default_factory=dict,
        description=(
            "依圖片 key 分組的最終決定；完全不動的圖片不會出現在這裡"
        ),
    )

    def to_json(self) -> str:
        """序列化成適合寫檔的 JSON 字串。

        Returns:
            以兩空格縮排、保留非 ASCII 字元的 JSON 字串，結尾帶換行。
        """
        return self.model_dump_json(indent=2) + "\n"
