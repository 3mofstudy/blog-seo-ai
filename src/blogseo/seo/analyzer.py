"""analyze 階段的流程調度。

負責把解析結果、圖片前處理與多個 provider 串起來，並確保：

* 圖片只前處理一次，所有模型共用同一份位元組，不重複縮圖。
* 任何單一模型、單一圖片的失敗都只寫進該格的 ``error`` 欄位，不影響其他呼叫。
* 全程不修改使用者的原始檔案。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

from blogseo import __version__
from blogseo.config import (
    ALT_MAX_CHARS,
    CONTEXT_RADIUS,
    KEYWORD_COUNT,
    SUMMARY_MAX_CHARS,
    SUMMARY_MIN_CHARS,
    SUMMARY_VARIANT_COUNT,
    USD_TO_TWD_RATE,
)
from blogseo.errors import BlogSeoError, ConfigError, ProviderError
from blogseo.image.analyzer import PreparedImage, prepare_image
from blogseo.image.renamer import build_filename
from blogseo.llm.base import BaseProvider
from blogseo.llm.registry import create_provider
from blogseo.markdown.parser import ImageTarget, ParsedArticle, extract_context
from blogseo.schemas.result import (
    DEFAULT_FIELDS,
    SCHEMA_VERSION,
    AnalysisField,
    AnalysisResult,
    ArticleInfo,
    ImageEntry,
    ImageOccurrence,
    Metadata,
    ModelImageAnalysis,
    ModelKeywordSummary,
    ModelUsage,
)
from blogseo.settings import AppSettings

#: 進度回呼：(已完成數, 總數, 說明文字)。
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class AnalyzeOptions:
    """analyze 的執行選項。

    Attributes:
        model_tokens: 要使用的模型別名清單，可含 ``別名:模型id`` 覆寫。
            未另外指定 ``text_tokens`` / ``image_tokens`` 時，文字與圖片都用這份。
        fields: 這次要產生的項目。
        text_tokens: 關鍵字／摘要專用的模型；``None`` 表示沿用 ``model_tokens``。
        image_tokens: 圖片分析專用的模型；``None`` 表示沿用 ``model_tokens``。
        alt_language: alt 使用的語言，``auto`` 表示跟隨文章語言。
        keyword_count: 每個模型要產生幾個關鍵字。
        summary_count: 每個模型要產生幾個不同角度的摘要版本。
        summary_min_chars: 每則摘要的字元下限。
        summary_max_chars: 每則摘要的字元上限。
        alt_max_chars: alt 文字的字元上限。
        max_images: 最多分析幾張圖，``None`` 為不限。
        concurrency: 平行呼叫上限。
        context_radius: 擷取圖片上下文的字元半徑。
        timeout: 單次 API 請求逾時秒數。
        usd_to_twd_rate: 費用換算成新台幣使用的匯率。
    """

    model_tokens: list[str]
    fields: frozenset[AnalysisField] = DEFAULT_FIELDS
    text_tokens: list[str] | None = None
    image_tokens: list[str] | None = None
    alt_language: str = "auto"
    keyword_count: int = KEYWORD_COUNT
    summary_count: int = SUMMARY_VARIANT_COUNT
    summary_min_chars: int = SUMMARY_MIN_CHARS
    summary_max_chars: int = SUMMARY_MAX_CHARS
    alt_max_chars: int = ALT_MAX_CHARS
    max_images: int | None = None
    concurrency: int = 4
    context_radius: int = CONTEXT_RADIUS
    timeout: float = 90.0
    usd_to_twd_rate: float = USD_TO_TWD_RATE

    @property
    def wants_text(self) -> bool:
        """是否需要呼叫文章分析。"""
        return bool(self.fields & {AnalysisField.KEYWORDS, AnalysisField.SUMMARY})

    @property
    def wants_images(self) -> bool:
        """是否需要呼叫圖片分析。"""
        return AnalysisField.IMAGES in self.fields

    def tokens_for_text(self) -> list[str]:
        """關鍵字／摘要實際使用的模型別名。"""
        if self.text_tokens is not None:
            return self.text_tokens
        return self.model_tokens

    def tokens_for_image(self) -> list[str]:
        """圖片分析實際使用的模型別名。"""
        if self.image_tokens is not None:
            return self.image_tokens
        return self.model_tokens


def parse_fields(raw: str) -> frozenset[AnalysisField]:
    """把 CLI 的 ``--fields`` 參數解析成項目集合。

    Args:
        raw: 以逗號分隔的項目名稱。

    Returns:
        要產生的項目集合。

    Raises:
        ConfigError: 出現無效項目，或一個項目都沒指定。
    """
    valid = {field.value for field in AnalysisField}
    selected: set[AnalysisField] = set()
    invalid: list[str] = []

    for piece in raw.split(","):
        name = piece.strip().lower()
        if not name:
            continue
        if name in valid:
            selected.add(AnalysisField(name))
        else:
            invalid.append(name)

    if invalid:
        raise ConfigError(
            f"無效的 --fields 項目：{'、'.join(invalid)}；可用的有：{'、'.join(sorted(valid))}"
        )
    if not selected:
        raise ConfigError(
            f"--fields 至少要指定一個項目，可用的有：{'、'.join(sorted(valid))}"
        )
    return frozenset(selected)


def options_from_settings(
    settings: AppSettings, *, fields: frozenset[AnalysisField]
) -> AnalyzeOptions:
    """依 ``blogseo.json`` 組出一次分析的選項。

    Args:
        settings: 使用者設定。
        fields: 這次要產生的項目。

    Returns:
        可交給 :func:`analyze_article` 的選項。
    """
    from blogseo.llm.registry import parse_model_tokens

    text_tokens = parse_model_tokens(settings.models.text)
    image_tokens = parse_model_tokens(settings.models.image)
    return AnalyzeOptions(
        model_tokens=list(dict.fromkeys([*text_tokens, *image_tokens])),
        fields=fields,
        text_tokens=text_tokens,
        image_tokens=image_tokens,
        alt_language=settings.alt.language,
        keyword_count=settings.keywords.count,
        summary_count=settings.summary.count,
        summary_min_chars=settings.summary.min_chars,
        summary_max_chars=settings.summary.max_chars,
        alt_max_chars=settings.alt.max_chars,
        max_images=settings.analyze.max_images,
        concurrency=settings.analyze.concurrency,
        context_radius=settings.analyze.context_radius,
        timeout=settings.analyze.timeout_seconds,
        usd_to_twd_rate=settings.cost.usd_to_twd_rate,
    )


@dataclass
class _ImageJob:
    """一張待分析圖片的準備結果。"""

    key: str
    target: ImageTarget
    entry: ImageEntry
    prepared: PreparedImage | None = None
    context: str = ""


@dataclass
class _ProviderSlot:
    """一個模型別名對應的 provider 實例或建立失敗的原因。"""

    token: str
    provider: BaseProvider | None
    model_id: str
    error: str | None = None
    run_text: bool = True
    run_image: bool = True


def _display_model_id(provider: BaseProvider) -> str:
    """用量表格顯示的模型 id；文字與圖片用不同模型時兩者都列出。"""
    if provider.text_model == provider.image_model:
        return provider.model
    return f"{provider.text_model} + {provider.image_model}"


def _build_image_entries(
    article: ParsedArticle,
    options: AnalyzeOptions,
) -> tuple[dict[str, ImageEntry], list[_ImageJob]]:
    """建立圖片的 JSON 條目，並挑出真正要送進模型的圖片。

    Args:
        article: 已解析的文章。
        options: 執行選項。

    Returns:
        圖片條目字典，以及待分析的工作清單。
    """
    entries: dict[str, ImageEntry] = {}
    jobs: list[_ImageJob] = []
    analyzable_count = 0

    for target in article.targets:
        entry = ImageEntry(
            source=target.source,
            resolved_path=str(target.resolved_path) if target.resolved_path else None,
            exists=target.exists,
            extension=target.extension,
            skipped_reason=target.skipped_reason,
            occurrences=[
                ImageOccurrence(
                    line=ref.line,
                    start=ref.start,
                    end=ref.end,
                    syntax=ref.syntax,
                    raw=ref.raw,
                    original_alt=ref.alt,
                )
                for ref in target.references
            ],
        )
        entries[target.key] = entry

        if not target.analyzable:
            continue

        if not options.wants_images:
            entry.skipped_reason = "本次 --fields 未包含 images"
            continue

        if options.max_images is not None and analyzable_count >= options.max_images:
            entry.skipped_reason = f"超過 --max-images 上限（{options.max_images}）"
            continue

        assert target.resolved_path is not None
        try:
            prepared = prepare_image(target.resolved_path)
        except BlogSeoError as exc:
            entry.skipped_reason = str(exc)
            continue

        entry.size_bytes = prepared.original_size_bytes
        entry.width = prepared.original_width
        entry.height = prepared.original_height
        analyzable_count += 1
        jobs.append(
            _ImageJob(
                key=target.key,
                target=target,
                entry=entry,
                prepared=prepared,
                context=extract_context(article, target, options.context_radius),
            )
        )

    return entries, jobs


def _build_slots(article: ParsedArticle, options: AnalyzeOptions) -> list[_ProviderSlot]:
    """建立每個模型別名對應的 provider。

    建立失敗（例如金鑰缺漏）不會中斷流程，而是記成該別名的錯誤。
    文字與圖片可以使用不同的模型；同一個 token 只會建立一個實例。

    Args:
        article: 已解析的文章，用來決定語言。
        options: 執行選項。

    Returns:
        provider slot 清單。

    Raises:
        ConfigError: 所有指定的模型都無法建立。
    """
    alt_language = (
        article.language if options.alt_language == "auto" else options.alt_language
    )
    text_tokens = options.tokens_for_text() if options.wants_text else []
    image_tokens = options.tokens_for_image() if options.wants_images else []
    ordered = list(dict.fromkeys([*text_tokens, *image_tokens]))
    slots: list[_ProviderSlot] = []

    for token in ordered:
        run_text = token in text_tokens
        run_image = token in image_tokens
        if run_text and run_image:
            role = "both"
        elif run_image:
            role = "image"
        else:
            role = "text"
        try:
            provider = create_provider(
                token,
                role=role,
                language=article.language,
                alt_language=alt_language,
                fields=options.fields,
                keyword_count=options.keyword_count,
                summary_count=options.summary_count,
                summary_min_chars=options.summary_min_chars,
                summary_max_chars=options.summary_max_chars,
                alt_max_chars=options.alt_max_chars,
                timeout=options.timeout,
            )
        except BlogSeoError as exc:
            slots.append(
                _ProviderSlot(
                    token=token,
                    provider=None,
                    model_id="",
                    error=str(exc),
                    run_text=run_text,
                    run_image=run_image,
                )
            )
            continue
        slots.append(
            _ProviderSlot(
                token=token,
                provider=provider,
                model_id=_display_model_id(provider),
                run_text=run_text,
                run_image=run_image,
            )
        )

    if all(slot.provider is None for slot in slots):
        reasons = "；".join(f"{slot.token}: {slot.error}" for slot in slots)
        raise ConfigError(f"沒有任何可用的模型 —— {reasons}")

    return slots


def _run_text_job(slot: _ProviderSlot, article: ParsedArticle) -> ModelKeywordSummary:
    """對單一模型執行文章分析。

    Args:
        slot: provider slot。
        article: 已解析的文章。

    Returns:
        分析結果或錯誤資訊。
    """
    assert slot.provider is not None
    started = time.perf_counter()
    try:
        result = slot.provider.analyze_text(article.content)
    except ProviderError as exc:
        return ModelKeywordSummary(
            model_id=slot.provider.text_model,
            error=str(exc),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    return ModelKeywordSummary(
        model_id=slot.provider.text_model,
        result=result,
        summary_lengths=[len(text) for text in result.summaries],
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def _run_image_job(slot: _ProviderSlot, job: _ImageJob) -> ModelImageAnalysis:
    """對單一模型、單一圖片執行分析。

    Args:
        slot: provider slot。
        job: 圖片工作。

    Returns:
        分析結果或錯誤資訊。
    """
    assert slot.provider is not None
    assert job.prepared is not None
    started = time.perf_counter()
    try:
        result = slot.provider.analyze_image(
            job.prepared.data,
            job.context,
            media_type=job.prepared.media_type,
            original_filename=job.key,
        )
    except ProviderError as exc:
        return ModelImageAnalysis(
            model_id=slot.provider.image_model,
            error=str(exc),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    return ModelImageAnalysis(
        model_id=slot.provider.image_model,
        result=result,
        final_filename=build_filename(result.suggested_filename, job.target.extension),
        alt_length=len(result.alt),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def analyze_article(
    article: ParsedArticle,
    options: AnalyzeOptions,
    *,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    """對一篇文章跑完所有指定模型的分析。

    Args:
        article: 已解析的文章。
        options: 執行選項。
        progress: 每完成一個呼叫時的進度回呼。

    Returns:
        可直接落地成 JSON 的分析結果。

    Raises:
        ConfigError: 所有指定的模型都無法建立。
    """
    overall_started = time.perf_counter()
    slots = _build_slots(article, options)
    entries, image_jobs = _build_image_entries(article, options)

    keyword_summary: dict[str, ModelKeywordSummary] = {}
    usable_slots = [slot for slot in slots if slot.provider is not None]

    for slot in slots:
        if slot.provider is None:
            placeholder = slot.model_id or "（未建立）"
            if slot.run_text:
                keyword_summary[slot.token] = ModelKeywordSummary(
                    model_id=placeholder, error=slot.error
                )
            if slot.run_image:
                for entry in entries.values():
                    if entry.skipped_reason is None:
                        entry.models[slot.token] = ModelImageAnalysis(
                            model_id=placeholder, error=slot.error
                        )

    total_jobs = sum(
        (1 if slot.run_text else 0) + (len(image_jobs) if slot.run_image else 0)
        for slot in usable_slots
    )
    completed = 0

    def report(label: str) -> None:
        nonlocal completed
        completed += 1
        if progress is not None:
            progress(completed, total_jobs, label)

    if total_jobs:
        with ThreadPoolExecutor(max_workers=max(1, options.concurrency)) as pool:
            futures: dict[object, tuple[str, str, _ImageJob | None]] = {}

            for slot in usable_slots:
                if slot.run_text:
                    future = pool.submit(_run_text_job, slot, article)
                    futures[future] = (slot.token, "text", None)
                if slot.run_image:
                    for job in image_jobs:
                        image_future = pool.submit(_run_image_job, slot, job)
                        futures[image_future] = (slot.token, "image", job)

            for future in as_completed(futures):
                token, kind, job = futures[future]
                outcome = future.result()
                if kind == "text":
                    assert isinstance(outcome, ModelKeywordSummary)
                    keyword_summary[token] = outcome
                    report(f"{token} 文章分析")
                else:
                    assert job is not None
                    assert isinstance(outcome, ModelImageAnalysis)
                    job.entry.models[token] = outcome
                    report(f"{token} {job.key}")

    usage: dict[str, ModelUsage] = {}
    known_costs: list[float] = []
    for slot in usable_slots:
        assert slot.provider is not None
        stats = slot.provider.usage
        cost_usd = slot.provider.estimate_cost_usd()
        if cost_usd is not None:
            known_costs.append(cost_usd)
        usage[slot.token] = ModelUsage(
            model_id=slot.model_id,
            calls=stats.calls,
            failed_calls=stats.failed_calls,
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
            elapsed_ms=stats.elapsed_ms,
            estimated_cost_usd=cost_usd,
            estimated_cost_twd=(
                None if cost_usd is None else cost_usd * options.usd_to_twd_rate
            ),
        )

    total_cost_usd = sum(known_costs) if known_costs else None
    alt_language = (
        article.language if options.alt_language == "auto" else options.alt_language
    )
    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        article_info=ArticleInfo(
            path=str(article.path),
            title=article.title,
            language=article.language,
            word_count=article.word_count,
            image_count=len(article.targets),
            content_hash=article.content_hash,
            frontmatter_keys=sorted(article.metadata),
        ),
        keyword_summary=keyword_summary,
        images=entries,
        metadata=Metadata(
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            tool_version=__version__,
            schema_version=SCHEMA_VERSION,
            total_elapsed_ms=int((time.perf_counter() - overall_started) * 1000),
            requested_fields=sorted(field.value for field in options.fields),
            alt_language=alt_language,
            summary_min_chars=options.summary_min_chars,
            summary_max_chars=options.summary_max_chars,
            summary_count=options.summary_count,
            alt_max_chars=options.alt_max_chars,
            usd_to_twd_rate=options.usd_to_twd_rate,
            total_cost_usd=total_cost_usd,
            total_cost_twd=(
                None if total_cost_usd is None else total_cost_usd * options.usd_to_twd_rate
            ),
            models=usage,
        ),
    )
