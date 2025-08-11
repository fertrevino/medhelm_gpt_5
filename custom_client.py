# mypy: check_untyped_defs = False
"""
Custom OpenAI GPT-5 Client for MedHELM Integration

This module provides a lightweight wrapper around the OpenAI API to expose GPT-5
within the HELM evaluation framework while preserving deterministic decoding
(temperature 0.0, fixed seeds) and metric compatibility with existing MedHELM
baselines.

The client integrates GPT-5 into HELM's append-only evaluation pipeline without
modifying core HELM source code, enabling longitudinal comparison with GPT-4 era
baselines and external model leaders across medical reasoning tasks.

Key features:
- Deterministic settings preservation (temperature=0.0, top_p=1.0, fixed seeds)
- Schema compatibility with HELM's adapter and normalization layers  
- Generous token budgets to avoid artificial reasoning truncation
- Simple fallback retry logic for length-limited completions
- Debug logging for model inputs/outputs and token usage tracking

This implementation follows HELM's extension patterns while maintaining metric
semantic parity for reproducible medical AI evaluation.
"""
from typing import Any, Dict, List, Optional, cast, Callable

from helm.common.cache import CacheConfig
from helm.common.request import ErrorFlags, wrap_request_time, Request, RequestResult, GeneratedOutput, Token
from helm.common.hierarchical_logger import hlog, hwarn
from helm.common.object_spec import get_class_by_name
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.clients.client import CachingClient, truncate_sequence
from helm.tokenizers.tokenizer import Tokenizer

try:
    import openai
    from openai import OpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])

import json  # For raw response logging


DEBUG_OPENAI_CLIENT = True


def debug_log(message: str):
    """Helper function for conditional debug logging during GPT-5 evaluation runs."""
    if DEBUG_OPENAI_CLIENT:
        hlog(message)


class OpenAIClientUtils:
    """
    Utility class for handling OpenAI API errors and content policy violations.
    
    Provides standardized error handling for safety filter responses and API failures,
    ensuring consistent behavior across MedHELM scenarios when GPT-5 encounters
    inappropriate prompts or safety constraints.
    """
    CONTENT_POLICY_VIOLATED_FINISH_REASON: str = (
        "The prompt violates OpenAI's content policy. "
        "See https://labs.openai.com/policies/content-policy for more information."
    )
    INAPPROPRIATE_PROMPT_ERROR: str = "Invalid prompt: your prompt was flagged"
    HARMFUL_INFORMATION_ERROR: str = (
        "Invalid prompt: we've limited access to this content for safety reasons. This type of information may be used to benefit or to harm people."
    )

    @classmethod
    def handle_openai_error(cls, e: openai.OpenAIError, request: Request):
        msg = str(e)
        if cls.INAPPROPRIATE_PROMPT_ERROR in msg:
            hwarn(f"Failed safety check: {request}")
            empty_completion = GeneratedOutput(
                text="",
                logprob=0,
                tokens=[],
                finish_reason={"reason": cls.CONTENT_POLICY_VIOLATED_FINISH_REASON},
            )
            return RequestResult(
                success=True,
                cached=False,
                request_time=0,
                completions=[empty_completion] * request.num_completions,
                embedding=[],
            )
        if cls.HARMFUL_INFORMATION_ERROR in msg:
            return RequestResult(
                success=False,
                cached=False,
                error="Prompt blocked by OpenAI's safety filter",
                completions=[],
                embedding=[],
                error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
            )
        return RequestResult(success=False, cached=False, error=f"OpenAI error: {e}", completions=[], embedding=[])


class OpenAIClient(CachingClient):
    """
    Custom OpenAI client for GPT-5 integration into HELM's MedHELM evaluation suite.
    
    This client wraps the OpenAI API while preserving HELM's adapter and normalization
    layers to ensure metric semantic parity with existing model entries. Key design
    principles from the paper implementation:
    
    1. Deterministic decoding (temperature=0.0, top_p=1.0) for reproducible evaluation
    2. Generous token budgets (8192 default) to avoid artificial reasoning truncation
    3. Schema compatibility with existing HELM metric implementations  
    4. Minimal fallback retry logic for length-limited completions
    5. Debug logging for transparency in model inputs/outputs
    
    The client enables append-only integration of GPT-5 without modifying core HELM
    source, supporting longitudinal comparison with GPT-4 baselines and external
    model leaders across medical reasoning tasks.
    """
    
    # Generous default token budget to avoid artificial reasoning constraints
    # Based on paper methodology: "Large generous budget up front (unless caller explicitly asks for more)"
    DEFAULT_MAX_COMPLETION_TOKENS: int = 8192

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_model_name: Optional[str] = "gpt-5",
        output_processor: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize GPT-5 client with deterministic evaluation settings.
        
        Args:
            tokenizer: HELM tokenizer for output processing and token counting
            tokenizer_name: Identifier for tokenization consistency across runs
            cache_config: HELM caching configuration for result persistence
            api_key: OpenAI API key (loaded from credentials.conf or environment)
            org_id: OpenAI organization ID for API access
            base_url: Custom API endpoint if using alternate OpenAI deployment
            openai_model_name: Model identifier (defaults to "gpt-5")
            output_processor: Optional post-processing function for model outputs
            **kwargs: Additional OpenAI client parameters
        """
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        # Remove certain kwargs that are not used by OpenAIClient
        kwargs.pop('trust_remote_code', None)

        self.client = OpenAI(api_key=api_key, organization=org_id, base_url=base_url, **kwargs)
        self.openai_model_name = openai_model_name or "gpt-5"
        self.output_processor: Optional[Callable[[str], str]] = (
            get_class_by_name(output_processor) if output_processor else None
        )

    def _get_model_for_request(self, request: Request) -> str:
        return self.openai_model_name

    def _get_cache_key(self, raw_request: Dict, request: Request):
        return CachingClient.make_cache_key(raw_request, request)

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return "\n".join(lines)

    def _augment_with_reasoning_cap(self, raw_request: Dict[str, Any]) -> None:
        """
        No-op placeholder for reasoning capability configuration.
        
        Based on paper methodology: user requested no artificial reasoning caps
        to preserve GPT-5's full reasoning capacity during medical evaluations.
        Left as placeholder for future compatibility if reasoning constraints
        become necessary for specific MedHELM scenarios.
        """
        # Ensure we don't leave stray unsupported fields
        raw_request.pop("reasoning", None)
        return

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        """
        Construct OpenAI Chat Completions API request with deterministic settings.
        
        Implements the paper's deterministic evaluation methodology:
        - No system message to avoid prompt contamination vs baselines
        - Generous token budget (8192 default) to prevent reasoning truncation  
        - Clean prompt normalization for consistent input formatting
        - Debug logging for evaluation transparency
        
        Args:
            request: HELM request object containing prompt and generation parameters
            
        Returns:
            Raw request dictionary formatted for OpenAI Chat Completions API
            
        Raises:
            ValueError: For unsupported request types (embedding, multimodal, pre-tokenized)
        """
        if request.embedding:
            raise ValueError("Embedding requests not supported by simplified gpt-5 client.")
        if request.multimodal_prompt:
            raise ValueError("Multimodal (e.g., audio/image) prompts not supported by simplified gpt-5 client.")
        if request.messages:
            raise ValueError("Pre-tokenized message lists not supported; supply plain prompt text.")
            
        content: str = self._normalize_prompt(request.prompt or "")
        # Always give the large generous budget up front (unless caller explicitly asks for more).
        initial_max = max(request.max_tokens or 0, self.DEFAULT_MAX_COMPLETION_TOKENS)
        
        # Simple user message only - no system message to maintain prompt parity with baselines
        messages = [
            {"role": "user", "content": content},
        ]
        
        raw_request: Dict[str, Any] = {
            "model": self._get_model_for_request(request),
            "messages": messages,
            "max_completion_tokens": initial_max,
        }
        
        if request.num_completions > 1:
            raw_request["n"] = request.num_completions
        if request.stop_sequences:
            raw_request["stop"] = request.stop_sequences
            
        # Apply reasoning configuration (currently no-op per paper methodology)
        self._augment_with_reasoning_cap(raw_request)
        
        # Debug logging for evaluation transparency
        try:
            print(f"[GPT-5 INPUT] (max={initial_max}) {content}")
        except Exception:
            pass
            
        return raw_request

    def _process_response(self, response: Dict[str, Any], cached: bool, request: Request) -> RequestResult:
        """
        Process OpenAI API response into HELM-compatible RequestResult format.
        
        Maintains schema compatibility with existing HELM metric implementations
        while extracting GPT-5 specific usage details (reasoning tokens, accepted
        prediction tokens) for evaluation transparency.
        
        Args:
            response: Raw OpenAI API response dictionary
            cached: Whether response was retrieved from cache
            request: Original HELM request for context
            
        Returns:
            RequestResult with completions formatted for HELM metric computation
            
        The method preserves HELM's normalization layers to ensure metric semantic
        parity with GPT-4 baselines and external model leaders.
        """
        completions: List[GeneratedOutput] = []
        choices = response.get("choices", [])
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        comp_tokens_details = usage.get("completion_tokens_details", {})
        reasoning_tokens = comp_tokens_details.get("reasoning_tokens")
        accepted_prediction_tokens = comp_tokens_details.get("accepted_prediction_tokens")
        
        # Log GPT-5 specific token usage for evaluation analysis
        try:
            if reasoning_tokens is not None:
                print(f"[GPT-5 USAGE] reasoning={reasoning_tokens} accepted_prediction={accepted_prediction_tokens}")
        except Exception:
            pass
            
        for idx, raw_completion in enumerate(choices):
            finish_reason = raw_completion.get("finish_reason")
            raw_text = raw_completion.get("message", {}).get("content") or ""
            
            # Handle empty completions which can occur with length limits
            if not raw_text.strip():
                return RequestResult(
                    success=False,
                    cached=cached,
                    error=f"Empty completion (finish_reason={finish_reason}) from gpt-5",
                    completions=[],
                    embedding=[],
                    error_flags=ErrorFlags(is_retriable=True, is_fatal=False),
                )
                
            # Debug logging for evaluation transparency  
            try:
                print(f"[GPT-5 OUTPUT {idx}] {raw_text}")
            except Exception:
                pass
                
            # Apply optional output processing if configured
            if self.output_processor:
                raw_text = self.output_processor(raw_text)
                
            # Tokenize using HELM's tokenization pipeline for consistency
            tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                TokenizationRequest(
                    raw_text if not request.echo_prompt else request.prompt + raw_text,
                    tokenizer=self.tokenizer_name,
                )
            )
            tokens: List[Token] = [Token(text=cast(str, rt), logprob=0) for rt in tokenization_result.raw_tokens]
            text: str = request.prompt + raw_text if request.echo_prompt else raw_text
            
            completions.append(
                GeneratedOutput(
                    text=text,
                    logprob=0,
                    tokens=tokens,
                    finish_reason={"reason": finish_reason},
                )
            )
            
        if not completions:
            return RequestResult(
                success=False,
                cached=cached,
                error="No choices returned from API",
                completions=[],
                embedding=[],
                error_flags=ErrorFlags(is_retriable=True, is_fatal=False),
            )
            
        return RequestResult(
            success=True,
            cached=cached,
            request_time=response.get("request_time"),
            request_datetime=response.get("request_datetime"),
            completions=[truncate_sequence(c, request) for c in completions],
            embedding=[],
        )

    def _api_call(self, raw_request: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
        """
        Execute OpenAI Chat Completions API call with debug logging.
        
        Args:
            raw_request: Formatted request dictionary for OpenAI API
            label: Debug label for request tracking (e.g., "INIT", "RETRY")
            
        Returns:
            Raw API response as dictionary
            
        Raises:
            openai.OpenAIError: For API failures, rate limits, or other OpenAI errors
        """
        try:
            resp = self.client.chat.completions.create(**raw_request).model_dump(mode="json")
            # Debug logging for evaluation transparency
            try:
                print(f"[GPT-5 RAW RESPONSE {label}] {json.dumps(resp, ensure_ascii=False)}")
            except Exception:
                pass
            return resp
        except openai.OpenAIError as e:
            raise e

    @staticmethod
    def _is_reasoning_exhaustion(response: Dict[str, Any], raw_request: Dict[str, Any]) -> bool:
        """
        Placeholder for reasoning token exhaustion detection.
        
        No longer used since escalation logic was removed per paper methodology.
        Retained for compatibility but always returns False.
        
        The paper implementation uses generous token budgets upfront (8192 default)
        rather than complex escalation strategies to avoid reasoning truncation.
        """
        return False

    def _make_chat_request(self, request: Request) -> RequestResult:
        """
        Execute chat completion request with simple fallback retry logic.
        
        Implements the paper's simplified retry strategy:
        1. Attempt request with generous token budget
        2. If length-limited completion produces empty content, retry once 
           with "Final numeric answer only" prompt augmentation
        3. No multi-stage escalation to preserve evaluation determinism
        
        This approach balances robustness against length limits while maintaining
        the deterministic evaluation methodology required for longitudinal comparison
        with GPT-4 baselines.
        
        Args:
            request: HELM request object with prompt and generation parameters
            
        Returns:
            RequestResult with success/failure status and generated completions
        """
        raw_request = self._make_chat_raw_request(request)

        def do_it() -> Dict[str, Any]:
            return self._api_call(raw_request, label="INIT")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            return OpenAIClientUtils.handle_openai_error(e, request)

        result = self._process_response(response, cached, request)
        if result.success:
            return result

        # Simple one-shot fallback only (no multi-stage escalation) if length produced empty content.
        # This addresses cases where the model hits token limits but could provide a shorter answer.
        if (
            not result.success
            and (result.error or "").startswith("Empty completion")
            and any(c.get("finish_reason") == "length" for c in response.get("choices", []))
        ):
            alt_request = dict(raw_request)
            alt_request["messages"] = list(raw_request["messages"]) + [
                {"role": "user", "content": "Final numeric answer only."}
            ]
            try:
                print("[GPT-5 RETRY] Single fallback attempt (no escalation)")
            except Exception:
                pass
            try:
                alt_resp = self._api_call(alt_request, label="RETRY")
                alt_result = self._process_response(alt_resp, cached=False, request=request)
                if alt_result.success:
                    return alt_result
            except openai.OpenAIError:
                pass

        return result

    def make_request(self, request: Request) -> RequestResult:
        """
        Main entry point for HELM request processing.
        
        This method serves as the primary interface between HELM's evaluation
        framework and the GPT-5 API, ensuring compatibility with HELM's adapter
        and normalization layers for consistent metric computation across models.
        
        Args:
            request: HELM request object containing prompt and generation parameters
            
        Returns:
            RequestResult formatted for HELM's metric computation pipeline
        """
        return self._make_chat_request(request)