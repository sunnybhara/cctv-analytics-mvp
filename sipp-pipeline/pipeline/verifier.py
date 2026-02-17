"""
Claude Vision API Verifier
===========================
Sends keyframes to Claude Vision for action classification.
"""

import base64
import json
import logging

import cv2

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, STRICT_MODE

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None and ANTHROPIC_API_KEY:
        import anthropic
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """You analyze bar CCTV footage to classify bartender and customer actions.
Return ONLY valid JSON with no markdown fences or extra text.

Required JSON schema:
{
    "action": "pouring_draft|pouring_bottle|mixing_cocktail|serving_customer|payment_card|payment_cash|cleaning|idle|unknown",
    "confidence": 0.0-1.0,
    "drink_category": "beer|wine|spirits|cocktail|soft_drink|unknown",
    "vessel_type": "pint_glass|wine_glass|rocks_glass|shot_glass|bottle|can|shaker|unknown",
    "description": "brief description of what is happening"
}"""

STRICT_MODE_ADDENDUM = """
Note: Faces have been blurred for privacy. Use body language, arm movement, and object interaction to determine the action. Do not flag blurred faces as suspicious."""


def verify_event(frames: list, event_context: dict) -> dict:
    """Send keyframes to Claude Vision API for action verification.

    Args:
        frames: list of numpy frames (BGR, from OpenCV)
        event_context: {"object_class": "bottle", "zone": "bar_zone", ...}

    Returns: parsed JSON dict with action, confidence, etc.
             Returns {"action": "unknown", "confidence": 0.0} on any failure.
    """
    client = _get_client()
    if not client:
        logger.warning("No ANTHROPIC_API_KEY set. Skipping VLM verification.")
        return {"action": "unknown", "confidence": 0.0, "description": "No API key"}

    if not frames:
        return {"action": "unknown", "confidence": 0.0, "description": "No frames"}

    content = []
    for frame in frames[:4]:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.standard_b64encode(buffer).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })

    prompt = (
        f"These are {len(content)} sequential frames from a bar CCTV camera. "
        f"A {event_context.get('object_class', 'object')} was detected near a person "
        f"in the {event_context.get('zone', 'bar')} zone. "
        f"Analyze the state change between the first and last frame. "
        f"Did an action occur (pour, serve, payment) or is this idle/cleaning? "
        f"Return JSON only."
    )
    content.append({"type": "text", "text": prompt})

    system = SYSTEM_PROMPT
    if STRICT_MODE:
        system += STRICT_MODE_ADDENDUM

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        raw_output = response.content[0].text
        return _safe_parse(raw_output)
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return {"action": "unknown", "confidence": 0.0, "description": str(e)}


def _safe_parse(output: str) -> dict:
    """Parse VLM JSON output with resilience to common failure modes."""
    cleaned = output.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            return {"action": "unknown", "confidence": 0.0, "description": "VLM returned non-object JSON"}
        if "action" not in result:
            result["action"] = "unknown"
        # Coerce confidence to float — VLM may return "high", null, or bool
        try:
            result["confidence"] = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            result["confidence"] = 0.0
        return result
    except (json.JSONDecodeError, TypeError):
        return {
            "action": "unknown",
            "confidence": 0.0,
            "description": "VLM parse failure",
            "raw": output[:200],
        }
