"""
notify.py – Sends deal notifications via email or stdout.

Reads a JSON payload with product, price, links, and contact info. Validates the
request using OPENAI_API_KEY from environment or .env file. If authorized, sends
an email using Resend (if configured) or prints the message to stdout.

Usage:
  python notify.py --input payload.json [--contact user@example.com] [--no-email]
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from typing import Any, List, Dict, Union


from os import environ

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()  # searches for .env in cwd (or parents) and loads vars.
except ImportError:  # pragma: no cover
    # python-dotenv isn't mandatory; user can still export vars normally.
    pass

try:
    import resend  # type: ignore
except ImportError:  # pragma: no cover
    resend = None  # Fallback to stdout if not installed.


# ---------------------------------------------------------------------------
# Helper functions ----------------------------------------------------------
# ---------------------------------------------------------------------------

def _validate_api_key(provided: str | None) -> None:
    """Ensure the caller supplied the correct shared secret."""

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set on this machine; cannot authorise caller.")

    if provided != OPENAI_API_KEY:
        raise PermissionError("API key mismatch – notification aborted.")


def _build_message_html(product: str, price: float, links: List[str]) -> str:
    """Return a simple HTML email body."""
    links_html = "<br/>".join(f'<a href="{link}">{link}</a>' for link in links)
    return f"""
        <p>🎉 <strong>Great news!</strong></p>
        <p>We found <strong>{product}</strong> at <strong>${price:.2f}</strong>.</p>
        <p>Buy here:<br/>{links_html}</p>
        <p>Reply with <em>yes</em> to confirm purchase, or <em>no</em> to dismiss.</p>
    """


def _build_message_text(product: str, price: float, links: List[str]) -> str:
    joined_links = "\n".join(links)
    return (
        f"Great news! We found {product} for ${price:.2f}.\n"
        f"Where to buy:\n{joined_links}\n\n"
        "Reply 'yes' to confirm purchase or 'no' to ignore."
    )


def _send_email(*, email_to: str, product: str, price: float, links: List[str]) -> None:
    """Dispatch an email using Resend. Falls back to stdout if unavailable."""

    api_key = os.getenv("RESEND_API_KEY")
    if not api_key or resend is None:
        raise RuntimeError("Resend not configured; cannot send email.")

    resend.api_key = api_key  # type: ignore[attr-defined]

    html_body = _build_message_html(product, price, links)
    text_body = _build_message_text(product, price, links)

    from_addr = os.getenv("RESEND_FROM_EMAIL", "Notifier <onboarding@resend.dev>")

    resend.Emails.send(  # type: ignore[attr-defined]
        {
            "from": from_addr,
            "to": email_to,
            "subject": f"Deal found: {product} for ${price:.2f}",
            "html": html_body,
            "text": text_body,
        }
    )


def _print_stdout(product: str, price: float, links: List[str], contact: str | None) -> None:
    """Simply print the message – handy for local dev & unit tests."""

    divider = "=" * 50
    print(divider)
    to_line = f"Notification TO: {contact}" if contact else "Notification (stdout only)"
    print(to_line)
    print(f"Suggested Product: {product}")
    print(f"Price: ${price:.2f}")
    print("Where to buy:")
    for link in links:
        print(f"  - {link}")
    print("\nDo you want to buy this product? Reply 'yes' or 'no' to confirm.")
    print(divider)
    print("(Notification printed – in production this would be sent by email.)\n")


def send_notification(
    *,
    product: str,
    price: float,
    links: Union[str, List[str]],
    user_contact: str | None = None,
    api_key: str | None = None,
    use_email: bool = True,
) -> bool:
    """Send a deal-found notification.

    Returns ``True`` on success, ``False`` if anything went wrong.
    """

    try:
        _validate_api_key(api_key)
        link_list = [links] if isinstance(links, str) else links

        if use_email and user_contact and "@" in user_contact:
            try:
                _send_email(email_to=user_contact, product=product, price=price, links=link_list)
                print(f"Email sent to {user_contact} (Resend).")
            except Exception as exc:
                print(f"WARN: email failed ({exc!s}). Falling back to stdout.")
                _print_stdout(product, price, link_list, user_contact)
        else:
            _print_stdout(product, price, link_list, user_contact)

        return True
    except Exception as exc:
        print(f"ERROR: {exc!s} – notification not sent.", file=sys.stderr)
        return False


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Notify user of a deal via email or stdout.")
    parser.add_argument("--input", "-i", help="Path to JSON payload file. Omit to read from stdin.")
    parser.add_argument("--contact", "-c", help="Override `user_contact` from JSON with this value.")
        # Use an explicit `dest` so the attribute is guaranteed to be `no_email`,
    # even if a non‑ASCII dash ever sneaks into the option string.
    parser.add_argument(
        "--no-email",
        dest="no_email",
        action="store_true",
        help="Force stdout even if email is configured.",
    )
    return parser.parse_args(argv)


def main() -> None:  # pragma: no cover
    args = _parse_args()

    # Read JSON payload — either from file or stdin.
    try:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as fp:
                payload: Dict[str, Any] = json.load(fp)
        else:
            payload = json.load(sys.stdin)
    except Exception as exc:
        sys.exit(f"Could not read JSON payload: {exc!s}")

    # Command-line overrides
    if args.contact:
        payload["user_contact"] = args.contact

    success = send_notification(
        product=payload.get("product"),
        price=float(payload.get("price")),
        links=payload.get("links"),
        user_contact=payload.get("user_contact"),
        api_key=payload.get("api_key"),
        use_email=not args.no_email,
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":  # pragma: no cover
    main()
